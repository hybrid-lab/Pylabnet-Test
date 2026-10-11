# Fast single image with a BIAS X SWEEP:
# MOT -> CMOT -> Dark MOT -> PGC -> release -> TOF -> image, repeated once per
# bias point. No lattice.
#
# Duplicate of jingfastPGC.py. Two things differ from it, and nothing else does.
#
#  1. The PGC value of Bias X (ao8) is swept over Start .. Stop in Steps
#     points, the way the lattice bias sweeps do it: one complete sequence per
#     point, all of them concatenated into a single buffer that the cards
#     regenerate forever. Bias Y and Bias Z keep their fixed PGC values.
#     Every point gets a column in a scan grid window -- latest frame, running
#     average, and summed profiles with a Gaussian fit.
#
#  2. The bias rails have TWO setpoints each, not one. jingfastPGC ramps them
#     from 0 V up to the PGC values at coils_off and back down to 0 V after the
#     second frame. Here the resting value is a separate MOT Bias X/Y/Z, held
#     through MOT loading and CMOT, and the ramp runs MOT value -> PGC value
#     and back again. So the field the MOT loads in and the field the PGC runs
#     in are independent: the MOT values are whatever a MOT bias sweep
#     (jingfastMOT_biasXSweep.py) found, and this script optimises the PGC
#     value on top of them without disturbing the loading.
#
#     Setting all three MOT Bias values to 0 reproduces jingfastPGC exactly.

import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import time

try:
    from scipy.optimize import curve_fit
except ImportError:      # fall back to moments; see fit_gaussian_1d
    curve_fit = None

# A 1/e^2 radius is 2 sigma, and integrating a 1/e^2-parameterised Gaussian
# gives w*sqrt(pi/2) -- both used to turn the two 1D fits back into a 2D peak.
ROOT_HALF_PI = np.sqrt(np.pi / 2.0)

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

DIFF_LEVELS = (-5, 50)

# Sensor size of the BFS-U3-51S5M. Plot2D allocates its buffer up front, so
# these must match the frames the camera actually sends -- change them if you
# ever enable ROI or binning.
FRAME_W = 2448
FRAME_H = 2048

# Key for the image plot, so configure() and experiment() agree on it.
IMAGE_NAME = 'PGC image, latest scan point (frame 1 - frame 2)'

# One name for both the image LUT and the colour bar, so they cannot disagree.
COLORMAP = 'inferno'

# Font size of the fit-results line above the image. pyqtgraph's PlotItem title
# defaults to 11pt, so this is double that. It is one long line, so if it ever
# runs wider than the image panel, drop it back rather than letting it clip.
FIT_TITLE_SIZE = '22pt'

# Requested camera pixel format. The camera remembers whatever was last set on
# it, so leaving this to chance means a run can silently come back 8-bit and
# throw away four bits of an already faint signal. "Mono12p" or "Mono12Packed"
# also work if Mono16 is unavailable on this model.
PIXEL_FORMAT = 'Mono16'

# Font size of the fit-results line above the image. pyqtgraph's PlotItem title
# defaults to 11pt, so this is double that. It is one long line, so if it ever
# runs wider than the image panel, drop it back rather than letting it clip.
FIT_TITLE_SIZE = '22pt'

# ---------------------------------------------------------------------------
# Safe state applied when the run stops.
# ---------------------------------------------------------------------------
# Without this, "stop" leaves the rails wherever the waveform happened to be.
# The AO/DO tasks run with regeneration, looping the buffer forever, and
# force_finish=True stops them mid-loop at an arbitrary sample; DAQmx then
# holds every output at that last sample when the task closes. So the coils
# could park anywhere between 0 V and the CMOT value and the VCOs at any phase
# detuning, differently on every stop.
#
# These are written as single-sample immediate writes AFTER the sequence tasks
# are closed -- they would collide with the running tasks otherwise.
SAFE_AO = {
    'ao1': 0.8,     # MOT AOM drive
    'ao5': 0.8,     # repump AOM drive
    'ao2': 0.0,     # MOT coils
    'ao3': 0.0,     # MOT VCO
    'ao4': 0.0,     # repump VCO
    'ao6': 0.0,     # lattice
    'ao7': 0.0,     # lattice
    'ao8': 0.0,     # bias X
    'ao9': 0.0,     # bias Y
    'ao10': 0.0,    # bias Z
}

# Digital lines are left in an arbitrary state by the same mechanism, so they
# are parked too. Light off is the quiescent choice -- with the coils at 0 V
# nothing is trapped either way, so leaving the beams on would only heat the
# cell. Set dio1/dio3 to 1 here if you would rather the beams stay on at the
# 0.8 V drive above for alignment.
SAFE_DO = {
    'dio0': 0,      # camera trigger
    'dio1': 0,      # MOT AOM enable -- off
    'dio2': 1,      # coils TTL: the state the sequence uses for "coils off"
    'dio3': 0,      # repump AOM enable -- off
    'dio4': 1,      # lattice off
    'dio5': 1,      # lattice off
    'dio6': 0,
}


def park_outputs(NI_do, NI_ao, logger):
    """Drive every line to its SAFE_AO / SAFE_DO value.

    Best-effort and never raises: this runs from a finally block, where an
    exception would mask whatever actually ended the run and would also skip
    the channels after the one that failed.

    write_static() is preferred because it forces the driver out of stack mode
    first -- if the run died between build_stack() and arm(), a plain
    set_ao_voltage() would be silently queued into a stack nobody will arm, and
    nothing would actually be parked. Older servers do not have it, so fall
    back to per-channel writes, each in its own try so one bad line cannot
    leave the rest at random.
    """
    try:
        NI_ao.write_static(ao=SAFE_AO)
        NI_do.write_static(do=SAFE_DO)
    except Exception as exc:
        logger.warn(f"write_static unavailable ({exc}); parking channel by channel")
        for channel, value in SAFE_AO.items():
            try:
                NI_ao.set_ao_voltage(ao_channel=channel, voltages=value)
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value} V: {e}")
        for channel, value in SAFE_DO.items():
            try:
                NI_do.set_do_voltage(do_channel=channel, value=bool(value))
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value}: {e}")
    logger.info(
        "Outputs parked: VCOs 0 V, coils 0 V, bias 0 V, AOM drive 0.8 V, "
        "all light off."
    )


# ---------------------------------------------------------------------------
# BIAS SWEEP -- the machinery the lattice bias sweeps use, brought over here.
# ---------------------------------------------------------------------------
# The sweep is done exactly the way lattice_biasXSweep_fastLattice.py does it:
# one complete sequence is built per bias point, the points are concatenated
# into a SINGLE buffer, and the cards regenerate that buffer forever. Nothing
# is re-armed between points, so the bias value steps on a sample boundary with
# the same timing jitter as every other rail -- which is none.
#
# The alternative, re-arming per point, costs a task teardown and rebuild
# between every image and lets the MOT reload for a different length of time on
# each one. That is the difference a bias sweep is trying to measure.
# Percentile the autoscaled top follows. Not the max: one hot pixel would then
# set the scale for the whole grid and push every cloud back into the dark end.
AUTOSCALE_PCT = 99.95


def bias_sweep_values(dataset):
    """The bias points to sweep, from Start / Stop / Steps on the GUI.

    Kept in one place so configure(), which sizes the display grid, and
    experiment(), which builds the waveform, cannot disagree about how many
    scan points there are or what is in them.

    Steps counts the POINTS and includes both endpoints, so 0 -> 1 V in 10
    steps is 0.000, 0.111, ... 1.000. A step count below 1 is treated as 1 (the
    start value alone) rather than dividing by zero or handing an empty list to
    a display grid with no columns in it.
    """
    start = float(dataset.get_input_parameter("bias_start_v"))
    stop = float(dataset.get_input_parameter("bias_stop_v"))
    steps = int(float(dataset.get_input_parameter("bias_steps")))
    if steps <= 1:
        return [start]
    return [float(v) for v in np.linspace(start, stop, steps)]


def autoscale_top(image, lo):
    """Top of the colour range for one image, from its own brightness."""
    hi = float(np.percentile(image, AUTOSCALE_PCT))
    if hi <= lo + 1.0:
        hi = lo + 10.0        # empty frame: keep a usable, non-degenerate range
    return hi


def _sigma_from_moment(profile):
    """Gaussian + constant offset fitted to a 1D profile, for the scan grid.

    Moment-based initial guess, then a scipy least-squares refinement. This is
    the grid's own fitter, deliberately separate from fit_gaussian_1d above:
    that one reports 1/e^2 radii and a reliability verdict for the single large
    image, this one reports the sigma and the background-subtracted trace the
    small per-tile panels draw.

    :return: (mu, sigma, background-subtracted profile), or
        (nan, nan, None) when there is nothing to fit.
    """
    n = profile.size
    if n < 10:
        return float('nan'), float('nan'), None
    y = profile.astype(np.float64, copy=True)

    baseline = np.percentile(y, 10)
    y_bg = y - baseline
    np.clip(y_bg, 0, None, out=y_bg)
    w_sum = y_bg.sum()
    if w_sum <= 0:
        return float('nan'), float('nan'), None

    x = np.arange(n, dtype=np.float64)
    mu0 = (y_bg * x).sum() / w_sum
    var0 = (y_bg * (x - mu0) ** 2).sum() / w_sum
    if var0 <= 0 or not np.isfinite(var0):
        return float('nan'), float('nan'), None
    sigma0 = np.sqrt(var0)
    amp0 = y_bg.max()

    if curve_fit is None:
        return float(mu0), float(sigma0), y_bg
    try:
        def gauss_with_offset(x, offset, amplitude, mu, sigma):
            return offset + amplitude * np.exp(
                -((x - mu) ** 2) / (2.0 * sigma ** 2))

        popt, _ = curve_fit(
            gauss_with_offset, x, profile,
            p0=[baseline, amp0, mu0, sigma0],
            bounds=([-np.inf, 0, 0, 1], [np.inf, np.inf, n, n]),
            maxfev=2000)
        fit_offset, _fit_amp, fit_mu, fit_sigma = popt
        if fit_sigma < 1 or fit_sigma > n / 2 or not np.isfinite(fit_sigma):
            # Fit ran away; the moments are the honest answer.
            return float(mu0), float(sigma0), y_bg
        y_display = profile - fit_offset
        np.clip(y_display, 0, None, out=y_display)
        return float(fit_mu), float(fit_sigma), y_display
    except Exception:
        return float(mu0), float(sigma0), y_bg


def _gaussian_curve(n, mu, sigma, amplitude):
    """(x, y) of a Gaussian, for drawing the fit over the profile."""
    x = np.arange(n, dtype=np.float64)
    y = amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return x, y


class StopRequested(Exception):
    """Raised in place of a frame when the user presses Stop.

    Unwinds straight out of the acquisition loop to the teardown, so the stop
    is acted on where it is noticed instead of every level in between having to
    test thread.running and pass a flag outwards. A sweep is a long cycle --
    n_scan sequences per pass -- so a stop that is only checked once per pass
    sits for a minute before anything happens.
    """


class ScanGridWindow:
    """One column per bias point: latest frame, running average, profiles.

    Separate popup rather than part of the main window, because the whole
    point of a sweep is seeing the points BESIDE each other, and the main
    window's image panel can only ever show one of them.

    Both rows share one colour scale across every column. A per-tile scale
    would renormalise each point to its own peak, so a column with half the
    atoms would look identical to a full one -- which is exactly the comparison
    the sweep exists to make.
    """

    MARGIN_PX = 100
    POLL_MS = 500

    def __init__(self, scan_labels, diff_levels, avg_levels,
                 title="Scan grid", tile_px=320,
                 colorbar_label='Brightness (counts, frame 1 - frame 2)'):
        self.scan_labels = list(scan_labels)
        self.n_scan = len(self.scan_labels)
        self.diff_levels = diff_levels
        self.avg_levels = avg_levels
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(tile_px * self.n_scan + self.MARGIN_PX * self.n_scan,
                        tile_px * 2 + self.MARGIN_PX + 80)
        self.current_imgs = []
        self.avg_imgs = []
        self.x_data_curves = []
        self.x_fit_curves = []
        self.y_data_curves = []
        self.y_fit_curves = []
        self.header_labels = []
        self._latest_avgs = [None] * self.n_scan
        self._rendered_versions = [0] * self.n_scan
        # Filled by the acquisition thread, applied by the timer below. Only
        # the timer runs on the GUI thread, and the colour bars are Qt.
        self._levels_hint = None
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            lbl = self.win.addLabel(
                f"<b>{self.scan_labels[col]}</b><br>"
                f"<span style='color:#FFFF00'>sx=-- sy=--</span>",
                row=0, col=base_col, size="10pt")
            self.header_labels.append(lbl)
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            vb = self.win.addViewBox(row=1, col=base_col, lockAspect=True)
            vb.invertY(True)
            img = pg.ImageItem(axisOrder='row-major')
            img.setLevels(self.diff_levels)
            img.setLookupTable(pg.colormap.get(COLORMAP).getLookupTable())
            vb.addItem(img)
            self.current_imgs.append(img)
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            y_margin_col = 2 * col
            y_plot = self.win.addPlot(row=2, col=y_margin_col)
            y_plot.setMaximumWidth(self.MARGIN_PX)
            y_plot.hideAxis('bottom')
            y_plot.hideAxis('left')
            y_plot.setMouseEnabled(x=False, y=False)
            y_plot.setMenuEnabled(False)
            y_plot.invertX(True)
            y_plot.invertY(True)
            self.y_data_curves.append(
                y_plot.plot([], [], pen=pg.mkPen('w', width=1)))
            self.y_fit_curves.append(
                y_plot.plot([], [], pen=pg.mkPen('y', width=2)))
            vb_avg = self.win.addViewBox(row=2, col=base_col, lockAspect=True)
            vb_avg.invertY(True)
            img = pg.ImageItem(axisOrder='row-major')
            img.setLevels(self.avg_levels)
            img.setLookupTable(pg.colormap.get(COLORMAP).getLookupTable())
            vb_avg.addItem(img)
            self.avg_imgs.append(img)
            y_plot.setYLink(vb_avg)
            x_plot = self.win.addPlot(row=3, col=base_col)
            x_plot.setMaximumHeight(self.MARGIN_PX)
            x_plot.hideAxis('bottom')
            x_plot.hideAxis('left')
            x_plot.setMouseEnabled(x=False, y=False)
            x_plot.setMenuEnabled(False)
            self.x_data_curves.append(
                x_plot.plot([], [], pen=pg.mkPen('w', width=1)))
            self.x_fit_curves.append(
                x_plot.plot([], [], pen=pg.mkPen('y', width=2)))
            x_plot.setXLink(vb_avg)
        # Colour bars in real camera counts, one per row of tiles. The tiles
        # carry no axis of their own, so without these nothing on screen says
        # which count value a colour means. Non-interactive: the levels are
        # driven from the acquisition side and a draggable bar would only
        # fight it.
        self.colorbar_cur = self._add_colorbar(
            self.current_imgs, self.diff_levels, row=1, col=2 * self.n_scan,
            label=colorbar_label)
        self.colorbar_avg = self._add_colorbar(
            self.avg_imgs, self.avg_levels, row=2, col=2 * self.n_scan,
            label=colorbar_label + ', averaged')
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    def _add_colorbar(self, imgs, levels, row, col, label):
        """Colour bar for one row of tiles, or None if it cannot be built.

        ColorBarItem arrived in pyqtgraph 0.12; on anything older the fallback
        is simply the picture with no scale beside it.
        """
        try:
            bar = pg.ColorBarItem(
                values=(float(levels[0]), float(levels[1])),
                colorMap=pg.colormap.get(COLORMAP),
                label=label, interactive=False)
            bar.setImageItem(imgs)
            self.win.addItem(bar, row=row, col=col)
            return bar
        except Exception:
            return None

    def set_levels_hint(self, diff_levels, avg_levels):
        """Called from the acquisition thread. Stores floats, touches no Qt.

        The tiles get their levels passed straight into setImage; this is only
        how the colour bars find out, and the timer is what applies it.
        """
        self._levels_hint = (tuple(diff_levels), tuple(avg_levels))

    def _poll_and_update(self):
        hint = self._levels_hint
        if hint is not None:
            self._levels_hint = None
            self.diff_levels, self.avg_levels = hint
            for bar, lv in ((self.colorbar_cur, self.diff_levels),
                            (self.colorbar_avg, self.avg_levels)):
                if bar is None:
                    continue
                try:
                    # Drive the bar, not the tiles: ColorBarItem.setLevels
                    # propagates to every linked ImageItem, so the tick labels
                    # and the pictures cannot drift apart.
                    bar.setLevels(low=lv[0], high=lv[1])
                except Exception:
                    pass
        for idx in range(self.n_scan):
            entry = self._latest_avgs[idx]
            if entry is None:
                continue
            avg_copy, version = entry
            if version <= self._rendered_versions[idx]:
                continue
            self._rendered_versions[idx] = version
            try:
                img = avg_copy.astype(np.float64)
                x_proj = img.sum(axis=0)
                y_proj = img.sum(axis=1)
                mu_x, sigma_x, x_bg = _sigma_from_moment(x_proj)
                if x_bg is not None and np.isfinite(sigma_x):
                    self.x_data_curves[idx].setData(
                        np.arange(x_proj.size, dtype=np.float64), x_bg)
                    gx, gy = _gaussian_curve(x_proj.size, mu_x, sigma_x,
                                             x_bg.max())
                    self.x_fit_curves[idx].setData(gx, gy)
                mu_y, sigma_y, y_bg = _sigma_from_moment(y_proj)
                if y_bg is not None and np.isfinite(sigma_y):
                    self.y_data_curves[idx].setData(
                        y_bg, np.arange(y_proj.size, dtype=np.float64))
                    gx, gy = _gaussian_curve(y_proj.size, mu_y, sigma_y,
                                             y_bg.max())
                    self.y_fit_curves[idx].setData(gy, gx)
                sx = f"{sigma_x:.1f}" if np.isfinite(sigma_x) else "--"
                sy = f"{sigma_y:.1f}" if np.isfinite(sigma_y) else "--"
                area = float(img.sum())
                self.header_labels[idx].setText(
                    f"<b>{self.scan_labels[idx]}</b><br>"
                    f"<span style='color:#FFFF00'>sx={sx} sy={sy}</span><br>"
                    f"<span style='color:#00FF88'>area={area:.0f}</span>")
            except Exception:
                pass

    def set_current(self, scan_idx, diff_image, levels=None):
        lv = self.diff_levels if levels is None else levels
        try:
            self.current_imgs[scan_idx].setImage(
                diff_image, levels=lv, autoLevels=False)
        except Exception:
            pass

    def set_average(self, scan_idx, avg_image, levels=None):
        lv = self.avg_levels if levels is None else levels
        try:
            self.avg_imgs[scan_idx].setImage(
                avg_image, levels=lv, autoLevels=False)
        except Exception:
            pass
        self._latest_avgs[scan_idx] = (
            np.asarray(avg_image).copy(), self._rendered_versions[scan_idx] + 1)


INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '800'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '6'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0'},
    'Repump_VCO': {'Repump VCO (V)': '0'},

    # ===================== PHASE 2: CMOT =====================
    # Identical to jingfastMOT, values and all -- this script's job is to add
    # Dark MOT and PGC on the END of that sequence, so anything before the
    # handover has to reproduce it exactly or the comparison means nothing.
    #
    # How long the MOT -> CMOT ramp takes: every quantity that differs between
    # the two phases (AOM powers, coil current, both VCOs) is swept linearly
    # over this window. 0 makes it an instant jump. Was hardcoded at 10 ms here
    # (CMOT_RAMP_MS) while jingfastMOT had it on the GUI, so the two could not
    # be matched without editing the file.
    'mot_switch_time': {'MOT Switch Time (ms) [MOT->CMOT ramp]': '15'},
    'cmot_hold_time': {'CMOT Hold Time (ms)': '5'},
    'cmot_mot_power': {'CMOT MOT AOM Power (V)': '0.6'},
    'cmot_repump_power': {'CMOT Repump AOM Power (V)': '0.2'},
    'cmot_detuning_voltage': {'CMOT MOT VCO (V) [-40MHz]': '-0.6'},
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '8'},
    'cmot_repump_vco': {'CMOT Repump VCO (V) [-3MHz]': '-0.2'},

    # ===================== PHASE 3: DARK MOT =====================
    'dark_mot_duration': {'Dark MOT Duration (ms)': '4'},
    'dark_mot_power': {'Dark MOT Cooling Power (V)': '0.2'},
    'dark_mot_detuning': {'Dark MOT Cooling VCO (V) [-55MHz]': '-0.6'},
    'dark_mot_repump_power': {'Dark MOT Repump Power (V)': '0.2'},
    'dark_mot_repump_vco': {'Dark MOT Repump VCO (V)': '0.0'},

    # ===================== PHASE 4: PGC =====================
    'pgc_duration': {'PGC Duration (ms)': '5'},
    'pgc_mot_power': {'PGC Cooling Power (V)': '0.2'},
    'pgc_detuning': {'PGC Cooling VCO (V) [-122MHz]': '-2.3'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.03'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '-0.05'},

    # ===================== RELEASE + IMAGING =====================
    'tof_us': {'TOF (us)': '0'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'coils_rampback_ms': {'Coils Rampback Duration (ms)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '100'},

    # ---------- Imaging repump prepulse ----------
    # Repump fires alone for this long before the MOT light joins it, so the
    # atoms are pumped up out of the lower ground state and are all cycling by
    # the time the camera opens. Both frames get the identical sequence -- the
    # subtraction only works if frame 2 saw exactly the same light as frame 1.
    # The repump then STAYS on at these settings through the MOT pulse; turning
    # it off there would let atoms fall dark part-way through the exposure.
    # Set the time to 0 to go back to repump and MOT light starting together.
    'img_repump_time_us': {'Imaging Repump Lead Time (us)': '200'},
    'img_repump_power': {'Imaging Repump Power (V)': '0.8'},
    'img_repump_vco': {'Imaging Repump VCO (V)': '0.06'},

    # ===================== BIAS COILS: MOT VALUES =====================
    # Where all three rails sit through MOT loading and CMOT, and where they
    # are returned to after the second frame. Separate from the PGC values
    # below on purpose: the field that loads the most atoms into the MOT is not
    # the field the molasses wants, so one set of numbers cannot be right for
    # both. Put the optimum from jingfastMOT_biasXSweep.py here, then sweep
    # the PGC value against it.
    #
    # All three at 0 reproduces jingfastPGC, which ramped up from 0 V and back
    # down to 0 V.
    'mot_bias_x_v': {'MOT Bias X (V)': '0.0'},
    'mot_bias_y_v': {'MOT Bias Y (V)': '0.0'},
    'mot_bias_z_v': {'MOT Bias Z (V)': '0.0'},

    # ================ BIAS X SWEEP (PGC value only) ================
    # Endpoints inclusive: Steps counts the POINTS, not the intervals, so
    # 0 -> 1 V in 10 steps is 0.000, 0.111, ... 1.000, and 1 step runs the
    # start value on its own.
    #
    # This is the value the rail ramps TO at coils_off and holds through the
    # dark MOT, the PGC and both imaging frames. MOT Bias X above is
    # unaffected, so the MOT loads identically at every point in the sweep and
    # the only thing that changes is the field the PGC runs in.
    #
    # Every point is a complete sequence -- about a second at the default MOT
    # loading time -- and adds a column to the scan grid, so a fine sweep costs
    # both time per pass and screen width. See the warning in experiment().
    'bias_start_v': {'PGC Bias X Start (V)': '0.0'},
    'bias_stop_v': {'PGC Bias X Stop (V)': '1.0'},
    'bias_steps': {'PGC Bias X Steps (points, endpoints included)': '10'},

    # The two axes that are NOT swept, at their PGC values.
    'pgc_bias_y_fixed_v': {'PGC Bias Y Fixed (V)': '0.0'},
    'pgc_bias_z_fixed_v': {'PGC Bias Z Fixed (V)': '0.0'},

    # How long each rail takes to get between its MOT value and its PGC value,
    # in both directions.
    'bias_ramp_ms': {'Bias Ramp Duration (ms)': '5'},

    # ===================== DISPLAY =====================
    # Autoscale by default (1): the top of the colour range follows the
    # signal in each frame (99.95th percentile, so one hot pixel cannot set
    # it), while Display Min stays pinned as the background. This is the
    # setting that makes a faint cloud visible at all -- inferno needs about
    # 20% of its range before it shows anything, so a fixed Display Max more
    # than ~5x the real peak renders the whole cloud as black even while the
    # profile fits find it easily (they sum 2048 rows, a ~45x gain in SNR,
    # so they see what the eye cannot).
    #
    # Set to 0 once a range is known, to compare brightness between runs:
    # then the colour range is exactly Display Min .. Display Max, clamped to
    # what a pixel can physically reach. The log's p99.95 line tells you what
    # to type.
    'display_autoscale': {'Autoscale Display? (1=yes, 0=manual)': '1'},
    'display_min': {'Display Min / Background (counts)': '-5'},
    'display_max': {'Display Max (counts)': '500'},
}


# ---------------------------------------------------------------------------
# Atom number and density
# ---------------------------------------------------------------------------
# Same apparatus and same camera as motAtomNumber.py, so the calibration
# constants are carried over unchanged. Only two things differ here and both
# matter: this script runs the camera at 50 dB of gain, and its exposure comes
# from the GUI rather than a module constant.
#
#   N_atoms = sum(S - B) / (Gamma_DN * (Omega/4pi) * T_opt * R_sc * t_exp)
#
GAMMA_HZ = 6.0666e6                  # Rb-87 D2 linewidth Gamma/2pi [Hz]
GAMMA = 2 * np.pi * GAMMA_HZ
I_SAT = 3.5779                       # [mW/cm^2] isotropic polarisation, D2

DN_PER_PHOTON_16BIT = 1.84           # from camera_photon_per_unit.py, at 0 dB
DN_PER_PHOTON_12BIT = 0.115
CALIB_GAIN_DB = 0.0                  # gain the calibration above was taken at

WINDOW_DIAMETER_MM = 19.3            # viewport clear aperture
APERTURE_DISTANCE_MM = 100.0         # chamber centre -> window
OPTICS_TRANSMISSION = 1.0

BEAM_POWER_MW = 50.0                 # per beam, at the atoms
BEAM_DIAMETER_MM = 30.0
N_BEAMS = 6
BEAM_PROFILE = 'flat'
IMAGING_DETUNING_MHZ = 0.0

# CHANGED from motAtomNumber's default of 0 dB: this script sets Gain to 50 dB
# in the camera setup below. FLIR gain is 10^(G/20), so 50 dB is a factor of
# 316 -- getting this wrong is a two-and-a-half order of magnitude error in N.
CAMERA_GAIN_DB = 50.0

# ---- Imaging geometry ----
# Sensor pitch divided by the magnification gives the size of one pixel
# projected back into the cloud.
PIXEL_PITCH_UM = 3.45
MAGNIFICATION = 0.59
PIXEL_UM = PIXEL_PITCH_UM / MAGNIFICATION      # ~5.847 um per pixel at the atoms

# The line-of-sight axis is not imaged, so its extent has to be assumed. Taking
# it equal to the axis along the 2448-pixel direction (x) is the usual choice
# for a MOT that is round in the plane perpendicular to the camera.
DEPTH_FOLLOWS_X = True

# Fraction of the 1/e^2 radius the brightness is integrated over. At 2w an
# elliptical aperture already encloses 1 - exp(-8) = 99.97% of a Gaussian, so
# this is the total in all but name, while still being a measurement of real
# pixels rather than an extrapolation of the fit.
INTEGRATION_RADII = 2.0


def collection_efficiency(aperture_diameter_mm, distance_mm):
    """Fraction of isotropic 4pi emission through a circular aperture.

    Exact (1 - cos(theta))/2 rather than the small-angle a^2/4L^2; they differ
    by ~1% at this 5.5 degree half-angle.
    """
    theta = np.arctan((aperture_diameter_mm / 2.0) / distance_mm)
    return 0.5 * (1.0 - np.cos(theta))


def beam_intensity(power_mw, diameter_mm, profile='flat'):
    """Intensity in mW/cm^2. Gaussian returns the on-axis peak (d = 1/e^2)."""
    radius_cm = diameter_mm / 2.0 / 10.0
    area_cm2 = np.pi * radius_cm ** 2
    if str(profile).lower().startswith('gauss'):
        return 2.0 * power_mw / area_cm2
    return power_mw / area_cm2


def scattering_rate(s_total, detuning_hz):
    """Two-level steady-state scattering rate [photons/s/atom]."""
    return (GAMMA / 2.0) * s_total / (
        1.0 + s_total + (2.0 * detuning_hz / GAMMA_HZ) ** 2)


def dn_per_photon_effective(dtype, gain_db=CAMERA_GAIN_DB,
                            calib_gain_db=CALIB_GAIN_DB):
    """DN per photon for this container, rescaled to the gain in use.

    Mono16 is 12-bit data left-shifted by four, so it reads 16x higher per
    photon than raw 12-bit; Mono8 is the same data shifted the other way and
    reads 16x lower. Picking the wrong one is a factor-of-16 error in N.
    """
    dt = np.dtype(dtype)
    if dt == np.uint16:
        base = DN_PER_PHOTON_16BIT
    elif dt == np.uint8:
        base = DN_PER_PHOTON_12BIT / 16.0
    else:
        base = DN_PER_PHOTON_12BIT
    return base * 10.0 ** ((gain_db - calib_gain_db) / 20.0)


def integrated_brightness(diff, fx, fy, n_radii=INTEGRATION_RADII):
    """Background-subtracted counts inside n_radii x the 1/e^2 radii.

    The aperture is the ellipse the fit describes, not a box, so it follows a
    cloud that is wider in one axis than the other. The pedestal is taken as
    the median OUTSIDE a 1.5x larger ellipse -- median rather than mean so
    stray light and hot pixels cannot drag it, and outside a margin so the
    cloud's own wings are not counted as background.

    :return: (dict) counts, offset, pixels, area_px
    """
    h, w = diff.shape
    yy, xx = np.ogrid[0:h, 0:w]
    rx, ry = n_radii * fx['w'], n_radii * fy['w']
    if rx <= 0 or ry <= 0:
        return None
    norm = (((xx - fx['center']) / rx) ** 2 + ((yy - fy['center']) / ry) ** 2)
    inside = norm <= 1.0
    if not inside.any():
        return None

    # Background estimate. The 1.5x margin is right for a compact cloud, but a
    # cloud that has expanded for milliseconds fills most of the sensor and the
    # annulus can fall entirely OFF it -- at sigma ~ 320 px the 2w ellipse
    # already covers 70% of the frame and nothing at all is left outside 3w.
    # The old code then used offset = 0.0, so no pedestal was removed and the
    # raw sum over most of the sensor was reported as cloud counts. That is
    # invisible in the picture and inflates the atom number by whatever the
    # pedestal integrates to.
    #
    # So step the margin in, and if even the 2w boundary leaves too little,
    # fall back to the median of the whole frame: still dominated by background
    # as long as the cloud is not most of the pixels, and far better than
    # assuming zero. Which estimator was used is returned, so a caller can say
    # when the number is not to be trusted.
    MIN_BG_PX = 20000
    offset, n_bg, bg_source = 0.0, 0, 'none'
    for margin, tag in ((1.5, '3w annulus'), (1.25, '2.5w annulus'),
                        (1.0, '2w boundary')):
        outside = norm > margin ** 2
        n_bg = int(outside.sum())
        if n_bg >= MIN_BG_PX:
            offset = float(np.median(diff[outside]))
            bg_source = tag
            break
    else:
        offset = float(np.median(diff))
        n_bg = int(diff.size)
        bg_source = 'whole-frame median (cloud fills the sensor)'

    n_px = int(inside.sum())
    return {
        # dtype=float64 is load-bearing, not decoration. diff is int32 and
        # numpy sums it with an int32 accumulator on Windows (np.dtype(int) is
        # int32 there), which wraps silently past 2.15e9. A bright cloud filling
        # a 2w ellipse reaches that -- 100k pixels near the 65532 ceiling is
        # 6.5e9 -- and the sum comes back NEGATIVE, taking the atom number and
        # the density negative with it. No error, no warning, just a sign flip.
        'counts': float(diff[inside].sum(dtype=np.float64) - offset * n_px),
        'offset': offset,
        'pixels': n_px,
        'rx': rx,
        'ry': ry,
        'inside': inside,
        'bg_pixels': n_bg,
        'bg_source': bg_source,
        # Fraction of the sensor the integration ellipse covers. Past ~0.5 the
        # background estimate is being squeezed and the counts are suspect.
        'fill_fraction': n_px / float(diff.size),
    }


def capture_fraction(shape, fx, fy, inside):
    """Fraction of a perfect 2D Gaussian that actually lands in the measured region.

    The 2w ellipse is 4 sigma across, which encloses 1 - exp(-8) = 99.97% of a
    Gaussian -- IF it fits on the sensor. After a few milliseconds of expansion
    it does not: at sigma ~ 300 px the 4 sigma semi-axis is 1200 px against a
    1024 px half-height, so the ellipse is clipped by the chip and `inside` is
    the ellipse intersected with the frame.

    Integrating the FITTED Gaussian over exactly that mask, against its
    analytic whole-plane integral 2*pi*sx*sy, gives the fraction collected.
    Dividing the measured atom number by it extrapolates to the whole cloud --
    on the explicit assumption that the cloud really is Gaussian out in the
    wings nobody measured, which is the whole point of the extrapolation and
    also its main weakness.

    :return: (float) 0..1, or nan if the fit is degenerate
    """
    h, w = shape
    yy, xx = np.ogrid[0:h, 0:w]
    sx = float(fx['sigma'])
    sy = float(fy['sigma'])
    if not (sx > 0 and sy > 0):
        return float('nan')
    g = np.exp(-((xx - fx['center']) ** 2) / (2.0 * sx ** 2)
               - ((yy - fy['center']) ** 2) / (2.0 * sy ** 2))
    total = 2.0 * np.pi * sx * sy      # exact integral over the whole plane
    return float(g[inside].sum() / total)


def atom_number(sum_dn, dn_per_photon, eta_collect, optics_transmission,
                r_sc, exposure_s):
    """N = sum(S - B) / (DN_per_photon * Omega/4pi * T * R_sc * t_exp)."""
    photons_sensor = sum_dn / dn_per_photon
    photons_emitted = photons_sensor / (eta_collect * optics_transmission)
    return photons_emitted / (r_sc * exposure_s)


def peak_density(n_atoms, wx_px, wy_px):
    """Peak number density [cm^-3] of a 3D Gaussian with 1/e^2 radii w.

    Integrating exp(-2r^2/w^2) over each axis gives w*sqrt(pi/2), so
    N = n0 (pi/2)^{3/2} wx wy wz. The depth wz is not measured; see
    DEPTH_FOLLOWS_X.
    """
    wx_cm = wx_px * PIXEL_UM * 1e-4
    wy_cm = wy_px * PIXEL_UM * 1e-4
    wz_cm = wx_cm if DEPTH_FOLLOWS_X else wy_cm
    volume = (np.pi / 2.0) ** 1.5 * wx_cm * wy_cm * wz_cm
    return n_atoms / volume if volume > 0 else float('nan')


def cloud_metrics(diff, fx, fy, dtype, exposure_us, gain_db=CAMERA_GAIN_DB):
    """Brightness, atom number and peak density for one difference frame.

    :param diff: (ndarray) frame1 - frame2, in camera counts
    :param fx, fy: (dict) fit_gaussian_1d results for the two axes
    :param dtype: raw frame dtype, which fixes the DN/photon scale
    :param exposure_us: (float) camera exposure actually used
    :return: (dict) or None if the region could not be formed
    """
    band = integrated_brightness(diff, fx, fy)
    if band is None:
        return None

    # Total brightness straight from the fitted centre and radii, out to
    # infinity: integrating amplitude*exp(-2r^2/w^2) along one axis gives
    # amplitude*w*sqrt(pi/2). Each marginal was already summed over the other
    # axis, so each integral is the WHOLE cloud on its own -- two independent
    # estimates of one number, not two factors to multiply.
    #
    # This is the extrapolated counterpart to band['counts']: same quantity,
    # one from the model and one from the pixels. Their ratio is the useful
    # part. It should sit at 1 - exp(-8) = 99.97% for a real Gaussian; well
    # under that means the fit is inventing wings the sensor never saw, and
    # over 100% means the aperture is collecting something the fit does not
    # describe -- stray light, a second spot, a badly placed centre.
    total_x = fx['amplitude'] * fx['w'] * ROOT_HALF_PI
    total_y = fy['amplitude'] * fy['w'] * ROOT_HALF_PI
    total_fit = 0.5 * (total_x + total_y)

    dn_ph = dn_per_photon_effective(dtype, gain_db=gain_db)
    eta = collection_efficiency(WINDOW_DIAMETER_MM, APERTURE_DISTANCE_MM)
    s_total = N_BEAMS * beam_intensity(
        BEAM_POWER_MW, BEAM_DIAMETER_MM, BEAM_PROFILE) / I_SAT
    r_sc = scattering_rate(s_total, IMAGING_DETUNING_MHZ * 1e6)
    n_atoms = atom_number(band['counts'], dn_ph, eta, OPTICS_TRANSMISSION,
                          r_sc, exposure_us * 1e-6)
    # What fraction of the cloud the measured region actually held, and the
    # whole-cloud number that implies. For a compact in-situ cloud this is
    # ~1.000 and n_atoms_total == n_atoms; for an expanded one it is the
    # correction that matters most.
    frac = capture_fraction(diff.shape, fx, fy, band['inside'])
    n_total = n_atoms / frac if (np.isfinite(frac) and frac > 0) else float('nan')
    return {
        'counts': band['counts'],
        'offset': band['offset'],
        'pixels': band['pixels'],
        'bg_source': band['bg_source'],
        'fill_fraction': band['fill_fraction'],
        'total_fit': total_fit,
        'total_from_x': total_x,
        'total_from_y': total_y,
        'frac_2w': band['counts'] / total_fit if total_fit > 0 else float('nan'),
        'n_atoms': n_atoms,
        'capture_fraction': frac,
        'n_atoms_total': n_total,
        # Peak density of the WHOLE cloud, so it is not depressed by the part
        # of the cloud that fell off the sensor.
        'density': peak_density(n_total, fx['w'], fy['w']),
        'dn_per_photon': dn_ph,
        'gain_db': gain_db,
        'eta': eta,
        's_total': s_total,
        'r_sc': r_sc,
        'wx_um': fx['w'] * PIXEL_UM,
        'wy_um': fy['w'] * PIXEL_UM,
    }


def results_table_html(fx, fy, peak, met, reliable):
    """Two-column-pair HTML table of the cloud results.

    Laid out wide rather than tall so it fits a short strip under the image and
    never has to overlap it. Monospace keeps the numbers in line.
    """
    if met is None:
        body = "<tr><td colspan=4>no cloud region -- fit unusable</td></tr>"
    else:
        rows = [
            ("1/e&sup2; radius x",
             f"{fx['w']:.1f} px = {met['wx_um']:.0f} &micro;m",
             "brightness (2w)", f"{met['counts']:.3e} counts"),
            ("1/e&sup2; radius y",
             f"{fy['w']:.1f} px = {met['wy_um']:.0f} &micro;m",
             "atom number", f"{met['n_atoms']:.3e}"),
            ("peak brightness", f"{peak:.1f} cnt/px",
             "peak density", f"{met['density']:.3e} cm&#8315;&sup3;"),
            ("total brightness", f"{met['total_fit']:.3e} counts",
             "2w / total", f"{met['frac_2w'] * 100:.1f}% (Gauss 99.97%)"),
            # Moved down from the old title line above the image.
            ("centre", f"({fx['center']:.0f}, {fy['center']:.0f}) px",
             "", ""),
        ]
        body = "".join(
            f"<tr><td>{a}</td><td><b>{b}</b></td>"
            f"<td>&nbsp;&nbsp;&nbsp;{c}</td><td><b>{d}</b></td></tr>"
            for a, b, c, d in rows)
    warn = "" if reliable else (
        "<tr><td colspan=4>"
        "<span style='color:#ff8a80'>fit unreliable -- values above are "
        "meaningless</span></td></tr>")
    return (f"<div style='font-family:monospace'>"
            f"<table cellspacing=2 cellpadding=1>{body}{warn}</table></div>")


def _gaussian(x, offset, amplitude, center, sigma):
    return offset + amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def fit_gaussian_1d(profile):
    """Least-squares Gaussian fit of one summed profile.

    The profile is a SUM along a whole axis, not a single row. That is what
    makes a fit possible at all on this data: one row through the cloud carries
    a couple of counts, but adding 2048 of them lifts the cloud by 2048x while
    the noise only grows as sqrt(2048), a ~45x gain in signal-to-noise.

    A flat pedestal between the two frames survives that summation too, so the
    fit carries an explicit offset term rather than assuming the baseline is
    zero -- otherwise the offset is absorbed into the width and the radius
    comes out far too large.

    :param profile: (ndarray) 1D summed profile
    :return: (dict) offset, amplitude, center, sigma, w (1/e^2 radius), curve;
        or None if it could not be fitted
    """
    y = np.asarray(profile, dtype=float)
    n = y.size
    if n < 8 or not np.all(np.isfinite(y)):
        return None
    x = np.arange(n, dtype=float)

    # Seed from moments. Baseline comes from the outer 10% at each end, which
    # is cloud-free for any sensibly framed image.
    edge = max(1, n // 10)
    offset0 = float(np.median(np.concatenate([y[:edge], y[-edge:]])))
    weights = np.clip(y - offset0, 0, None)
    total = weights.sum()
    if total <= 0:
        return None
    center0 = float((weights * x).sum() / total)
    var = float((weights * (x - center0) ** 2).sum() / total)
    sigma0 = float(np.sqrt(var)) if var > 0 else n / 10.0
    amp0 = float(y.max() - offset0)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        return None

    p = [offset0, amp0, center0, sigma0]
    if curve_fit is not None:
        try:
            p, _ = curve_fit(_gaussian, x, y, p0=p, maxfev=5000)
        except Exception:
            pass        # keep the moment estimate rather than losing the frame
    offset, amplitude, center, sigma = (float(v) for v in p)
    sigma = abs(sigma)
    if sigma <= 0 or not np.isfinite(sigma):
        return None

    curve = _gaussian(x, offset, amplitude, center, sigma)
    # Least squares always returns numbers. On a frame with no cloud it returns
    # a broad, shallow bump fitted to noise, and a 1/e^2 radius quoted off that
    # is meaningless -- so judge the fit against the residual scatter and say
    # whether it can be trusted, rather than printing it as though it can.
    noise = float(np.std(y - curve))
    # Significance of the whole peak, not of one sample. The amplitude is
    # constrained by every point across the width, so a per-point ratio badly
    # understates a fit built from thousands of them: at 3x per-point SNR this
    # still recovers the radius to a few percent.
    # Capped at the profile length: a runaway width would otherwise inflate the
    # significance of a fit that is really just a broad bump across the frame.
    n_eff = min(max(1.0, 2.0 * sigma), float(n))
    snr = abs(amplitude) * np.sqrt(n_eff) / noise if noise > 0 else float('inf')
    # 25 is calibrated, not guessed: on synthetic clouds everything above it
    # recovers the radius and peak to better than 5%, while 20 already lets
    # through fits that are 10-16% out.
    reliable = bool(amplitude > 0 and snr >= 25.0
                    and 2.0 * sigma < 0.4 * n
                    and 0 <= center < n)
    return {
        'offset': offset,
        'amplitude': amplitude,
        'center': center,
        'sigma': sigma,
        'w': 2.0 * sigma,          # 1/e^2 radius
        'snr': snr,
        'reliable': reliable,
        'curve': curve,
        'x': x,
    }


# ---------------------------------------------------------------------------
# Camera full scale -- the brightest a single pixel can possibly report.
# ---------------------------------------------------------------------------
# Measured from this camera's own frames rather than assumed: every Mono16 word
# it produces satisfies v = 16*k + 12 (12-bit ADC left-shifted by four, plus a
# 12 count pedestal). 5013502 of 5013504 pixels in the reference frame follow
# that exactly. So the ceiling is 16*4095 + 12 = 65532, NOT the 65535 the
# container would suggest, and observed saturation piles up at 65516.
SENSOR_MAX_MONO16 = 16 * 4095 + 12   # 65532
SENSOR_MAX_MONO12 = 4095
SENSOR_MAX_MONO8 = 255


def localized_signal(diff, blocks=32):
    """Find the strongest spatially-clustered excess in a difference frame.

    The integrated area cannot tell a cloud from a uniform baseline shift, and
    across five million pixels that distinction is everything: a pedestal
    difference of 0.04 counts per pixel -- far below one ADU, invisible in every
    percentile -- integrates to 2e5 counts and scores 300 sigma against a noise
    model that assumes pixels are independent. It looks like an enormous signal
    and is just the two frames sitting at slightly different black levels.

    Binning separates them. Atoms concentrate into a few blocks; a pedestal
    lifts every block equally, so it cancels against the median. The comparison
    is against the median absolute deviation over blocks, which a pedestal does
    not inflate either.

    :param diff: (ndarray) frame1 - frame2
    :param blocks: (int) grid is blocks x blocks
    :return: (dict) peak block level, background, excess in sigma, and where
    """
    h, w = diff.shape
    by, bx = h // blocks, w // blocks
    if by < 1 or bx < 1:
        return None
    binned = diff[:by * blocks, :bx * blocks].reshape(
        blocks, by, blocks, bx).mean(axis=(1, 3))
    med = float(np.median(binned))
    mad = float(np.median(np.abs(binned - med))) * 1.4826
    iy, ix = np.unravel_index(int(np.argmax(binned)), binned.shape)
    peak = float(binned[iy, ix])
    return {
        'peak': peak,
        'background': med,
        'excess_sigma': (peak - med) / mad if mad > 0 else float('inf'),
        'x': int(ix * bx + bx // 2),
        'y': int(iy * by + by // 2),
        'block_px': by * bx,
    }


def sensor_full_scale(frame):
    """Brightest value one pixel can report, deduced from the frame itself.

    An 8-bit container is unambiguous. A 16-bit one is not: it carries either
    left-shifted Mono16 or raw 12-bit, and both arrive as uint16.

    The peak value cannot tell them apart -- a dim frame comes nowhere near
    either ceiling, so a faint MOT would be misread as 12-bit and the range
    wrongly capped at 4095. The low nibble can: left-shifting by four leaves
    it constant across the whole frame (12 on this camera), whereas raw
    12-bit data spreads over all sixteen residues.

    :param frame: (ndarray) one raw camera frame
    :return: (int) maximum single-pixel brightness in counts
    """
    dt = frame.dtype
    if dt == np.uint8:
        return SENSOR_MAX_MONO8
    if dt != np.uint16:
        return int(np.iinfo(dt).max)

    counts = np.bincount(np.ravel(frame) & 0xF, minlength=16)
    if counts.max() > 0.9 * counts.sum():
        return SENSOR_MAX_MONO16
    return SENSOR_MAX_MONO12


class AutoLevelPlot2D(Plot2D):
    """Plot2D whose colour scale follows the signal instead of staying fixed.

    Plot2D redraws with autoLevels=False, so a fixed range makes a weak MOT
    invisible: a few counts above background occupies a sliver of a -5..50
    scale and renders as black. The acquisition thread cannot call setLevels
    itself -- Qt widget calls from the QThread worker crash the app -- so it
    leaves a (lo, hi) hint here and update(), which the GUI thread runs, is
    what actually touches the widget.
    """

    def set_levels_hint(self, lo, hi):
        """Called from the worker thread. Stores two floats, touches no Qt."""
        self._levels_hint = (float(lo), float(hi))

    def fit_to_frame(self, shape):
        """Resize the buffer to the camera's actual frame, if it differs.

        Plot2D sizes its buffer in visualize(), before a single frame exists,
        so any hardcoded sensor size is a guess. Guess wrong -- ROI, binning,
        a different camera -- and set_data rejects EVERY frame: it logs
        'Incompatible data shape', leaves the buffer full of NaN, and update()
        then refuses to draw, so the panel stays black forever with no other
        symptom. Resizing here removes the guess.

        Called from the worker thread, so it only rewrites plain attributes.
        The view limits are Qt and are deferred to update().

        :param shape: (tuple) numpy shape of the incoming frame, (rows, cols)
        :return: (bool) True if a resize happened
        """
        h, w = int(shape[0]), int(shape[1])
        if (self.pts_y, self.pts_x) == (h, w):
            return False
        self.pts_x, self.pts_y = w, h
        self.min_x, self.max_x = 0, w
        self.min_y, self.max_y = 0, h
        self.data = np.full((h, w), np.nan)
        self.position = 0
        self._pending_limits = True
        return True

    def attach_colorbar(self, lo, hi, colormap=COLORMAP, label='Brightness'):
        """Add a colour bar whose axis is labelled in real image units.

        ImageView's own histogram is hidden, so this is what shows the reader
        which count value each colour means. Read-only: update() owns the
        scale, and a draggable bar would just fight it.
        """
        self.colorbar = pg.ColorBarItem(
            values=(float(lo), float(hi)),
            colorMap=pg.colormap.get(colormap),
            label=label,
            interactive=False,
        )
        self.colorbar.setImageItem(
            self.graph.getImageItem(), insert_in=self.graph.getView())

    def attach_profiles(self, width=170, height=140, table_height=240):
        """Put the summed x and y profiles along two sides of the image.

        ImageView holds its view as the single central item of its GraphicsView,
        so nothing can sit beside the image until that central item becomes a
        layout. Swapping in a GraphicsLayout and re-adding the very same view
        object into it leaves ImageView's own machinery -- autoRange, levels,
        the ImageItem -- pointing at the object it already knows.

        The profile axes are linked to the image axes, so a pan or zoom on the
        picture carries the profiles with it and they stay registered to it.
        """
        view = self.graph.getView()             # PlotItem, per Plot2D
        layout = pg.GraphicsLayout()
        self.graph.ui.graphicsView.setCentralItem(layout)
        layout.addItem(view, 0, 0)

        # Right: y profile, drawn sideways (counts on x, row index on y).
        self.y_plot = layout.addPlot(row=0, col=1)
        self.y_plot.setYLink(view)
        self.y_plot.showAxis('left', False)
        self.y_plot.setLabel('bottom', 'sum')
        # Bottom: x profile.
        self.x_plot = layout.addPlot(row=1, col=0)
        self.x_plot.setXLink(view)
        self.x_plot.showAxis('bottom', False)
        self.x_plot.setLabel('left', 'sum')

        # Linking shares the RANGE, not the orientation. The image runs row 0 at
        # the top (invertY), so without this the side panel draws the same rows
        # bottom-up and the profile appears mirrored against the picture it is
        # supposed to annotate -- a cloud low in the frame peaks high in the
        # panel. Copy the image's inversion instead of assuming it.
        image_vb = view.getViewBox()
        self.y_plot.invertY(image_vb.yInverted())
        self.x_plot.invertX(image_vb.xInverted())

        layout.layout.setColumnFixedWidth(1, width)
        layout.layout.setRowFixedHeight(1, height)

        data_pen = pg.mkPen('#9aa0a6', width=1)
        fit_pen = pg.mkPen('#ffd54f', width=2)
        self.x_curve = self.x_plot.plot(pen=data_pen)
        self.x_fit_curve = self.x_plot.plot(pen=fit_pen)
        self.y_curve = self.y_plot.plot(pen=data_pen)
        self.y_fit_curve = self.y_plot.plot(pen=fit_pen)

        # Results table gets its own full-width row UNDER everything rather
        # than being drawn over the picture, so it can never cover the cloud.
        self.table_label = pg.LabelItem(justify='left')
        layout.addItem(self.table_label, row=2, col=0, colspan=2)
        layout.layout.setRowFixedHeight(2, table_height)

    def set_profile_hint(self, x_prof, x_fit, y_prof, y_fit, table=None):
        """Called from the worker thread. Stores arrays, touches no Qt."""
        self._profile_hint = (x_prof, x_fit, y_prof, y_fit, table)

    def _draw_profiles(self):
        """GUI-thread half of set_profile_hint."""
        hint = getattr(self, '_profile_hint', None)
        if hint is None or not hasattr(self, 'x_curve'):
            return
        x_prof, x_fit, y_prof, y_fit, table = hint
        self._profile_hint = None

        self.x_curve.setData(np.arange(x_prof.size), x_prof)
        self.y_curve.setData(y_prof, np.arange(y_prof.size))
        # y first, x second: the sideways panel takes (value, index).
        if x_fit is None:
            self.x_fit_curve.clear()
        else:
            self.x_fit_curve.setData(np.arange(x_fit.size), x_fit)
        if y_fit is None:
            self.y_fit_curve.clear()
        else:
            self.y_fit_curve.setData(y_fit, np.arange(y_fit.size))
        # No title over the image: every number it used to carry is a row in
        # the table below, and a title is the one item that grows downward INTO
        # the picture rather than taking a cell of its own.
        if table is not None and hasattr(self, 'table_label'):
            # Monospaced so the value columns line up instead of drifting with
            # the digits in them.
            self.table_label.setText(table, size=FIT_TITLE_SIZE)

    def update(self, **kwargs):
        # Deferred from fit_to_frame(): setLimits is a Qt call and belongs on
        # the GUI thread, which is the one running update().
        if getattr(self, '_pending_limits', False):
            self.graph.view.setLimits(xMin=self.min_x, xMax=self.max_x,
                                      yMin=self.min_y, yMax=self.max_y)
            self._pending_limits = False
        hint = getattr(self, '_levels_hint', None)
        if hint is not None:
            colorbar = getattr(self, 'colorbar', None)
            if colorbar is not None:
                # Drive the bar, not the ImageView: ColorBarItem.setLevels
                # propagates to the linked ImageItem, so the tick labels and
                # the picture can never drift apart.
                colorbar.setLevels(low=hint[0], high=hint[1])
            else:
                self.graph.setLevels(*hint)
        self._draw_profiles()
        super().update(**kwargs)


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']

    disp_min = float(dataset.get_input_parameter("display_min"))
    disp_max = float(dataset.get_input_parameter("display_max"))
    # No frame has arrived yet, so the exact ceiling is unknown; clamp to the
    # widest one any container allows so the bar cannot open on an impossible
    # scale. experiment() tightens this once it can inspect a real frame.
    disp_min = max(disp_min, -float(SENSOR_MAX_MONO16))
    disp_max = min(disp_max, float(SENSOR_MAX_MONO16))
    dataset.diff_levels = (disp_min, disp_max)

    # Image goes in the MAIN DataTaker window. Plot2D with no 'window' kwarg
    # adds its ImageView straight to gui.graph_layout, rather than opening a
    # popup the way the old GraphicsLayoutWidget did.
    dataset.add_child(
        name=IMAGE_NAME,
        data_type=AutoLevelPlot2D,
        min_x=0, max_x=FRAME_W, pts_x=FRAME_W,
        min_y=0, max_y=FRAME_H, pts_y=FRAME_H,
        new_plot=True,
    )
    img = dataset.children[IMAGE_NAME]
    img.graph.view.setAspectLocked(True)
    img.graph.view.invertY(True)
    img.update_colormap(COLORMAP)
    # Strip ImageView's built-in furniture: the histogram/LUT bar and the
    # ROI + Menu buttons. The levels are driven from the code, so the manual
    # handles only invite the scale being dragged out from under it. Hiding
    # them does not affect setLevels.
    img.graph.ui.histogram.hide()
    img.graph.ui.roiBtn.hide()
    img.graph.ui.menuBtn.hide()
    # Starting range only. Display Min pins the bottom for good; the top is
    # rescaled to each frame's peak by AutoLevelPlot2D.
    img.graph.setLevels(disp_min, disp_max)
    # Scale bar in real camera counts. This is a difference image, so the
    # numbers are frame 1 minus frame 2 and legitimately go negative.
    img.attach_colorbar(disp_min, disp_max,
                        label='Brightness (counts, frame 1 - frame 2)')
    # Summed profiles + Gaussian fits down the right and along the bottom.
    img.attach_profiles()
    # Scan grid: one column per bias point, latest frame on top and the
    # running average under it, with summed profiles and a Gaussian fit beside
    # each. The main image above still shows whichever point came in last, with
    # its full fit table and atom number; the grid is what lets the points be
    # compared, which is the only reason to sweep at all.
    dataset.scan_grid = ScanGridWindow(
        scan_labels=[f"BX={v:.3f}V" for v in bias_sweep_values(dataset)],
        diff_levels=(disp_min, disp_max), avg_levels=(disp_min, disp_max),
        title="PGC bias X sweep (ao8)",
    )
    # The main line plot is unused; hide it so it isn't an empty panel.
    dataset.graph.hide()


def experiment(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    logger = dataset.log

    NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
    camera_client = kwargs["fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber"]
    OPX_client = kwargs["OPX_OPX"]

    # =========================================================================
    # READ GUI PARAMETERS
    # =========================================================================
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    mot_switch_ms = int(dataset.get_input_parameter("mot_switch_time"))
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_v = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

    dark_ms = int(dataset.get_input_parameter("dark_mot_duration"))
    dark_power = float(dataset.get_input_parameter("dark_mot_power"))
    dark_detuning = float(dataset.get_input_parameter("dark_mot_detuning"))
    dark_rep_power = float(dataset.get_input_parameter("dark_mot_repump_power"))
    dark_rep_vco = float(dataset.get_input_parameter("dark_mot_repump_vco"))

    pgc_ms = int(dataset.get_input_parameter("pgc_duration"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))

    tof_us = int(dataset.get_input_parameter("tof_us"))
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    coils_rampback_ms = int(dataset.get_input_parameter("coils_rampback_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))

    img_repump_us = int(dataset.get_input_parameter("img_repump_time_us"))
    img_rep_power = float(dataset.get_input_parameter("img_repump_power"))
    img_rep_vco = float(dataset.get_input_parameter("img_repump_vco"))

    # ---- Bias: MOT values, then the swept PGC value ----
    mot_bias_x_v = float(dataset.get_input_parameter("mot_bias_x_v"))
    mot_bias_y_v = float(dataset.get_input_parameter("mot_bias_y_v"))
    mot_bias_z_v = float(dataset.get_input_parameter("mot_bias_z_v"))
    bias_values_v = bias_sweep_values(dataset)
    n_scan = len(bias_values_v)
    pgc_bias_y_fixed = float(dataset.get_input_parameter("pgc_bias_y_fixed_v"))
    pgc_bias_z_fixed = float(dataset.get_input_parameter("pgc_bias_z_fixed_v"))
    bias_ramp_ms = int(dataset.get_input_parameter("bias_ramp_ms"))
    # The grid is about 420 px per point and every point costs a full sequence
    # on every pass, so a step count that looks harmless on the GUI is what
    # makes the cycle long and the window wider than the screen.
    if n_scan > 16:
        logger.warn(
            f"{n_scan} sweep points: the scan grid will be about "
            f"{n_scan * 420} px wide, and one pass through the buffer runs "
            f"{n_scan} full sequences back to back. Fewer points, or a "
            f"narrower Start .. Stop range, if that is not what you meant."
        )

    # =========================================================================
    # TIMING
    # =========================================================================
    ni_rate = 20000
    SPM = ni_rate // 1000
    SPU = ni_rate / 1e6
    delay_ns = int(round(1e9 / ni_rate)) - 500
    ttl_up = max(1, SPM // 20)
    def ms2s(t): return t * SPM
    def us2s(t): return int(round(t * SPU))

    mot_s = ms2s(mot_loading_ms)
    cmot_ramp_s = ms2s(mot_switch_ms)
    cmot_hold_s = ms2s(cmot_hold_ms)
    dark_s = ms2s(dark_ms)
    pgc_s = ms2s(pgc_ms)
    tof_samp = us2s(tof_us)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    coils_rampback_s = ms2s(coils_rampback_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))
    # Repump-only prepulse, then the MOT pulse. One sample is 50 us at 20 kS/s,
    # so anything under 25 us rounds away to nothing -- 200 us is 4 samples.
    img_rep_s = us2s(img_repump_us)
    img_win_s = img_rep_s + aom_pulse_s     # whole light window per frame

    # =========================================================================
    # TIMELINE: MOT -> CMOT -> Dark MOT -> PGC -> release -> TOF -> image
    # =========================================================================
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    dark_start = cmot_end
    dark_end = dark_start + dark_s
    pgc_start = dark_end
    pgc_end = pgc_start + pgc_s
    release = pgc_end
    coils_off = cmot_end - coils_lead_s
    # f1/f2 are where the LIGHT starts (repump first). The camera fires
    # img_rep_s later, when the MOT light joins -- the prepulse produces almost
    # no fluorescence, so an exposure covering only it would come back empty.
    f1 = release + tof_samp
    f2 = f1 + clear_s
    img_light_1 = f1 + img_rep_s            # MOT light + camera, frame 1
    img_light_2 = f2 + img_rep_s            # MOT light + camera, frame 2
    coils_restore_t = f2 + img_win_s + coils_restore_s  # coils OFF through frame 2
    N = coils_restore_t + coils_rampback_s + 2

    logger.info(
        f"Fast PGC: mot={mot_loading_ms}ms, cmot={mot_switch_ms}+{cmot_hold_ms}ms, "
        f"dark={dark_ms}ms, pgc={pgc_ms}ms, "
        f"tof={tof_us}us, release={release/SPM:.0f}ms, N={N}"
    )
    logger.info(
        f"Imaging: repump alone {img_rep_s / SPM * 1000:.0f}us "
        f"({img_rep_power} V, VCO {img_rep_vco} V) from {f1 / SPM:.3f}ms, "
        f"then +MOT light and camera at {img_light_1 / SPM:.3f}ms. "
        f"Effective expansion time = tof + repump lead = "
        f"{(img_light_1 - release) / SPM * 1000:.0f}us"
    )
    if img_repump_us > 0 and img_rep_s == 0:
        logger.warn(
            f"Imaging Repump Lead Time {img_repump_us}us is under half a sample "
            f"at {ni_rate} S/s and rounded to zero -- repump and MOT light will "
            f"start together. Use >= {int(0.5e6 / ni_rate) + 1}us."
        )

    # =========================================================================
    # BUILD WAVEFORMS
    # =========================================================================
    cam = [0] * N
    for s in range(ttl_up):
        cam[img_light_1 + s] = 1
        cam[img_light_2 + s] = 1

    ado = [0] * N
    aao = [mot_aom_power] * N
    rdo = [0] * N
    rao = [repump_aom_power] * N

    # MOT
    for s in range(cmot_start):
        ado[s] = 1
        aao[s] = mot_aom_power
        rdo[s] = 1
        rao[s] = repump_aom_power
    # CMOT ramp
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        ado[t] = 1
        aao[t] = mot_aom_power + frac * (cmot_mot_power - mot_aom_power)
        rdo[t] = 1
        rao[t] = repump_aom_power + frac * (cmot_repump_power - repump_aom_power)
    # CMOT hold
    for s in range(cmot_hold_s):
        t = cmot_hold_start + s
        ado[t] = 1
        aao[t] = cmot_mot_power
        rdo[t] = 1
        rao[t] = cmot_repump_power
    # Dark MOT
    for s in range(dark_s):
        t = dark_start + s
        ado[t] = 1
        aao[t] = dark_power
        rdo[t] = 1
        rao[t] = dark_rep_power
    # PGC
    for s in range(pgc_s):
        t = pgc_start + s
        ado[t] = 1
        aao[t] = pgc_mot_power
        rdo[t] = 1
        rao[t] = pgc_rep_power
    # TOF: digital off, analog stays at full power (thermal stability)
    for s in range(release, min(f1, N)):
        ado[s] = 0
        rdo[s] = 0
    # Imaging: repump alone for img_rep_s, then the MOT light joins it. Repump
    # holds its imaging power for the whole window rather than handing back to
    # repump_aom_power, so nothing changes about the light mid-exposure.
    for s in range(img_win_s):
        for base in (f1, f2):
            t = base + s
            if t >= N:
                continue
            rdo[t] = 1
            rao[t] = img_rep_power
            if s >= img_rep_s:          # prepulse over -- cooling light on
                ado[t] = 1
                aao[t] = mot_aom_power

    # Coils: TTL off from coils_off through frame 2 + restore delay,
    # then back ON with analog ramped 0 -> MOT value (soft turn-on).
    cdo = [0] * N
    for s in range(coils_off, min(coils_restore_t, N)):
        cdo[s] = 1
    cao = [mot_coils_v] * N
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        cao[t] = mot_coils_v + frac * (cmot_coils_v - mot_coils_v)
    for s in range(cmot_hold_start, N):
        cao[s] = cmot_coils_v
    # Rampback: analog 0 -> mot_coils_v over coils_rampback_s
    for s in range(coils_rampback_s):
        t = coils_restore_t + s
        if t >= N:
            break
        frac = s / max(coils_rampback_s - 1, 1)
        cao[t] = frac * mot_coils_v
    for s in range(coils_restore_t + coils_rampback_s, N):
        cao[s] = mot_coils_v

    # MOT VCO
    mvco = [vco_loading] * N
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        mvco[t] = vco_loading + frac * (cmot_detuning - vco_loading)
    for s in range(cmot_hold_start, cmot_end):
        mvco[s] = cmot_detuning
    for s in range(dark_s):
        mvco[dark_start + s] = dark_detuning
    for s in range(pgc_s):
        mvco[pgc_start + s] = pgc_detuning
    for s in range(pgc_end, N):
        mvco[s] = pgc_detuning
    # Lead is measured from the MOT AOM opening, which is now img_rep_s after
    # the repump does, so the cooling VCO still gets its full settling time.
    vlo = max(0, img_light_1 - vco_lead_s)
    vhi = min(N, f2 + img_win_s)
    for s in range(vlo, vhi):
        mvco[s] = vco_imaging

    # Repump VCO
    rvco = [repump_vco] * N
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        rvco[t] = repump_vco + frac * (cmot_repump_vco - repump_vco)
    for s in range(cmot_hold_start, cmot_end):
        rvco[s] = cmot_repump_vco
    for s in range(dark_s):
        rvco[dark_start + s] = dark_rep_vco
    for s in range(pgc_s):
        rvco[pgc_start + s] = pgc_rep_vco
    for s in range(pgc_end, N):
        rvco[s] = pgc_rep_vco
    # Imaging repump VCO. Gets the same lead as the cooling VCO, referenced to
    # the repump AOM opening at f1 -- it has to be settled before the prepulse,
    # which is the whole point of firing the repump early.
    rvlo = max(0, f1 - vco_lead_s)
    rvhi = min(N, f2 + img_win_s)
    for s in range(rvlo, rvhi):
        rvco[s] = img_rep_vco

    # Lattice channels: all off
    lat_off_do = [1] * N
    lat_off_ao = [0.0] * N

    # Bias coils. Three rails, two setpoints each: the MOT value through MOT
    # loading and CMOT, the PGC value from coils_off -- the bias leaves with the
    # quadrupole TTL, 2 ms before the dark MOT -- held through the dark MOT, the
    # PGC and both imaging frames, then back to the MOT value.
    #
    # Ramped both ways rather than stepped. A step into a coil is an inductive
    # transient, and ramping back means every sub-sequence ENDS in the state the
    # next one begins in, which is what keeps the points in the concatenated
    # buffer from leaking into each other: without it, the last point's PGC
    # field would still be on the rail while the next point's MOT was loading.
    bias_ramp_s = ms2s(bias_ramp_ms)
    bias_down_start = f2 + img_win_s  # ramp down begins right after F2's light

    def build_bias(mot_v, pgc_v):
        """One rail: mot_v, ramp to pgc_v at coils_off, ramp back after F2."""
        w = [mot_v] * N
        for s in range(bias_ramp_s):
            t = coils_off + s
            if t >= N:
                break
            frac = s / max(bias_ramp_s - 1, 1)
            w[t] = mot_v + frac * (pgc_v - mot_v)
        for s in range(coils_off + bias_ramp_s, min(bias_down_start, N)):
            w[s] = pgc_v
        for s in range(bias_ramp_s):
            t = bias_down_start + s
            if t >= N:
                break
            frac = s / max(bias_ramp_s - 1, 1)
            w[t] = pgc_v + frac * (mot_v - pgc_v)
        for s in range(bias_down_start + bias_ramp_s, N):
            w[s] = mot_v
        return w

    # =========================================================================
    # CONCATENATE ONE SUB-SEQUENCE PER BIAS POINT
    # =========================================================================
    # Nothing outside the bias rails depends on the swept value, so the
    # sub-sequence built above is identical for every scan point and only the
    # three bias waveforms differ. Repeating the shared lists is therefore
    # exact -- and it is also why N, cycle_ms and the frame timeout computed
    # above are still the right numbers for ONE scan point.
    #
    # The cards see a single buffer of n_scan sub-sequences and regenerate it
    # forever. Nothing is re-armed between points, so the bias steps on a
    # sample boundary with the same jitter as every other rail, which is none,
    # and the MOT reloads for exactly the same time at every point. Re-arming
    # per point would cost a task teardown and rebuild between every image and
    # let the loading time vary -- which is the difference a bias sweep is
    # trying to measure.
    total = N * n_scan
    cam_p = cam * n_scan
    ado_p = ado * n_scan
    aao_p = aao * n_scan
    rdo_p = rdo * n_scan
    rao_p = rao * n_scan
    cdo_p = cdo * n_scan
    cao_p = cao * n_scan
    mvco_p = mvco * n_scan
    rvco_p = rvco * n_scan
    lat_off_do_p = lat_off_do * n_scan
    lat_off_ao_p = lat_off_ao * n_scan

    biasx_p = []
    biasy_p = []
    biasz_p = []
    for b_v in bias_values_v:
        biasx_p += build_bias(mot_bias_x_v, b_v)
        biasy_p += build_bias(mot_bias_y_v, pgc_bias_y_fixed)
        biasz_p += build_bias(mot_bias_z_v, pgc_bias_z_fixed)

    logger.info(
        f"PGC bias X sweep: {n_scan} points "
        f"{bias_values_v[0]:.3f} -> {bias_values_v[-1]:.3f} V "
        f"{[round(v, 4) for v in bias_values_v]}; PGC bias Y held at "
        f"{pgc_bias_y_fixed:.3f} V and Z at {pgc_bias_z_fixed:.3f} V; "
        f"MOT bias ({mot_bias_x_v:.3f}, {mot_bias_y_v:.3f}, "
        f"{mot_bias_z_v:.3f}) V, ramped over {bias_ramp_ms} ms each way; "
        f"{N} samples per point, {total} in the buffer "
        f"({total / SPM / 1000:.1f} s per pass)"
    )

    # =========================================================================
    # LOAD, ARM, EXECUTE
    # =========================================================================
    camera_client.set_hardware_trigger(
        line="Line0", activation="RisingEdge",
        selector="FrameStart", overlap="ReadOut",
        acquisition_mode="Continuous")
    camera_client.set_exposure(camera_exposure_us)
    # Gain: request it, then READ IT BACK and use whatever the camera
    # actually applied. try_set_float is best-effort -- a value past the
    # node's maximum raises inside SetValue, the driver swallows it, and
    # the camera keeps the 0 dB that connect() left. Nothing in the image
    # betrays that, but every photometric result is then wrong by
    # 10^(dG/20): 316x for a 50 dB request that silently did not take.
    gain_set = camera_client.try_set_float("Gain", CAMERA_GAIN_DB)
    gain_db = None
    try:
        gain_db = camera_client.get_float("Gain")
    except Exception as exc:
        logger.warn(
            f"Camera server has no get_float ({exc}); cannot confirm the "
            f"gain. Restart it to enable the read-back."
        )
    if gain_db is None:
        gain_db = CAMERA_GAIN_DB if gain_set else 0.0
        logger.warn(
            f"Gain could not be read back; assuming {gain_db:.2f} dB from "
            f"the set() {'succeeding' if gain_set else 'FAILING'}."
        )
    if abs(gain_db - CAMERA_GAIN_DB) > 0.01:
        logger.error(
            f"Camera gain is {gain_db:.2f} dB, NOT the {CAMERA_GAIN_DB:.2f} "
            f"dB requested -- the set was refused (this model's maximum is "
            f"typically ~48 dB). Atom number and density are computed from "
            f"the ACTUAL {gain_db:.2f} dB, so they are correct; but had "
            f"this gone unnoticed N would have been wrong by a factor "
            f"{10 ** ((CAMERA_GAIN_DB - gain_db) / 20):.0f}. Set "
            f"CAMERA_GAIN_DB to a value the camera accepts."
        )
    else:
        logger.info(f"Camera gain confirmed at {gain_db:.2f} dB")
    dataset.camera_client.start_acquisition()

    try:
        NI_card_1.finalize_clock()
    except Exception:
        pass

    # Clock routing is SERVER state, not script state: clock_master_dev lives on
    # the driver object, which outlives every script that touches it. This one
    # needs the stock routing -- OPX -> card 1 PFI0 -> exported on PXI_Trig7,
    # cards 2 and 3 consuming it -- and never sets it, so it would otherwise
    # inherit whatever the last script left behind.
    #
    # jingfastMOT hands the master role to card 3 and has it export PXI_Trig7 on
    # its own oscillator. It puts that back in its finally, but that does not
    # help if it crashed, was force-stopped, or is still running. Card 3 left as
    # master reserves PXI_Trig7 the moment it arms, and card 1's arm_clock()
    # below then dies with:
    #
    #   DaqError -89126: Trigger line requested could not be reserved because it
    #   is already in use.  Property: DAQmx_Exported_SampClk_OutputTerm
    #   Destination Device: PXI1Slot2_1
    #
    # Asserting the routing here is idempotent and costs nothing, and it is the
    # only way this script is immune to how the previous one exited.
    for card, label in ((NI_card_1, 'card 1'), (NI_card_2, 'card 2'),
                        (NI_card_3, 'card 3')):
        try:
            card.reset_clock()
        except Exception as exc:
            logger.warn(
                f"Could not reset clock routing on {label} ({exc}). If arm_clock "
                f"fails with -89126, another card still owns PXI_Trig7 -- "
                f"restart the NI servers."
            )
    logger.info("Clock routing asserted: card 1 masters PXI_Trig7 from the OPX")

    h1 = None
    h2 = None

    NI_card_2.build_stack()
    NI_card_2.set_do_voltage(do_channel="dio0", value=cam_p, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio1", value=ado_p, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio2", value=cdo_p, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio3", value=rdo_p, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio4", value=lat_off_do_p, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio5", value=lat_off_do_p, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio6", value=[0] * total, sample_rate=ni_rate)

    NI_card_3.build_stack()
    NI_card_3.set_ao_voltage(ao_channel="ao1", voltages=aao_p, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao2", voltages=cao_p, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao3", voltages=mvco_p, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao4", voltages=rvco_p, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao5", voltages=rao_p, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=lat_off_ao_p, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=lat_off_ao_p, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao8", voltages=biasx_p, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao9", voltages=biasy_p, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao10", voltages=biasz_p, sample_rate=ni_rate)

    h1 = NI_card_2.arm(regeneration=True)
    h2 = NI_card_3.arm(regeneration=True)
    NI_card_1.arm_clock(length=total, sample_rate=ni_rate)

    OPX_client.build_stack()
    clk = OPX_client.create_new_do_elem(do_channel=1, length=500)
    M = 100000
    run_buffer = 10_000_000
    with OPX_client.for_("j", 0, M, 1):
        with OPX_client.for_("i", 0, total + 1, 1):
            OPX_client.set_digital_voltage(element=clk)
            OPX_client.delay(delay_ns)
        OPX_client.delay(run_buffer)
    OPX_client.execute(wait=False)

    img_ds = dataset.children[IMAGE_NAME]
    autoscale = bool(int(dataset.get_input_parameter("display_autoscale")))
    disp_min = float(dataset.get_input_parameter("display_min"))
    disp_max = float(dataset.get_input_parameter("display_max"))
    # Brightest raw pixel seen this run, and the resulting physical ceiling.
    # Both are settled on the first frame and refined if anything brighter
    # turns up; `limit` is what the manual range is clamped against.
    run_peak = None
    limit = None

    # One trigger pair per pass through the buffer, so the longest the camera
    # ever waits is frame 2 of one cycle to frame 1 of the next -- roughly a
    # whole cycle. Deriving the timeout from the sequence beats a hardcoded 2 s,
    # which starts timing out on EVERY grab the moment MOT loading passes ~1.9 s
    # and looks exactly like a dead trigger.
    cycle_ms = N / SPM
    frame_timeout_ms = int(2 * cycle_ms + 1000)
    logger.info(
        f"Sequence cycle {cycle_ms:.0f} ms ({N} samples @ {ni_rate} S/s); "
        f"frame timeout {frame_timeout_ms} ms"
    )

    def get_frame(timeout_ms=1000, poll_ms=500):
        """One frame, waited for in short slices so Stop is seen promptly.

        The camera is triggered by the sequence, so a grab legitimately blocks
        for a whole scan point, and the timeout has to allow for a missed
        trigger on top of that. Asking for the entire wait in one call means
        Stop cannot be noticed until the frame lands -- with n_scan points in
        the buffer that is a long time to sit after pressing it.

        Spinnaker's GetNextImage does not consume anything when it times out,
        so a queue of short waits collects exactly the frame one long wait
        would have; it just gets a chance to look at thread.running between
        them.
        """
        deadline = time.perf_counter() + timeout_ms / 1000.0
        while True:
            if not thread.running:
                raise StopRequested()
            remaining_ms = (deadline - time.perf_counter()) * 1000.0
            if remaining_ms <= 0:
                raise TimeoutError(f"No camera frame within {timeout_ms} ms")
            started = time.perf_counter()
            try:
                b, shape, dtype = dataset.camera_client.get_frame_bytes(
                    int(min(poll_ms, remaining_ms)))
            except Exception:
                # A timeout has already burned its slice. A real fault --
                # incomplete frame, RPC hiccup -- comes back at once, so pause
                # before retrying rather than spinning on it.
                if time.perf_counter() - started < 0.05:
                    time.sleep(0.05)
                continue
            return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    try:
        # Frame arrival diagnostics. A missing camera trigger is completely
        # silent -- GetNextImage just times out -- so without this the panels
        # simply stay black and the log says nothing at all.
        timeouts = 0
        pairs = 0

        def report_miss(exc, phase):
            """Say why a grab failed, escalating as misses pile up."""
            if timeouts == 1:
                logger.warn(
                    f"No camera frame on {phase} ({exc}). Waited "
                    f"{frame_timeout_ms} ms, one scan point is "
                    f"{cycle_ms:.0f} ms."
                )
            elif timeouts in (5, 20) or timeouts % 50 == 0:
                logger.error(
                    f"{timeouts} missed frame grabs, {pairs} pairs so far. The "
                    f"camera is not being triggered: check dio0 on NI card 2 is "
                    f"wired to camera Line0, that the cards armed above, and "
                    f"that the sample clock is actually running."
                )

        # ------------------------------------------------------------------
        # Which bias value produced which image
        # ------------------------------------------------------------------
        # Nothing in a frame says what the bias was when it was taken. The
        # buffer regenerates in order and wraps, so pair k belongs to scan
        # point k % n_scan, and this counter is the ONLY thing tying the two
        # together. It is therefore advanced on a completed pair and nowhere
        # else -- including across the discarded first pair below, which is a
        # real scan point being consumed, not a free one.
        pair_index = 0
        diff_sums = [None] * n_scan
        # Every warning below fires ONCE per run: the condition that produces
        # one persists for as long as whatever caused it, and a copy per scan
        # point per pass would bury the log within a minute.
        warned = set()
        diff_counts = [0] * n_scan
        # Latest p99.95 of each tile, so one range can cover a whole row.
        cur_tops = [None] * n_scan
        avg_tops = [None] * n_scan
        scan_grid = dataset.scan_grid

        logger.info(
            f"Discarding first frame pair (scan point 0, "
            f"BX={bias_values_v[0]:.3f} V)"
        )
        try:
            get_frame(timeout_ms=frame_timeout_ms + 5000)
            get_frame(timeout_ms=frame_timeout_ms + 5000)
            pair_index += 1
            logger.info("First frame pair received -- camera is being triggered")
        except StopRequested:
            raise
        except Exception as exc:
            logger.error(
                f"No frames within {frame_timeout_ms + 5000} ms of arming "
                f"({exc}). Nothing is triggering the camera, so no scan point "
                f"will ever be filled in. Check that dio0 on NI card 2 reaches camera Line0, that card 1 armed its clock above, and that the OPX is actually driving it."
            )

        while thread.running:
            # Read BEFORE the grab: this is the point the frames about to
            # arrive were taken at.
            scan_idx = pair_index % n_scan
            b_v = bias_values_v[scan_idx]
            try:
                frame1 = get_frame(timeout_ms=frame_timeout_ms)
            except StopRequested:
                break
            except Exception as exc:
                timeouts += 1
                report_miss(exc, f"frame 1 of BX={b_v:.3f}V")
                # Counter deliberately NOT advanced. A timeout here almost
                # always means no trigger fired at all -- the sequence stalled,
                # or nothing is wired to Line0 -- in which case the next frame
                # to arrive is still this scan point's.
                continue
            try:
                frame2 = get_frame(timeout_ms=frame_timeout_ms)
            except StopRequested:
                break
            except Exception as exc:
                timeouts += 1
                report_miss(exc, f"frame 2 of BX={b_v:.3f}V")
                logger.error(
                    f"Lost a frame in the MIDDLE of a pair at "
                    f"BX={b_v:.3f} V. If that frame was taken and merely "
                    f"not collected, it is still queued, and every image from "
                    f"here on is paired and labelled one frame out: the tiles "
                    f"would then show frame2-minus-next-frame1 under the wrong "
                    f"bias value. Restart the run rather than trusting the "
                    f"grid past this point."
                )
                continue
            pairs += 1
            pair_index += 1
            diff = frame1.astype(np.int32) - frame2.astype(np.int32)

            if diff_sums[scan_idx] is None:
                diff_sums[scan_idx] = diff.astype(np.float64)
            else:
                diff_sums[scan_idx] += diff
            diff_counts[scan_idx] += 1
            avg = diff_sums[scan_idx] / diff_counts[scan_idx]

            # Track the brightest pixel seen this run, on the RAW frames --
            # full scale is a property of the ADC, not of the difference.
            frame_peak = int(max(frame1.max(), frame2.max()))
            if run_peak is None or frame_peak > run_peak:
                run_peak = frame_peak

            if limit is None:
                limit = sensor_full_scale(frame1)
                # The difference of two frames spans -limit .. +limit.
                lo_req, hi_req = disp_min, disp_max
                disp_min = max(disp_min, -float(limit))
                disp_max = min(disp_max, float(limit))
                logger.info(
                    f"Camera full scale: {limit} counts ({frame1.dtype}). "
                    f"Display range limited to [{-limit}, {limit}]."
                )
                if frame1.dtype == np.uint8:
                    logger.error(
                        f"Camera is sending 8-bit frames despite the request for "
                        f"{PIXEL_FORMAT}. Every count is 16x coarser than Mono16, "
                        f"so a cloud that should read tens of counts arrives as "
                        f"1-3 and is quantised into a handful of grey levels. "
                        f"Nothing on the display side can undo that -- fix the "
                        f"PixelFormat (check SpinView is not holding the camera, "
                        f"and that Mono16/Mono12p is available on this model)."
                    )
                if (lo_req, hi_req) != (disp_min, disp_max):
                    logger.warn(
                        f"Requested display range ({lo_req:.0f}, {hi_req:.0f}) "
                        f"exceeds what a pixel can reach; using "
                        f"({disp_min:.0f}, {disp_max:.0f})."
                    )
                img_ds.set_levels_hint(disp_min, disp_max)

            if autoscale:
                # Top follows the signal, bottom stays pinned at Display Min so
                # the background does not breathe. The top is the largest over
                # the scan points seen so far, not this tile's own, so the
                # columns stay comparable to each other.
                cur_tops[scan_idx] = autoscale_top(diff, disp_min)
                avg_tops[scan_idx] = autoscale_top(avg, disp_min)
                cur_levels = (disp_min,
                              max(t for t in cur_tops if t is not None))
                avg_levels = (disp_min,
                              max(t for t in avg_tops if t is not None))
            else:
                cur_levels = (disp_min, disp_max)
                avg_levels = (disp_min, disp_max)
                if cur_levels[1] <= cur_levels[0]:
                    # A zero-width range renders as a flat block.
                    cur_levels = (disp_min, disp_min + 1)
                    avg_levels = cur_levels
            lo, hi = cur_levels

            scan_grid.set_current(scan_idx, diff, levels=cur_levels)
            scan_grid.set_average(scan_idx, avg, levels=avg_levels)
            scan_grid.set_levels_hint(cur_levels, avg_levels)

            # Match the plot to the camera before handing it data, or every
            # frame is silently rejected and nothing is ever drawn.
            if img_ds.fit_to_frame(diff.shape):
                logger.warn(
                    f"Camera frame is {diff.shape[1]}x{diff.shape[0]}, not the "
                    f"assumed {FRAME_W}x{FRAME_H}; image plot resized to match."
                )

            # Profiles and fits. Sum along one axis, fit the result: summing
            # 2048 rows lifts the cloud 2048x while noise grows only as
            # sqrt(2048), so a per-pixel signal far below one count still gives
            # a fittable profile.
            x_prof = diff.sum(axis=0).astype(float)   # sum rows -> vs column
            y_prof = diff.sum(axis=1).astype(float)   # sum columns -> vs row
            fx = fit_gaussian_1d(x_prof)
            fy = fit_gaussian_1d(y_prof)

            met = None
            peak = None
            good = False
            if fx and fy:
                # Undo the summation to recover the 2D peak: integrating the
                # other axis multiplied the amplitude by w*sqrt(pi/2).
                peak = fx['amplitude'] / (fy['w'] * ROOT_HALF_PI)
                good = fx['reliable'] and fy['reliable']
                flag = '' if good else '   [UNRELIABLE]'
                met = cloud_metrics(diff, fx, fy, frame1.dtype,
                                    camera_exposure_us, gain_db=gain_db)
                if met:
                    logger.info(
                        f"BX={b_v:.3f}V n={diff_counts[scan_idx]} | cloud: "
                        f"brightness({INTEGRATION_RADII:.0f}w)="
                        f"{met['counts']:.3e} cnt over {met['pixels']} px "
                        f"(pedestal {met['offset']:+.3f}/px) | "
                        f"N={met['n_atoms']:.3e} atoms | "
                        f"n0={met['density']:.3e} cm^-3 | "
                        f"w=({met['wx_um']:.0f},{met['wy_um']:.0f}) um | "
                        f"gain={met['gain_db']:.1f}dB "
                        f"({met['dn_per_photon']:.3g} DN/photon){flag}"
                    )
                logger.info(
                    f"BX={b_v:.3f}V | fit: 1/e2 wx={fx['w']:.1f}px "
                    f"wy={fy['w']:.1f}px | "
                    f"centre=({fx['center']:.1f},{fy['center']:.1f}) | "
                    f"peak={peak:.4f} cnt/px (measured max {float(diff.max()):.0f}) | "
                    f"snr x={fx['snr']:.1f} y={fy['snr']:.1f}{flag}"
                )
                if not good and 'unreliable' not in warned:
                    warned.add('unreliable')
                    logger.warn(
                        f"Gaussian fit is not trustworthy (peak significance "
                        f"x={fx['snr']:.1f}, y={fy['snr']:.1f}; needs >=25, or "
                        f"the width ran away past 40% of the frame). Least "
                        f"squares still returns numbers on a frame with no "
                        f"cloud -- it just fits a broad bump to noise -- so "
                        f"treat the radii above as meaningless until there is a "
                        f"real cloud. Reported once per run."
                    )

            # The averaged tile's own widths, which is what the grid header
            # shows; logged so the sweep can be read off the log alone.
            _, sx_avg, _ = _sigma_from_moment(avg.sum(axis=0))
            _, sy_avg, _ = _sigma_from_moment(avg.sum(axis=1))

            # Neither call touches Qt: the hints store plain arrays and floats,
            # set_data stores the image. The GUI thread does the redraw.
            img_ds.set_levels_hint(lo, hi)
            img_ds.set_profile_hint(
                x_prof, fx['curve'] if fx else None,
                y_prof, fy['curve'] if fy else None,
                results_table_html(fx, fy, peak, met, good) if (fx and fy)
                else results_table_html(None, None, None, None, False))
            img_ds.set_data(diff)

            # Diagnostics. "All dark" has several distinct causes and the
            # numbers below separate them: no light on the sensor at all,
            # light but no atoms, a signal pointing the wrong way, or a
            # display range that simply does not cover the data.
            f1_mean, f2_mean = float(frame1.mean()), float(frame2.mean())
            d_lo = float(np.percentile(diff, 0.05))
            d_hi = float(np.percentile(diff, 99.95))
            # float64 accumulator: see integrated_brightness. Five million
            # int32 pixels overflow an int32 sum at a mean of only ~430 counts.
            area = float(diff.sum(dtype=np.float64))
            # What the integrated area would be from noise alone. Per-pixel
            # scatter grows as sqrt(N) when summed, so anything much larger
            # than this is real signal -- and its SIGN says which frame the
            # atoms were in.
            noise_area = float(diff.std()) * np.sqrt(diff.size)
            # Area alone is not evidence of atoms -- see localized_signal.
            loc = localized_signal(diff)
            logger.info(
                f"BX={b_v:.3f}V | avg n={diff_counts[scan_idx]} "
                f"sx={sx_avg:.1f}px sy={sy_avg:.1f}px "
                f"area={float(avg.sum()):.3e} | "
                f"raw f1={f1_mean:.0f} f2={f2_mean:.0f} peak={run_peak}/{limit} | "
                f"diff min={float(diff.min()):.0f} p0.05={d_lo:.1f} "
                f"med={float(np.median(diff)):.1f} p99.95={d_hi:.1f} "
                f"max={float(diff.max()):.0f} | "
                f"area={area:.3e} ({area / max(noise_area, 1):+.0f}sig, "
                f"mean {area / diff.size:+.3f}/px) | "
                + (f"cloud {loc['peak'] - loc['background']:+.3f}/px "
                   f"({loc['excess_sigma']:+.1f}sig) at ({loc['x']},{loc['y']}) | "
                   if loc else "")
                + f"levels=({lo:.0f},{hi:.0f}) {'auto' if autoscale else 'manual'}"
            )

            # Every warning below fires ONCE per run. The condition that
            # produces it persists for as long as whatever caused it, and one
            # copy per scan point per pass would bury the log within a minute.
            #
            # An area that is huge but spatially flat is a black-level shift
            # between the two frames, not atoms. Say so explicitly -- the raw
            # sigma on the area is the single most misleading number here.
            if loc is not None and abs(area) > 5 * noise_area \
                    and loc['excess_sigma'] < 5 and 'flat' not in warned:
                warned.add('flat')
                logger.warn(
                    f"Integrated area is {area / max(noise_area, 1):+.0f} sigma but "
                    f"the excess is UNIFORM: the brightest {loc['block_px']}-pixel "
                    f"block is only {loc['excess_sigma']:+.1f} sigma above the rest "
                    f"of the frame. That is a {area / diff.size:+.3f} count/px "
                    f"baseline difference between frame 1 and frame 2, not a "
                    f"cloud. Ignore the area sigma; it counts a sub-ADU pedestal "
                    f"across {diff.size / 1e6:.1f}M pixels as signal."
                )

            # Is the sensor seeing light at all? Compare against the pedestal,
            # not against zero: this camera's floor sits around 1180 counts.
            if run_peak is not None and limit and run_peak < 0.02 * limit:
                if 'dark' not in warned:
                    warned.add('dark')
                    logger.warn(
                        f"Sensor is nearly dark: brightest raw pixel {run_peak} of "
                        f"{limit}. No light is reaching the camera -- check the "
                        f"imaging pulse, exposure and trigger, not the display."
                    )
            # A cloud in frame 2 rather than frame 1 integrates NEGATIVE. The
            # percentile tails cannot see this -- read noise makes them nearly
            # symmetric either way -- but the area cannot miss it.
            elif area < -5.0 * noise_area:
                if 'negative' not in warned:
                    warned.add('negative')
                    logger.warn(
                        f"Signal is NEGATIVE: area={area:.3e} "
                        f"({area / max(noise_area, 1):.0f} sigma below zero). Frame 2 "
                        f"is brighter than frame 1, so the frame pair is swapped "
                        f"and the atoms are being subtracted the wrong way round. "
                        f"Nothing shows because the cloud sits below Display Min. "
                        f"In a sweep this also means the tiles are labelled with "
                        f"the wrong bias value -- see the mid-pair loss warning."
                    )
            # No signal either way -- the display is not the problem.
            elif abs(area) < 3.0 * noise_area:
                if 'nosignal' not in warned:
                    warned.add('nosignal')
                    logger.warn(
                        f"No signal: area={area:.3e} is within noise "
                        f"({noise_area:.1e}). The two frames are identical to "
                        f"within read noise, so there are no atoms in frame 1."
                    )
            # Range that cannot show the data is the commonest 'all dark'.
            elif not autoscale and d_hi > 0 and hi > 10.0 * d_hi:
                if 'range_high' not in warned:
                    warned.add('range_high')
                    logger.warn(
                        f"Display Max {hi:.0f} is {hi / d_hi:.0f}x the actual "
                        f"signal (p99.95={d_hi:.1f}); everything renders near the "
                        f"bottom of the colour range. Try Display Max ~{d_hi:.0f}."
                    )
            elif not autoscale and d_hi > hi:
                if 'range_low' not in warned:
                    warned.add('range_low')
                    logger.warn(
                        f"Signal (p99.95={d_hi:.1f}) exceeds Display Max {hi:.0f}; "
                        f"the cloud is clipped. Try Display Max ~{d_hi * 1.2:.0f}."
                    )
    except StopRequested:
        logger.info("Stop requested -- ending the sweep here.")
    finally:
        dataset.camera_client.stop_acquisition()
        if h1 is not None:
            try:
                NI_card_2.finalize(h1, timeout=120.0, force_finish=True)
            except Exception:
                pass
        if h2 is not None:
            try:
                NI_card_3.finalize(h2, timeout=120.0, force_finish=True)
            except Exception:
                pass
        try:
            NI_card_1.finalize_clock()
        except Exception:
            pass
        # Last, once every sequence task has released its channels: park the
        # rails at a known state instead of wherever the loop was interrupted.
        park_outputs(NI_card_2, NI_card_3, logger)
