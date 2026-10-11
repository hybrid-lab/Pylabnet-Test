# ---------------------------------------------------------------------------
# Duplicate of lattice_biasYSweep.py with every INIT_DICT default that fastLattice.py also
# has set to fastLattice.py's value, so this sweep runs on top of the same
# MOT/CMOT/dark-MOT/PGC/lattice/imaging conditions. The swept axis and the
# analysis code are unchanged from lattice_biasYSweep.py.
# CMOT_RAMP_MS is likewise 15 ms here (fastLattice 'MOT Switch Time'),
# not the 10 ms of lattice_biasYSweep.py.
#
# The image display and its scaling come from jingfastPGC.py: Display Min /
# Display Max / Autoscale on the GUI instead of the hardcoded (0, 20) levels,
# the range clamped to what a pixel can physically reach (sensor_full_scale),
# a colour bar in real camera counts beside each row of tiles, and the
# per-frame diagnostics that say WHY a frame looks black.
# ---------------------------------------------------------------------------
# Lattice + BIAS Y SWEEP.
# Full sequence:
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> lattice HOLD (FIXED 20ms) -> release -> short TOF -> image
# Sweep ao9 = Bias Y over Start .. Stop in Steps points, all three
# set on the GUI (endpoints included).
# Other two bias axes fixed via GUI.
# Lattice runs continuously at full AM/TTL for AOM thermal stability.
#
# Sequence per scan point:
#   1. MOT loading (full power, coils on)
#   2a. CMOT ramp (15ms hardcoded, = fastLattice MOT Switch Time)
#   2b. CMOT hold (GUI)
#      -- coils TTL off 2ms before CMOT ends --
#   3. [Dark MOT] (field-free, far detuned, low repump)
#   4. [PGC] (field-free, very far detuned, low power, sub-Doppler cooling)
#   5. Release + TOF (everything off)
#   6. Frame 1 imaging
#   7. Atoms clear
#   8. Frame 2 background
#
# PGC (polarization gradient cooling / optical molasses):
#   - No B-field (coils already off)
#   - Cooling beam: very far red-detuned (~-100 MHz), low power (~10-20%)
#   - Repump: very low (~3-5%) or off
#   - Typical duration: 5-20 ms
#   - Cools atoms well below Doppler limit via Sisyphus effect

import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
import time
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

M_RB87 = 1.4431e-25   # kg
K_B = 1.3806e-23       # J/K


def _sigma_from_moment(profile):
    """Fit a Gaussian + constant offset to a 1D profile.
    Uses moment-based initial guess, then scipy least-squares refinement.
    Returns (mu, sigma, bg_subtracted_profile) or (nan, nan, None).
    The returned sigma and mu come from the fit, not from moments."""
    n = profile.size
    if n < 10:
        return float('nan'), float('nan'), None
    y = profile.astype(np.float64, copy=True)

    # Baseline: use lower percentile
    baseline = np.percentile(y, 10)
    y_bg = y - baseline
    np.clip(y_bg, 0, None, out=y_bg)
    w_sum = y_bg.sum()
    if w_sum <= 0:
        return float('nan'), float('nan'), None

    x = np.arange(n, dtype=np.float64)

    # Moment-based initial guess
    mu0 = (y_bg * x).sum() / w_sum
    var0 = (y_bg * (x - mu0) ** 2).sum() / w_sum
    if var0 <= 0 or not np.isfinite(var0):
        return float('nan'), float('nan'), None
    sigma0 = np.sqrt(var0)
    amp0 = y_bg.max()

    # Least-squares Gaussian + offset fit: y = offset + amp * exp(-(x-mu)^2/(2*sigma^2))
    try:
        from scipy.optimize import curve_fit

        def gauss_with_offset(x, offset, amplitude, mu, sigma):
            return offset + amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))

        p0 = [baseline, amp0, mu0, sigma0]
        bounds_lo = [-np.inf, 0, 0, 1]
        bounds_hi = [np.inf, np.inf, n, n]
        popt, _ = curve_fit(gauss_with_offset, x, profile,
                            p0=p0, bounds=(bounds_lo, bounds_hi),
                            maxfev=2000)
        fit_offset, fit_amp, fit_mu, fit_sigma = popt

        if fit_sigma < 1 or fit_sigma > n / 2 or not np.isfinite(fit_sigma):
            # Fit returned nonsense, fall back to moments
            return float(mu0), float(sigma0), y_bg

        # Return the fit-subtracted background profile for display
        y_display = profile - fit_offset
        np.clip(y_display, 0, None, out=y_display)
        return float(fit_mu), float(fit_sigma), y_display

    except Exception:
        # scipy not available or fit failed -- fall back to moment estimate
        return float(mu0), float(sigma0), y_bg


def _gaussian_curve(n, mu, sigma, amplitude):
    """Generate (x, y) Gaussian curve. Used for overlay display."""
    x = np.arange(n, dtype=np.float64)
    y = amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return x, y


def _linear_fit(x, y):
    n = len(x)
    if n < 2:
        return float('nan'), float('nan')
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float('nan'), float('nan')
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    n = len(x)
    det = n * sxx - sx * sx
    if abs(det) < 1e-30:
        return float('nan'), float('nan')
    a = (sxx * sy - sx * sxy) / det
    b = (n * sxy - sx * sy) / det
    return float(a), float(b)


# ---------------------------------------------------------------------------
# DISPLAY / SCALING  -- ported from jingfastPGC.py
# ---------------------------------------------------------------------------
# One name for both the image LUT and the colour bars, so they cannot disagree.
COLORMAP = 'inferno'

# Pixel format the camera is expected to be in. Nothing here sets it -- it is
# only named in the 8-bit error below -- but the camera remembers whatever was
# last set on it, so a run can silently come back 8-bit and throw away four
# bits of an already faint signal.
PIXEL_FORMAT = 'Mono16'

# Measured from this camera's own frames rather than assumed: every Mono16 word
# it produces satisfies v = 16*k + 12 (12-bit ADC left-shifted by four, plus a
# 12 count pedestal). So the ceiling is 16*4095 + 12 = 65532, NOT the 65535 the
# container would suggest, and observed saturation piles up at 65516.
SENSOR_MAX_MONO16 = 16 * 4095 + 12   # 65532
SENSOR_MAX_MONO12 = 4095
SENSOR_MAX_MONO8 = 255

# Percentile the autoscaled top follows. Not the max: one hot pixel would then
# set the scale for the whole frame and push the cloud back into the dark end.
AUTOSCALE_PCT = 99.95

# Fallback only. configure() and experiment() both read Display Min/Max from
# the GUI; these are what a missing or malformed field degrades to, and they
# match the INIT_DICT defaults. The old (0, 20) was narrower than a single
# photon: the calibration is 1.84 DN/photon at 0 dB and FLIR gain is 10^(G/20),
# so at the 50 dB this script sets, one photon is 1.84 * 316 = 582 counts. A 20
# count ceiling put every pixel that caught even one photon past the top.
DIFF_LEVELS = (-5, 500)
AVG_DIFF_LEVELS = (-5, 500)


def sensor_full_scale(frame):
    """Brightest value one pixel can report, deduced from the frame itself.

    An 8-bit container is unambiguous. A 16-bit one is not: it carries either
    left-shifted Mono16 or raw 12-bit, and both arrive as uint16.

    The peak value cannot tell them apart -- a dim frame comes nowhere near
    either ceiling, so a faint cloud would be misread as 12-bit and the range
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


def autoscale_top(image, lo):
    """Top of the colour range for one image, from its own brightness."""
    hi = float(np.percentile(image, AUTOSCALE_PCT))
    if hi <= lo + 1.0:
        hi = lo + 10.0        # empty frame: keep a usable, non-degenerate range
    return hi


def read_display_levels(dataset):
    """(lo, hi, autoscale) from the GUI, clamped to what a pixel can reach.

    Kept in one place so configure() and experiment() cannot drift apart, and
    so a missing or malformed field degrades to something viewable instead of
    raising in the middle of a run. No frame exists when configure() calls
    this, so the clamp here is the widest any container allows; experiment()
    tightens it once it can inspect a real frame.
    """
    def _get(name, default):
        try:
            return float(dataset.get_input_parameter(name))
        except Exception:
            return default
    lo = _get("display_min", DIFF_LEVELS[0])
    hi = _get("display_max", DIFF_LEVELS[1])
    auto = bool(int(_get("display_autoscale", 1.0)))
    # This is a difference image, so it legitimately spans -full .. +full.
    lo = max(lo, -float(SENSOR_MAX_MONO16))
    hi = min(hi, float(SENSOR_MAX_MONO16))
    if hi <= lo:
        hi = lo + 1.0         # a zero-width range renders as a flat block
    return lo, hi, auto


def bias_sweep_values(dataset):
    """The bias points to sweep, from Start / Stop / Steps on the GUI.

    Kept in one place so configure(), which sizes the display grid, and
    experiment(), which builds the waveform, cannot disagree about how many
    scan points there are or what is in them -- they used to read the same ten
    fields twice over.

    Steps counts the points and includes both endpoints, so it reproduces the
    fixed ten-field sweep exactly at its default. A step count below 1 is
    treated as 1 (the start value alone) rather than dividing by zero or
    handing an empty list to a display grid with no columns in it.
    """
    start = float(dataset.get_input_parameter("bias_start_v"))
    stop = float(dataset.get_input_parameter("bias_stop_v"))
    steps = int(float(dataset.get_input_parameter("bias_steps")))
    if steps <= 1:
        return [start]
    return [float(v) for v in np.linspace(start, stop, steps)]


class StopRequested(Exception):
    """Raised in place of a frame when the user presses Stop.

    Unwinds straight out of the run and scan loops to the teardown, so the
    stop is acted on where it is noticed instead of every level in between
    having to test thread.running and pass a flag outwards.
    """


class ScanGridWindow:
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
            cmap = pg.colormap.get('inferno')
            img.setLookupTable(cmap.getLookupTable())
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
            self.y_data_curves.append(y_plot.plot([], [], pen=pg.mkPen('w', width=1)))
            self.y_fit_curves.append(y_plot.plot([], [], pen=pg.mkPen('y', width=2)))
            vb_avg = self.win.addViewBox(row=2, col=base_col, lockAspect=True)
            vb_avg.invertY(True)
            img = pg.ImageItem(axisOrder='row-major')
            img.setLevels(self.avg_levels)
            cmap = pg.colormap.get('inferno')
            img.setLookupTable(cmap.getLookupTable())
            vb_avg.addItem(img)
            self.avg_imgs.append(img)
            y_plot.setYLink(vb_avg)
            x_plot = self.win.addPlot(row=3, col=base_col)
            x_plot.setMaximumHeight(self.MARGIN_PX)
            x_plot.hideAxis('bottom')
            x_plot.hideAxis('left')
            x_plot.setMouseEnabled(x=False, y=False)
            x_plot.setMenuEnabled(False)
            self.x_data_curves.append(x_plot.plot([], [], pen=pg.mkPen('w', width=1)))
            self.x_fit_curves.append(x_plot.plot([], [], pen=pg.mkPen('y', width=2)))
            x_plot.setXLink(vb_avg)
        # Colour bars in real camera counts, one for each row of tiles. The
        # tiles carry no axis of their own, so without these nothing on screen
        # says which count value a colour means. Non-interactive: the levels
        # are driven from the acquisition side and a draggable bar would only
        # fight it. Each bar is linked to EVERY tile in its row, so all scan
        # points share one scale -- a per-tile scale would make a column with
        # half the atoms look identical to a full one, which is exactly the
        # comparison a sweep exists to make.
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
        is simply the picture with no scale beside it, which is what this
        window showed before.
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
                    self.x_data_curves[idx].setData(np.arange(x_proj.size, dtype=np.float64), x_bg)
                    gx, gy = _gaussian_curve(x_proj.size, mu_x, sigma_x, x_bg.max())
                    self.x_fit_curves[idx].setData(gx, gy)
                mu_y, sigma_y, y_bg = _sigma_from_moment(y_proj)
                if y_bg is not None and np.isfinite(sigma_y):
                    self.y_data_curves[idx].setData(y_bg, np.arange(y_proj.size, dtype=np.float64))
                    gx, gy = _gaussian_curve(y_proj.size, mu_y, sigma_y, y_bg.max())
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
        self._latest_avgs[scan_idx] = (np.asarray(avg_image).copy(), self._rendered_versions[scan_idx] + 1)


class TempFitWindow:
    POLL_MS = 1000

    def __init__(self, pixel_size_um):
        self.pixel_size_m = pixel_size_um * 1e-6
        self.win = pg.GraphicsLayoutWidget(show=True, title="Temperature from TOF")
        self.win.resize(800, 500)
        self.plot_x = self.win.addPlot(row=0, col=0, title="<b>sigma_x^2 vs t^2</b>")
        self.plot_x.setLabel('bottom', 't^2', units='s^2')
        self.plot_x.setLabel('left', 'sigma_x^2', units='m^2')
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)
        self.data_x = self.plot_x.plot([], [], pen=None, symbol='o', symbolSize=10, symbolBrush='#2196F3')
        self.fit_x = self.plot_x.plot([], [], pen=pg.mkPen('#FF9800', width=2))
        self.label_x = pg.TextItem('', color='#FF9800', anchor=(0, 1))
        self.plot_x.addItem(self.label_x)
        self.plot_y = self.win.addPlot(row=0, col=1, title="<b>sigma_y^2 vs t^2</b>")
        self.plot_y.setLabel('bottom', 't^2', units='s^2')
        self.plot_y.setLabel('left', 'sigma_y^2', units='m^2')
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)
        self.data_y = self.plot_y.plot([], [], pen=None, symbol='o', symbolSize=10, symbolBrush='#4CAF50')
        self.fit_y = self.plot_y.plot([], [], pen=pg.mkPen('#F44336', width=2))
        self.label_y = pg.TextItem('', color='#F44336', anchor=(0, 1))
        self.plot_y.addItem(self.label_y)
        self._latest_data = None
        self._rendered_version = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    def update_data(self, tof_us_list, sigma_x_list, sigma_y_list):
        self._latest_data = (list(tof_us_list), list(sigma_x_list), list(sigma_y_list), self._rendered_version + 1)

    def _poll_and_update(self):
        entry = self._latest_data
        if entry is None:
            return
        tof_us, sx_px, sy_px, version = entry
        if version <= self._rendered_version:
            return
        self._rendered_version = version
        try:
            t_s = np.array(tof_us, dtype=np.float64) * 1e-6
            t2 = t_s ** 2
            sx_m = np.array(sx_px, dtype=np.float64) * self.pixel_size_m
            sy_m = np.array(sy_px, dtype=np.float64) * self.pixel_size_m
            sx2 = sx_m ** 2
            sy2 = sy_m ** 2
            mask_x = np.isfinite(sx2) & (sx2 > 0)
            mask_y = np.isfinite(sy2) & (sy2 > 0)
            if mask_x.sum() >= 2:
                self.data_x.setData(t2[mask_x], sx2[mask_x])
                a_x, b_x = _linear_fit(t2[mask_x], sx2[mask_x])
                if np.isfinite(b_x) and b_x > 0:
                    T_x = M_RB87 * b_x / K_B
                    t2_fit = np.linspace(0, t2.max() * 1.1, 50)
                    self.fit_x.setData(t2_fit, a_x + b_x * t2_fit)
                    self.label_x.setText(f"T_x = {T_x*1e6:.1f} uK\nsigma_0x = {np.sqrt(max(a_x,0))/self.pixel_size_m:.0f} px")
                    self.label_x.setPos(t2.max() * 0.05, sx2[mask_x].max() * 0.95)
            if mask_y.sum() >= 2:
                self.data_y.setData(t2[mask_y], sy2[mask_y])
                a_y, b_y = _linear_fit(t2[mask_y], sy2[mask_y])
                if np.isfinite(b_y) and b_y > 0:
                    T_y = M_RB87 * b_y / K_B
                    t2_fit = np.linspace(0, t2.max() * 1.1, 50)
                    self.fit_y.setData(t2_fit, a_y + b_y * t2_fit)
                    self.label_y.setText(f"T_y = {T_y*1e6:.1f} uK\nsigma_0y = {np.sqrt(max(a_y,0))/self.pixel_size_m:.0f} px")
                    self.label_y.setPos(t2.max() * 0.05, sy2[mask_y].max() * 0.95)
        except Exception:
            pass


# =============================================================================
INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '800'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '6'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0'},
    'Repump_VCO': {'Repump VCO (V)': '0'},

    # ===================== PHASE 2: CMOT =====================
    'cmot_hold_time': {'CMOT Hold Time (ms)': '5'},
    'cmot_mot_power': {'CMOT MOT AOM Power (V)': '0.6'},
    'cmot_repump_power': {'CMOT Repump AOM Power (V)': '0.2'},
    'cmot_detuning_voltage': {'CMOT MOT VCO (V) [-40MHz]': '-0.6'},
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '8'},
    'cmot_repump_vco': {'CMOT Repump VCO (V) [-3MHz]': '-0.2'},

    # ===================== PHASE 3: DARK MOT =====================
    'dark_mot_enabled': {'Dark MOT Enabled (1=yes, 0=no)': '1'},
    'dark_mot_duration': {'Dark MOT Duration (ms)': '4'},
    'dark_mot_power': {'Dark MOT Cooling Power (V)': '0.2'},
    'dark_mot_detuning': {'Dark MOT Cooling VCO (V) [-55MHz]': '-0.6'},
    'dark_mot_repump_power': {'Dark MOT Repump Power (V)': '0.2'},
    'dark_mot_repump_vco': {'Dark MOT Repump VCO (V)': '0.0'},

    # ===================== PHASE 4a: PGC COOL =====================
    'pgc_cool_ms': {'PGC Cool Duration (ms)': '5'},
    'pgc_mot_power': {'PGC Cooling Power (V)': '0.2'},
    'pgc_detuning': {'PGC Cooling VCO (V) [-122MHz]': '-2.3'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.03'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '-0.05'},

    # ===================== PHASE 4b: PGC RAMP-DOWN (lattice loading) =====================
    # PGC cooling/repump ramp from full to 0. Lattice already at full power.
    # By end of rampdown, atoms only see the lattice.
    'pgc_rampdown_ms': {'PGC Rampdown Duration (ms)': '10'},

    # ===================== LATTICE AM =====================
    # DIM-3000 set to AMP=320 (+32dBm) via dim_rf_on.py; AM=1.0V => full max.
    # AOM thermal stability: lattice TTL stays ON & AM stays at lat_am_v
    # for the ENTIRE cycle.
    'lat_am_v': {'Lattice AM (V)': '0.6'},

    # ===================== FIXED LATTICE HOLD =====================
    'hold_ms': {'Lattice Hold (ms)': '20'},

    # ===================== FIXED TOF after release =====================
    'tof_us': {'TOF (us)': '5000'},

    # ===================== BIAS Y SWEEP =====================
    # Endpoints inclusive: Steps counts the POINTS, not the intervals, so
    # 0 -> 1 V in 10 steps is 0.000, 0.111, ... 1.000 -- the ten fields this
    # replaced. 1 step runs the start value on its own.
    # Each step is a full sequence (~1 s) on every one of the M runs and adds
    # a column to the scan grid, so a fine sweep costs time and screen width.
    'bias_start_v': {'Bias Y Start (V)': '0.0'},
    'bias_stop_v': {'Bias Y Stop (V)': '1.0'},
    'bias_steps': {'Bias Y Steps (points, endpoints included)': '10'},

    # ===================== IMAGING =====================
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '100'},

    # ===================== CALIBRATION =====================
    'pixel_size_um': {'Pixel Size (um/px)': '3.45'},

    # ===================== BIAS COILS =====================
    # 0V during MOT+CMOT; linearly ramped to these values starting at
    # coils_off (just before dark MOT), held through both imaging frames,
    # then ramped back to 0V after frame 2.
    'pgc_bias_x_v': {'PGC Bias X (V)': '0.0'},
    'pgc_bias_y_v': {'PGC Bias Y (V)': '0.0'},
    'pgc_bias_z_v': {'PGC Bias Z (V)': '0.0'},

    # ===================== FIXED BIASES (X, Z) =====================
    'bias_x_fixed_v': {'Bias X Fixed (V)': '0.0'},
    'bias_z_fixed_v': {'Bias Z Fixed (V)': '0.0'},

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

    # ===================== GENERAL =====================
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},
}


# ---------------------------------------------------------------------------
# Safe state applied when the run stops.  (learned from jingfastMOT)
# ---------------------------------------------------------------------------
# Without this, "stop" leaves the rails wherever the waveform happened to be.
# The AO/DO tasks run with regeneration, looping the buffer forever, and
# force_finish=True stops them mid-loop at an arbitrary sample; DAQmx then
# holds every output at that last sample when the task closes. So the coils
# could park anywhere between 0 V and the CMOT value and the VCOs at any phase
# detuning, differently on every stop.
#
# These are single-sample immediate writes made AFTER the sequence tasks are
# closed -- they would collide with the running tasks otherwise.
SAFE_AO_BASE = {
    'ao1': 0.8,     # MOT AOM drive -- held up for AOM thermal stability
    'ao5': 0.8,     # repump AOM drive
    'ao2': 0.0,     # MOT coils -- current setpoint to zero
    'ao3': 0.0,     # MOT VCO
    'ao4': 0.0,     # repump VCO
    'ao8': 0.0,     # bias X
    'ao9': 0.0,     # bias Y
    'ao10': 0.0,    # bias Z
}

# THE LATTICE IS DELIBERATELY LEFT ON, which is where this differs from
# jingfastMOT (it parks dio4/dio5 HIGH and ao6/ao7 at 0 V, i.e. lattice off).
# These scripts drive the lattice AOMs at a constant RF level for the entire
# cycle precisely so the crystals stay in one thermal state and the fibre
# coupling does not drift. Parking the lattice off between runs throws that
# away: the crystal cools, and the next run starts with the coupling walking
# back to equilibrium for minutes. So ao6/ao7 stay at the AM voltage the run
# used and dio4/dio5 stay LOW (= beam on).
#
# dio2 = 1 is "coils off": the state the sequence drives from just before
# release through imaging. The coil analog goes to 0 V regardless, so the
# current setpoint is zero either way.
SAFE_DO_LATTICE_ON = {
    'dio0': 0,      # camera trigger idle
    'dio1': 0,      # MOT AOM enable -- light off
    'dio2': 1,      # coils TTL: the sequence's "coils off" state
    'dio3': 0,      # repump AOM enable -- light off
    'dio4': 0,      # lattice up   -- LOW = ON, left running on purpose
    'dio5': 0,      # lattice down -- LOW = ON, left running on purpose
    'dio6': 0,      # DIM-3000 segment trigger idle (no move)
}


def park_outputs(NI_do, NI_ao, logger, lattice_am=0.0, lattice_on=True):
    """Drive every line to its safe value, lattice left ON and stationary.

    Best effort and never raises: this runs from a finally block, where an
    exception would mask whatever actually ended the run and would also skip
    the channels after the one that failed.

    write_static() is preferred because it forces the driver out of stack mode
    first -- if the run died between build_stack() and arm(), a plain
    set_ao_voltage() would be silently queued into a stack nobody will arm and
    nothing would actually be parked. Older servers do not have it, so fall
    back to per-channel writes, each in its own try so one bad line cannot
    leave the rest at random.

    :param lattice_am: AM voltage the run used, held on ao6/ao7
    :param lattice_on: False parks the lattice off (ao6/ao7 = 0, TTL high),
        for a run that had the lattice disabled to begin with
    """
    safe_ao = dict(SAFE_AO_BASE)
    safe_do = dict(SAFE_DO_LATTICE_ON)
    if lattice_on:
        safe_ao['ao6'] = float(lattice_am)
        safe_ao['ao7'] = float(lattice_am)
    else:
        safe_ao['ao6'] = 0.0
        safe_ao['ao7'] = 0.0
        safe_do['dio4'] = 1
        safe_do['dio5'] = 1
    try:
        NI_ao.write_static(ao=safe_ao)
        NI_do.write_static(do=safe_do)
    except Exception as exc:
        logger.warn(f"write_static unavailable ({exc}); parking channel by channel")
        for channel, value in safe_ao.items():
            try:
                NI_ao.set_ao_voltage(ao_channel=channel, voltages=value)
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value} V: {e}")
        for channel, value in safe_do.items():
            try:
                NI_do.set_do_voltage(do_channel=channel, value=bool(value))
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value}: {e}")
    logger.info(
        "Outputs parked: VCOs 0 V, coils 0 V, bias 0 V, AOM drive 0.8 V, "
        "cooling/repump light off, lattice %s."
        % (f"ON at {float(lattice_am):.2f} V (stationary)" if lattice_on
           else "OFF")
    )


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")
    dataset.camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    bias_values = bias_sweep_values(dataset)
    scan_labels = [f"BY={v:.3f}V" for v in bias_values]
    # Display range from the GUI rather than the old hardcoded (0, 20).
    disp_min, disp_max, _ = read_display_levels(dataset)
    dataset.diff_levels = (disp_min, disp_max)
    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        diff_levels=(disp_min, disp_max), avg_levels=(disp_min, disp_max),
        title="Lattice + Bias Y sweep (ao9, hold=20ms)",
    )
    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))
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
    # Phase 1: MOT
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    # Phase 2: CMOT (always on)
    CMOT_RAMP_MS = 15
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_v = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

    # Phase 3: Dark MOT (toggleable)
    dark_enabled = int(dataset.get_input_parameter("dark_mot_enabled")) != 0
    dark_ms = int(dataset.get_input_parameter("dark_mot_duration"))
    dark_power = float(dataset.get_input_parameter("dark_mot_power"))
    dark_detuning = float(dataset.get_input_parameter("dark_mot_detuning"))
    dark_rep_power = float(dataset.get_input_parameter("dark_mot_repump_power"))
    dark_rep_vco = float(dataset.get_input_parameter("dark_mot_repump_vco"))

    # Phase 4: PGC (two-stage: cool + rampdown). Always on for this experiment.
    pgc_cool_ms = int(dataset.get_input_parameter("pgc_cool_ms"))
    pgc_rampdown_ms = int(dataset.get_input_parameter("pgc_rampdown_ms"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))
    # Lattice
    lat_am_v = float(dataset.get_input_parameter("lat_am_v"))

    hold_ms = int(dataset.get_input_parameter("hold_ms"))
    tof_us = int(dataset.get_input_parameter("tof_us"))
    bias_values_v = bias_sweep_values(dataset)
    n_scan = len(bias_values_v)
    # The grid window is sized at roughly 420 px per scan point and every point
    # costs a full sequence on every run, so a step count that looks harmless
    # on the GUI is what makes a cycle long and the window wider than the
    # screen. Say so rather than letting it be a surprise.
    if n_scan > 16:
        logger.warn(
            f"{n_scan} sweep steps: the scan grid will be about "
            f"{n_scan * 420} px wide, and one pass through the buffer takes "
            f"{n_scan} sequences per run. Fewer steps, or a narrower Start .. "
            f"Stop range, if that is not what you meant.")
    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))

    # Imaging
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    bias_x_fixed = float(dataset.get_input_parameter("bias_x_fixed_v"))
    bias_z_fixed = float(dataset.get_input_parameter("bias_z_fixed_v"))
    bias_ramp_ms = int(dataset.get_input_parameter("bias_ramp_ms"))

    # =========================================================================
    # CHANNELS
    # =========================================================================
    camera_trigger_do = "dio0"
    MOT_AOM_do = "dio1"
    mot_coils_do = "dio2"
    Repump_AOM_do = "dio3"
    MOT_AOM_ao = "ao1"
    mot_coils_ao = "ao2"
    MOT_freq_VCO_ao = "ao3"
    Repump_freq_VCO_ao = "ao4"
    Repump_AOM_ao = "ao5"
    opx_trigger_do = 1

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
    cmot_ramp_s = ms2s(CMOT_RAMP_MS)
    cmot_hold_s = ms2s(cmot_hold_ms)
    dark_s = ms2s(dark_ms) if dark_enabled else 0
    pgc_cool_s = ms2s(pgc_cool_ms)
    pgc_rampdown_s = ms2s(pgc_rampdown_ms)
    pgc_s = pgc_cool_s + pgc_rampdown_s  # total PGC duration
    hold_s_fixed = ms2s(hold_ms)
    tof_s_fixed = us2s(tof_us)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    bias_ramp_s = ms2s(bias_ramp_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    # =========================================================================
    # TIMELINE ANCHORS
    # MOT -> CMOT ramp -> CMOT hold -> [Dark MOT] -> [PGC] -> release
    # =========================================================================
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s

    dark_start = cmot_end
    dark_end = dark_start + dark_s

    pgc_start = dark_end
    pgc_cool_end = pgc_start + pgc_cool_s
    pgc_end = pgc_cool_end + pgc_rampdown_s
    # release time depends on hold (per scan point) - computed in build_sub_sequence

    # Coils OFF before Dark MOT (or PGC if no Dark MOT) -- field-free phases
    if dark_enabled:
        coils_off = cmot_end - coils_lead_s
    else:
        coils_off = pgc_start - coils_lead_s

    dark_str = "ON" if dark_enabled else "OFF"
    logger.info(f"Time at start {time.perf_counter_ns()}")
    logger.info(
        f"Timeline: MOT 0-{mot_loading_ms}ms, "
        f"CMOT {mot_loading_ms}-{cmot_end/SPM:.0f}ms, "
        f"Dark MOT {dark_str} ({dark_start/SPM:.0f}-{dark_end/SPM:.0f}ms), "
        f"PGC_cool ({pgc_start/SPM:.0f}-{pgc_cool_end/SPM:.0f}ms), "
        f"PGC_rampdown+lattice load ({pgc_cool_end/SPM:.0f}-{pgc_end/SPM:.0f}ms), "
        f"lattice HOLD = {hold_ms}ms, TOF = {tof_us}us, "
        f"sweep Bias Y = {n_scan} steps, "
        f"{bias_values_v[0]:.3f} -> {bias_values_v[-1]:.3f}V "
        f"{[round(v, 4) for v in bias_values_v]}"
    )

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence(bias_swept_v):
        release = pgc_end + hold_s_fixed  # release happens AFTER lattice hold
        f1 = release + tof_s_fixed
        f2 = f1 + clear_s
        crt = f2 + aom_pulse_s + coils_restore_s  # coils stay OFF through frame 2
        N = max(crt + ttl_up + 1, f2 + ttl_up + 1)

        cam = [0] * N
        for s in range(ttl_up):
            cam[f1 + s] = 1
            cam[f2 + s] = 1

        ado = [0] * N
        aao = [mot_aom_power] * N
        rdo = [0] * N
        rao = [repump_aom_power] * N

        # Phase 1: MOT -- full power
        for s in range(cmot_start):
            ado[s] = 1
            aao[s] = mot_aom_power
            rdo[s] = 1
            rao[s] = repump_aom_power

        # Phase 2a: CMOT ramp
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            ado[t] = 1
            aao[t] = mot_aom_power + frac * (cmot_mot_power - mot_aom_power)
            rdo[t] = 1
            rao[t] = repump_aom_power + frac * (cmot_repump_power - repump_aom_power)

        # Phase 2b: CMOT hold
        for s in range(cmot_hold_s):
            t = cmot_hold_start + s
            ado[t] = 1
            aao[t] = cmot_mot_power
            rdo[t] = 1
            rao[t] = cmot_repump_power

        # Phase 3: Dark MOT (if enabled)
        if dark_enabled:
            for s in range(dark_s):
                t = dark_start + s
                ado[t] = 1
                aao[t] = dark_power
                rdo[t] = 1
                rao[t] = dark_rep_power

        # Phase 4a: PGC cool (full PGC light, lattice already on)
        for s in range(pgc_cool_s):
            t = pgc_start + s
            ado[t] = 1
            aao[t] = pgc_mot_power
            rdo[t] = 1
            rao[t] = pgc_rep_power
        # Phase 4b: PGC rampdown (cooling & repump amplitude -> 0; lattice unchanged)
        for s in range(pgc_rampdown_s):
            t = pgc_cool_end + s
            frac = s / max(pgc_rampdown_s - 1, 1)
            ado[t] = 1
            aao[t] = pgc_mot_power * (1.0 - frac)
            rdo[t] = 1
            rao[t] = pgc_rep_power * (1.0 - frac)
        # Lattice HOLD: PGC TTL off (no cooling light), analog stays at full
        # for thermal stability. Lattice provides the trapping during this time.
        for s in range(pgc_end, release):
            ado[s] = 0
            rdo[s] = 0
        # TOF: same (digital off, analog stays at full)
        for s in range(release, min(f1, N)):
            ado[s] = 0
            rdo[s] = 0

        # Imaging pulses: full MOT power
        for s in range(aom_pulse_s):
            if f1 + s < N:
                ado[f1 + s] = 1
                aao[f1 + s] = mot_aom_power
                rdo[f1 + s] = 1
                rao[f1 + s] = repump_aom_power
            if f2 + s < N:
                ado[f2 + s] = 1
                aao[f2 + s] = mot_aom_power
                rdo[f2 + s] = 1
                rao[f2 + s] = repump_aom_power

        # Coils DO: 0=ON, 1=OFF
        cdo = [0] * N
        for s in range(coils_off, min(crt, N)):
            cdo[s] = 1

        # Coils AO
        cao = [mot_coils_v] * N
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            cao[t] = mot_coils_v + frac * (cmot_coils_v - mot_coils_v)
        for s in range(cmot_hold_start, N):
            cao[s] = cmot_coils_v

        # MOT VCO
        mvco = [vco_loading] * N
        # CMOT ramp
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            mvco[t] = vco_loading + frac * (cmot_detuning - vco_loading)
        # CMOT hold
        for s in range(cmot_hold_start, cmot_end):
            mvco[s] = cmot_detuning
        # Dark MOT
        if dark_enabled:
            for s in range(dark_s):
                mvco[dark_start + s] = dark_detuning
        # PGC (both stages held at pgc_detuning)
        for s in range(pgc_s):
            mvco[pgc_start + s] = pgc_detuning
        for s in range(pgc_end, N):
            mvco[s] = pgc_detuning
        # Imaging VCO override
        vlo = max(0, f1 - vco_lead_s)
        vhi = min(N, f2 + aom_pulse_s)
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
        if dark_enabled:
            for s in range(dark_s):
                rvco[dark_start + s] = dark_rep_vco
        for s in range(pgc_s):
            rvco[pgc_start + s] = pgc_rep_vco
        for s in range(pgc_end, N):
            rvco[s] = pgc_rep_vco

        # Bias coils: held CONSTANT at swept / fixed values for the ENTIRE
        # sub-sequence (no ramping, on through MOT loading too).
        biasx = [bias_x_fixed] * N
        biasy = [bias_swept_v] * N
        biasz = [bias_z_fixed] * N

        # Lattice: TTL ON and AM constant at lat_am_v for the entire sub-seq
        # (matches the fast lattice code for AOM thermal stability).
        ludo = [0] * N
        lddo = [0] * N
        luao = [lat_am_v] * N
        ldao = [lat_am_v] * N

        return cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, biasx, biasy, biasz, ludo, lddo, luao, ldao, N

    # Concatenate all TOF sub-sequences
    cam_p = []
    ado_p = []
    aao_p = []
    rdo_p = []
    rao_p = []
    cdo_p = []
    cao_p = []
    mvco_p = []
    rvco_p = []
    biasx_p = []
    biasy_p = []
    biasz_p = []
    ludo_p = []
    lddo_p = []
    luao_p = []
    ldao_p = []
    for b_v in bias_values_v:
        (cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco,
         bx_w, by_w, bz_w, ludo, lddo, luao, ldao, N) = build_sub_sequence(b_v)
        cam_p += cam
        ado_p += ado
        aao_p += aao
        rdo_p += rdo
        rao_p += rao
        cdo_p += cdo
        cao_p += cao
        mvco_p += mvco
        rvco_p += rvco
        biasx_p += bx_w
        biasy_p += by_w
        biasz_p += bz_w
        ludo_p += ludo
        lddo_p += lddo
        luao_p += luao
        ldao_p += ldao

    total = len(cam_p)
    logger.info(
        f"Lattice + Bias Y sweep: {n_scan} steps "
        f"{bias_values_v[0]:.3f} -> {bias_values_v[-1]:.3f}V, "
        f"hold={hold_ms}ms, tof={tof_us}us, total={total} samples")

    diff_sums = [None] * n_scan
    diff_counts = [0] * n_scan

    # ---- display / scaling state (jingfastPGC) ----
    disp_min, disp_max, autoscale = read_display_levels(dataset)
    # Brightest raw pixel seen this run, and the resulting physical ceiling.
    # Both settle on the first frame and are refined if anything brighter turns
    # up; `limit` is what the requested range is clamped against.
    run_peak = None
    limit = None
    # Latest p99.95 of each tile, so one range can cover a whole row.
    cur_tops = [None] * n_scan
    avg_tops = [None] * n_scan
    # Each diagnostic fires once per run; the conditions persist for as long as
    # whatever caused them, and one warning per scan point would bury the log.
    warned = set()

    # =========================================================================
    # MAIN ACQUISITION LOOP
    # =========================================================================
    while thread.running:
        camera_client.set_hardware_trigger(
            line="Line0", activation="RisingEdge",
            selector="FrameStart", overlap="ReadOut",
            acquisition_mode="Continuous")
        camera_client.set_exposure(camera_exposure_us)
        camera_client.try_set_float("Gain", 50.0)
        logger.info("Starting acquisition")
        dataset.camera_client.start_acquisition()

        # Clear stale clock task from previous cycle
        try:
            NI_card_1.finalize_clock()
        except Exception:
            pass

        # Load waveforms into data cards first
        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=cam_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=MOT_AOM_do, value=ado_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=mot_coils_do, value=cdo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=Repump_AOM_do, value=rdo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio4", value=ludo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio5", value=lddo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio6", value=[0] * total, sample_rate=ni_rate)
        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel=MOT_AOM_ao, voltages=aao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=mot_coils_ao, voltages=cao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=MOT_freq_VCO_ao, voltages=mvco_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_freq_VCO_ao, voltages=rvco_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_AOM_ao, voltages=rao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=luao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=ldao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao8", voltages=biasx_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao9", voltages=biasy_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao10", voltages=biasz_p, sample_rate=ni_rate)

        # Arm data cards BEFORE clock
        h1 = NI_card_2.arm(regeneration=True)
        h2 = NI_card_3.arm(regeneration=True)

        # Arm clock AFTER data cards are ready
        NI_card_1.arm_clock(length=total, sample_rate=ni_rate)

        # OPX starts clocking -- all cards are ready
        OPX_client.build_stack()
        clk = OPX_client.create_new_do_elem(do_channel=opx_trigger_do, length=500)
        M = 4
        run_buffer = 10_000_000
        with OPX_client.for_("j", 0, M, 1):
            with OPX_client.for_("i", 0, total + 1, 1):
                OPX_client.set_digital_voltage(element=clk)
                OPX_client.delay(delay_ns)
            OPX_client.delay(run_buffer)

        OPX_client.execute(wait=False)
        logger.info("Image gathering starts")

        def get_frame(timeout_ms=1000, poll_ms=500):
            """One frame, waited for in short slices so Stop is seen promptly.

            The camera is triggered by the sequence, so a grab legitimately
            blocks for a whole scan point -- about a second here -- and the
            timeout has to allow for a missed trigger on top of that. Asking
            for the entire wait in one call means Stop cannot be noticed until
            the frame lands, which is what made stopping slow: thread.running
            was tested once per pass through the buffer, so a stop pressed at
            the wrong moment sat through the remaining runs and scan points.

            Spinnaker's GetNextImage does not consume anything when it times
            out, so a queue of short waits collects exactly the frame one long
            wait would have -- it just gets a chance to look at thread.running
            between them.
            """
            deadline = time.perf_counter() + timeout_ms / 1000.0
            while True:
                if not thread.running:
                    raise StopRequested()
                remaining_ms = (deadline - time.perf_counter()) * 1000.0
                if remaining_ms <= 0:
                    raise TimeoutError(
                        f"No camera frame within {timeout_ms} ms")
                started = time.perf_counter()
                try:
                    b, shape, dtype = dataset.camera_client.get_frame_bytes(
                        int(min(poll_ms, remaining_ms)))
                except Exception:
                    # A timeout has already burned its slice. A real fault --
                    # incomplete frame, RPC hiccup -- comes back at once, so
                    # pause before retrying rather than spinning on it.
                    if time.perf_counter() - started < 0.05:
                        time.sleep(0.05)
                    continue
                return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        scan_grid = dataset.scan_grid
        sigma_x_vals = [float('nan')] * n_scan
        sigma_y_vals = [float('nan')] * n_scan

        try:
            for run_idx in range(M):
                if not thread.running:
                    raise StopRequested()
                logger.info(f"Run {run_idx+1} of {M}")
                for scan_idx, b_v in enumerate(bias_values_v):
                    if not thread.running:
                        raise StopRequested()
                    is_first = (run_idx == 0 and scan_idx == 0)
                    frame1 = get_frame(timeout_ms=10000 if is_first else 100000)
                    frame2 = get_frame(timeout_ms=100000)
                    diff = frame1.astype(np.int32) - frame2.astype(np.int32)
                    if diff_sums[scan_idx] is None:
                        diff_sums[scan_idx] = diff.astype(np.float64)
                    else:
                        diff_sums[scan_idx] += diff
                    diff_counts[scan_idx] += 1
                    avg = diff_sums[scan_idx] / diff_counts[scan_idx]

                    # ------------------ display / scaling ------------------
                    # Full scale is a property of the ADC, not of the
                    # difference, so it is read off the RAW frames.
                    frame_peak = int(max(frame1.max(), frame2.max()))
                    if run_peak is None or frame_peak > run_peak:
                        run_peak = frame_peak
                    if limit is None:
                        limit = sensor_full_scale(frame1)
                        lo_req, hi_req = disp_min, disp_max
                        # The difference of two frames spans -limit .. +limit.
                        disp_min = max(disp_min, -float(limit))
                        disp_max = min(disp_max, float(limit))
                        logger.info(
                            f"Camera full scale: {limit} counts "
                            f"({frame1.dtype}). Display range limited to "
                            f"[{-limit}, {limit}].")
                        if frame1.dtype == np.uint8:
                            logger.error(
                                f"Camera is sending 8-bit frames despite the "
                                f"request for {PIXEL_FORMAT}. Every count is "
                                f"16x coarser than Mono16, so a cloud that "
                                f"should read tens of counts arrives as 1-3 "
                                f"and is quantised into a handful of grey "
                                f"levels. Nothing on the display side can undo "
                                f"that -- fix the PixelFormat (check SpinView "
                                f"is not holding the camera, and that "
                                f"Mono16/Mono12p is available on this model).")
                        if (lo_req, hi_req) != (disp_min, disp_max):
                            logger.warn(
                                f"Requested display range ({lo_req:.0f}, "
                                f"{hi_req:.0f}) exceeds what a pixel can "
                                f"reach; using ({disp_min:.0f}, "
                                f"{disp_max:.0f}).")
                    if autoscale:
                        # Top follows the signal, bottom stays pinned at
                        # Display Min so the background does not breathe. The
                        # top is the largest over the scan points seen so far,
                        # not this tile's own, so the columns stay comparable.
                        cur_tops[scan_idx] = autoscale_top(diff, disp_min)
                        avg_tops[scan_idx] = autoscale_top(avg, disp_min)
                        cur_levels = (disp_min,
                                      max(t for t in cur_tops if t is not None))
                        avg_levels = (disp_min,
                                      max(t for t in avg_tops if t is not None))
                    else:
                        cur_levels = (disp_min, disp_max)
                        avg_levels = (disp_min, disp_max)
                    scan_grid.set_current(scan_idx, diff, levels=cur_levels)
                    scan_grid.set_average(scan_idx, avg, levels=avg_levels)
                    scan_grid.set_levels_hint(cur_levels, avg_levels)
                    img = avg.astype(np.float64)
                    _, sx, _ = _sigma_from_moment(img.sum(axis=0))
                    _, sy, _ = _sigma_from_moment(img.sum(axis=1))
                    sigma_x_vals[scan_idx] = sx
                    sigma_y_vals[scan_idx] = sy
                    logger.info(
                        f"  BiasY={b_v:.2f}V: n={diff_counts[scan_idx]}, "
                        f"sx={sx:.1f}px, sy={sy:.1f}px, "
                        f"area={float(avg.sum()):.0f}")

                    # Diagnostics, from jingfastPGC. "All dark" has several
                    # distinct causes and the numbers below separate them: no
                    # light on the sensor at all, light but no atoms, a signal
                    # pointing the wrong way, or a display range that simply
                    # does not cover the data.
                    d_lo = float(np.percentile(diff, 0.05))
                    d_hi = float(np.percentile(diff, 99.95))
                    # float64 accumulator: five million int32 pixels overflow
                    # an int32 sum at a mean of only ~430 counts.
                    area = float(diff.sum(dtype=np.float64))
                    # What the integrated area would be from noise alone.
                    # Per-pixel scatter grows as sqrt(N) when summed, so
                    # anything much larger than this is real signal -- and its
                    # SIGN says which frame the atoms were in.
                    noise_area = float(diff.std()) * np.sqrt(diff.size)
                    # Area alone is not evidence of atoms -- see
                    # localized_signal.
                    loc = localized_signal(diff)
                    lo, hi = cur_levels
                    logger.info(
                        f"    raw f1={float(frame1.mean()):.0f} "
                        f"f2={float(frame2.mean()):.0f} "
                        f"peak={run_peak}/{limit} | "
                        f"diff min={float(diff.min()):.0f} p0.05={d_lo:.1f} "
                        f"med={float(np.median(diff)):.1f} "
                        f"p99.95={d_hi:.1f} max={float(diff.max()):.0f} | "
                        f"area={area:.3e} "
                        f"({area / max(noise_area, 1):+.0f}sig, "
                        f"mean {area / diff.size:+.3f}/px) | "
                        + (f"cloud {loc['peak'] - loc['background']:+.3f}/px "
                           f"({loc['excess_sigma']:+.1f}sig) at "
                           f"({loc['x']},{loc['y']}) | " if loc else "")
                        + f"levels=({lo:.0f},{hi:.0f}) "
                        + ('auto' if autoscale else 'manual'))

                    # An area that is huge but spatially flat is a black-level
                    # shift between the two frames, not atoms. Say so
                    # explicitly -- the raw sigma on the area is the single
                    # most misleading number here.
                    if loc is not None and abs(area) > 5 * noise_area \
                            and loc['excess_sigma'] < 5 and 'flat' not in warned:
                        warned.add('flat')
                        logger.warn(
                            f"Integrated area is "
                            f"{area / max(noise_area, 1):+.0f} sigma but the "
                            f"excess is UNIFORM: the brightest "
                            f"{loc['block_px']}-pixel block is only "
                            f"{loc['excess_sigma']:+.1f} sigma above the rest "
                            f"of the frame. That is a "
                            f"{area / diff.size:+.3f} count/px baseline "
                            f"difference between frame 1 and frame 2, not a "
                            f"cloud. Ignore the area sigma; it counts a "
                            f"sub-ADU pedestal across {diff.size / 1e6:.1f}M "
                            f"pixels as signal.")

                    # Is the sensor seeing light at all? Compare against the
                    # pedestal, not against zero: this camera's floor sits
                    # around 1180 counts.
                    if run_peak is not None and limit \
                            and run_peak < 0.02 * limit:
                        if 'dark' not in warned:
                            warned.add('dark')
                            logger.warn(
                                f"Sensor is nearly dark: brightest raw pixel "
                                f"{run_peak} of {limit}. No light is reaching "
                                f"the camera -- check the imaging pulse, "
                                f"exposure and trigger, not the display.")
                    # A cloud in frame 2 rather than frame 1 integrates
                    # NEGATIVE. The percentile tails cannot see this -- read
                    # noise makes them nearly symmetric either way -- but the
                    # area cannot miss it.
                    elif area < -5.0 * noise_area:
                        if 'negative' not in warned:
                            warned.add('negative')
                            logger.warn(
                                f"Signal is NEGATIVE: area={area:.3e} "
                                f"({area / max(noise_area, 1):.0f} sigma below "
                                f"zero). Frame 2 is brighter than frame 1, so "
                                f"the frame pair is swapped and the atoms are "
                                f"being subtracted the wrong way round. "
                                f"Nothing shows because the cloud sits below "
                                f"Display Min.")
                    # No signal either way -- the display is not the problem.
                    elif abs(area) < 3.0 * noise_area:
                        if 'nosignal' not in warned:
                            warned.add('nosignal')
                            logger.warn(
                                f"No signal: area={area:.3e} is within noise "
                                f"({noise_area:.1e}). The two frames are "
                                f"identical to within read noise, so there "
                                f"are no atoms in frame 1.")
                    # A range that cannot show the data is the commonest
                    # 'all dark'.
                    elif not autoscale and d_hi > 0 and hi > 10.0 * d_hi:
                        if 'range_high' not in warned:
                            warned.add('range_high')
                            logger.warn(
                                f"Display Max {hi:.0f} is {hi / d_hi:.0f}x the "
                                f"actual signal (p99.95={d_hi:.1f}); "
                                f"everything renders near the bottom of the "
                                f"colour range. Try Display Max ~{d_hi:.0f}.")
                    elif not autoscale and d_hi > hi:
                        if 'range_low' not in warned:
                            warned.add('range_low')
                            logger.warn(
                                f"Signal (p99.95={d_hi:.1f}) exceeds Display "
                                f"Max {hi:.0f}; the cloud is clipped. Try "
                                f"Display Max ~{d_hi * 1.2:.0f}.")

                time.sleep(wait_time)
        except StopRequested:
            logger.info(
                "Stop requested -- ending this cycle here instead of "
                "finishing the remaining runs and scan points.")
        finally:
            logger.info("Stopping acquisition")
            dataset.camera_client.stop_acquisition()

        logger.info(f"Cycle over {time.perf_counter_ns()}")
        time.sleep(wait_time)
        try:
            NI_card_2.finalize(h1, timeout=120.0, force_finish=True)
        except Exception as e:
            logger.info(f"NI card 2 finalize: {e!r}")
        try:
            NI_card_3.finalize(h2, timeout=120.0, force_finish=True)
        except Exception as e:
            logger.info(f"NI card 3 finalize: {e!r}")
        try:
            NI_card_1.finalize_clock()
        except Exception as e:
            logger.info(f"Clock finalize: {e!r}")
        # Sequence tasks are closed now, so static writes can land.
        # The lattice is deliberately left ON and stationary; see
        # park_outputs() for why parking it off would cost coupling.
        park_outputs(NI_card_2, NI_card_3, logger, lattice_am=lat_am_v)
