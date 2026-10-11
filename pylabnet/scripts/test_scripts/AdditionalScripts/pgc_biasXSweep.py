# PGC Bias X sweep with full cooling sequence.
# Full sequence: MOT -> CMOT -> Dark MOT -> PGC -> release -> TOF(20ms) -> image
# Sweep ao8 (Bias X) from 0 to 0.22V in 10 points.
# Bias ramps 0 -> PGC value (ramped at coils_off) -> hold through F2 -> back to 0.
#
# Sequence per scan point:
#   1. MOT loading (full power, coils on)
#   2a. CMOT ramp (10ms hardcoded)
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


def _resolve_roi(shape, x0, x1, y0, y1):
    """Clamp a user ROI to the frame; any -1 means the whole frame.

    Returns (x0, x1, y0, y1) as a half-open [x0, x1) x [y0, y1) box.
    """
    h, w = shape
    if min(x0, x1, y0, y1) < 0:
        return 0, w, 0, h
    x0, x1 = sorted((int(np.clip(x0, 0, w)), int(np.clip(x1, 0, w))))
    y0, y1 = sorted((int(np.clip(y0, 0, h)), int(np.clip(y1, 0, h))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0, w, 0, h
    return x0, x1, y0, y1


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


DIFF_LEVELS = (0, 20)
AVG_DIFF_LEVELS = (0, 20)


class ScanGridWindow:
    MARGIN_PX = 100
    POLL_MS = 500

    def __init__(self, scan_labels, diff_levels, avg_levels,
                 title="Scan grid", tile_px=320, roi=None):
        self.scan_labels = list(scan_labels)
        self.n_scan = len(self.scan_labels)
        self.diff_levels = diff_levels
        self.avg_levels = avg_levels
        # (x0, x1, y0, y1) or None for the whole frame. Everything the
        # header reports -- sx, sy and area -- is measured inside this box,
        # so scattered light outside it cannot capture the Gaussian fit or
        # be counted as signal. The box is drawn on each average image so
        # it can be placed by eye.
        self.roi = roi
        self.roi_outlines = []
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
            outline = pg.PlotDataItem(pen=pg.mkPen('#00FF88', width=1))
            vb_avg.addItem(outline)
            self.roi_outlines.append(outline)
            if self.roi is not None and min(self.roi) >= 0:
                rx0, rx1, ry0, ry1 = self.roi
                outline.setData([rx0, rx1, rx1, rx0, rx0],
                                [ry0, ry0, ry1, ry1, ry0])
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
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    def _poll_and_update(self):
        for idx in range(self.n_scan):
            entry = self._latest_avgs[idx]
            if entry is None:
                continue
            avg_copy, version = entry
            if version <= self._rendered_versions[idx]:
                continue
            self._rendered_versions[idx] = version
            try:
                full = avg_copy.astype(np.float64)
                rx0, rx1, ry0, ry1 = _resolve_roi(
                    full.shape, *(self.roi if self.roi is not None
                                  else (-1, -1, -1, -1)))
                img = full[ry0:ry1, rx0:rx1]
                x_proj = img.sum(axis=0)
                y_proj = img.sum(axis=1)
                # Curves are drawn at ABSOLUTE pixel coordinates, not at
                # 0..roi_width: these panels are x/y-linked to the image, so
                # a cropped profile plotted from zero would sit under the
                # wrong part of the picture.
                mu_x, sigma_x, x_bg = _sigma_from_moment(x_proj)
                if x_bg is not None and np.isfinite(sigma_x):
                    self.x_data_curves[idx].setData(
                        np.arange(rx0, rx1, dtype=np.float64), x_bg)
                    gx, gy = _gaussian_curve(x_proj.size, mu_x, sigma_x, x_bg.max())
                    self.x_fit_curves[idx].setData(gx + rx0, gy)
                mu_y, sigma_y, y_bg = _sigma_from_moment(y_proj)
                if y_bg is not None and np.isfinite(sigma_y):
                    self.y_data_curves[idx].setData(
                        y_bg, np.arange(ry0, ry1, dtype=np.float64))
                    gx, gy = _gaussian_curve(y_proj.size, mu_y, sigma_y, y_bg.max())
                    self.y_fit_curves[idx].setData(gy, gx + ry0)
                sx = f"{sigma_x:.1f}" if np.isfinite(sigma_x) else "--"
                sy = f"{sigma_y:.1f}" if np.isfinite(sigma_y) else "--"
                area = float(img.sum())    # ROI only, see _resolve_roi
                self.header_labels[idx].setText(
                    f"<b>{self.scan_labels[idx]}</b><br>"
                    f"<span style='color:#FFFF00'>sx={sx} sy={sy}</span><br>"
                    f"<span style='color:#00FF88'>area={area:.0f}</span>")
            except Exception:
                pass

    def set_current(self, scan_idx, diff_image):
        try:
            self.current_imgs[scan_idx].setImage(diff_image, levels=self.diff_levels, autoLevels=False)
        except Exception:
            pass

    def set_average(self, scan_idx, avg_image):
        try:
            self.avg_imgs[scan_idx].setImage(avg_image, levels=self.avg_levels, autoLevels=False)
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
# ---------------------------------------------------------------------------
# Safe state applied when the run stops.
# ---------------------------------------------------------------------------
# Without this, "stop" leaves the rails wherever the waveform happened to be.
# The AO/DO tasks run with regeneration, looping the buffer forever, and
# force_finish=True stops them mid-loop at an arbitrary sample; DAQmx then holds
# every output at that last sample when the task closes. So the coils could park
# anywhere between 0 V and the CMOT value and the VCOs at any phase detuning,
# differently on every stop.
#
# These are written as single-sample immediate writes AFTER the sequence tasks
# are closed -- they would collide with the running tasks otherwise.
SAFE_AO = {
    'ao1': 0.8,     # MOT AOM drive -- held up for AOM thermal stability
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
#
# dio2 = 1 is "coils off": the sequence drives it HIGH from just before release
# through imaging, the window where the quadrupole must be off, and LOW during
# the MOT when it must be on. The coil analog goes to 0 V regardless, so the
# current setpoint is zero either way.
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


INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '800'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '6'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0'},
    'Repump_VCO': {'Repump VCO (V)': '0'},

    # ===================== PHASE 2: CMOT =====================
    # MOT -> CMOT ramp duration. Was hardcoded at 10 ms (CMOT_RAMP_MS),
    # which made it impossible to match jingfastPGC without editing the file.
    'mot_switch_time': {'MOT Switch Time (ms) [MOT->CMOT ramp]': '15'},
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

    # ===================== PHASE 4: PGC (optical molasses) =====================
    # Sub-Doppler cooling: no B-field, very far red-detuned, low power.
    # Sisyphus effect cools atoms well below Doppler limit.
    'pgc_enabled': {'PGC Enabled (1=yes, 0=no)': '1'},
    'pgc_duration': {'PGC Duration (ms)': '5'},
    'pgc_mot_power': {'PGC Cooling Power (V)': '0.2'},
    'pgc_detuning': {'PGC Cooling VCO (V) [-122MHz]': '-2.3'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.03'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '-0.05'},

    # ===================== FIXED TOF =====================
    'tof_us': {'TOF (us)': '0'},

    # ===================== BIAS X SWEEP (10 points, 0-1V) =====================
    'bias_1_v': {'Bias X 1 (V)': '0.000'},
    'bias_2_v': {'Bias X 2 (V)': '0.024'},
    'bias_3_v': {'Bias X 3 (V)': '0.049'},
    'bias_4_v': {'Bias X 4 (V)': '0.073'},
    'bias_5_v': {'Bias X 5 (V)': '0.098'},
    'bias_6_v': {'Bias X 6 (V)': '0.122'},
    'bias_7_v': {'Bias X 7 (V)': '0.147'},
    'bias_8_v': {'Bias X 8 (V)': '0.171'},
    'bias_9_v': {'Bias X 9 (V)': '0.196'},
    'bias_10_v': {'Bias X 10 (V)': '0.220'},

    # ===================== IMAGING =====================
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'coils_rampback_ms': {'Coils Rampback Duration (ms)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '100'},

    # ---------- Imaging repump prepulse (same as jingfastPGC) ----------
    # Repump fires alone for this long before the MOT light joins it, so the
    # atoms are pumped up out of F=1 and are all cycling by the time the
    # camera opens. Both frames get the identical sequence -- the subtraction
    # only works if frame 2 saw exactly the same light as frame 1. The repump
    # then STAYS on at these settings through the MOT pulse; turning it off
    # there would let atoms fall dark part-way through the exposure. Set the
    # time to 0 to go back to repump and MOT light starting together.
    'img_repump_time_us': {'Imaging Repump Lead Time (us)': '200'},
    'img_repump_power': {'Imaging Repump Power (V)': '0.8'},
    'img_repump_vco': {'Imaging Repump VCO (V)': '0.06'},

    # ===================== ANALYSIS ROI =====================
    # Box the cloud sits in, as [x0, x1) x [y0, y1) in pixels. sx, sy and
    # area are all measured inside it, so the scattered-light band at the
    # edge of the frame cannot capture the Gaussian fit or be counted as
    # signal. Set any of the four to -1 to use the whole frame. The box is
    # outlined in green on each averaged image.
    'roi_x0': {'ROI x0 (-1 = full frame)': '-1'},
    'roi_x1': {'ROI x1': '-1'},
    'roi_y0': {'ROI y0': '-1'},
    'roi_y1': {'ROI y1': '-1'},

    # ===================== CALIBRATION =====================
    'pixel_size_um': {'Pixel Size (um/px)': '3.45'},

    # ===================== FIXED BIASES (Y, Z) =====================
    # Same ramped profile as swept axis (0 -> value at coils_off -> hold -> back to 0)
    'bias_y_fixed_v': {'Bias Y Fixed (V)': '0.0'},
    'bias_z_fixed_v': {'Bias Z Fixed (V)': '0.0'},
    'bias_ramp_ms': {'Bias Ramp Duration (ms)': '5'},

    # ===================== GENERAL =====================
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},

    # ===================== WARM-UP =====================
    # Cycles run before any data is kept, at the FIRST sweep point's
    # settings. Between one arming and the next the coils are off and the
    # light is down for seconds while the waveform is finalized, uploaded
    # and re-armed, so the MOT is completely gone. The first sub-sequence
    # after that loads from ZERO atoms, while every later one only tops up
    # a MOT that never fully decayed -- which is why the first point or two
    # come out different. These cycles absorb that transient. Set to 0 to
    # keep every shot; each one costs a full sub-sequence of run time.
    'warmup_cycles': {'Warm-up Cycles (0 = none)': '3'},
}


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")
    dataset.camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    bias_values = [
        float(dataset.get_input_parameter("bias_1_v")),
        float(dataset.get_input_parameter("bias_2_v")),
        float(dataset.get_input_parameter("bias_3_v")),
        float(dataset.get_input_parameter("bias_4_v")),
        float(dataset.get_input_parameter("bias_5_v")),
        float(dataset.get_input_parameter("bias_6_v")),
        float(dataset.get_input_parameter("bias_7_v")),
        float(dataset.get_input_parameter("bias_8_v")),
        float(dataset.get_input_parameter("bias_9_v")),
        float(dataset.get_input_parameter("bias_10_v")),
    ]
    scan_labels = [f"BX={v:.2f}V" for v in bias_values]
    roi = (int(dataset.get_input_parameter("roi_x0")),
           int(dataset.get_input_parameter("roi_x1")),
           int(dataset.get_input_parameter("roi_y0")),
           int(dataset.get_input_parameter("roi_y1")))
    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        roi=roi,
        diff_levels=DIFF_LEVELS, avg_levels=AVG_DIFF_LEVELS,
        title="PGC Bias X sweep (ao8, TOF=20ms)",
        tile_px=180,
    )
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
    mot_switch_ms = int(dataset.get_input_parameter("mot_switch_time"))
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

    # Phase 4: PGC (toggleable)
    pgc_enabled = int(dataset.get_input_parameter("pgc_enabled")) != 0
    pgc_ms = int(dataset.get_input_parameter("pgc_duration"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))

    tof_us = int(dataset.get_input_parameter("tof_us"))
    bias_values_v = [
        float(dataset.get_input_parameter("bias_1_v")),
        float(dataset.get_input_parameter("bias_2_v")),
        float(dataset.get_input_parameter("bias_3_v")),
        float(dataset.get_input_parameter("bias_4_v")),
        float(dataset.get_input_parameter("bias_5_v")),
        float(dataset.get_input_parameter("bias_6_v")),
        float(dataset.get_input_parameter("bias_7_v")),
        float(dataset.get_input_parameter("bias_8_v")),
        float(dataset.get_input_parameter("bias_9_v")),
        float(dataset.get_input_parameter("bias_10_v")),
    ]
    n_scan = len(bias_values_v)

    # Imaging
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))
    coils_rampback_ms = int(dataset.get_input_parameter("coils_rampback_ms"))
    img_repump_us = int(dataset.get_input_parameter("img_repump_time_us"))
    img_rep_power = float(dataset.get_input_parameter("img_repump_power"))
    img_rep_vco = float(dataset.get_input_parameter("img_repump_vco"))
    wait_time = float(dataset.get_input_parameter("wait_time"))
    warmup_cycles = max(0, int(dataset.get_input_parameter("warmup_cycles")))

    bias_y_fixed = float(dataset.get_input_parameter("bias_y_fixed_v"))
    bias_z_fixed = float(dataset.get_input_parameter("bias_z_fixed_v"))
    bias_ramp_ms = int(dataset.get_input_parameter("bias_ramp_ms"))
    roi_req = (int(dataset.get_input_parameter("roi_x0")),
               int(dataset.get_input_parameter("roi_x1")),
               int(dataset.get_input_parameter("roi_y0")),
               int(dataset.get_input_parameter("roi_y1")))

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
    cmot_ramp_s = ms2s(mot_switch_ms)
    cmot_hold_s = ms2s(cmot_hold_ms)
    dark_s = ms2s(dark_ms) if dark_enabled else 0
    pgc_s = ms2s(pgc_ms) if pgc_enabled else 0
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    bias_ramp_s = ms2s(bias_ramp_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))
    coils_rampback_s = ms2s(coils_rampback_ms)
    # Repump-only prepulse, then the MOT pulse. One sample is 50 us at
    # 20 kS/s, so anything under 25 us rounds away -- 200 us is 4 samples.
    img_rep_s = us2s(img_repump_us)
    img_win_s = img_rep_s + aom_pulse_s     # whole light window per frame

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
    pgc_end = pgc_start + pgc_s

    release = pgc_end

    # Coils OFF: before Dark MOT if it's enabled, else before PGC, else before release
    # Dark MOT and PGC are both field-free phases
    if dark_enabled:
        coils_off = cmot_end - coils_lead_s
    elif pgc_enabled:
        coils_off = pgc_start - coils_lead_s
    else:
        coils_off = release - coils_lead_s

    dark_str = "ON" if dark_enabled else "OFF"
    pgc_str = "ON" if pgc_enabled else "OFF"
    logger.info(f"Time at start {time.perf_counter_ns()}")
    logger.info(
        f"Timeline: MOT 0-{mot_loading_ms}ms, "
        f"CMOT {mot_loading_ms}-{cmot_end/SPM:.0f}ms, "
        f"Dark MOT {dark_str} ({dark_start/SPM:.0f}-{dark_end/SPM:.0f}ms), "
        f"PGC {pgc_str} ({pgc_start/SPM:.0f}-{pgc_end/SPM:.0f}ms), "
        f"coils OFF at {coils_off/SPM:.0f}ms, "
        f"release at {release/SPM:.0f}ms, "
        f"TOF={tof_us}us, bias_x_sweep={bias_values_v}V"
    )

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    tof_s = us2s(tof_us)

    def build_sub_sequence(bias_swept_v):
        # f1/f2 are where the LIGHT starts (repump first). The camera fires
        # img_rep_s later, when the MOT light joins -- the prepulse produces
        # almost no fluorescence, so an exposure covering only it would come
        # back empty.
        f1 = release + tof_s
        f2 = f1 + clear_s
        img_light_1 = f1 + img_rep_s        # MOT light + camera, frame 1
        img_light_2 = f2 + img_rep_s        # MOT light + camera, frame 2
        crt = f2 + img_win_s + coils_restore_s  # coils stay OFF through frame 2
        N = crt + coils_rampback_s + 2

        cam = [0] * N
        for s in range(ttl_up):
            cam[img_light_1 + s] = 1
            cam[img_light_2 + s] = 1

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

        # Phase 4: PGC (if enabled) -- very far detuned, low power, no B-field
        if pgc_enabled:
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

        # Imaging: repump alone for img_rep_s, then the MOT light joins it.
        # Repump holds its imaging power for the whole window rather than
        # handing back to repump_aom_power, so nothing changes about the
        # light mid-exposure.
        for s in range(img_win_s):
            for base in (f1, f2):
                t = base + s
                if t >= N:
                    continue
                rdo[t] = 1
                rao[t] = img_rep_power
                if s >= img_rep_s:      # prepulse over -- cooling light on
                    ado[t] = 1
                    aao[t] = mot_aom_power

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
        # Rampback: analog 0 -> mot_coils_v over coils_rampback_s, so the
        # quadrupole comes back softly instead of stepping to full current.
        for s in range(coils_rampback_s):
            t = crt + s
            if t >= N:
                break
            frac = s / max(coils_rampback_s - 1, 1)
            cao[t] = frac * mot_coils_v
        for s in range(crt + coils_rampback_s, N):
            cao[s] = mot_coils_v

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
        # PGC
        if pgc_enabled:
            for s in range(pgc_s):
                mvco[pgc_start + s] = pgc_detuning
        # After last phase: fill with last detuning
        if pgc_enabled:
            last_det = pgc_detuning
            last_end = pgc_end
        elif dark_enabled:
            last_det = dark_detuning
            last_end = dark_end
        else:
            last_det = cmot_detuning
            last_end = cmot_end
        for s in range(last_end, N):
            mvco[s] = last_det
        # Imaging VCO override. The lead is measured from the MOT AOM
        # opening, which is now img_rep_s after the repump's, so the cooling
        # VCO still gets its full settling time.
        vlo = max(0, img_light_1 - vco_lead_s)
        vhi = min(N, f2 + img_win_s)
        for s in range(vlo, vhi):
            mvco[s] = vco_imaging

        # Repump VCO
        rvco = [repump_vco] * N
        # CMOT ramp
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            rvco[t] = repump_vco + frac * (cmot_repump_vco - repump_vco)
        # CMOT hold
        for s in range(cmot_hold_start, cmot_end):
            rvco[s] = cmot_repump_vco
        # Dark MOT
        if dark_enabled:
            for s in range(dark_s):
                rvco[dark_start + s] = dark_rep_vco
        # PGC
        if pgc_enabled:
            for s in range(pgc_s):
                rvco[pgc_start + s] = pgc_rep_vco
        # After last phase
        if pgc_enabled:
            last_rvco = pgc_rep_vco
        elif dark_enabled:
            last_rvco = dark_rep_vco
        else:
            last_rvco = cmot_repump_vco
        for s in range(last_end, N):
            rvco[s] = last_rvco
        # Imaging repump VCO. Gets the same lead as the cooling VCO,
        # referenced to the repump AOM opening at f1 -- it has to be settled
        # before the prepulse, which is the whole point of firing early.
        rvlo = max(0, f1 - vco_lead_s)
        rvhi = min(N, f2 + img_win_s)
        for s in range(rvlo, rvhi):
            rvco[s] = img_rep_vco

        # Bias coils: 0V during MOT+CMOT, ramp UP to PGC values at coils_off,
        # hold through both imaging frames, then ramp DOWN to 0V after F2.
        bias_down_start = f2 + img_win_s
        biasx = [0.0] * N
        biasy = [0.0] * N
        biasz = [0.0] * N
        # Ramp UP at coils_off
        for s in range(bias_ramp_s):
            t = coils_off + s
            if t >= N:
                break
            frac = s / max(bias_ramp_s - 1, 1)
            biasx[t] = frac * bias_swept_v
            biasy[t] = frac * bias_y_fixed
            biasz[t] = frac * bias_z_fixed
        # Hold at PGC values from end of up-ramp through F2
        for s in range(coils_off + bias_ramp_s, min(bias_down_start, N)):
            biasx[s] = bias_swept_v
            biasy[s] = bias_y_fixed
            biasz[s] = bias_z_fixed
        # Ramp DOWN to 0V after F2
        for s in range(bias_ramp_s):
            t = bias_down_start + s
            if t >= N:
                break
            frac = 1.0 - s / max(bias_ramp_s - 1, 1)
            biasx[t] = frac * bias_swept_v
            biasy[t] = frac * bias_y_fixed
            biasz[t] = frac * bias_z_fixed

        return cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, biasx, biasy, biasz, N

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
    # Warm-up first, at the first point's settings, then the real sweep.
    # They are part of the SAME regenerating buffer, so they run at the head
    # of every pass and the MOT is always in its cyclic steady state by the
    # time point 1 is recorded. Their frames are discarded in the
    # acquisition loop below.
    warm_values = [bias_values_v[0]] * warmup_cycles
    for b_v in warm_values + bias_values_v:
        cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, bx_w, by_w, bz_w, N = build_sub_sequence(b_v)
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

    total = len(cam_p)
    logger.info(
        f"PGC Bias X sweep: values={bias_values_v}V, tof={tof_us}us, "
        f"{warmup_cycles} warm-up cycle(s) at {bias_values_v[0]}V, "
        f"total={total} samples ({total / SPM / 1000:.2f} s per pass)")

    diff_sums = [None] * n_scan
    diff_counts = [0] * n_scan

    # =========================================================================
    # MAIN ACQUISITION LOOP
    # =========================================================================
    # Everything below is wrapped so that however the run ends -- Stop,
    # a timeout, or a DaqError while arming -- the rails are parked at the
    # same idle state jingfastMOT uses instead of being left wherever the
    # regenerating waveform happened to be interrupted.
    h1 = h2 = None
    try:
        while thread.running:
            camera_client.set_hardware_trigger(
                line="Line0", activation="RisingEdge",
                selector="FrameStart", overlap="ReadOut",
                acquisition_mode="Continuous")
            camera_client.set_exposure(camera_exposure_us)
            camera_client.try_set_float("Gain", 50.0)
            logger.info("Starting acquisition")
            dataset.camera_client.start_acquisition()

            # Throw away anything the camera is still holding from the previous
            # cycle. Nothing is armed yet and the OPX is not clocking, so no
            # genuine trigger can have fired -- every frame returned here is
            # stale by definition, and one left in the buffer would offset EVERY
            # pair for the rest of the run.
            #
            # Draining rather than discarding a fixed first pair is deliberate.
            # The ten sub-sequences are concatenated into one waveform and the
            # frames arrive strictly in order, so dropping a pair here would
            # shift every panel by one bias value permanently and pair each
            # background with the NEXT point's atoms. Only a whole pass
            # (2 x n_scan frames) could be dropped safely, and that would throw
            # away a quarter of the data.
            stale = 0
            while True:
                try:
                    dataset.camera_client.get_frame_bytes(200)
                    stale += 1
                except Exception:
                    break
            if stale:
                logger.warn(
                    f"Discarded {stale} stale frame(s) buffered before this "
                    f"cycle; without this every pair would have been offset "
                    f"by one.")

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
            NI_card_2.set_do_voltage(do_channel="dio4", value=[1] * total, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel="dio5", value=[1] * total, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel="dio6", value=[0] * total, sample_rate=ni_rate)
            NI_card_3.build_stack()
            NI_card_3.set_ao_voltage(ao_channel=MOT_AOM_ao, voltages=aao_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel=mot_coils_ao, voltages=cao_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel=MOT_freq_VCO_ao, voltages=mvco_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel=Repump_freq_VCO_ao, voltages=rvco_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel=Repump_AOM_ao, voltages=rao_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=[0.0] * total, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=[0.0] * total, sample_rate=ni_rate)
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

            def get_frame(timeout_ms=1000):
                b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
                return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

            scan_grid = dataset.scan_grid

            # Stop responsiveness. thread.running is what the GUI's Stop button
            # clears, and it used to be tested ONLY by the outer `while`.
            # Everything below -- M passes x n_scan points x 2 blocking reads --
            # ran to completion regardless, so Stop took at minimum a full set of
            # passes, and if frames had stopped arriving it took up to
            # 2*n_scan*M reads x 100 s of timeout before thread.running was even
            # looked at again. It is now tested at every level.
            #
            # The timeout is derived from the sequence instead of a flat 100 s:
            # the longest legitimate gap between frames is one sub-sequence, so
            # anything past a few of those is a dead trigger, not a slow one.
            # N is one sub-sequence. total/n_scan would be wrong now that
            # the buffer also carries the warm-up cycles.
            sub_ms = N / SPM
            frame_timeout_ms = int(3 * sub_ms + 2000)
            first_timeout_ms = max(15000, frame_timeout_ms)
            logger.info(
                f"Sub-sequence {sub_ms:.0f} ms; frame timeout "
                f"{frame_timeout_ms} ms (first {first_timeout_ms} ms)")

            def grab(timeout_ms):
                """One frame, or None if it timed out or Stop was pressed."""
                if not thread.running:
                    return None
                try:
                    return get_frame(timeout_ms=timeout_ms)
                except Exception as exc:
                    if thread.running:
                        logger.warn(
                            f"Frame timed out after {timeout_ms} ms ({exc}); "
                            f"abandoning this pass and re-arming.")
                    return None

            stopping = False
            try:
                for run_idx in range(M):
                    if not thread.running:
                        stopping = True
                        break
                    logger.info(f"Run {run_idx+1} of {M}")
                    # The warm-up cycles sit at the head of the buffer and
                    # fire their camera triggers like any other cycle, so
                    # their frames must be consumed -- otherwise every pair
                    # after them would be offset by one.
                    for _ in range(2 * warmup_cycles):
                        if grab(frame_timeout_ms) is None:
                            stopping = True
                            break
                    if stopping:
                        break
                    for scan_idx, b_v in enumerate(bias_values_v):
                        if not thread.running:
                            stopping = True
                            break
                        is_first = (run_idx == 0 and scan_idx == 0)
                        frame1 = grab(first_timeout_ms if is_first
                                      else frame_timeout_ms)
                        if frame1 is None:
                            stopping = True
                            break
                        frame2 = grab(frame_timeout_ms)
                        if frame2 is None:
                            stopping = True
                            break
                        diff = frame1.astype(np.int32) - frame2.astype(np.int32)
                        scan_grid.set_current(scan_idx, diff)
                        if diff_sums[scan_idx] is None:
                            diff_sums[scan_idx] = diff.astype(np.float64)
                        else:
                            diff_sums[scan_idx] += diff
                        diff_counts[scan_idx] += 1
                        avg = diff_sums[scan_idx] / diff_counts[scan_idx]
                        scan_grid.set_average(scan_idx, avg)
                        ax0, ax1, ay0, ay1 = _resolve_roi(avg.shape, *roi_req)
                        img = avg.astype(np.float64)[ay0:ay1, ax0:ax1]
                        _, sx, _ = _sigma_from_moment(img.sum(axis=0))
                        _, sy, _ = _sigma_from_moment(img.sum(axis=1))
                        logger.info(
                            f"  BiasX={b_v:.2f}V: n={diff_counts[scan_idx]}, "
                            f"sx={sx:.1f}px, sy={sy:.1f}px, "
                            f"area={float(avg.sum()):.0f}")

                    if stopping:
                        break
                    time.sleep(wait_time)
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
    finally:
        # Tasks must be closed before a static write, or it collides with
        # the channels they still hold. The loop finalizes at the end of
        # every iteration, but an exception can leave them open, so try
        # again here; a second finalize on a closed handle just raises and
        # is swallowed.
        for _card, _h in ((NI_card_2, h1), (NI_card_3, h2)):
            if _h is None:
                continue
            try:
                _card.finalize(_h, timeout=10.0, force_finish=True)
            except Exception:
                pass
        try:
            NI_card_1.finalize_clock()
        except Exception:
            pass
        park_outputs(NI_card_2, NI_card_3, logger)
