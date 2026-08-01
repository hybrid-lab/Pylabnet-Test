# Dark MOT + CMOT -- TOF sweep with temperature measurement.
# Sweeps 4 TOF values, extracts Gaussian sigma from each,
# fits sigma^2 vs t^2 to extract temperature:
#   sigma^2(t) = sigma_0^2 + (k_B T / m) * t^2
#   slope of sigma^2 vs t^2 = k_B T / m
#   T = m * slope / k_B
#
# Requires pixel_size_um parameter to convert sigma from pixels to meters.
# For Rb87: m = 1.443e-25 kg, k_B = 1.381e-23 J/K

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

# Rb87 constants
M_RB87 = 1.4431e-25   # kg
K_B = 1.3806e-23       # J/K


def _sigma_from_moment(profile):
    n = profile.size
    if n < 5:
        return float('nan'), float('nan'), None
    y = profile.astype(np.float64, copy=True)
    baseline = np.percentile(y, 10)
    y -= baseline
    np.clip(y, 0, None, out=y)
    w_sum = y.sum()
    if w_sum <= 0:
        return float('nan'), float('nan'), None
    x = np.arange(n, dtype=np.float64)
    mu = (y * x).sum() / w_sum
    var = (y * (x - mu) ** 2).sum() / w_sum
    if var <= 0 or not np.isfinite(var):
        return float('nan'), float('nan'), None
    return float(mu), float(np.sqrt(var)), y


def _gaussian_curve(n, mu, sigma, amplitude):
    x = np.arange(n, dtype=np.float64)
    y = amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return x, y


def _linear_fit(x, y):
    """Least-squares linear fit y = a + b*x. Returns (a, b) or (nan, nan)."""
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


# =============================================================================
# Image grid window (same as before)
# =============================================================================
class ScanGridWindow:
    MARGIN_PX = 100
    POLL_MS = 500

    def __init__(self, scan_labels, diff_levels, avg_levels,
                 title="Scan grid", tile_px=320):
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


# =============================================================================
# Temperature fit window -- plots sigma^2 vs t^2 with linear fit
# =============================================================================
class TempFitWindow:
    POLL_MS = 1000

    def __init__(self, pixel_size_um):
        self.pixel_size_m = pixel_size_um * 1e-6
        self.win = pg.GraphicsLayoutWidget(show=True, title="Temperature from TOF")
        self.win.resize(800, 500)

        # sigma^2 vs t^2 plot (x-axis)
        self.plot_x = self.win.addPlot(row=0, col=0,
                                       title="<b>sigma_x^2 vs t^2</b>")
        self.plot_x.setLabel('bottom', 't^2', units='s^2')
        self.plot_x.setLabel('left', 'sigma_x^2', units='m^2')
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)
        self.data_x = self.plot_x.plot([], [], pen=None,
                                       symbol='o', symbolSize=10, symbolBrush='#2196F3')
        self.fit_x = self.plot_x.plot([], [], pen=pg.mkPen('#FF9800', width=2))
        self.label_x = pg.TextItem('', color='#FF9800', anchor=(0, 1))
        self.plot_x.addItem(self.label_x)

        # sigma^2 vs t^2 plot (y-axis)
        self.plot_y = self.win.addPlot(row=0, col=1,
                                       title="<b>sigma_y^2 vs t^2</b>")
        self.plot_y.setLabel('bottom', 't^2', units='s^2')
        self.plot_y.setLabel('left', 'sigma_y^2', units='m^2')
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)
        self.data_y = self.plot_y.plot([], [], pen=None,
                                       symbol='o', symbolSize=10, symbolBrush='#4CAF50')
        self.fit_y = self.plot_y.plot([], [], pen=pg.mkPen('#F44336', width=2))
        self.label_y = pg.TextItem('', color='#F44336', anchor=(0, 1))
        self.plot_y.addItem(self.label_y)

        # Shared data: experiment thread writes, timer reads
        self._latest_data = None
        self._rendered_version = 0

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    def update_data(self, tof_us_list, sigma_x_list, sigma_y_list):
        """Called from experiment thread. Stores data for main-thread timer."""
        self._latest_data = (
            list(tof_us_list), list(sigma_x_list), list(sigma_y_list),
            self._rendered_version + 1
        )

    def _poll_and_update(self):
        entry = self._latest_data
        if entry is None:
            return
        tof_us, sx_px, sy_px, version = entry
        if version <= self._rendered_version:
            return
        self._rendered_version = version

        try:
            # Convert TOF to seconds, sigma from pixels to meters
            t_s = np.array(tof_us, dtype=np.float64) * 1e-6
            t2 = t_s ** 2

            sx_m = np.array(sx_px, dtype=np.float64) * self.pixel_size_m
            sy_m = np.array(sy_px, dtype=np.float64) * self.pixel_size_m
            sx2 = sx_m ** 2
            sy2 = sy_m ** 2

            # Filter valid points
            mask_x = np.isfinite(sx2) & (sx2 > 0)
            mask_y = np.isfinite(sy2) & (sy2 > 0)

            # X-axis fit
            if mask_x.sum() >= 2:
                self.data_x.setData(t2[mask_x], sx2[mask_x])
                a_x, b_x = _linear_fit(t2[mask_x], sx2[mask_x])
                if np.isfinite(b_x) and b_x > 0:
                    T_x = M_RB87 * b_x / K_B
                    t2_fit = np.linspace(0, t2.max() * 1.1, 50)
                    self.fit_x.setData(t2_fit, a_x + b_x * t2_fit)
                    T_x_uK = T_x * 1e6
                    self.label_x.setText(
                        f"T_x = {T_x_uK:.1f} uK\n"
                        f"sigma_0x = {np.sqrt(max(a_x,0))/self.pixel_size_m:.0f} px")
                    self.label_x.setPos(t2.max() * 0.05, sx2[mask_x].max() * 0.95)

            # Y-axis fit
            if mask_y.sum() >= 2:
                self.data_y.setData(t2[mask_y], sy2[mask_y])
                a_y, b_y = _linear_fit(t2[mask_y], sy2[mask_y])
                if np.isfinite(b_y) and b_y > 0:
                    T_y = M_RB87 * b_y / K_B
                    t2_fit = np.linspace(0, t2.max() * 1.1, 50)
                    self.fit_y.setData(t2_fit, a_y + b_y * t2_fit)
                    T_y_uK = T_y * 1e6
                    self.label_y.setText(
                        f"T_y = {T_y_uK:.1f} uK\n"
                        f"sigma_0y = {np.sqrt(max(a_y,0))/self.pixel_size_m:.0f} px")
                    self.label_y.setPos(t2.max() * 0.05, sy2[mask_y].max() * 0.95)
        except Exception:
            pass


# =============================================================================
INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '2000'},
    'mot_aom_power': {'MOT AOM Power (V)': '1.0'},
    'repump_aom_power': {'Repump AOM Power (V)': '1.0'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '4.5'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0.31'},
    'Repump_VCO': {'Repump VCO (V)': '0.2'},

    # ===================== PHASE 2: CMOT (fixed) =====================
    'cmot_hold_time': {'CMOT Hold Time (ms)': '10'},
    'cmot_mot_power': {'CMOT MOT AOM Power (V)': '0.5'},
    'cmot_repump_power': {'CMOT Repump AOM Power (V)': '0.15'},
    'cmot_detuning_voltage': {'CMOT MOT VCO (V) [-40MHz]': '-0.17'},
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '6.5'},
    'cmot_repump_vco': {'CMOT Repump VCO (V) [-3MHz]': '0.26'},

    # ===================== PHASE 3: DARK MOT =====================
    'dark_mot_duration': {'Dark MOT Duration (ms)': '20'},
    'dark_mot_power': {'Dark MOT Cooling Power (V)': '0.2'},
    'dark_mot_repump_power': {'Dark MOT Repump Power (V)': '0.1'},
    'dark_mot_repump_vco': {'Dark MOT Repump VCO (V)': '0.26'},
    'dark_mot_detuning': {'Dark MOT Cooling VCO (V) [-55MHz]': '-0.46'},

    # ===================== PHASE 4: TOF (swept) =====================
    'tof_1_us': {'TOF 1 (us)': '2000'},
    'tof_2_us': {'TOF 2 (us)': '5000'},
    'tof_3_us': {'TOF 3 (us)': '8000'},
    'tof_4_us': {'TOF 4 (us)': '10000'},
    'tof_5_us': {'TOF 5 (us)': '13000'},
    'tof_6_us': {'TOF 6 (us)': '15000'},

    # ===================== IMAGING =====================
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 1)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.6'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '10'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '190'},

    # ===================== TEMPERATURE CALIBRATION =====================
    'pixel_size_um': {'Pixel Size (um/px)': '3.45'},

    # ===================== GENERAL =====================
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},
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

    tof_values = [
        int(dataset.get_input_parameter("tof_1_us")),
        int(dataset.get_input_parameter("tof_2_us")),
        int(dataset.get_input_parameter("tof_3_us")),
        int(dataset.get_input_parameter("tof_4_us")),
        int(dataset.get_input_parameter("tof_5_us")),
        int(dataset.get_input_parameter("tof_6_us")),
    ]
    scan_labels = [f"TOF={t}us" for t in tof_values]
    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        diff_levels=DIFF_LEVELS, avg_levels=AVG_DIFF_LEVELS,
        title="TOF sweep + Temperature (MOT -> CMOT -> Dark MOT -> image)",
    )

    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))
    dataset.temp_fit = TempFitWindow(pixel_size_um)

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

    CMOT_RAMP_MS = 10
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_v = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

    dark_ms = int(dataset.get_input_parameter("dark_mot_duration"))
    dark_power = float(dataset.get_input_parameter("dark_mot_power"))
    dark_rep_power = float(dataset.get_input_parameter("dark_mot_repump_power"))
    dark_rep_vco = float(dataset.get_input_parameter("dark_mot_repump_vco"))
    dark_detuning = float(dataset.get_input_parameter("dark_mot_detuning"))

    tof_values_us = [
        int(dataset.get_input_parameter("tof_1_us")),
        int(dataset.get_input_parameter("tof_2_us")),
        int(dataset.get_input_parameter("tof_3_us")),
        int(dataset.get_input_parameter("tof_4_us")),
        int(dataset.get_input_parameter("tof_5_us")),
        int(dataset.get_input_parameter("tof_6_us")),
    ]
    n_scan = len(tof_values_us)
    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))

    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

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
    dark_s = ms2s(dark_ms)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    dark_start = cmot_end
    dark_end = dark_start + dark_s
    release = dark_end
    coils_off = cmot_end - coils_lead_s

    logger.info(f"Time at start {time.perf_counter_ns()}")
    logger.info(
        f"Timeline: MOT 0-{mot_loading_ms}ms, "
        f"CMOT {mot_loading_ms}-{cmot_end/SPM:.0f}ms, "
        f"Dark MOT {dark_start/SPM:.0f}-{dark_end/SPM:.0f}ms, "
        f"release at {release/SPM:.0f}ms, "
        f"TOF sweep={tof_values_us}us, "
        f"pixel={pixel_size_um}um/px"
    )

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence(tof_us):
        tof_s = us2s(tof_us)
        f1 = release + tof_s
        f2 = f1 + clear_s
        crt = f1 + coils_restore_s
        N = f2 + ttl_up + 1

        cam = [0] * N
        for s in range(ttl_up):
            cam[f1 + s] = 1
            cam[f2 + s] = 1

        ado = [0] * N
        aao = [0.0] * N
        rdo = [0] * N
        rao = [0.0] * N
        for s in range(cmot_start):
            ado[s] = 1
            aao[s] = mot_aom_power
            rdo[s] = 1
            rao[s] = repump_aom_power
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            ado[t] = 1
            aao[t] = mot_aom_power + frac * (cmot_mot_power - mot_aom_power)
            rdo[t] = 1
            rao[t] = repump_aom_power + frac * (cmot_repump_power - repump_aom_power)
        for s in range(cmot_hold_s):
            t = cmot_hold_start + s
            ado[t] = 1
            aao[t] = cmot_mot_power
            rdo[t] = 1
            rao[t] = cmot_repump_power
        for s in range(dark_s):
            t = dark_start + s
            ado[t] = 1
            aao[t] = dark_power
            rdo[t] = 1
            rao[t] = dark_rep_power
        for s in range(release, min(f1, N)):
            ado[s] = 0
            aao[s] = 0.0
            rdo[s] = 0
            rao[s] = 0.0
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

        cdo = [0] * N
        for s in range(coils_off, min(crt, N)):
            cdo[s] = 1
        cao = [mot_coils_v] * N
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            cao[t] = mot_coils_v + frac * (cmot_coils_v - mot_coils_v)
        for s in range(cmot_hold_start, N):
            cao[s] = cmot_coils_v

        mvco = [vco_loading] * N
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            mvco[t] = vco_loading + frac * (cmot_detuning - vco_loading)
        for s in range(cmot_hold_start, cmot_end):
            mvco[s] = cmot_detuning
        for s in range(dark_s):
            mvco[dark_start + s] = dark_detuning
        for s in range(dark_end, N):
            mvco[s] = dark_detuning
        vlo = max(0, f1 - vco_lead_s)
        vhi = min(N, f2 + aom_pulse_s)
        for s in range(vlo, vhi):
            mvco[s] = vco_imaging

        rvco = [repump_vco] * N
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            rvco[t] = repump_vco + frac * (cmot_repump_vco - repump_vco)
        for s in range(cmot_hold_start, cmot_end):
            rvco[s] = cmot_repump_vco
        for s in range(dark_s):
            rvco[dark_start + s] = dark_rep_vco
        for s in range(dark_end, N):
            rvco[s] = dark_rep_vco

        return cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, N

    cam_p = []
    ado_p = []
    aao_p = []
    rdo_p = []
    rao_p = []
    cdo_p = []
    cao_p = []
    mvco_p = []
    rvco_p = []
    for tof_us in tof_values_us:
        cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, N = build_sub_sequence(tof_us)
        cam_p += cam
        ado_p += ado
        aao_p += aao
        rdo_p += rdo
        rao_p += rao
        cdo_p += cdo
        cao_p += cao
        mvco_p += mvco
        rvco_p += rvco

    total = len(cam_p)
    logger.info(f"TOF sweep+temp: tof={tof_values_us}us, total={total} samples")

    diff_sums = [None] * n_scan
    diff_counts = [0] * n_scan

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

        NI_card_1.arm_clock(length=total, sample_rate=ni_rate)
        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=cam_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=MOT_AOM_do, value=ado_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=mot_coils_do, value=cdo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=Repump_AOM_do, value=rdo_p, sample_rate=ni_rate)
        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel=MOT_AOM_ao, voltages=aao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=mot_coils_ao, voltages=cao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=MOT_freq_VCO_ao, voltages=mvco_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_freq_VCO_ao, voltages=rvco_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_AOM_ao, voltages=rao_p, sample_rate=ni_rate)

        OPX_client.build_stack()
        clk = OPX_client.create_new_do_elem(do_channel=opx_trigger_do, length=500)
        M = 4
        run_buffer = 10_000_000
        with OPX_client.for_("j", 0, M, 1):
            with OPX_client.for_("i", 0, total + 1, 1):
                OPX_client.set_digital_voltage(element=clk)
                OPX_client.delay(delay_ns)
            OPX_client.delay(run_buffer)

        h1 = NI_card_2.arm(regeneration=True)
        h2 = NI_card_3.arm(regeneration=True)
        OPX_client.execute(wait=False)
        logger.info("Image gathering starts")

        def get_frame(timeout_ms=1000):
            b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
            return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        scan_grid = dataset.scan_grid
        temp_fit = dataset.temp_fit

        # Track sigma values for temperature fit
        sigma_x_vals = [float('nan')] * n_scan
        sigma_y_vals = [float('nan')] * n_scan

        try:
            for run_idx in range(M):
                logger.info(f"Run {run_idx+1} of {M}")
                for scan_idx, tof_v in enumerate(tof_values_us):
                    is_first = (run_idx == 0 and scan_idx == 0)
                    frame1 = get_frame(timeout_ms=10000 if is_first else 100000)
                    frame2 = get_frame(timeout_ms=100000)
                    diff = frame1.astype(np.int32) - frame2.astype(np.int32)
                    scan_grid.set_current(scan_idx, diff)
                    if diff_sums[scan_idx] is None:
                        diff_sums[scan_idx] = diff.astype(np.float64)
                    else:
                        diff_sums[scan_idx] += diff
                    diff_counts[scan_idx] += 1
                    avg = diff_sums[scan_idx] / diff_counts[scan_idx]
                    scan_grid.set_average(scan_idx, avg)

                    # Extract sigma for temperature
                    img = avg.astype(np.float64)
                    x_proj = img.sum(axis=0)
                    y_proj = img.sum(axis=1)
                    _, sx, _ = _sigma_from_moment(x_proj)
                    _, sy, _ = _sigma_from_moment(y_proj)
                    sigma_x_vals[scan_idx] = sx
                    sigma_y_vals[scan_idx] = sy

                    logger.info(
                        f"  TOF={tof_v}us: n={diff_counts[scan_idx]}, "
                        f"sx={sx:.1f}px, sy={sy:.1f}px, "
                        f"area={float(avg.sum()):.0f}")

                # After each full sweep, update temperature fit
                temp_fit.update_data(tof_values_us, sigma_x_vals, sigma_y_vals)

                # Also compute and log temperature
                pxm = pixel_size_um * 1e-6
                t_s = np.array(tof_values_us, dtype=np.float64) * 1e-6
                sx_m = np.array(sigma_x_vals) * pxm
                sy_m = np.array(sigma_y_vals) * pxm
                mask_x = np.isfinite(sx_m) & (sx_m > 0)
                mask_y = np.isfinite(sy_m) & (sy_m > 0)
                if mask_x.sum() >= 2:
                    _, bx = _linear_fit(t_s[mask_x]**2, sx_m[mask_x]**2)
                    if np.isfinite(bx) and bx > 0:
                        Tx = M_RB87 * bx / K_B * 1e6
                        logger.info(f"  >> T_x = {Tx:.1f} uK")
                if mask_y.sum() >= 2:
                    _, by = _linear_fit(t_s[mask_y]**2, sy_m[mask_y]**2)
                    if np.isfinite(by) and by > 0:
                        Ty = M_RB87 * by / K_B * 1e6
                        logger.info(f"  >> T_y = {Ty:.1f} uK")

                time.sleep(wait_time)
                logger.info(f"Updated plots for run {run_idx+1} of {M}")
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
