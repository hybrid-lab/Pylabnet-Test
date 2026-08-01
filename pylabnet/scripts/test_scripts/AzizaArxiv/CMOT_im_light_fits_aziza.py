"""
MOT fluorescence imaging script -- TOF sweep with CMOT phase.

Sequence phases (in time order):

  1. MOT LOADING
     - MOT AOM: ON at full power (AO = 1.0)
     - Repump AOM: ON at full power (AO = 1.0)
     - MOT VCO: loading voltage (red-detuned for trapping)
     - Repump VCO: fixed voltage
     - Coils DO: ON (0), Coils AO: MOT coil voltage
     Duration: mot_loading_time (GUI)

  2. CMOT (Compressed MOT)
     - MOT AOM: reduced power (cmot_mot_power)
     - Repump AOM: reduced power (cmot_repump_power)
     - MOT VCO: cmot_detuning_voltage (further red-detuned)
     - Repump VCO: same as MOT phase
     - Coils DO: ON (0), Coils AO: cmot_coil_voltage (higher)
     Duration: cmot_duration (GUI)

  3. RELEASE + TOF (all beams & coils off)
     - MOT AOM: OFF
     - Coils DO: OFF (1) -- fires 2ms before CMOT ends
     - Coils AO: stays at cmot_coil_voltage (TTL handles switch)
     - Duration: tof (swept, 4 values)

  4. IMAGING (Frame 1 -- signal)
     - MOT AOM: brief pulse (matched to camera exposure)
     - MOT VCO: imaging voltage (near resonance)
     - Camera trigger: rising edge

  5. ATOMS CLEAR (wait for atoms to leave FOV)
     - Coils DO: back ON 10ms after frame_1
     Duration: atoms_clear_time (GUI)

  6. BACKGROUND (Frame 2 -- no atoms)
     - Same AOM pulse as Frame 1
     - Camera trigger: rising edge

Adding future phases (e.g. optical molasses, optical pumping):
  - Insert new phase between CMOT and RELEASE
  - Increase release_time accordingly
  - Everything downstream (TOF, imaging, background) shifts automatically

Hardware: NI DAQ cards (PXI), OPX clock, BFS-U3-51S5M camera.
"""

import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D  # noqa: F401
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


# -----------------------------------------------------------------------------
# Gaussian width estimator
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Display levels
# -----------------------------------------------------------------------------
DIFF_LEVELS = (0, 20)
AVG_DIFF_LEVELS = (0, 20)


# -----------------------------------------------------------------------------
# Grid window with QTimer-based fits
# -----------------------------------------------------------------------------
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
                f"<span style='color:#FFFF00'>sigmax=-- sigmay=--</span>",
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
                x_axis = np.arange(x_proj.size, dtype=np.float64)
                if x_bg is not None and np.isfinite(sigma_x):
                    self.x_data_curves[idx].setData(x_axis, x_bg)
                    gx_x, gx_y = _gaussian_curve(x_proj.size, mu_x, sigma_x, x_bg.max())
                    self.x_fit_curves[idx].setData(gx_x, gx_y)
                mu_y, sigma_y, y_bg = _sigma_from_moment(y_proj)
                y_axis = np.arange(y_proj.size, dtype=np.float64)
                if y_bg is not None and np.isfinite(sigma_y):
                    self.y_data_curves[idx].setData(y_bg, y_axis)
                    gy_x, gy_y = _gaussian_curve(y_proj.size, mu_y, sigma_y, y_bg.max())
                    self.y_fit_curves[idx].setData(gy_y, gy_x)
                sx_s = f"{sigma_x:.1f}" if np.isfinite(sigma_x) else "--"
                sy_s = f"{sigma_y:.1f}" if np.isfinite(sigma_y) else "--"
                total_area = float(img.sum())
                self.header_labels[idx].setText(
                    f"<b>{self.scan_labels[idx]}</b><br>"
                    f"<span style='color:#FFFF00'>sigmax={sx_s} sigmay={sy_s}</span><br>"
                    f"<span style='color:#00FF88'>area={total_area:.0f}</span>")
            except Exception:
                pass

    def set_current(self, scan_idx, diff_image):
        try:
            self.current_imgs[scan_idx].setImage(
                diff_image, levels=self.diff_levels, autoLevels=False)
        except Exception:
            pass

    def set_average(self, scan_idx, avg_image):
        try:
            self.avg_imgs[scan_idx].setImage(
                avg_image, levels=self.avg_levels, autoLevels=False)
        except Exception:
            pass
        current_version = self._rendered_versions[scan_idx] + 1
        self._latest_avgs[scan_idx] = (
            np.asarray(avg_image).copy(), current_version)


# =============================================================================
# INIT_DICT -- GUI parameters organized by sequence phase
# =============================================================================
INIT_DICT = {
    # ===================== PHASE 1: MOT LOADING =====================
    # Coil current ~10A, cooling detuning ~-15MHz, full power
    'mot_loading_time': {'MOT Loading Time (ms)': '2000'},
    'mot_aom_power': {'MOT AOM Power (V)': '1.0'},
    'repump_aom_power': {'Repump AOM Power (V)': '1.0'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '4.5'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0.31'},
    'Repump_VCO': {'Repump VCO (V)': '0.2'},

    # ===================== PHASE 2: CMOT =====================
    # Coil current ~15A, cooling detuning ~-40MHz, reduced power
    # CMOT has two sub-phases:
    #   2a. RAMP (hardcoded 10ms): linearly ramp all params from MOT to CMOT.
    #   2b. HOLD: hold at CMOT values for cmot_hold_time.
    'cmot_enabled': {'CMOT Enabled (1=yes, 0=no)': '1'},
    'cmot_hold_time': {'CMOT Hold Time (ms)': '10'},
    'cmot_mot_power': {'CMOT MOT AOM Power (V)': '0.5'},
    'cmot_repump_power': {'CMOT Repump AOM Power (V)': '0.15'},
    'cmot_detuning_voltage': {'CMOT MOT VCO (V) [-40MHz]': '-0.17'},
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '6.5'},
    'cmot_repump_vco': {'CMOT Repump VCO (V) [-3MHz]': '0.26'},

    # ===================== PHASE 3: RELEASE + TOF (swept) =====================
    'tof_1_us': {'TOF 1 (us)': '2000'},
    'tof_2_us': {'TOF 2 (us)': '5000'},
    'tof_3_us': {'TOF 3 (us)': '10000'},
    'tof_4_us': {'TOF 4 (us)': '15000'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 1)': '10'},

    # ===================== PHASE 4-6: IMAGING =====================
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.6'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '10'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '190'},

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
    ]
    scan_labels = [f"TOF={t}us" for t in tof_values]

    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        diff_levels=DIFF_LEVELS,
        avg_levels=AVG_DIFF_LEVELS,
        title="TOF sweep (MOT -> CMOT -> release -> image)",
    )
    dataset.graph.hide()


def experiment(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    logger = dataset.log

    dataset.NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    dataset.NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    dataset.NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
    dataset.camera_client = kwargs["fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber"]
    dataset.OPX_client = kwargs["OPX_OPX"]

    OPX_client = dataset.OPX_client
    NI_card_1 = dataset.NI_card_1
    NI_card_2 = dataset.NI_card_2
    NI_card_3 = dataset.NI_card_3
    camera_client = dataset.camera_client

    # =========================================================================
    # READ ALL GUI PARAMETERS
    # =========================================================================
    # Phase 1: MOT
    mot_loading_time_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_voltage = float(dataset.get_input_parameter("mot_coils_voltage"))
    mot_vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    # Phase 2: CMOT (ramp 10ms hardcoded + hold from GUI)
    cmot_enabled = int(dataset.get_input_parameter("cmot_enabled")) != 0
    CMOT_RAMP_MS = 10  # hardcoded ramp duration
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning_voltage = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_voltage = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

    # Phase 3: Release + TOF
    tof_values_us = [
        int(dataset.get_input_parameter("tof_1_us")),
        int(dataset.get_input_parameter("tof_2_us")),
        int(dataset.get_input_parameter("tof_3_us")),
        int(dataset.get_input_parameter("tof_4_us")),
    ]
    n_scan = len(tof_values_us)
    coils_ttl_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_delay_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))

    # Phase 4-6: Imaging
    mot_vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_time_ms = int(dataset.get_input_parameter("atoms_clear_time"))

    # General
    wait_time = float(dataset.get_input_parameter("wait_time"))

    # =========================================================================
    # HARDCODED CHANNEL ASSIGNMENTS
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
    # TIMING CONSTANTS
    # =========================================================================
    ni_sample_rate = 20000
    SAMPLES_PER_MS = ni_sample_rate // 1000         # 20
    SAMPLES_PER_US = ni_sample_rate / 1_000_000.0   # 0.02
    sample_period_ns = int(round(1e9 / ni_sample_rate))
    delay_ns = sample_period_ns - 500
    camera_ttl_up = max(1, SAMPLES_PER_MS // 20)    # 1 sample

    def ms_to_s(t_ms):
        return t_ms * SAMPLES_PER_MS

    def us_to_s(t_us):
        return int(round(t_us * SAMPLES_PER_US))

    # Phase durations in samples
    mot_loading_s = ms_to_s(mot_loading_time_ms)
    cmot_ramp_s = ms_to_s(CMOT_RAMP_MS) if cmot_enabled else 0
    cmot_hold_s = ms_to_s(cmot_hold_ms) if cmot_enabled else 0
    cmot_total_s = cmot_ramp_s + cmot_hold_s
    atoms_clear_s = ms_to_s(atoms_clear_time_ms)
    vco_lead_s = us_to_s(vco_lead_us)
    coils_ttl_lead_s = ms_to_s(coils_ttl_lead_ms)
    coils_restore_delay_s = ms_to_s(coils_restore_delay_ms)

    aom_imaging_pulse_s = max(camera_ttl_up, int(np.ceil(camera_exposure_us * SAMPLES_PER_US)))

    # =========================================================================
    # KEY TIMELINE ANCHORS
    #
    # To add a new phase (e.g. molasses), insert it here and push
    # release_time forward. Everything downstream adjusts automatically.
    # =========================================================================
    cmot_start = mot_loading_s                     # end of MOT / start of CMOT ramp
    cmot_hold_start = cmot_start + cmot_ramp_s     # end of ramp / start of hold
    if cmot_enabled:
        release_time = cmot_start + cmot_total_s   # end of CMOT hold
    else:
        release_time = cmot_start                  # no CMOT, release right after MOT
    coils_off_time = release_time - coils_ttl_lead_s

    logger.info(f"Time at start {time.perf_counter_ns()}")
    cmot_status = "ON" if cmot_enabled else "OFF"
    logger.info(
        f"Timeline: MOT 0-{mot_loading_time_ms}ms, "
        f"CMOT {cmot_status} (ramp={CMOT_RAMP_MS}ms + hold={cmot_hold_ms}ms), "
        f"release at {mot_loading_time_ms + (CMOT_RAMP_MS + cmot_hold_ms if cmot_enabled else 0)}ms, "
        f"coils TTL off {coils_ttl_lead_ms}ms before release"
    )

    # =========================================================================
    # WAVEFORM CONSTRUCTION -- one sub-sequence per TOF value
    # =========================================================================
    def build_sub_sequence(tof_us):
        tof_s = us_to_s(tof_us)

        # --- Timeline for this TOF point ---
        frame_1 = release_time + tof_s
        frame_2 = frame_1 + atoms_clear_s
        sub_end = frame_2 + camera_ttl_up + 1

        # --- Camera trigger ---
        cam = [0] * sub_end
        for s in range(camera_ttl_up):
            cam[frame_1 + s] = 1
            cam[frame_2 + s] = 1

        # --- MOT AOM (DO + AO) ---
        aom_do = [0] * sub_end
        aom_ao = [0.0] * sub_end

        # --- Repump AOM (DO + AO) -- mirrors MOT AOM timing ---
        rep_do = [0] * sub_end
        rep_ao = [0.0] * sub_end

        # Phase 1: MOT loading -- both AOMs on at full power
        for s in range(0, cmot_start):
            aom_do[s] = 1
            aom_ao[s] = mot_aom_power
            rep_do[s] = 1
            rep_ao[s] = repump_aom_power

        if cmot_enabled:
            # Phase 2a: CMOT ramp -- linearly ramp both AOMs down
            for s in range(cmot_ramp_s):
                t = cmot_start + s
                frac = s / max(cmot_ramp_s - 1, 1)  # 0.0 -> 1.0
                aom_do[t] = 1
                aom_ao[t] = mot_aom_power + frac * (cmot_mot_power - mot_aom_power)
                rep_do[t] = 1
                rep_ao[t] = repump_aom_power + frac * (cmot_repump_power - repump_aom_power)

            # Phase 2b: CMOT hold -- hold at CMOT power
            for s in range(cmot_hold_s):
                t = cmot_hold_start + s
                aom_do[t] = 1
                aom_ao[t] = cmot_mot_power
                rep_do[t] = 1
                rep_ao[t] = cmot_repump_power

        # Phase 3: Release -- both AOMs off (already 0)

        # Phase 4: Frame 1 imaging pulse -- both AOMs on at full MOT power
        for s in range(aom_imaging_pulse_s):
            if frame_1 + s < sub_end:
                aom_do[frame_1 + s] = 1
                aom_ao[frame_1 + s] = mot_aom_power
                rep_do[frame_1 + s] = 1
                rep_ao[frame_1 + s] = repump_aom_power

        # Phase 6: Frame 2 imaging pulse (background) -- same as frame 1
        for s in range(aom_imaging_pulse_s):
            if frame_2 + s < sub_end:
                aom_do[frame_2 + s] = 1
                aom_ao[frame_2 + s] = mot_aom_power
                rep_do[frame_2 + s] = 1
                rep_ao[frame_2 + s] = repump_aom_power

        # --- MOT coils DO: 0=ON, 1=OFF ---
        coils_do = [0] * sub_end
        coils_restore = frame_1 + coils_restore_delay_s
        for s in range(coils_off_time, min(coils_restore, sub_end)):
            coils_do[s] = 1

        # --- MOT coils AO ---
        coils_ao = [mot_coils_voltage] * sub_end
        if cmot_enabled:
            for s in range(cmot_ramp_s):
                t = cmot_start + s
                frac = s / max(cmot_ramp_s - 1, 1)
                coils_ao[t] = mot_coils_voltage + frac * (cmot_coils_voltage - mot_coils_voltage)
            for s in range(cmot_hold_start, sub_end):
                coils_ao[s] = cmot_coils_voltage

        # --- MOT VCO ---
        mot_vco = [mot_vco_loading] * sub_end
        if cmot_enabled:
            for s in range(cmot_ramp_s):
                t = cmot_start + s
                frac = s / max(cmot_ramp_s - 1, 1)
                mot_vco[t] = mot_vco_loading + frac * (cmot_detuning_voltage - mot_vco_loading)
            for s in range(cmot_hold_start, sub_end):
                mot_vco[s] = cmot_detuning_voltage

        # Imaging voltage around frame triggers (overrides whatever was there)
        vco_lo = max(0, frame_1 - vco_lead_s)
        vco_hi = min(sub_end, frame_2 + aom_imaging_pulse_s)
        for s in range(vco_lo, vco_hi):
            mot_vco[s] = mot_vco_imaging

        # --- Repump VCO: fixed during MOT, ramps during CMOT ---
        repump_vco_arr = [repump_vco] * sub_end
        if cmot_enabled:
            # Phase 2a: ramp from MOT repump VCO to CMOT repump VCO
            for s in range(cmot_ramp_s):
                t = cmot_start + s
                frac = s / max(cmot_ramp_s - 1, 1)
                repump_vco_arr[t] = repump_vco + frac * (cmot_repump_vco - repump_vco)
            # Phase 2b + onward: hold at CMOT repump VCO
            for s in range(cmot_hold_start, sub_end):
                repump_vco_arr[s] = cmot_repump_vco

        return (cam, aom_do, aom_ao, rep_do, rep_ao,
                coils_do, coils_ao, mot_vco, repump_vco_arr, sub_end)

    # --- Concatenate sub-sequences ---
    camera_trigger_pulse = []
    MOT_AOM_do_pulse = []
    MOT_AOM_ao_pulse = []
    Repump_AOM_do_pulse = []
    Repump_AOM_ao_pulse = []
    mot_coils_do_pulse = []
    mot_coils_ao_pulse = []
    MOT_freq_VCO_pulse = []
    Repump_freq_VCO_pulse = []

    for tof_us in tof_values_us:
        (cam, aom_do, aom_ao, rep_do, rep_ao,
         cdo, cao, mvco, rvco, sub_end) = build_sub_sequence(tof_us)
        camera_trigger_pulse += cam
        MOT_AOM_do_pulse += aom_do
        MOT_AOM_ao_pulse += aom_ao
        Repump_AOM_do_pulse += rep_do
        Repump_AOM_ao_pulse += rep_ao
        mot_coils_do_pulse += cdo
        mot_coils_ao_pulse += cao
        MOT_freq_VCO_pulse += mvco
        Repump_freq_VCO_pulse += rvco

    sequence_end_samples = len(camera_trigger_pulse)
    experiment_length_samples = int(sequence_end_samples)

    logger.info(
        f"Running TOF sweep (MOT+CMOT): "
        f"tof_values={tof_values_us}us, "
        f"mot={mot_loading_time_ms}ms, "
        f"cmot={cmot_status}(ramp={CMOT_RAMP_MS}ms+hold={cmot_hold_ms}ms), "
        f"mot_power={mot_aom_power}->{cmot_mot_power}V, "
        f"rep_power={repump_aom_power}->{cmot_repump_power}V, "
        f"cmot_vco={cmot_detuning_voltage}V, cmot_coils={cmot_coils_voltage}V, "
        f"img_vco={mot_vco_imaging}V, exposure={camera_exposure_us}us, "
        f"sequence={sequence_end_samples} samples"
    )

    # Persistent accumulators
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

        NI_card_1.arm_clock(length=experiment_length_samples, sample_rate=ni_sample_rate)

        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=camera_trigger_pulse, sample_rate=ni_sample_rate)
        NI_card_2.set_do_voltage(do_channel=MOT_AOM_do, value=MOT_AOM_do_pulse, sample_rate=ni_sample_rate)
        NI_card_2.set_do_voltage(do_channel=mot_coils_do, value=mot_coils_do_pulse, sample_rate=ni_sample_rate)
        NI_card_2.set_do_voltage(do_channel=Repump_AOM_do, value=Repump_AOM_do_pulse, sample_rate=ni_sample_rate)

        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel=MOT_AOM_ao, voltages=MOT_AOM_ao_pulse, sample_rate=ni_sample_rate)
        NI_card_3.set_ao_voltage(ao_channel=mot_coils_ao, voltages=mot_coils_ao_pulse, sample_rate=ni_sample_rate)
        NI_card_3.set_ao_voltage(ao_channel=MOT_freq_VCO_ao, voltages=MOT_freq_VCO_pulse, sample_rate=ni_sample_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_freq_VCO_ao, voltages=Repump_freq_VCO_pulse, sample_rate=ni_sample_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_AOM_ao, voltages=Repump_AOM_ao_pulse, sample_rate=ni_sample_rate)

        OPX_client.build_stack()
        clock_elem = OPX_client.create_new_do_elem(do_channel=opx_trigger_do, length=500)
        N = experiment_length_samples
        M = 4  # number of averaging runs
        run_buffer = 10_000_000
        with OPX_client.for_("j", 0, M, 1):
            with OPX_client.for_("i", 0, N + 1, 1):
                OPX_client.set_digital_voltage(element=clock_elem)
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

        try:
            for run_idx in range(M):
                logger.info(f"Run {run_idx + 1} of {M}")
                for scan_idx, tof_us in enumerate(tof_values_us):
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
                    avg_image = diff_sums[scan_idx] / diff_counts[scan_idx]
                    scan_grid.set_average(scan_idx, avg_image)

                    total_area = float(avg_image.sum())
                    logger.info(
                        f"  TOF={tof_us}us: n_avg={diff_counts[scan_idx]}, area={total_area:.0f}")

                time.sleep(wait_time)
                logger.info(f"Updated plots for run {run_idx + 1} of {M}")
        finally:
            logger.info("Stopping acquisition")
            dataset.camera_client.stop_acquisition()

        logger.info(f"Cycle over {time.perf_counter_ns()}")

        time.sleep(wait_time)
        try:
            NI_card_2.finalize(h1, timeout=120.0, force_finish=True)
        except Exception as e:
            logger.info(f"NI card 2 finalize failed: {e!r}")
        try:
            NI_card_3.finalize(h2, timeout=120.0, force_finish=True)
        except Exception as e:
            logger.info(f"NI card 3 finalize failed: {e!r}")
        try:
            NI_card_1.finalize_clock()
        except Exception as e:
            logger.info(f"Clock finalize failed: {e!r}")
