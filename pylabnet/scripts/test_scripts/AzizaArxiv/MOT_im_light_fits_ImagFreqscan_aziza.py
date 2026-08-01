"""
MOT fluorescence imaging script — VCO imaging detuning scan.

Sweeps 4 different MOT VCO imaging voltages at a fixed time-of-flight.
Each voltage produces one signal/background frame pair; all 4 sub-sequences
are concatenated back-to-back into a single NI waveform. The OPX repeats
the concatenated waveform `number_of_runs` times for averaging.

Sub-sequence (per scan point), parameters arranged in time order from t=0:

    MOT_AOM_start ... MOT_AOM_end     MOT AOM long pulse (VCO at loading V)
    mot_loading_time                  MOT coils DO switches 0 -> 1 (off)
    + tof (fixed)                     Frame 1 trigger (VCO at imaging V)
    + atoms_clear_time                Frame 2 trigger (VCO at imaging V)

Plots: 2 x 4 grid — top row = current diff, bottom row = running average.
       Marginal Gaussian fits + σ labels updated by a main-thread QTimer.
"""

import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D  # noqa: F401
import time
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


# -----------------------------------------------------------------------------
# NumPy compatibility shims
# -----------------------------------------------------------------------------
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float


# -----------------------------------------------------------------------------
# 1D Gaussian width estimator (pure numpy)
# -----------------------------------------------------------------------------
def _sigma_from_moment(profile):
    """Moment-based Gaussian width estimate. Returns (mu, sigma, bg_sub_profile)
    or (nan, nan, None) on failure."""
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
    """Generate (x, y) Gaussian curve of length n."""
    x = np.arange(n, dtype=np.float64)
    y = amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return x, y


# -----------------------------------------------------------------------------
# Display levels
# -----------------------------------------------------------------------------
FRAME1_LEVELS = (0, 20)
FRAME2_LEVELS = (0, 20)
DIFF_LEVELS = (0, 20)
AVG_DIFF_LEVELS = (0, 20)


# -----------------------------------------------------------------------------
# Standalone grid window with QTimer-based Gaussian fits
# -----------------------------------------------------------------------------
class ScanGridWindow:
    """2 x n_scan grid of ImageItems with marginal profile fits.

    Architecture: experiment thread calls set_current/set_average which only
    do setImage (thread-safe). A QTimer on the main thread polls for new
    averages and updates profile curves + σ labels — all on the correct
    thread so nothing deadlocks."""

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

        # Shared data: experiment thread writes, main-thread timer reads.
        self._latest_avgs = [None] * self.n_scan
        self._rendered_versions = [0] * self.n_scan

        # --- Row 0: column headers ---
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            lbl = self.win.addLabel(
                f"<b>{self.scan_labels[col]}</b><br>"
                f"<span style='color:#FFFF00'>σx=— σy=—</span>",
                row=0, col=base_col, size="10pt"
            )
            self.header_labels.append(lbl)

        # --- Row 1: current-run diff images ---
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            vb = self.win.addViewBox(row=1, col=base_col, lockAspect=True)
            vb.invertY(True)
            img = pg.ImageItem(axisOrder='row-major')
            img.setLevels(self.diff_levels)
            vb.addItem(img)
            self.current_imgs.append(img)

        # --- Rows 2-3: average images with marginal profile plots ---
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            y_margin_col = 2 * col

            # y-profile plot (left of the average image)
            y_plot = self.win.addPlot(row=2, col=y_margin_col)
            y_plot.setMaximumWidth(self.MARGIN_PX)
            y_plot.hideAxis('bottom')
            y_plot.hideAxis('left')
            y_plot.setMouseEnabled(x=False, y=False)
            y_plot.setMenuEnabled(False)
            y_plot.invertX(True)
            y_plot.invertY(True)
            y_data = y_plot.plot([], [], pen=pg.mkPen('w', width=1))
            y_fit = y_plot.plot([], [], pen=pg.mkPen('y', width=2))
            self.y_data_curves.append(y_data)
            self.y_fit_curves.append(y_fit)

            # Average image
            vb_avg = self.win.addViewBox(row=2, col=base_col, lockAspect=True)
            vb_avg.invertY(True)
            img = pg.ImageItem(axisOrder='row-major')
            img.setLevels(self.avg_levels)
            vb_avg.addItem(img)
            self.avg_imgs.append(img)

            y_plot.setYLink(vb_avg)

            # x-profile plot (below the average image)
            x_plot = self.win.addPlot(row=3, col=base_col)
            x_plot.setMaximumHeight(self.MARGIN_PX)
            x_plot.hideAxis('bottom')
            x_plot.hideAxis('left')
            x_plot.setMouseEnabled(x=False, y=False)
            x_plot.setMenuEnabled(False)
            x_data = x_plot.plot([], [], pen=pg.mkPen('w', width=1))
            x_fit = x_plot.plot([], [], pen=pg.mkPen('y', width=2))
            self.x_data_curves.append(x_data)
            self.x_fit_curves.append(x_fit)
            x_plot.setXLink(vb_avg)

        # --- Main-thread polling timer for fits ---
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    def _poll_and_update(self):
        """Runs on the Qt main thread. Picks up new averages, computes
        Gaussian fits, updates profile curves and σ labels."""
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

                sx_s = f"{sigma_x:.1f}" if np.isfinite(sigma_x) else "—"
                sy_s = f"{sigma_y:.1f}" if np.isfinite(sigma_y) else "—"
                total_area = float(img.sum())
                area_s = f"{total_area:.0f}"
                self.header_labels[idx].setText(
                    f"<b>{self.scan_labels[idx]}</b><br>"
                    f"<span style='color:#FFFF00'>σx={sx_s} σy={sy_s}</span><br>"
                    f"<span style='color:#00FF88'>area={area_s}</span>"
                )
            except Exception:
                pass

    # --- Called from experiment thread ---

    def set_current(self, scan_idx, diff_image):
        try:
            self.current_imgs[scan_idx].setImage(
                diff_image, levels=self.diff_levels, autoLevels=False
            )
        except Exception:
            pass

    def set_average(self, scan_idx, avg_image):
        try:
            self.avg_imgs[scan_idx].setImage(
                avg_image, levels=self.avg_levels, autoLevels=False
            )
        except Exception:
            pass
        # Store a copy for the main-thread timer.
        current_version = self._rendered_versions[scan_idx] + 1
        self._latest_avgs[scan_idx] = (
            np.asarray(avg_image).copy(),
            current_version,
        )


# -----------------------------------------------------------------------------
# Experiment parameter dictionary
# -----------------------------------------------------------------------------
INIT_DICT = {
    # --- MOT AOM gating ---
    'MOT_AOM_start': {'MOT AOM Start Time (ms)': '0'},
    'MOT_AOM_end': {'MOT AOM End Time (ms)': '2000'},

    # --- MOT loading ---
    'mot_loading_time': {'MOT Loading Time (ms)': '2000'},

    # --- Release-and-image timing ---
    'tof_us': {'TOF (us)': '5000'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '190'},

    # --- Camera ---
    'camera_exposure_us': {'Camera Exposure Time (us)': '1000'},
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},

    # --- MOT coils ---
    'mot_coils_on_voltage': {'MOT Coils ON Analog Voltage (V)': '7.0'},

    # --- VCO frequency control ---
    # Loading voltage during MOT loading phase.
    'MOT_freq_VCO_loading_voltage': {'MOT VCO Loading Voltage (V)': '0.4'},
    # 4 imaging voltages to scan (each gets its own column in the grid).
    'MOT_freq_VCO_img_1': {'MOT VCO Imaging 1 (V)': '0.40'},
    'MOT_freq_VCO_img_2': {'MOT VCO Imaging 2 (V)': '0.45'},
    'MOT_freq_VCO_img_3': {'MOT VCO Imaging 3 (V)': '0.50'},
    'MOT_freq_VCO_img_4': {'MOT VCO Imaging 4 (V)': '0.55'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '10'},
    'Repump_freq_VCO_voltage': {'Repump Freq VCO Voltage (V)': '0.0'},
}


def define_dataset():
    return 'Dataset'


# =============================================================================
# CONFIGURE
# =============================================================================
def configure(**kwargs):
    dataset = kwargs['dataset']

    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")

    dataset.camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']

    tof_us = int(dataset.get_input_parameter("tof_us"))
    scan_voltages = [
        float(dataset.get_input_parameter("MOT_freq_VCO_img_1")),
        float(dataset.get_input_parameter("MOT_freq_VCO_img_2")),
        float(dataset.get_input_parameter("MOT_freq_VCO_img_3")),
        float(dataset.get_input_parameter("MOT_freq_VCO_img_4")),
    ]
    scan_labels = [f"VCO={v:.2f}V" for v in scan_voltages]

    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        diff_levels=DIFF_LEVELS,
        avg_levels=AVG_DIFF_LEVELS,
        title=f"VCO imaging scan (TOF={tof_us}us)",
    )

    dataset.graph.hide()


# =============================================================================
# EXPERIMENT
# =============================================================================
def experiment(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    logger = dataset.log

    # Re-resolve hardware handles.
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

    # Pull GUI parameters.
    MOT_AOM_start = int(dataset.get_input_parameter("MOT_AOM_start"))
    MOT_AOM_end = int(dataset.get_input_parameter("MOT_AOM_end"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    mot_loading_time = int(dataset.get_input_parameter("mot_loading_time"))
    atoms_clear_time = int(dataset.get_input_parameter("atoms_clear_time"))
    tof_us_fixed = int(dataset.get_input_parameter("tof_us"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    scan_voltages = [
        float(dataset.get_input_parameter("MOT_freq_VCO_img_1")),
        float(dataset.get_input_parameter("MOT_freq_VCO_img_2")),
        float(dataset.get_input_parameter("MOT_freq_VCO_img_3")),
        float(dataset.get_input_parameter("MOT_freq_VCO_img_4")),
    ]
    n_scan = len(scan_voltages)

    # Hardcoded channel assignments.
    camera_trigger_do = "dio0"
    MOT_AOM_do = "dio1"
    mot_coils_do = "dio2"
    MOT_AOM_ao = "ao1"
    mot_coils_ao = "ao2"
    MOT_freq_VCO_ao = "ao3"
    Repump_freq_VCO_ao = "ao4"
    opx_trigger_do = 1

    mot_coils_on_voltage = float(dataset.get_input_parameter("mot_coils_on_voltage"))
    MOT_freq_VCO_loading_voltage = float(dataset.get_input_parameter("MOT_freq_VCO_loading_voltage"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    Repump_freq_VCO_voltage = float(dataset.get_input_parameter("Repump_freq_VCO_voltage"))

    # Timing constants.
    ni_sample_rate = 20000
    SAMPLES_PER_MS = ni_sample_rate // 1000
    SAMPLES_PER_US = ni_sample_rate / 1_000_000.0
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"
    opx_ttl_pulse_ns = 500
    sample_period_ns = int(round(1e9 / ni_sample_rate))
    delay_ns = sample_period_ns - opx_ttl_pulse_ns
    camera_ttl_up = max(1, SAMPLES_PER_MS // 20)

    MOT_AOM_start_s = MOT_AOM_start * SAMPLES_PER_MS
    MOT_AOM_end_s = MOT_AOM_end * SAMPLES_PER_MS
    mot_loading_time_s = mot_loading_time * SAMPLES_PER_MS
    atoms_clear_time_s = atoms_clear_time * SAMPLES_PER_MS
    coils_off_time_s = mot_loading_time_s

    def us_to_samples(t_us):
        return int(round(t_us * SAMPLES_PER_US))
    tof_samples_fixed = us_to_samples(tof_us_fixed)
    vco_lead_samples = us_to_samples(vco_lead_us)

    aom_imaging_pulse_samples = max(
        camera_ttl_up,
        int(np.ceil(camera_exposure_us * SAMPLES_PER_US))
    )

    # Sanity checks.
    if camera_exposure_us <= 0:
        raise ValueError("camera_exposure_us must be > 0")
    if delay_ns < 0:
        raise ValueError("ni_sample_rate too high for 500 ns OPX pulse")
    if tof_us_fixed < 0:
        raise ValueError("tof_us must be non-negative")

    logger.info(f"Time at start {time.perf_counter_ns()}")

    # =========================================================================
    # WAVEFORM CONSTRUCTION — one sub-sequence per VCO imaging voltage
    # =========================================================================
    def build_sub_sequence(imaging_vco_v):
        frame_1 = mot_loading_time_s + tof_samples_fixed
        frame_2 = frame_1 + atoms_clear_time_s
        sub_end = max(frame_2 + camera_ttl_up, MOT_AOM_end_s)

        down = frame_2 - frame_1 - camera_ttl_up
        cam = (
            [0] * frame_1 +
            [1] * camera_ttl_up +
            [0] * down +
            [1] * camera_ttl_up +
            [0] * max(0, sub_end - frame_2 - camera_ttl_up)
        )

        aom = [0] * sub_end
        for idx in range(MOT_AOM_start_s, min(MOT_AOM_end_s, sub_end)):
            aom[idx] = 1
        for offset in range(aom_imaging_pulse_samples):
            if frame_1 + offset < sub_end:
                aom[frame_1 + offset] = 1
            if frame_2 + offset < sub_end:
                aom[frame_2 + offset] = 1

        coils_do_arr = [0] * coils_off_time_s + [1] * (sub_end - coils_off_time_s)
        coils_ao_arr = [mot_coils_on_voltage] * sub_end

        mot_vco = [MOT_freq_VCO_loading_voltage] * sub_end
        vco_lo = max(0, frame_1 - vco_lead_samples)
        vco_hi = min(sub_end, frame_2 + aom_imaging_pulse_samples)
        for idx in range(vco_lo, vco_hi):
            mot_vco[idx] = imaging_vco_v

        return cam, aom, coils_do_arr, coils_ao_arr, mot_vco, sub_end

    camera_trigger_pulse = []
    MOT_AOM_pulse = []
    mot_coils_do_pulse = []
    mot_coils_ao_pulse = []
    MOT_freq_VCO_pulse = []

    for vco_v in scan_voltages:
        cam, aom, cdo, cao, mvco, sub_end = build_sub_sequence(vco_v)
        camera_trigger_pulse += cam
        MOT_AOM_pulse += aom
        mot_coils_do_pulse += cdo
        mot_coils_ao_pulse += cao
        MOT_freq_VCO_pulse += mvco

    sequence_end_samples = len(camera_trigger_pulse)
    experiment_length_samples = int(sequence_end_samples)
    Repump_freq_VCO_pulse = [Repump_freq_VCO_voltage] * sequence_end_samples

    logger.info(
        f"Running VCO imaging scan: "
        f"tof={tof_us_fixed}us (fixed), "
        f"scan_voltages={scan_voltages}, "
        f"sequence_length={sequence_end_samples} samples"
    )

    # Persistent accumulators (survive across outer cycles).
    diff_sums = [None] * n_scan
    diff_counts = [0] * n_scan

    # =========================================================================
    # MAIN ACQUISITION LOOP
    # =========================================================================
    while thread.running:
        camera_client.set_hardware_trigger(
            line=trigger_line, activation=trigger_edge,
            selector="FrameStart", overlap="ReadOut",
            acquisition_mode="Continuous",
        )
        camera_client.set_exposure(camera_exposure_us)
        camera_client.try_set_float("Gain", 50.0)

        logger.info("Starting acquisition")
        dataset.camera_client.start_acquisition()

        NI_card_1.arm_clock(length=experiment_length_samples, sample_rate=ni_sample_rate)

        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=camera_trigger_pulse, sample_rate=ni_sample_rate)
        NI_card_2.set_do_voltage(do_channel=MOT_AOM_do, value=MOT_AOM_pulse, sample_rate=ni_sample_rate)
        NI_card_2.set_do_voltage(do_channel=mot_coils_do, value=mot_coils_do_pulse, sample_rate=ni_sample_rate)

        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel=MOT_AOM_ao, voltages=MOT_AOM_pulse, sample_rate=ni_sample_rate)
        NI_card_3.set_ao_voltage(ao_channel=mot_coils_ao, voltages=mot_coils_ao_pulse, sample_rate=ni_sample_rate)
        NI_card_3.set_ao_voltage(ao_channel=MOT_freq_VCO_ao, voltages=MOT_freq_VCO_pulse, sample_rate=ni_sample_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_freq_VCO_ao, voltages=Repump_freq_VCO_pulse, sample_rate=ni_sample_rate)

        OPX_client.build_stack()
        clock_elem = OPX_client.create_new_do_elem(do_channel=opx_trigger_do, length=500)
        N = experiment_length_samples
        number_of_runs = 4
        M = number_of_runs
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

                for scan_idx, vco_v in enumerate(scan_voltages):
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
                        f"  VCO={vco_v}V: n_avg={diff_counts[scan_idx]}, area={total_area:.0f}"
                    )

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
