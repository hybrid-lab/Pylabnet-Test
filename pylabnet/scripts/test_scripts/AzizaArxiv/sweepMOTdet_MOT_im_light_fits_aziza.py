# MOT fluorescence imaging -- VCO loading detuning scan.
# MOT only (no CMOT, no Dark MOT). Sweeps 4 MOT VCO loading voltages.
# Fixed TOF, fixed imaging VCO. Repump AOM mirrors MOT AOM timing.
#
# Sequence per scan point:
#   1. MOT loading (AOM on, coils on, VCO at scan voltage)
#   2. Release (coils TTL off 2ms early, all beams off)
#   3. TOF (everything off)
#   4. Frame 1 imaging pulse (both AOMs full power, VCO at imaging voltage)
#   5. Atoms clear
#   6. Frame 2 background pulse (same as frame 1)

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


DIFF_LEVELS = (0, 20)
AVG_DIFF_LEVELS = (0, 20)


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
                    f"<span style='color:#FFFF00'>sx={sx_s} sy={sy_s}</span><br>"
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
# INIT_DICT
# =============================================================================
INIT_DICT = {
    # ===================== MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '2000'},
    'mot_aom_power': {'MOT AOM Power (V)': '1.0'},
    'repump_aom_power': {'Repump AOM Power (V)': '1.0'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '4.5'},
    'Repump_VCO': {'Repump VCO (V)': '0.2'},

    # ===================== SCAN: 4 MOT VCO loading voltages =====================
    'MOT_VCO_load_1': {'MOT VCO Loading 1 (V)': '0.25'},
    'MOT_VCO_load_2': {'MOT VCO Loading 2 (V)': '0.30'},
    'MOT_VCO_load_3': {'MOT VCO Loading 3 (V)': '0.35'},
    'MOT_VCO_load_4': {'MOT VCO Loading 4 (V)': '0.40'},

    # ===================== RELEASE + IMAGING =====================
    'tof_us': {'TOF (us)': '5000'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 1)': '10'},
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

    scan_voltages = [
        float(dataset.get_input_parameter("MOT_VCO_load_1")),
        float(dataset.get_input_parameter("MOT_VCO_load_2")),
        float(dataset.get_input_parameter("MOT_VCO_load_3")),
        float(dataset.get_input_parameter("MOT_VCO_load_4")),
    ]
    tof_us = int(dataset.get_input_parameter("tof_us"))
    scan_labels = [f"Load={v:.2f}V" for v in scan_voltages]

    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        diff_levels=DIFF_LEVELS,
        avg_levels=AVG_DIFF_LEVELS,
        title=f"MOT VCO loading scan (TOF={tof_us}us)",
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
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    scan_voltages = [
        float(dataset.get_input_parameter("MOT_VCO_load_1")),
        float(dataset.get_input_parameter("MOT_VCO_load_2")),
        float(dataset.get_input_parameter("MOT_VCO_load_3")),
        float(dataset.get_input_parameter("MOT_VCO_load_4")),
    ]
    n_scan = len(scan_voltages)

    tof_us = int(dataset.get_input_parameter("tof_us"))
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    # =========================================================================
    # HARDCODED CHANNELS
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
    tof_s = us2s(tof_us)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    # Timeline: MOT -> release -> TOF -> image
    release = mot_s
    coils_off = release - coils_lead_s
    f1 = release + tof_s
    f2 = f1 + clear_s
    coils_restore_t = f1 + coils_restore_s

    logger.info(f"Time at start {time.perf_counter_ns()}")

    # =========================================================================
    # WAVEFORM CONSTRUCTION -- one sub-sequence per scan voltage
    # =========================================================================
    def build_sub_sequence(loading_vco_v):
        N = f2 + ttl_up + 1

        # Camera trigger
        cam = [0] * N
        for s in range(ttl_up):
            cam[f1 + s] = 1
            cam[f2 + s] = 1

        # MOT AOM DO + AO
        ado = [0] * N
        aao = [0.0] * N
        # Repump AOM DO + AO (mirrors MOT AOM)
        rdo = [0] * N
        rao = [0.0] * N

        # Phase 1: MOT loading -- both AOMs on at full power
        for s in range(mot_s):
            ado[s] = 1
            aao[s] = mot_aom_power
            rdo[s] = 1
            rao[s] = repump_aom_power

        # Release + TOF: everything off (arrays already 0)

        # Imaging pulses: both AOMs on at full power
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
        for s in range(coils_off, min(coils_restore_t, N)):
            cdo[s] = 1

        # Coils AO: flat at MOT voltage
        cao = [mot_coils_v] * N

        # MOT VCO: loading voltage (scan parameter), imaging around frames
        mvco = [loading_vco_v] * N
        vlo = max(0, f1 - vco_lead_s)
        vhi = min(N, f2 + aom_pulse_s)
        for s in range(vlo, vhi):
            mvco[s] = vco_imaging

        # Repump VCO: fixed
        rvco = [repump_vco] * N

        return cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, N

    # Concatenate sub-sequences
    camera_pulse = []
    mot_aom_do_pulse = []
    mot_aom_ao_pulse = []
    rep_aom_do_pulse = []
    rep_aom_ao_pulse = []
    coils_do_pulse = []
    coils_ao_pulse = []
    mot_vco_pulse = []
    rep_vco_pulse = []

    for vco_v in scan_voltages:
        cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, N = build_sub_sequence(vco_v)
        camera_pulse += cam
        mot_aom_do_pulse += ado
        mot_aom_ao_pulse += aao
        rep_aom_do_pulse += rdo
        rep_aom_ao_pulse += rao
        coils_do_pulse += cdo
        coils_ao_pulse += cao
        mot_vco_pulse += mvco
        rep_vco_pulse += rvco

    total_samples = len(camera_pulse)

    logger.info(
        f"MOT VCO loading scan: "
        f"scan={scan_voltages}, tof={tof_us}us, img_vco={vco_imaging}V, "
        f"mot={mot_loading_ms}ms, coils={mot_coils_v}V, "
        f"exposure={camera_exposure_us}us, sequence={total_samples} samples"
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

        NI_card_1.arm_clock(length=total_samples, sample_rate=ni_rate)

        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=camera_pulse, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=MOT_AOM_do, value=mot_aom_do_pulse, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=mot_coils_do, value=coils_do_pulse, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=Repump_AOM_do, value=rep_aom_do_pulse, sample_rate=ni_rate)

        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel=MOT_AOM_ao, voltages=mot_aom_ao_pulse, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=mot_coils_ao, voltages=coils_ao_pulse, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=MOT_freq_VCO_ao, voltages=mot_vco_pulse, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_freq_VCO_ao, voltages=rep_vco_pulse, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_AOM_ao, voltages=rep_aom_ao_pulse, sample_rate=ni_rate)

        OPX_client.build_stack()
        clk = OPX_client.create_new_do_elem(do_channel=opx_trigger_do, length=500)
        M = 4
        run_buffer = 10_000_000
        with OPX_client.for_("j", 0, M, 1):
            with OPX_client.for_("i", 0, total_samples + 1, 1):
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
                        f"  Load={vco_v}V: n_avg={diff_counts[scan_idx]}, area={total_area:.0f}")

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
