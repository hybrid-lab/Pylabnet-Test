# Lattice ramp duration sweep.
# MOT -> CMOT -> Dark MOT -> PGC -> Lattice ramp (duration swept) -> Lattice hold -> image
# Stationary lattice. Both DIM-3000 at same fixed frequency.
# Sweep ramp duration to find optimal loading adiabaticity.

import numpy as np
import serial
import time
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

DIM_COM_UP = 'COM4'
DIM_COM_DOWN = 'COM5'


def _dim_open(port):
    try:
        return serial.Serial(port=port, baudrate=19200,
                             bytesize=8, parity='N', stopbits=1, timeout=1)
    except Exception:
        return None


def _dim_send(ser, cmd):
    ser.write((cmd + '\n').encode())
    time.sleep(0.1)


def _dim_query(ser, cmd):
    ser.flushInput()
    _dim_send(ser, cmd)
    time.sleep(0.15)
    return ser.readline().decode().strip()


def _dim_configure_fixed(logger, freq_hz, amplitude_x10):
    ser_up = _dim_open(DIM_COM_UP)
    ser_down = _dim_open(DIM_COM_DOWN)
    if ser_up is None or ser_down is None:
        logger.info("DIM-3000: cannot open COM ports -- skipping config")
        if ser_up:
            ser_up.close()
        if ser_down:
            ser_down.close()
        return
    try:
        id1 = _dim_query(ser_up, '*IDN?')
        id2 = _dim_query(ser_down, '*IDN?')
        logger.info(f"DIM-3000 up ({DIM_COM_UP}): {id1}")
        logger.info(f"DIM-3000 down ({DIM_COM_DOWN}): {id2}")
        _dim_send(ser_up, 'Mseg:0')
        _dim_send(ser_down, 'Mseg:0')
        for ser, name in [(ser_up, "up"), (ser_down, "down")]:
            _dim_send(ser, f'FRQ:{freq_hz}')
            _dim_send(ser, f'AMP:{amplitude_x10}')
            _dim_send(ser, 'OUT_on')
            f = _dim_query(ser, 'FRQ?')
            logger.info(f"DIM-3000 {name}: {freq_hz/1e6:.3f} MHz, "
                        f"+{amplitude_x10/10:.1f} dBm, RF ON (freq={f})")
    finally:
        ser_up.close()
        ser_down.close()


def _sigma_from_moment(profile):
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
    try:
        from scipy.optimize import curve_fit

        def gauss_with_offset(x, offset, amplitude, mu, sigma):
            return offset + amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
        p0 = [baseline, amp0, mu0, sigma0]
        popt, _ = curve_fit(gauss_with_offset, x, profile,
                            p0=p0, bounds=([-np.inf, 0, 0, 1], [np.inf, np.inf, n, n]),
                            maxfev=2000)
        fit_offset, fit_amp, fit_mu, fit_sigma = popt
        if fit_sigma < 1 or fit_sigma > n / 2 or not np.isfinite(fit_sigma):
            return float(mu0), float(sigma0), y_bg
        y_display = profile - fit_offset
        np.clip(y_display, 0, None, out=y_display)
        return float(fit_mu), float(fit_sigma), y_display
    except Exception:
        return float(mu0), float(sigma0), y_bg


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
INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '2000'},
    'mot_aom_power': {'MOT AOM Power (V)': '1.0'},
    'repump_aom_power': {'Repump AOM Power (V)': '1.0'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '4.5'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0.31'},
    'Repump_VCO': {'Repump VCO (V)': '0.2'},

    # ===================== PHASE 2: CMOT =====================
    'cmot_hold_time': {'CMOT Hold Time (ms)': '10'},
    'cmot_mot_power': {'CMOT MOT AOM Power (V)': '0.5'},
    'cmot_repump_power': {'CMOT Repump AOM Power (V)': '0.15'},
    'cmot_detuning_voltage': {'CMOT MOT VCO (V) [-40MHz]': '-0.17'},
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '6.5'},
    'cmot_repump_vco': {'CMOT Repump VCO (V) [-3MHz]': '0.26'},

    # ===================== PHASE 3: DARK MOT =====================
    'dark_mot_duration': {'Dark MOT Duration (ms)': '10'},
    'dark_mot_power': {'Dark MOT Cooling Power (V)': '0.2'},
    'dark_mot_detuning': {'Dark MOT Cooling VCO (V) [-55MHz]': '-0.46'},
    'dark_mot_repump_power': {'Dark MOT Repump Power (V)': '0.1'},
    'dark_mot_repump_vco': {'Dark MOT Repump VCO (V)': '0.26'},

    # ===================== PHASE 4: PGC =====================
    'pgc_duration': {'PGC Duration (ms)': '10'},
    'pgc_mot_power': {'PGC Cooling Power (V)': '0.15'},
    'pgc_detuning': {'PGC Cooling VCO (V) [-100MHz]': '-1.32'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.05'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '0.26'},

    # ===================== PHASE 5: LATTICE LOADING =====================
    'lattice_start_v': {'Lattice Ramp Start (V)': '-1.0'},
    'lattice_end_v': {'Lattice Ramp End (V)': '0.3'},

    'lattice_hold_ms': {'Lattice Hold Duration (ms)': '35'},

    # ===================== RAMP DURATION SWEEP =====================
    'ramp_1_ms': {'Ramp 1 (ms)': '5'},
    'ramp_2_ms': {'Ramp 2 (ms)': '10'},
    'ramp_3_ms': {'Ramp 3 (ms)': '15'},
    'ramp_4_ms': {'Ramp 4 (ms)': '20'},
    'ramp_5_ms': {'Ramp 5 (ms)': '30'},
    'ramp_6_ms': {'Ramp 6 (ms)': '40'},

    # ===================== DIM-3000 AOM DRIVER =====================
    'dim_freq_mhz': {'DIM3000 Frequency (MHz)': '100.000'},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)': '34.0'},

    # ===================== IMAGING =====================
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
    ramp_values = [
        int(dataset.get_input_parameter("ramp_1_ms")),
        int(dataset.get_input_parameter("ramp_2_ms")),
        int(dataset.get_input_parameter("ramp_3_ms")),
        int(dataset.get_input_parameter("ramp_4_ms")),
        int(dataset.get_input_parameter("ramp_5_ms")),
        int(dataset.get_input_parameter("ramp_6_ms")),
    ]
    scan_labels = [f"Ramp={t}ms" for t in ramp_values]
    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        diff_levels=DIFF_LEVELS, avg_levels=AVG_DIFF_LEVELS,
        title="Lattice ramp duration sweep (hold=35ms)",
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
    dark_detuning = float(dataset.get_input_parameter("dark_mot_detuning"))
    dark_rep_power = float(dataset.get_input_parameter("dark_mot_repump_power"))
    dark_rep_vco = float(dataset.get_input_parameter("dark_mot_repump_vco"))

    pgc_ms = int(dataset.get_input_parameter("pgc_duration"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))

    lat_start_v = float(dataset.get_input_parameter("lattice_start_v"))
    lat_end_v = float(dataset.get_input_parameter("lattice_end_v"))

    lat_hold_ms = int(dataset.get_input_parameter("lattice_hold_ms"))

    ramp_values_ms = [
        int(dataset.get_input_parameter("ramp_1_ms")),
        int(dataset.get_input_parameter("ramp_2_ms")),
        int(dataset.get_input_parameter("ramp_3_ms")),
        int(dataset.get_input_parameter("ramp_4_ms")),
        int(dataset.get_input_parameter("ramp_5_ms")),
        int(dataset.get_input_parameter("ramp_6_ms")),
    ]
    n_scan = len(ramp_values_ms)

    dim_freq_mhz = float(dataset.get_input_parameter("dim_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))

    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    # =========================================================================
    # CONFIGURE DIM-3000: both at same fixed frequency
    # =========================================================================
    freq_hz = int(round(dim_freq_mhz * 1e6))
    amp_x10 = int(round(dim_amplitude_dbm * 10))
    _dim_configure_fixed(logger, freq_hz, amp_x10)

    # =========================================================================
    # TIMING (fixed for all scan points)
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
    pgc_s = ms2s(pgc_ms)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    # Fixed timeline up to lattice ramp end
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    dark_start = cmot_end
    dark_end = dark_start + dark_s
    pgc_start = dark_end
    pgc_end = pgc_start + pgc_s
    lat_ramp_start = pgc_end
    coils_off = cmot_end - coils_lead_s

    logger.info(
        f"Lattice hold sweep: mot={mot_loading_ms}ms, cmot={CMOT_RAMP_MS}+{cmot_hold_ms}ms, "
        f"dark={dark_ms}ms, pgc={pgc_ms}ms, "
        f"hold={lat_hold_ms}ms, ramp_sweep={ramp_values_ms}ms, DIM={dim_freq_mhz:.3f}MHz"
    )

    # =========================================================================
    # BUILD SUB-SEQUENCE PER HOLD TIME
    # =========================================================================
    lat_hold_s = ms2s(lat_hold_ms)

    def build_sub_sequence(ramp_ms):
        lat_ramp_s = ms2s(ramp_ms)
        lat_ramp_end = lat_ramp_start + lat_ramp_s
        lat_hold_end = lat_ramp_end + lat_hold_s
        release = lat_hold_end  # TOF=0
        f1 = release
        f2 = f1 + clear_s
        lat_off_time = f2 + aom_pulse_s
        N = lat_off_time + 2
        crt = f1 + coils_restore_s

        cam = [0] * N
        for s in range(ttl_up):
            cam[f1 + s] = 1
            cam[f2 + s] = 1

        ado = [0] * N
        aao = [0.0] * N
        rdo = [0] * N
        rao = [0.0] * N
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
        # Lattice ramp: PGC beams stay on
        for s in range(lat_ramp_s):
            t = lat_ramp_start + s
            ado[t] = 1
            aao[t] = pgc_mot_power
            rdo[t] = 1
            rao[t] = pgc_rep_power
        # Lattice hold: beams off
        for s in range(lat_ramp_end, min(f1, N)):
            ado[s] = 0
            aao[s] = 0.0
            rdo[s] = 0
            rao[s] = 0.0
        # Imaging pulses
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

        # Coils
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
        for s in range(lat_ramp_s):
            mvco[lat_ramp_start + s] = pgc_detuning
        for s in range(lat_ramp_end, N):
            mvco[s] = pgc_detuning
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
        for s in range(dark_s):
            rvco[dark_start + s] = dark_rep_vco
        for s in range(pgc_s):
            rvco[pgc_start + s] = pgc_rep_vco
        for s in range(lat_ramp_s):
            rvco[lat_ramp_start + s] = pgc_rep_vco
        for s in range(lat_ramp_end, N):
            rvco[s] = pgc_rep_vco

        # Lattice channels
        ludo = [1] * N
        lddo = [1] * N
        luao = [lat_start_v] * N
        ldao = [lat_start_v] * N
        for s in range(lat_ramp_s):
            t = lat_ramp_start + s
            frac = s / max(lat_ramp_s - 1, 1)
            ludo[t] = 0
            lddo[t] = 0
            v = lat_start_v + frac * (lat_end_v - lat_start_v)
            luao[t] = v
            ldao[t] = v
        for s in range(lat_ramp_end, min(lat_off_time, N)):
            ludo[s] = 0
            lddo[s] = 0
            luao[s] = lat_end_v
            ldao[s] = lat_end_v
        for s in range(lat_off_time, N):
            ludo[s] = 1
            lddo[s] = 1
            luao[s] = lat_start_v
            ldao[s] = lat_start_v

        # No seg triggers
        seg = [0] * N

        return cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, ludo, lddo, luao, ldao, seg, N

    # Concatenate
    cam_p = []
    ado_p = []
    aao_p = []
    rdo_p = []
    rao_p = []
    cdo_p = []
    cao_p = []
    mvco_p = []
    rvco_p = []
    ludo_p = []
    lddo_p = []
    luao_p = []
    ldao_p = []
    seg_p = []
    for r_ms in ramp_values_ms:
        cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, ludo, lddo, luao, ldao, seg, N = build_sub_sequence(r_ms)
        cam_p += cam
        ado_p += ado
        aao_p += aao
        rdo_p += rdo
        rao_p += rao
        cdo_p += cdo
        cao_p += cao
        mvco_p += mvco
        rvco_p += rvco
        ludo_p += ludo
        lddo_p += lddo
        luao_p += luao
        ldao_p += ldao
        seg_p += seg

    total = len(cam_p)
    logger.info(f"Ramp sweep: ramps={ramp_values_ms}ms, total={total} samples")

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
        dataset.camera_client.start_acquisition()

        try:
            NI_card_1.finalize_clock()
        except Exception:
            pass

        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(do_channel="dio0", value=cam_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio1", value=ado_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio2", value=cdo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio3", value=rdo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio4", value=ludo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio5", value=lddo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio6", value=seg_p, sample_rate=ni_rate)
        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel="ao1", voltages=aao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao2", voltages=cao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao3", voltages=mvco_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao4", voltages=rvco_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao5", voltages=rao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=luao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=ldao_p, sample_rate=ni_rate)

        h1 = NI_card_2.arm(regeneration=True)
        h2 = NI_card_3.arm(regeneration=True)
        NI_card_1.arm_clock(length=total, sample_rate=ni_rate)

        OPX_client.build_stack()
        clk = OPX_client.create_new_do_elem(do_channel=1, length=500)
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

        try:
            for run_idx in range(M):
                logger.info(f"Run {run_idx+1} of {M}")
                for scan_idx, r_ms in enumerate(ramp_values_ms):
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
                    logger.info(
                        f"  Ramp={r_ms}ms: n={diff_counts[scan_idx]}, "
                        f"area={float(avg.sum()):.0f}")
                time.sleep(wait_time)
        finally:
            dataset.camera_client.stop_acquisition()

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
