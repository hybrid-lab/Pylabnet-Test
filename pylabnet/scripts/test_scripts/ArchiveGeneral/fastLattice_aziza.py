# Fast single image: MOT -> CMOT -> Dark MOT -> PGC -> Lattice ramp -> Lattice hold -> image
# Stationary lattice only (no move). Both DIM-3000 at same fixed frequency.
# After power cycle, just run this code.

import numpy as np
import serial
import time
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

DIFF_LEVELS = (-5, 50)

DIM_COM_UP = 'COM4'
DIM_COM_DOWN = 'COM5'

INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '500'},
    'mot_aom_power': {'MOT AOM Power (V)': '1.0'},
    'repump_aom_power': {'Repump AOM Power (V)': '1.0'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '4.5'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0.31'},
    'Repump_VCO': {'Repump VCO (V)': '0.2'},

    # ===================== PHASE 2: CMOT (always on) =====================
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
    'pgc_detuning': {'PGC Cooling VCO (V) [-122MHz]': '-1.75'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.05'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '0.26'},

    # ===================== PHASE 5: LATTICE LOADING =====================
    'lattice_ramp_ms': {'Lattice Ramp Duration (ms)': '20'},
    'lattice_hold_ms': {'Lattice Hold Duration (ms)': '30'},
    'lattice_start_v': {'Lattice Ramp Start (V)': '-1.0'},
    'lattice_end_v': {'Lattice Ramp End (V)': '0.3'},

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

    # ===================== BIAS COILS =====================
    'bias_x_v': {'Bias X (V)': '0.15'},
    'bias_y_v': {'Bias Y (V)': '0.4'},
    'bias_z_v': {'Bias Z (V)': '0.0'},

    # ===================== DISPLAY =====================
    'display_min': {'Display Min (counts)': '-5'},
    'display_max': {'Display Max (counts)': '50'},
}


# =========================================================================
# DIM-3000 HELPERS
# =========================================================================
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


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")
    dataset.camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']

    disp_min = float(dataset.get_input_parameter("display_min"))
    disp_max = float(dataset.get_input_parameter("display_max"))
    dataset.diff_levels = (disp_min, disp_max)

    dataset.img_win = pg.GraphicsLayoutWidget(show=True, title="MOT image (diff)")
    dataset.img_win.resize(600, 500)
    vb = dataset.img_win.addViewBox(lockAspect=True)
    vb.invertY(True)
    dataset.img_item = pg.ImageItem(axisOrder='row-major')
    dataset.img_item.setLevels(dataset.diff_levels)
    cmap = pg.colormap.get('inferno')
    dataset.img_item.setLookupTable(cmap.getLookupTable())
    vb.addItem(dataset.img_item)
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

    lat_ramp_ms = int(dataset.get_input_parameter("lattice_ramp_ms"))
    lat_hold_ms = int(dataset.get_input_parameter("lattice_hold_ms"))
    lat_start_v = float(dataset.get_input_parameter("lattice_start_v"))
    lat_end_v = float(dataset.get_input_parameter("lattice_end_v"))

    dim_freq_mhz = float(dataset.get_input_parameter("dim_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))

    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))

    bias_x_v = float(dataset.get_input_parameter("bias_x_v"))
    bias_y_v = float(dataset.get_input_parameter("bias_y_v"))
    bias_z_v = float(dataset.get_input_parameter("bias_z_v"))

    # =========================================================================
    # CONFIGURE DIM-3000: both at same fixed frequency
    # =========================================================================
    freq_hz = int(round(dim_freq_mhz * 1e6))
    amp_x10 = int(round(dim_amplitude_dbm * 10))
    _dim_configure_fixed(logger, freq_hz, amp_x10)

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
    pgc_s = ms2s(pgc_ms)
    lat_ramp_s = ms2s(lat_ramp_ms)
    lat_hold_s = ms2s(lat_hold_ms)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    # =========================================================================
    # TIMELINE
    # MOT -> CMOT -> Dark MOT -> PGC -> Lat ramp (PGC on) -> Lat hold (dark) -> image
    # =========================================================================
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    dark_start = cmot_end
    dark_end = dark_start + dark_s
    pgc_start = dark_end
    pgc_end = pgc_start + pgc_s
    lat_ramp_start = pgc_end
    lat_ramp_end = lat_ramp_start + lat_ramp_s
    lat_hold_end = lat_ramp_end + lat_hold_s
    release = lat_hold_end  # TOF=0, image in lattice
    f1 = release
    f2 = f1 + clear_s
    lat_off_time = f2 + aom_pulse_s
    N = lat_off_time + 2

    coils_off = cmot_end - coils_lead_s
    coils_restore_t = f1 + coils_restore_s

    logger.info(
        f"Fast lattice: mot={mot_loading_ms}ms, cmot={CMOT_RAMP_MS}+{cmot_hold_ms}ms, "
        f"dark={dark_ms}ms, pgc={pgc_ms}ms, "
        f"lat_ramp={lat_ramp_ms}ms, lat_hold={lat_hold_ms}ms, "
        f"DIM={dim_freq_mhz:.3f}MHz, release={release/SPM:.0f}ms, N={N}"
    )

    # =========================================================================
    # BUILD WAVEFORMS
    # =========================================================================
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
    for s in range(coils_off, min(coils_restore_t, N)):
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

    # Lattice channels: on during ramp+hold+imaging, off after frame 2
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

    # Seg trigger: all zeros (no move)
    seg_do = [0] * N

    # =========================================================================
    # LOAD, ARM, EXECUTE
    # =========================================================================
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
    h1 = None
    h2 = None

    NI_card_2.build_stack()
    NI_card_2.set_do_voltage(do_channel="dio0", value=cam, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio1", value=ado, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio2", value=cdo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio3", value=rdo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio4", value=ludo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio5", value=lddo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio6", value=seg_do, sample_rate=ni_rate)

    NI_card_3.build_stack()
    NI_card_3.set_ao_voltage(ao_channel="ao1", voltages=aao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao2", voltages=cao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao3", voltages=mvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao4", voltages=rvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao5", voltages=rao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=luao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=ldao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao8", voltages=[bias_x_v] * N, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao9", voltages=[bias_y_v] * N, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao10", voltages=[bias_z_v] * N, sample_rate=ni_rate)

    h1 = NI_card_2.arm(regeneration=True)
    h2 = NI_card_3.arm(regeneration=True)
    NI_card_1.arm_clock(length=N, sample_rate=ni_rate)

    OPX_client.build_stack()
    clk = OPX_client.create_new_do_elem(do_channel=1, length=500)
    M = 100000
    run_buffer = 10_000_000
    with OPX_client.for_("j", 0, M, 1):
        with OPX_client.for_("i", 0, N + 1, 1):
            OPX_client.set_digital_voltage(element=clk)
            OPX_client.delay(delay_ns)
        OPX_client.delay(run_buffer)
    OPX_client.execute(wait=False)

    def get_frame(timeout_ms=1000):
        b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    try:
        logger.info("Discarding first frame pair")
        try:
            get_frame(timeout_ms=15000)
            get_frame(timeout_ms=10000)
        except Exception:
            pass

        while thread.running:
            try:
                frame1 = get_frame(timeout_ms=2000)
            except Exception:
                if not thread.running:
                    break
                continue
            if not thread.running:
                break
            try:
                frame2 = get_frame(timeout_ms=2000)
            except Exception:
                if not thread.running:
                    break
                continue
            diff = frame1.astype(np.int32) - frame2.astype(np.int32)
            dataset.img_item.setImage(diff, autoLevels=True)
            logger.info(f"area={float(diff.sum()):.0f}")
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
