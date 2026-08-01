# Fast single image: MOT -> CMOT -> [Dark MOT] -> [PGC] -> [Lattice ramp] -> release -> image
# Lattice phase: ramp lattice power on (-1V to 0V) while ramping PGC beams off.
# Lattice stays on through TOF + imaging, turns off after background frame.
# Re-arm-safe: finalize clock, load cards, arm data, arm clock, OPX execute.

import numpy as np
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
    'dark_mot_enabled': {'Dark MOT Enabled (1=yes, 0=no)': '1'},
    'dark_mot_duration': {'Dark MOT Duration (ms)': '20'},
    'dark_mot_power': {'Dark MOT Cooling Power (V)': '0.2'},
    'dark_mot_detuning': {'Dark MOT Cooling VCO (V) [-55MHz]': '-0.46'},
    'dark_mot_repump_power': {'Dark MOT Repump Power (V)': '0.1'},
    'dark_mot_repump_vco': {'Dark MOT Repump VCO (V)': '0.26'},

    # ===================== PHASE 4: PGC =====================
    'pgc_enabled': {'PGC Enabled (1=yes, 0=no)': '1'},
    'pgc_duration': {'PGC Duration (ms)': '10'},
    'pgc_mot_power': {'PGC Cooling Power (V)': '0.15'},
    'pgc_detuning': {'PGC Cooling VCO (V) [-100MHz]': '-1.32'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.05'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '0.26'},

    # ===================== PHASE 5: LATTICE LOADING =====================
    # Ramp lattice on while ramping PGC beams off.
    # Lattice stays on through TOF + imaging, off after background.
    'lattice_enabled': {'Lattice Loading Enabled (1=yes, 0=no)': '0'},
    'lattice_ramp_ms': {'Lattice Ramp Duration (ms)': '10'},
    'lattice_hold_ms': {'Lattice Hold Duration (ms)': '10'},
    'lattice_start_v': {'Lattice Ramp Start (V)': '-1.0'},
    'lattice_end_v': {'Lattice Ramp End (V)': '0.0'},

    # ===================== RELEASE + IMAGING =====================
    'tof_us': {'TOF (us)': '0'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 1)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.6'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '10'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '190'},

    # ===================== DISPLAY =====================
    'display_min': {'Display Min (counts)': '-5'},
    'display_max': {'Display Max (counts)': '50'},
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
    # READ ALL GUI PARAMETERS
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

    dark_enabled = int(dataset.get_input_parameter("dark_mot_enabled")) != 0
    dark_ms = int(dataset.get_input_parameter("dark_mot_duration"))
    dark_power = float(dataset.get_input_parameter("dark_mot_power"))
    dark_detuning = float(dataset.get_input_parameter("dark_mot_detuning"))
    dark_rep_power = float(dataset.get_input_parameter("dark_mot_repump_power"))
    dark_rep_vco = float(dataset.get_input_parameter("dark_mot_repump_vco"))

    pgc_enabled = int(dataset.get_input_parameter("pgc_enabled")) != 0
    pgc_ms = int(dataset.get_input_parameter("pgc_duration"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))

    lat_enabled = int(dataset.get_input_parameter("lattice_enabled")) != 0
    lat_ramp_ms = int(dataset.get_input_parameter("lattice_ramp_ms"))
    lat_hold_ms = int(dataset.get_input_parameter("lattice_hold_ms"))
    lat_start_v = float(dataset.get_input_parameter("lattice_start_v"))
    lat_end_v = float(dataset.get_input_parameter("lattice_end_v"))

    tof_us = int(dataset.get_input_parameter("tof_us"))
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))

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
    pgc_s = ms2s(pgc_ms) if pgc_enabled else 0
    lat_ramp_s = ms2s(lat_ramp_ms) if lat_enabled else 0
    lat_hold_s = ms2s(lat_hold_ms) if lat_enabled else 0
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    # =========================================================================
    # TIMELINE ANCHORS
    # MOT -> CMOT -> [Dark MOT] -> [PGC] -> [Lattice ramp+hold] -> release
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
    release = lat_hold_end
    tof_samp = us2s(tof_us)
    f1 = release + tof_samp
    f2 = f1 + clear_s
    # Lattice turns off after frame 2
    lat_off_time = f2 + aom_pulse_s
    N = max(f2 + ttl_up + 1, lat_off_time + 2)

    # Coils OFF before first field-free phase
    if dark_enabled:
        coils_off = cmot_end - coils_lead_s
    elif pgc_enabled:
        coils_off = pgc_start - coils_lead_s
    elif lat_enabled:
        coils_off = lat_ramp_start - coils_lead_s
    else:
        coils_off = release - coils_lead_s
    coils_restore_t = f1 + coils_restore_s

    dark_str = "ON" if dark_enabled else "OFF"
    pgc_str = "ON" if pgc_enabled else "OFF"
    lat_str = "ON" if lat_enabled else "OFF"
    logger.info(
        f"Fast MOT+Lattice: mot={mot_loading_ms}ms, "
        f"cmot=ON(ramp={CMOT_RAMP_MS}+hold={cmot_hold_ms}ms), "
        f"dark={dark_str}({dark_ms}ms), pgc={pgc_str}({pgc_ms}ms), "
        f"lattice={lat_str}(ramp={lat_ramp_ms}+hold={lat_hold_ms}ms, "
        f"{lat_start_v}V->{lat_end_v}V), "
        f"coils_off={coils_off/SPM:.0f}ms, release={release/SPM:.0f}ms, "
        f"tof={tof_us}us, N={N}, cycle~{N/ni_rate*1000:.0f}ms"
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

    # Phase 1: MOT
    for s in range(cmot_start):
        ado[s] = 1
        aao[s] = mot_aom_power
        rdo[s] = 1
        rao[s] = repump_aom_power

    # Phase 2: CMOT ramp + hold
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

    # Phase 3: Dark MOT
    if dark_enabled:
        for s in range(dark_s):
            t = dark_start + s
            ado[t] = 1
            aao[t] = dark_power
            rdo[t] = 1
            rao[t] = dark_rep_power

    # Phase 4: PGC
    if pgc_enabled:
        for s in range(pgc_s):
            t = pgc_start + s
            ado[t] = 1
            aao[t] = pgc_mot_power
            rdo[t] = 1
            rao[t] = pgc_rep_power

    # Phase 5: Lattice ramp -- PGC beams stay on at constant power
    if lat_enabled and lat_ramp_s > 0:
        # Determine what power the MOT/repump beams are at when lattice ramp starts
        if pgc_enabled:
            beam_mot = pgc_mot_power
            beam_rep = pgc_rep_power
        elif dark_enabled:
            beam_mot = dark_power
            beam_rep = dark_rep_power
        else:
            beam_mot = cmot_mot_power
            beam_rep = cmot_repump_power
        # PGC beams stay on at constant power during lattice ramp
        for s in range(lat_ramp_s):
            t = lat_ramp_start + s
            ado[t] = 1
            aao[t] = beam_mot
            rdo[t] = 1
            rao[t] = beam_rep

    # Lattice hold: PGC beams off instantly
    if lat_enabled:
        for s in range(lat_hold_s):
            t = lat_ramp_end + s
            ado[t] = 0
            aao[t] = 0.0
            rdo[t] = 0
            rao[t] = 0.0

    # TOF: everything off (MOT beams already off, lattice stays on)
    for s in range(release, min(f1, N)):
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

    # Coils DO
    cdo = [0] * N
    for s in range(coils_off, min(coils_restore_t, N)):
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
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        mvco[t] = vco_loading + frac * (cmot_detuning - vco_loading)
    for s in range(cmot_hold_start, cmot_end):
        mvco[s] = cmot_detuning
    if dark_enabled:
        for s in range(dark_s):
            mvco[dark_start + s] = dark_detuning
    if pgc_enabled:
        for s in range(pgc_s):
            mvco[pgc_start + s] = pgc_detuning
    # Fill after last cooling phase (lattice ramp keeps PGC detuning for beam ramp-down)
    if lat_enabled and pgc_enabled:
        for s in range(lat_ramp_s):
            mvco[lat_ramp_start + s] = pgc_detuning
    if pgc_enabled:
        last_det = pgc_detuning
        last_end = lat_hold_end if lat_enabled else pgc_end
    elif dark_enabled:
        last_det = dark_detuning
        last_end = dark_end
    else:
        last_det = cmot_detuning
        last_end = cmot_end
    for s in range(last_end, N):
        mvco[s] = last_det
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
    if pgc_enabled:
        for s in range(pgc_s):
            rvco[pgc_start + s] = pgc_rep_vco
    if lat_enabled and pgc_enabled:
        for s in range(lat_ramp_s):
            rvco[lat_ramp_start + s] = pgc_rep_vco
    if pgc_enabled:
        last_rvco = pgc_rep_vco
    elif dark_enabled:
        last_rvco = dark_rep_vco
    else:
        last_rvco = cmot_repump_vco
    for s in range(last_end, N):
        rvco[s] = last_rvco

    # =========================================================================
    # LATTICE CHANNELS
    # Active-low TTL: 0=on, 1=off
    # Amplitude: ramp from lat_start_v to lat_end_v during ramp phase
    # =========================================================================
    lat_up_do = [1] * N   # default OFF
    lat_down_do = [1] * N
    lat_up_ao = [lat_start_v] * N   # default at start voltage
    lat_down_ao = [lat_start_v] * N

    if lat_enabled:
        # During ramp: TTL on, amplitude ramps
        for s in range(lat_ramp_s):
            t = lat_ramp_start + s
            frac = s / max(lat_ramp_s - 1, 1)
            lat_up_do[t] = 0
            lat_down_do[t] = 0  # TTL LOW = on
            v = lat_start_v + frac * (lat_end_v - lat_start_v)
            lat_up_ao[t] = v
            lat_down_ao[t] = v

        # During hold: TTL on, amplitude at end value
        for s in range(lat_hold_s):
            t = lat_ramp_end + s
            lat_up_do[t] = 0
            lat_down_do[t] = 0
            lat_up_ao[t] = lat_end_v
            lat_down_ao[t] = lat_end_v

        if tof_samp == 0:
            # TOF = 0: atoms trapped in lattice, keep lattice on through imaging
            for s in range(release, min(lat_off_time, N)):
                lat_up_do[s] = 0
                lat_down_do[s] = 0
                lat_up_ao[s] = lat_end_v
                lat_down_ao[s] = lat_end_v
            # Turn off after frame 2 imaging pulse
            for s in range(lat_off_time, N):
                lat_up_do[s] = 1
                lat_down_do[s] = 1
                lat_up_ao[s] = lat_start_v
                lat_down_ao[s] = lat_start_v
        else:
            # TOF > 0: lattice off at release, atoms expand freely
            for s in range(release, N):
                lat_up_do[s] = 1
                lat_down_do[s] = 1
                lat_up_ao[s] = lat_start_v
                lat_down_ao[s] = lat_start_v

    # =========================================================================
    # CLEAR STALE CLOCK, LOAD, ARM, EXECUTE
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

    # Load data cards
    NI_card_2.build_stack()
    NI_card_2.set_do_voltage(do_channel="dio0", value=cam, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio1", value=ado, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio2", value=cdo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio3", value=rdo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio4", value=lat_up_do, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio5", value=lat_down_do, sample_rate=ni_rate)

    NI_card_3.build_stack()
    NI_card_3.set_ao_voltage(ao_channel="ao1", voltages=aao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao2", voltages=cao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao3", voltages=mvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao4", voltages=rvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao5", voltages=rao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=lat_up_ao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=lat_down_ao, sample_rate=ni_rate)

    # Arm data cards BEFORE clock
    h1 = NI_card_2.arm(regeneration=True)
    h2 = NI_card_3.arm(regeneration=True)

    # Arm clock AFTER data cards
    NI_card_1.arm_clock(length=N, sample_rate=ni_rate)

    # OPX last
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
        # Discard first frame pair (OPX/NI sync settling)
        logger.info("Discarding first frame pair (sync settling)")
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
