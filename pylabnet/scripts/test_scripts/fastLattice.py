# Fast single image: MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown -> release -> TOF -> image
# WITH LATTICE LOADING via two-stage PGC:
#   PGC_cool (5ms, default): full PGC light, lattice OFF -- sub-Doppler cooling
#   PGC_rampdown (10ms, default): PGC light ramps to 0 WHILE lattice ramps up
#     to full -- adiabatic transfer from molasses into lattice trap.
# After release: PGC off, lattice holds at max through TOF + both imaging frames.
# Both up (COM4/dio4/ao6) and down (COM5/dio5/ao7) lattices follow identical ramp.
# NB: DIM-3000 max amplitude assumed AMP=320 (+32 dBm) via serial setup;
#     ao6/ao7 at +1.0V => max scaling, at -1.0V => parked off.

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

    # ===================== PHASE 4a: PGC COOL =====================
    # Field-free sub-Doppler cooling, lattice still OFF.
    'pgc_cool_ms': {'PGC Cool Duration (ms)': '5'},
    'pgc_mot_power': {'PGC Cooling Power (V)': '0.15'},
    'pgc_detuning': {'PGC Cooling VCO (V) [-122MHz]': '-1.75'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.05'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '0.26'},

    # ===================== PHASE 4b: PGC RAMP-DOWN =====================
    # PGC light ramps to 0 while lattice ramps to full. Adiabatic transfer.
    'pgc_rampdown_ms': {'PGC Rampdown Duration (ms)': '10'},

    # ===================== PHASE 5: LATTICE HOLD =====================
    # Atoms held in the lattice with all cooling light off, before release.
    # Mimics what the lattice_holdSweep / lattice_biasSweep scripts do.
    'lattice_hold_ms': {'Lattice Hold (ms)': '20'},

    # ===================== RELEASE + IMAGING =====================
    'tof_us': {'TOF (us)': '5000'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'coils_rampback_ms': {'Coils Rampback Duration (ms)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.6'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '10'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '100'},

    # ===================== BIAS COILS =====================
    # Held CONSTANT at these values for the ENTIRE cycle (no ramping).
    # Matches the sweep scripts (lattice_biasXSweep etc.).
    'pgc_bias_x_v': {'PGC Bias X (V)': '0.0'},
    'pgc_bias_y_v': {'PGC Bias Y (V)': '0.44'},
    'pgc_bias_z_v': {'PGC Bias Z (V)': '0.0'},

    # ===================== LATTICE =====================
    # Ramps during PGC: ramp_start_v -> ramp_end_v over pgc_duration.
    # After PGC, holds at ramp_end_v through TOF + both imaging frames.
    # ramp_start_v = -1.0V => DIM AM input below threshold (effectively off).
    # ramp_end_v = +0.3V => previously-optimized lattice depth.
    # Lattice AM scaling on ao6/ao7. With DIM-3000 set to AMP=320 (+32 dBm)
    # via dim_rf_on.py serial setup, AM=+1.0V => full +32 dBm output.
    # AM=-1.0V parks below threshold (effectively off).
    'lattice_ramp_start_v': {'Lattice Ramp Start (V)': '-1.0'},
    'lattice_ramp_end_v': {'Lattice Ramp End (V)': '1.0'},
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

    dataset.img_win = pg.GraphicsLayoutWidget(show=True, title="PGC + Lattice (atoms - background)")
    dataset.img_win.resize(700, 600)
    cmap = pg.colormap.get('inferno')   # matches bias sweep colormap

    vb = dataset.img_win.addViewBox(lockAspect=True)
    vb.invertY(True)
    dataset.img_item = pg.ImageItem(axisOrder='row-major')
    dataset.img_item.setLookupTable(cmap.getLookupTable())
    dataset.img_item.setLevels((0, 20))   # matches DIFF_LEVELS in bias sweeps
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

    pgc_cool_ms = int(dataset.get_input_parameter("pgc_cool_ms"))
    pgc_rampdown_ms = int(dataset.get_input_parameter("pgc_rampdown_ms"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))

    lattice_hold_ms = int(dataset.get_input_parameter("lattice_hold_ms"))

    tof_us = int(dataset.get_input_parameter("tof_us"))
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    coils_rampback_ms = int(dataset.get_input_parameter("coils_rampback_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))

    pgc_bias_x_v = float(dataset.get_input_parameter("pgc_bias_x_v"))
    pgc_bias_y_v = float(dataset.get_input_parameter("pgc_bias_y_v"))
    pgc_bias_z_v = float(dataset.get_input_parameter("pgc_bias_z_v"))

    lat_start_v = float(dataset.get_input_parameter("lattice_ramp_start_v"))
    lat_end_v = float(dataset.get_input_parameter("lattice_ramp_end_v"))

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
    pgc_cool_s = ms2s(pgc_cool_ms)
    pgc_rampdown_s = ms2s(pgc_rampdown_ms)
    pgc_s = pgc_cool_s + pgc_rampdown_s
    lattice_hold_s = ms2s(lattice_hold_ms)
    tof_samp = us2s(tof_us)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    coils_rampback_s = ms2s(coils_rampback_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    # =========================================================================
    # TIMELINE: MOT -> CMOT -> Dark MOT -> PGC -> release -> TOF -> image
    # =========================================================================
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    dark_start = cmot_end
    dark_end = dark_start + dark_s
    pgc_start = dark_end
    pgc_cool_end = pgc_start + pgc_cool_s
    pgc_end = pgc_cool_end + pgc_rampdown_s
    release = pgc_end + lattice_hold_s   # atoms sit in lattice for hold_s, then release
    coils_off = cmot_end - coils_lead_s
    f1 = release + tof_samp
    f2 = f1 + clear_s
    coils_restore_t = f2 + aom_pulse_s + coils_restore_s  # coils stay OFF through frame 2
    N = coils_restore_t + coils_rampback_s + 2

    logger.info(
        f"Fast PGC+Lattice: mot={mot_loading_ms}ms, cmot={CMOT_RAMP_MS}+{cmot_hold_ms}ms, "
        f"dark={dark_ms}ms, pgc_cool={pgc_cool_ms}ms, pgc_rampdown={pgc_rampdown_ms}ms, "
        f"lattice_hold={lattice_hold_ms}ms, tof={tof_us}us, "
        f"(lattice CONSTANT at {lat_end_v:.2f}V whole cycle), "
        f"release={release/SPM:.0f}ms, N={N}"
    )

    # =========================================================================
    # BUILD WAVEFORMS
    # =========================================================================
    cam = [0] * N
    for s in range(ttl_up):
        cam[f1 + s] = 1
        cam[f2 + s] = 1

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
    # PGC cool (full PGC light, lattice off)
    for s in range(pgc_cool_s):
        t = pgc_start + s
        ado[t] = 1
        aao[t] = pgc_mot_power
        rdo[t] = 1
        rao[t] = pgc_rep_power
    # PGC rampdown: cooling AND repump ramp linearly to 0 over rampdown_s.
    # TTLs stay HIGH (light on) until amplitude hits 0; cleaner to keep TTL high
    # throughout and just let amplitude fall, so atoms see a smooth handoff.
    for s in range(pgc_rampdown_s):
        t = pgc_cool_end + s
        frac = s / max(pgc_rampdown_s - 1, 1)
        ado[t] = 1
        aao[t] = pgc_mot_power * (1.0 - frac)
        rdo[t] = 1
        rao[t] = pgc_rep_power * (1.0 - frac)
    # PGC end -> release: lattice hold (atoms only in lattice, all PGC light off).
    # Then release -> F1: TOF (lattice still on, atoms expand inside lattice).
    # Digital off for both; analog stays at full power (thermal stability).
    for s in range(pgc_end, min(f1, N)):
        ado[s] = 0
        rdo[s] = 0
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
    # PGC cool + rampdown: hold cooling beam at pgc_detuning throughout
    for s in range(pgc_s):
        mvco[pgc_start + s] = pgc_detuning
    for s in range(pgc_end, N):
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
    for s in range(pgc_end, N):
        rvco[s] = pgc_rep_vco

    # Lattice: BOTH AM and TTL held constant for the entire cycle so the
    # AOM crystal sees identical RF drive at all times -> stable thermal
    # state -> stable fiber coupling. Lattice light is always present at
    # the chamber, but during MOT loading there are no cold atoms to care.
    # lat_start_v is kept as a GUI parameter for backward compatibility but
    # is now unused (AM is constant at lat_end_v throughout).
    # Up beam: dio4/ao6  |  Down beam: dio5/ao7
    ludo = [0] * N
    lddo = [0] * N                  # TTL low = beam ON, entire cycle
    luao = [lat_end_v] * N
    ldao = [lat_end_v] * N  # AM constant at lat_end_v

    # Bias coils: held CONSTANT at GUI values for the ENTIRE cycle (no ramping).
    # Matches the sweep scripts. Bias light therefore present through MOT
    # loading too, which is fine: the quadrupole still dominates during MOT.
    biasx = [pgc_bias_x_v] * N
    biasy = [pgc_bias_y_v] * N
    biasz = [pgc_bias_z_v] * N

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
    NI_card_2.set_do_voltage(do_channel="dio6", value=[0] * N, sample_rate=ni_rate)

    NI_card_3.build_stack()
    NI_card_3.set_ao_voltage(ao_channel="ao1", voltages=aao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao2", voltages=cao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao3", voltages=mvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao4", voltages=rvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao5", voltages=rao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=luao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=ldao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao8", voltages=biasx, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao9", voltages=biasy, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao10", voltages=biasz, sample_rate=ni_rate)

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
            # Subtracted image, fixed levels (0, 20), no autoscaling.
            diff = frame1.astype(np.int32) - frame2.astype(np.int32)
            dataset.img_item.setImage(diff, levels=(0, 20), autoLevels=False)
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
