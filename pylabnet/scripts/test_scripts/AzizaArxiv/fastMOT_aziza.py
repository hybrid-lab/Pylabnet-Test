# Fast single MOT image -- no scans, no fits, no CMOT.
# Just load MOT, release, take one image, subtract background, display, repeat.

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

DIFF_LEVELS = (0, 20)

INIT_DICT = {
    'mot_loading_time': {'MOT Loading Time (ms)': '500'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '5.5'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0.32'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.6'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '10'},
    'Repump_VCO': {'Repump VCO (V)': '0.2'},
    'Repump_AOM_voltage': {'Repump AOM Voltage (V)': '1.0'},
    'tof_us': {'TOF (us)': '2000'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '190'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
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

    # Single image display
    dataset.img_win = pg.GraphicsLayoutWidget(show=True, title="MOT image (diff)")
    dataset.img_win.resize(600, 500)
    vb = dataset.img_win.addViewBox(lockAspect=True)
    vb.invertY(True)
    dataset.img_item = pg.ImageItem(axisOrder='row-major')
    dataset.img_item.setLevels(DIFF_LEVELS)
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

    # Read parameters
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))
    repump_aom_v = float(dataset.get_input_parameter("Repump_AOM_voltage"))
    tof_us = int(dataset.get_input_parameter("tof_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))

    # Timing
    ni_rate = 20000
    SPM = ni_rate // 1000
    SPU = ni_rate / 1e6
    delay_ns = int(round(1e9 / ni_rate)) - 500
    ttl_up = max(1, SPM // 20)

    mot_s = mot_loading_ms * SPM
    tof_s = int(round(tof_us * SPU))
    clear_s = atoms_clear_ms * SPM
    lead_s = int(round(vco_lead_us * SPU))
    coils_lead_s = coils_lead_ms * SPM
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    release = mot_s
    coils_off = release - coils_lead_s
    f1 = release + tof_s
    f2 = f1 + clear_s
    N = f2 + ttl_up + 1

    # Build waveforms (single sub-sequence, no concatenation)
    cam = [0] * N
    for s in range(ttl_up):
        cam[f1 + s] = 1
        cam[f2 + s] = 1

    aom = [0] * N
    aom_ao = [0.0] * N
    for s in range(mot_s):
        aom[s] = 1
        aom_ao[s] = 1.0
    for s in range(aom_pulse_s):
        if f1 + s < N:
            aom[f1 + s] = 1
            aom_ao[f1 + s] = 1.0
        if f2 + s < N:
            aom[f2 + s] = 1
            aom_ao[f2 + s] = 1.0

    cdo = [0] * N
    for s in range(coils_off, N):
        cdo[s] = 1

    cao = [mot_coils_v] * N

    vco = [vco_loading] * N
    vlo = max(0, f1 - lead_s)
    vhi = min(N, f2 + aom_pulse_s)
    for s in range(vlo, vhi):
        vco[s] = vco_imaging

    rvco = [repump_vco] * N

    # Repump AOM: mirrors MOT AOM timing exactly (same DO pattern, own AO voltage)
    # Hardcoded channels: dio3 = Repump AOM DO, ao5 = Repump AOM AO
    # *** CHANGE THESE IF YOUR WIRING IS DIFFERENT ***
    rep_do = list(aom)           # identical TTL pattern to MOT AOM
    rep_ao = [0.0] * N
    for s in range(N):
        if aom[s]:
            rep_ao[s] = repump_aom_v  # 1.0 V when on, 0 when off

    logger.info(f"Fast MOT: load={mot_loading_ms}ms, tof={tof_us}us, N={N} samples, "
                f"cycle ~{N/ni_rate*1000:.0f}ms")

    # Arm everything ONCE, run the OPX in a huge loop, pull frames continuously
    camera_client.set_hardware_trigger(
        line="Line0", activation="RisingEdge",
        selector="FrameStart", overlap="ReadOut",
        acquisition_mode="Continuous")
    camera_client.set_exposure(camera_exposure_us)
    camera_client.try_set_float("Gain", 50.0)
    dataset.camera_client.start_acquisition()

    NI_card_1.arm_clock(length=N, sample_rate=ni_rate)

    NI_card_2.build_stack()
    NI_card_2.set_do_voltage(do_channel="dio0", value=cam, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio1", value=aom, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio2", value=cdo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio3", value=rep_do, sample_rate=ni_rate)

    NI_card_3.build_stack()
    NI_card_3.set_ao_voltage(ao_channel="ao1", voltages=aom_ao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao2", voltages=cao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao3", voltages=vco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao4", voltages=rvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao5", voltages=rep_ao, sample_rate=ni_rate)

    OPX_client.build_stack()
    clk = OPX_client.create_new_do_elem(do_channel=1, length=500)

    # OPX loops 100k times -- effectively runs forever until we stop it
    M = 100000
    run_buffer = 10_000_000  # 10 ms gap between reps
    with OPX_client.for_("j", 0, M, 1):
        with OPX_client.for_("i", 0, N + 1, 1):
            OPX_client.set_digital_voltage(element=clk)
            OPX_client.delay(delay_ns)
        OPX_client.delay(run_buffer)

    h1 = NI_card_2.arm(regeneration=True)
    h2 = NI_card_3.arm(regeneration=True)
    OPX_client.execute(wait=False)

    def get_frame(timeout_ms=1000):
        b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    # Pull frame pairs as fast as they come -- no re-arming between shots
    try:
        first = True
        while thread.running:
            try:
                frame1 = get_frame(timeout_ms=10000 if first else 2000)
            except Exception:
                if not thread.running:
                    break
                continue  # missed trigger, try again
            first = False

            if not thread.running:
                break

            try:
                frame2 = get_frame(timeout_ms=2000)
            except Exception:
                if not thread.running:
                    break
                continue

            diff = frame1.astype(np.int32) - frame2.astype(np.int32)
            dataset.img_item.setImage(diff, levels=DIFF_LEVELS, autoLevels=False)
            logger.info(f"area={float(diff.sum()):.0f}")
    finally:
        dataset.camera_client.stop_acquisition()
        try:
            NI_card_2.finalize(h1, timeout=120.0, force_finish=True)
        except Exception:
            pass
        try:
            NI_card_3.finalize(h2, timeout=120.0, force_finish=True)
        except Exception:
            pass
        try:
            NI_card_1.finalize_clock()
        except Exception:
            pass
