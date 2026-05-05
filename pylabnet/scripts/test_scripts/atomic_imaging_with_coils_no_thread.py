import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D  # noqa: F401
import time
from qt_plotting import QtMatplotlibFrameViewer


if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

FRAME1_LEVELS = (0, 20)
FRAME2_LEVELS = (0, 20)
DIFF_LEVELS = (0, 20)
AVG_DIFF_LEVELS = (0, 20)

# -----------------------------
# Experiment script settings
# Timing values are in ms.
# -----------------------------

INIT_DICT = {
    'imaging_AOM_start': {'Imaging AOM Start Time (ms)': '0'},
    'imaging_AOM_end': {'Imaging AOM End Time (ms)': '5'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '1000'},

    # To not use scan set all scan parameters to -1
    'frame_1': {'Camera Frame 1 Time (ms)': '5'},
    'frame_2': {'Camera Frame 2 Time (ms)': '20'},
    'wait_time': {'Wait Time Between Cycles (s)': '0'},

    'camera_trigger_do': {'Camera Trigger DO Channel': '0'},
    'imaging_AOM_do': {'Imaging AOM DO Channel': '1'},
    'imaging_AOM_ao': {'Imaging AOM AO Channel': '1'},

    # MOT coils controls
    'mot_coils_do': {'MOT Coils DO Channel': '2'},
    'mot_coils_ao': {'MOT Coils AO Channel': '2'},
    'mot_coils_on_voltage': {'MOT Coils ON Analog Voltage (V)': '9.0'},
    'mot_coils_off_voltage': {'MOT Coils OFF Analog Voltage (V)': '0.0'},

    'opx_trigger_do': {'OPX Triggering DO Channel': '1'},
}


def define_dataset():
    """Specifies the type of plot to use for the data."""
    return 'Dataset'


def configure(**kwargs):
    """Sets up the hardware and the plot before the experiment runs."""
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")

    camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m']
    dataset.camera_client = camera_client

    NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_1 = NI_card_1
    NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_2 = NI_card_2
    NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    dataset.NI_card_3 = NI_card_3

    dataset.add_child(
        name="Frame 1",
        data_type=Plot2D,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    dataset.add_child(
        name="Frame 2",
        data_type=Plot2D,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    dataset.add_child(
        name="Image Difference",
        data_type=Plot2DWithAvg,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    frame1_ds = dataset.children["Frame 1"]
    frame2_ds = dataset.children["Frame 2"]
    diff_ds = dataset.children["Image Difference"]
    avg_ds = diff_ds.children["Image Differencecurrentavg"]

    frame1_ds.graph.setLevels(*FRAME1_LEVELS)
    frame2_ds.graph.setLevels(*FRAME2_LEVELS)
    diff_ds.graph.setLevels(*DIFF_LEVELS)
    avg_ds.graph.setLevels(*AVG_DIFF_LEVELS)

    frame1_ds.graph.view.setAspectLocked(True)
    frame2_ds.graph.view.setAspectLocked(True)
    diff_ds.graph.view.setAspectLocked(True)
    avg_ds.graph.view.setAspectLocked(True)

    dataset.graph.hide()


def experiment(**kwargs):
    """Run one experiment cycle and exit."""
    dataset = kwargs['dataset']
    logger = dataset.log

    dataset.NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    dataset.NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    dataset.NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
    dataset.camera_client = kwargs["fluorescence_imaging_camera_bfs_u3_51s5m"]
    dataset.OPX_client = kwargs["OPX_OPX"]

    OPX_client = dataset.OPX_client
    NI_card_1 = dataset.NI_card_1
    NI_card_2 = dataset.NI_card_2
    NI_card_3 = dataset.NI_card_3
    camera_client = dataset.camera_client

    imaging_AOM_start = int(dataset.get_input_parameter("imaging_AOM_start"))
    imaging_AOM_end = int(dataset.get_input_parameter("imaging_AOM_end"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    frame_1 = int(dataset.get_input_parameter("frame_1"))
    frame_2 = int(dataset.get_input_parameter("frame_2"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    opx_trigger_do = int(dataset.get_input_parameter("opx_trigger_do"))
    camera_trigger_do = "dio" + str(int(dataset.get_input_parameter("camera_trigger_do")))
    imaging_AOM_do = "dio" + str(int(dataset.get_input_parameter("imaging_AOM_do")))
    imaging_AOM_ao = "ao" + str(int(dataset.get_input_parameter("imaging_AOM_ao")))

    mot_coils_do = "dio" + str(int(dataset.get_input_parameter("mot_coils_do")))
    mot_coils_ao = "ao" + str(int(dataset.get_input_parameter("mot_coils_ao")))
    mot_coils_on_voltage = float(dataset.get_input_parameter("mot_coils_on_voltage"))
    mot_coils_off_voltage = float(dataset.get_input_parameter("mot_coils_off_voltage"))

    ni_sample_rate = 1000
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"
    opx_ttl_pulse_ns = 500
    sample_period_ns = int(round(1e9 / ni_sample_rate))
    delay_ns = sample_period_ns - opx_ttl_pulse_ns
    camera_ttl_up = 1
    aom_imaging_pulse_ms = max(1, int(np.ceil(camera_exposure_us / 1000.0)))

    if camera_exposure_us <= 0:
        raise ValueError("camera_exposure_us must be greater than 0")
    if delay_ns < 0:
        raise ValueError("ni_sample_rate is too high for a 500 ns OPX TTL pulse")
    if frame_1 < 0:
        raise ValueError("frame_1 must be non-negative")
    if frame_2 <= frame_1 + camera_ttl_up:
        raise ValueError("frame_2 must be greater than frame_1 + camera_ttl_up")
    if imaging_AOM_end < imaging_AOM_start:
        raise ValueError("imaging_AOM_end must be greater than or equal to imaging_AOM_start")

    logger.info(f"Time at start {time.perf_counter_ns()}")

    sequence_end_ms = max(
        frame_2 + camera_ttl_up,
        imaging_AOM_end
    )

    down_time = frame_2 - frame_1 - camera_ttl_up
    camera_trigger_pulse = (
        [0] * frame_1 +
        [1] * camera_ttl_up +
        [0] * down_time +
        [1] * camera_ttl_up +
        [0] * max(0, sequence_end_ms - frame_2 - camera_ttl_up)
    )

    imaging_AOM_pulse = [0] * sequence_end_ms
    for pulse_idx in range(imaging_AOM_start, min(imaging_AOM_end, sequence_end_ms)):
        imaging_AOM_pulse[pulse_idx] = 1
    for offset in range(aom_imaging_pulse_ms):
        pulse_idx = frame_2 + offset
        if pulse_idx < sequence_end_ms:
            imaging_AOM_pulse[pulse_idx] = 1

    # Keep the MOT coils on for the full sequence.
    mot_coils_do_pulse = [0] * sequence_end_ms

    # Keep the analog coils drive on for the full sequence.
    mot_coils_ao_pulse = [mot_coils_on_voltage] * sequence_end_ms

    experiment_length_ms = int(sequence_end_ms * 1)
    logger.info(
        f"Running sequence with frame_1={frame_1} ms, "
        f"frame_2={frame_2} ms, exposure={camera_exposure_us} us, "
        f"aom_start={imaging_AOM_start} ms, aom_end={imaging_AOM_end} ms, "
        f"frame2_aom_pulse={aom_imaging_pulse_ms} ms, "
        f"opx_delay={delay_ns} ns"
    )

    camera_client.set_hardware_trigger(
        line=trigger_line,
        activation=trigger_edge,
        selector="FrameStart",
        overlap="ReadOut",
        acquisition_mode="Continuous",
    )
    camera_client.set_exposure(camera_exposure_us)
    camera_client.try_set_float("Gain", 50.0)

    logger.info("Starting acquisition")
    dataset.camera_client.start_acquisition()

    NI_card_1.arm_clock(length=experiment_length_ms, sample_rate=ni_sample_rate)
    logger.info("Clock configured")

    NI_card_2.build_stack()
    NI_card_2.set_do_voltage(
        do_channel=camera_trigger_do,
        value=camera_trigger_pulse,
        sample_rate=ni_sample_rate
    )
    NI_card_2.set_do_voltage(
        do_channel=imaging_AOM_do,
        value=imaging_AOM_pulse,
        sample_rate=ni_sample_rate
    )
    NI_card_2.set_do_voltage(
        do_channel=mot_coils_do,
        value=mot_coils_do_pulse,
        sample_rate=ni_sample_rate
    )

    NI_card_3.build_stack()
    NI_card_3.set_ao_voltage(
        ao_channel=imaging_AOM_ao,
        voltages=imaging_AOM_pulse,
        sample_rate=ni_sample_rate
    )
    NI_card_3.set_ao_voltage(
        ao_channel=mot_coils_ao,
        voltages=mot_coils_ao_pulse,
        sample_rate=ni_sample_rate
    )

    OPX_client.build_stack()
    clock_elem = OPX_client.create_new_do_elem(
        do_channel=opx_trigger_do,
        length=500
    )
    logger.info(f"Experiment Length: {experiment_length_ms}")
    N = experiment_length_ms
    number_of_runs = 10
    M = number_of_runs
    run_buffer = 10_000_000
    with OPX_client.for_("j", 0, M, 1):
        with OPX_client.for_("i", 0, (N + 1) * 10, 1):
            OPX_client.set_digital_voltage(element=clock_elem)
            OPX_client.delay(delay_ns)
        OPX_client.delay(run_buffer)

    logger.info(f"Time after configureing voltages before arm {time.perf_counter_ns()}")

    h1 = NI_card_2.arm(retriggerable=True)
    h2 = NI_card_3.arm(retriggerable=True)
    OPX_client.execute(wait=True)
    NI_card_2.finalize(h1, timeout=120.0)
    NI_card_3.finalize(h2, timeout=120.0)
    NI_card_1.finalize_clock()

    logger.info(f"Time sequence done, image gathering starts {time.perf_counter_ns()}")

    def get_frame(timeout_ms=1000):
        b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
        logger.info(f"{shape}")
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    frame1_ds = dataset.children["Frame 1"]
    frame2_ds = dataset.children["Frame 2"]
    diff_ds = dataset.children["Image Difference"]
    avg = diff_ds.children["Image Differencecurrentavg"]

    try:
        for run_idx in range(M):
            logger.info(f"Requesting frame pair for run {run_idx + 1} of {M}")

            frame1 = get_frame(timeout_ms=10000 if run_idx == 0 else 100000)
            logger.info(
                f"Got frame{2 * run_idx + 1}: shape={frame1.shape}, dtype={frame1.dtype}, "
                f"min={frame1.min()}, max={frame1.max()}"
            )

            frame2 = get_frame(timeout_ms=100000)
            logger.info(
                f"Got frame{2 * run_idx + 2}: shape={frame2.shape}, dtype={frame2.dtype}, "
                f"min={frame2.min()}, max={frame2.max()}"
            )

            diff = frame1.astype(np.int32) - frame2.astype(np.int32)

            logger.info(f"Time images gathered plots starting {time.perf_counter_ns()}")

            frame1_ds.set_data(frame1)
            frame1_ds.update()

            frame2_ds.set_data(frame2)
            frame2_ds.update()

            diff_ds.set_data(diff)
            diff_ds.update()

            frame1_ds.graph.setLevels(*FRAME1_LEVELS)
            frame2_ds.graph.setLevels(*FRAME2_LEVELS)
            diff_ds.graph.setLevels(*DIFF_LEVELS)
            avg.graph.setLevels(*AVG_DIFF_LEVELS)
            avg.graph.ui.histogram.autoHistogramRange = False

            logger.info(f"Updated DataTaker plots for run {run_idx + 1} of {M}")
    finally:
        logger.info("Stopping acquisition")
        dataset.camera_client.stop_acquisition()

    logger.info(f"Plots plotted, cycle over {time.perf_counter_ns()}")

    time.sleep(wait_time)
