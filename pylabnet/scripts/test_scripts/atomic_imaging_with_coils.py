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
    'camera_exposure_us': {'Camera Exposure Time (us)': '1000'},

    # MOT coils:
    # coils are OFF before mot_coils_start
    # coils are ON from mot_coils_start to mot_coils_end
    # coils are OFF after mot_coils_end
    'mot_coils_start': {'MOT Coils ON Start Time (ms)': '0'},

    #To not use scan set all scan paramters to -1

    'frame_1': {'Camera Frame 1 Time (ms)': '20'},
    'frame_1_scan_start': {'Camera Frame 1 Scan Start (ms)': '-1'},
    'frame_1_scan_stop': {'Camera Frame 1 Scan Stop (ms)': '-1'},
    'frame_1_scan_step': {'Camera Frame 1 Scan Step (ms)': '-1'},
    'frame_2_delay_ms': {'Frame 2 Delay After AOM/Coils Off (ms)': '5'},
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


def _get_frame_1_values(dataset):
    """Returns the list of frame_1 times to scan through."""
    scan_start = int(dataset.get_input_parameter("frame_1_scan_start"))
    scan_stop = int(dataset.get_input_parameter("frame_1_scan_stop"))
    scan_step = int(dataset.get_input_parameter("frame_1_scan_step"))

    if scan_start >= 0 or scan_stop >= 0 or scan_step >= 0:
        if scan_start < 0 or scan_stop < 0 or scan_step <= 0:
            raise ValueError("frame_1 scan requires non-negative start/stop and a positive step")
        if scan_stop < scan_start:
            raise ValueError("frame_1_scan_stop must be greater than or equal to frame_1_scan_start")

        return list(range(scan_start, scan_stop + 1, scan_step))

    return [int(dataset.get_input_parameter("frame_1"))]


def configure(**kwargs):
    """Sets up the hardware and the plot before the experiment runs."""
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")

    # Get device clients
    camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m']
    dataset.camera_client = camera_client

    NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_1 = NI_card_1
    NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_2 = NI_card_2
    NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    dataset.NI_card_3 = NI_card_3

    # First camera image
    dataset.add_child(
        name="Frame 1",
        data_type=Plot2D,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    # Second camera image
    dataset.add_child(
        name="Frame 2",
        data_type=Plot2D,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    # Difference image + running average
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

    # Hide the parent dataset graph
    dataset.graph.hide()


def experiment(**kwargs):
    """Main experiment entrypoint called by DataTaker."""
    thread = kwargs['thread']
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
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))

    mot_coils_start = int(dataset.get_input_parameter("mot_coils_start"))

    frame_1_values = _get_frame_1_values(dataset)
    frame_2_delay_ms = int(dataset.get_input_parameter("frame_2_delay_ms"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    opx_trigger_do = int(dataset.get_input_parameter("opx_trigger_do"))
    camera_trigger_do = "dio" + str(int(dataset.get_input_parameter("camera_trigger_do")))
    imaging_AOM_do = "dio" + str(int(dataset.get_input_parameter("imaging_AOM_do")))
    imaging_AOM_ao = "ao" + str(int(dataset.get_input_parameter("imaging_AOM_ao")))

    mot_coils_do = "dio" + str(int(dataset.get_input_parameter("mot_coils_do")))
    mot_coils_ao = "ao" + str(int(dataset.get_input_parameter("mot_coils_ao")))
    mot_coils_on_voltage = float(dataset.get_input_parameter("mot_coils_on_voltage"))
    mot_coils_off_voltage = float(dataset.get_input_parameter("mot_coils_off_voltage"))

    # 1 sample = 1 ms
    ni_sample_rate = 1000
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"
    opx_ttl_pulse_ns = 500
    sample_period_ns = int(round(1e9 / ni_sample_rate))
    delay_ns = sample_period_ns - opx_ttl_pulse_ns
    camera_ttl_up = 1
    aom_pulse_width_ms = max(1, int(np.ceil(camera_exposure_us / 1000.0)))

    if camera_exposure_us <= 0:
        raise ValueError("camera_exposure_us must be greater than 0")
    if frame_2_delay_ms < 0:
        raise ValueError("frame_2_delay_ms must be non-negative")
    if delay_ns < 0:
        raise ValueError("ni_sample_rate is too high for a 500 ns OPX TTL pulse")

    for frame_1 in frame_1_values:
        if frame_1 < 0:
            raise ValueError("frame_1 values must be non-negative")

    buffer = 0
    frame_1_idx = 0

    while thread.running:
        logger.info(f"Time at start {time.perf_counter_ns()}")
        frame_1 = frame_1_values[frame_1_idx % len(frame_1_values)]
        frame_1_idx += 1
        imaging_AOM_end = frame_1 + aom_pulse_width_ms
        mot_coils_end = frame_1 + aom_pulse_width_ms
        frame_2 = mot_coils_end + frame_2_delay_ms

        if imaging_AOM_end < imaging_AOM_start:
            raise ValueError("imaging_AOM_start must be earlier than or equal to frame_1 + exposure")
        if mot_coils_end < mot_coils_start:
            raise ValueError("mot_coils_start must be earlier than or equal to frame_1 + exposure")

        # Total sequence length must cover everything
        sequence_end_ms = max(
            frame_2 + max(camera_ttl_up, aom_pulse_width_ms),
            imaging_AOM_end,
            mot_coils_end
        )

        # Camera trigger pulse: two triggers, one for each frame
        down_time = frame_2 - frame_1 - camera_ttl_up
        camera_trigger_pulse = (
            [0] * frame_1 +
            [1] * camera_ttl_up +
            [0] * down_time +
            [1] * camera_ttl_up +
            [0] * max(0, sequence_end_ms - frame_2 - camera_ttl_up)
        )

        # Imaging AOM pulses:
        # keep the initial AOM window and add one pulse for each exposure.
        imaging_AOM_pulse = [0] * sequence_end_ms
        for pulse_idx in range(imaging_AOM_start, min(imaging_AOM_end, sequence_end_ms)):
            imaging_AOM_pulse[pulse_idx] = 1
        for frame_time in (frame_1, frame_2):
            for offset in range(aom_pulse_width_ms):
                pulse_idx = frame_time + offset
                if pulse_idx < sequence_end_ms:
                    imaging_AOM_pulse[pulse_idx] = 1

        # MOT coils logic:
        # TTL HIGH = coils OFF
        # TTL LOW  = coils ON
        mot_coils_do_pulse = (
            [1] * mot_coils_start +
            [0] * max(0, mot_coils_end - mot_coils_start) +
            [1] * max(0, sequence_end_ms - mot_coils_end)
        )

        mot_coils_ao_pulse = (
            [mot_coils_on_voltage] * mot_coils_start +
            [mot_coils_on_voltage] * max(0, mot_coils_end - mot_coils_start) +
            [mot_coils_on_voltage] * max(0, sequence_end_ms - mot_coils_end)
        )

        experiment_length_ms = int(sequence_end_ms * 1)
        logger.info(
            f"Running sequence with frame_1={frame_1} ms, "
            f"frame_2={frame_2} ms, exposure={camera_exposure_us} us, "
            f"aom_off={imaging_AOM_end} ms, coils_off={mot_coils_end} ms, "
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

        # NI card 1 used to get the clock from OPX
        NI_card_1.arm_clock(length=experiment_length_ms + buffer, sample_rate=ni_sample_rate)
        logger.info("Clock configured")

        # NI digital outputs
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

        # NI analog outputs
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

        # OPX sends digital pulses to NI to set NI clock
        OPX_client.build_stack()
        clock_elem = OPX_client.create_new_do_elem(
            do_channel=opx_trigger_do,
            length=500
        )
        logger.info(f"Experiment Length: {experiment_length_ms}")
        N = experiment_length_ms
        with OPX_client.for_("i", 0, N, 1):
            OPX_client.set_digital_voltage(element=clock_elem)
            OPX_client.delay(delay_ns)

        logger.info(f"Time after configureing voltages before arm {time.perf_counter_ns()}")

        # Start NI and then OPX
        h1 = NI_card_2.arm() #NI gets ready for OPX
        h2 = NI_card_3.arm()
        OPX_client.execute() #OPX starts now
        NI_card_2.finalize(h1, timeout=120.0)
        NI_card_3.finalize(h2, timeout=120.0)
        NI_card_1.finalize_clock()

        logger.info(f"Time sequence done, image gathering starts {time.perf_counter_ns()}")

        def get_frame(timeout_ms=1000):
            b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
            logger.info(f"{shape}")
            return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        try:
            logger.info("Requesting one frame")
            frame1 = get_frame(timeout_ms=10000)
            frame2 = get_frame(timeout_ms=100000)

            logger.info(
                f"Got frame1: shape={frame1.shape}, dtype={frame1.dtype}, "
                f"min={frame1.min()}, max={frame1.max()}"
            )
            logger.info(
                f"Got frame2: shape={frame2.shape}, dtype={frame2.dtype}, "
                f"min={frame2.min()}, max={frame2.max()}"
            )
        finally:
            logger.info("Stopping acquisition")
            dataset.camera_client.stop_acquisition()

        diff = frame1.astype(np.int32) - frame2.astype(np.int32)

        logger.info(f"Time images gathered plots starting {time.perf_counter_ns()}")

        # Update plots
        frame1_ds = dataset.children["Frame 1"]
        frame2_ds = dataset.children["Frame 2"]
        diff_ds = dataset.children["Image Difference"]

        frame1_ds.set_data(frame1)
        frame1_ds.update()

        frame2_ds.set_data(frame2)
        frame2_ds.update()

        diff_ds.set_data(diff)
        diff_ds.update()

        frame1_ds.graph.setLevels(*FRAME1_LEVELS)
        frame2_ds.graph.setLevels(*FRAME2_LEVELS)
        diff_ds.graph.setLevels(*DIFF_LEVELS)

        avg = diff_ds.children["Image Differencecurrentavg"]
        avg.graph.setLevels(*AVG_DIFF_LEVELS)
        avg.graph.ui.histogram.autoHistogramRange = False

        logger.info(f"Plots plotted, cycle over {time.perf_counter_ns()}")

        time.sleep(wait_time)

        # thread.running = False
