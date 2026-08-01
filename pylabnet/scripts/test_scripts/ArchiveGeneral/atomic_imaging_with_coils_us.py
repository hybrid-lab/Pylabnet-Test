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
SAMPLE_PERIOD_US = 10
NI_SAMPLE_RATE = 100000
OPX_CLOCK_DELAY_NS = 9500

# -----------------------------
# Experiment script settings
# Timing values are in us.
# -----------------------------

INIT_DICT = {
    'imaging_AOM_start': {'Imaging AOM Start Time (us)': '0'},
    'imaging_AOM_end': {'Imaging AOM End Time (us)': '500000'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '100'},

    # MOT coils:
    # coils are OFF before mot_coils_start
    # coils are ON from mot_coils_start to mot_coils_end
    # coils are OFF after mot_coils_end
    'mot_coils_start': {'MOT Coils ON Start Time (us)': '0'},
    'mot_coils_end': {'MOT Coils ON End Time (us)': '500000'},

    # To not use scan set all scan parameters to -1.
    'frame_1': {'Camera Frame 1 Time (us)': '650000'},
    'frame_1_scan_start': {'Camera Frame 1 Scan Start (us)': '500000'},
    'frame_1_scan_stop': {'Camera Frame 1 Scan Stop (us)': '503000'},
    'frame_1_scan_step': {'Camera Frame 1 Scan Step (us)': '100'},
    'frame_2': {'Camera Frame 2 Time (us)': '600000'},
    'wait_time': {'Wait Time Between Cycles (s)': '0.2'},

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


def _us_to_samples(time_us, name):
    """Converts a time in us to samples on the 10 us timing grid."""
    if time_us % SAMPLE_PERIOD_US != 0:
        raise ValueError(f"{name} must be a multiple of {SAMPLE_PERIOD_US} us")
    return time_us // SAMPLE_PERIOD_US


def configure(**kwargs):
    """Sets up the hardware and the plot before the experiment runs."""
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")

    # Get device clients
    camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
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
    dataset.camera_client = kwargs["fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber"]
    dataset.OPX_client = kwargs["OPX_OPX"]

    OPX_client = dataset.OPX_client
    NI_card_1 = dataset.NI_card_1
    NI_card_2 = dataset.NI_card_2
    NI_card_3 = dataset.NI_card_3
    camera_client = dataset.camera_client

    imaging_AOM_start = int(dataset.get_input_parameter("imaging_AOM_start"))
    imaging_AOM_end = int(dataset.get_input_parameter("imaging_AOM_end"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))

    mot_coils_start = int(dataset.get_input_parameter("mot_coils_start"))
    mot_coils_end = int(dataset.get_input_parameter("mot_coils_end"))

    frame_1_values = _get_frame_1_values(dataset)
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

    # 1 sample = 10 us
    ni_sample_rate = NI_SAMPLE_RATE
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"
    camera_ttl_up_us = 10
    camera_ttl_up = _us_to_samples(camera_ttl_up_us, "camera_ttl_up_us")
    aom_pulse_width = _us_to_samples(camera_exposure_us, "camera_exposure_us")

    if camera_exposure_us <= 0:
        raise ValueError("camera_exposure_us must be greater than 0")
    if imaging_AOM_end < imaging_AOM_start:
        raise ValueError("imaging_AOM_end must be >= imaging_AOM_start")

    if mot_coils_end < mot_coils_start:
        raise ValueError("mot_coils_end must be >= mot_coils_start")

    for frame_1 in frame_1_values:
        if frame_1 < 0:
            raise ValueError("frame_1 values must be non-negative")
        _us_to_samples(frame_1, "frame_1")
        if frame_2 <= frame_1 + camera_ttl_up_us:
            raise ValueError("frame_2 must be greater than each frame_1 value + camera_ttl_up_us")

    _us_to_samples(imaging_AOM_start, "imaging_AOM_start")
    _us_to_samples(imaging_AOM_end, "imaging_AOM_end")
    _us_to_samples(camera_exposure_us, "camera_exposure_us")
    _us_to_samples(mot_coils_start, "mot_coils_start")
    _us_to_samples(mot_coils_end, "mot_coils_end")
    _us_to_samples(frame_2, "frame_2")

    buffer = 0
    frame_1_idx = 0

    while thread.running:
        frame_1 = frame_1_values[frame_1_idx % len(frame_1_values)]
        frame_1_idx += 1

        imaging_AOM_start_samples = _us_to_samples(imaging_AOM_start, "imaging_AOM_start")
        imaging_AOM_end_samples = _us_to_samples(imaging_AOM_end, "imaging_AOM_end")
        mot_coils_start_samples = _us_to_samples(mot_coils_start, "mot_coils_start")
        mot_coils_end_samples = _us_to_samples(mot_coils_end, "mot_coils_end")
        frame_1_samples = _us_to_samples(frame_1, "frame_1")
        frame_2_samples = _us_to_samples(frame_2, "frame_2")
        tail_samples = _us_to_samples(100, "sequence tail")

        # Total sequence length must cover everything
        sequence_end_samples = max(
            frame_2_samples + max(camera_ttl_up, aom_pulse_width) + tail_samples,
            imaging_AOM_end_samples + tail_samples,
            mot_coils_end_samples + tail_samples
        )

        # Camera trigger pulse: two triggers, one for each frame
        down_time = frame_2_samples - frame_1_samples - camera_ttl_up
        camera_trigger_pulse = (
            [0] * frame_1_samples +
            [1] * camera_ttl_up +
            [0] * down_time +
            [1] * camera_ttl_up +
            [0] * max(0, sequence_end_samples - frame_2_samples - camera_ttl_up)
        )

        # Imaging AOM pulses:
        # keep the initial AOM window and add one pulse for each exposure.
        imaging_AOM_pulse = [0] * sequence_end_samples
        for pulse_idx in range(
            imaging_AOM_start_samples,
            min(imaging_AOM_end_samples, sequence_end_samples)
        ):
            imaging_AOM_pulse[pulse_idx] = 1
        for frame_time in (frame_1_samples, frame_2_samples):
            for offset in range(aom_pulse_width):
                pulse_idx = frame_time + offset
                if pulse_idx < sequence_end_samples:
                    imaging_AOM_pulse[pulse_idx] = 1

        # MOT coils logic:
        # TTL HIGH = coils OFF
        # TTL LOW  = coils ON
        mot_coils_do_pulse = (
            [1] * mot_coils_start_samples +
            [0] * max(0, mot_coils_end_samples - mot_coils_start_samples) +
            [1] * max(0, sequence_end_samples - mot_coils_end_samples)
        )

        mot_coils_ao_pulse = (
            [mot_coils_on_voltage] * mot_coils_start_samples +
            [mot_coils_on_voltage] * max(0, mot_coils_end_samples - mot_coils_start_samples) +
            [mot_coils_on_voltage] * max(0, sequence_end_samples - mot_coils_end_samples)
        )

        experiment_length_samples = int(sequence_end_samples * 1.01)
        logger.info(
            f"Running sequence with frame_1={frame_1} us, "
            f"frame_2={frame_2} us, exposure={camera_exposure_us} us, "
            f"sample_period={SAMPLE_PERIOD_US} us"
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
        NI_card_1.arm_clock(length=experiment_length_samples + buffer, sample_rate=ni_sample_rate)
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

        N = experiment_length_samples
        with OPX_client.for_("i", 0, N, 1):
            OPX_client.set_digital_voltage(element=clock_elem)
            OPX_client.delay(OPX_CLOCK_DELAY_NS)

        # Start NI and then OPX
        h1 = NI_card_2.arm()
        h2 = NI_card_3.arm()
        OPX_client.execute()
        NI_card_2.finalize(h1, timeout=120.0)
        NI_card_3.finalize(h2, timeout=120.0)
        NI_card_1.finalize_clock()

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

        time.sleep(wait_time)

        # thread.running = False
