import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2D  # noqa: F401
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
FRAME3_LEVELS = (0, 20)
FRAME4_LEVELS = (0, 20)
FRAME5_LEVELS = (0, 20)

RAW_FRAME_NAMES = [f"Frame {i}" for i in range(1, 6)]
VIEW_X_RANGE = (-4000, 6000)
VIEW_Y_RANGE = (0, 2048)

# -----------------------------
# Experiment script settings
# Timing values are in ms.
# -----------------------------

INIT_DICT = {
    'imaging_AOM_start': {'Imaging AOM Start Time (ms)': '1'},
    'imaging_AOM_end': {'Imaging AOM End Time (ms)': '700'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '2000'},

    # MOT coils:
    # coils are OFF before mot_coils_start
    # coils are ON from mot_coils_start to mot_coils_end
    # coils are OFF after mot_coils_end
    'mot_coils_start': {'MOT Coils ON Start Time (ms)': '0'},
    'mot_coils_end': {'MOT Coils ON End Time (ms)': '700'},

    'frame_1': {'Camera Frame 1 Time (ms)': '650'},
    'frame_2': {'Camera Frame 2 Time (ms)': '710'},
    'frame_3': {'Camera Frame 3 Time (ms)': '720'},
    'frame_4': {'Camera Frame 4 Time (ms)': '730'},
    'frame_5': {'Camera Frame 5 Time (ms)': '900'},
    'wait_time': {'Wait Time Between Cycles (s)': '0.1'},

    'camera_trigger_do': {'Camera Trigger DO Channel': '0'},
    'imaging_AOM_do': {'Imaging AOM DO Channel': '1'},
    'imaging_AOM_ao': {'Imaging AOM AO Channel': '1'},

    # MOT coils controls
    'mot_coils_do': {'MOT Coils DO Channel': '2'},
    'mot_coils_ao': {'MOT Coils AO Channel': '2'},
    'mot_coils_on_voltage': {'MOT Coils ON Analog Voltage (V)': '9'},
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

    # Get device clients
    camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m']
    dataset.camera_client = camera_client

    NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_1 = NI_card_1
    NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_2 = NI_card_2
    NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    dataset.NI_card_3 = NI_card_3

    for name in RAW_FRAME_NAMES:
        dataset.add_child(
            name=name,
            data_type=Plot2D,
            min_x=0, max_x=2448, pts_x=2448,
            min_y=0, max_y=2048, pts_y=2048,
            new_plot=True
        )

    dataset.children["Frame 1"].graph.setLevels(*FRAME1_LEVELS)
    dataset.children["Frame 2"].graph.setLevels(*FRAME2_LEVELS)
    dataset.children["Frame 3"].graph.setLevels(*FRAME3_LEVELS)
    dataset.children["Frame 4"].graph.setLevels(*FRAME4_LEVELS)
    dataset.children["Frame 5"].graph.setLevels(*FRAME5_LEVELS)

    for name in RAW_FRAME_NAMES:
        dataset.children[name].graph.view.setAspectLocked(False)
        dataset.children[name].graph.view.setLimits(
            xMin=VIEW_X_RANGE[0],
            xMax=VIEW_X_RANGE[1],
            yMin=VIEW_Y_RANGE[0],
            yMax=VIEW_Y_RANGE[1]
        )
        dataset.children[name].graph.view.setRange(
            xRange=VIEW_X_RANGE,
            yRange=VIEW_Y_RANGE,
            padding=0
        )

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
    imaging_AOM_end = int(dataset.get_input_parameter("imaging_AOM_end"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))

    mot_coils_start = int(dataset.get_input_parameter("mot_coils_start"))
    mot_coils_end = int(dataset.get_input_parameter("mot_coils_end"))

    frame_times = [
        int(dataset.get_input_parameter("frame_1")),
        int(dataset.get_input_parameter("frame_2")),
        int(dataset.get_input_parameter("frame_3")),
        int(dataset.get_input_parameter("frame_4")),
        int(dataset.get_input_parameter("frame_5")),
    ]
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
    camera_ttl_up = 1
    forced_aom_pulse_ms = max(1, int(np.ceil(camera_exposure_us / 1000.0)))

    if imaging_AOM_end < imaging_AOM_start:
        raise ValueError("imaging_AOM_end must be >= imaging_AOM_start")

    if mot_coils_end < mot_coils_start:
        raise ValueError("mot_coils_end must be >= mot_coils_start")

    for i in range(len(frame_times) - 1):
        if frame_times[i + 1] <= frame_times[i]:
            raise ValueError("Each frame time must be greater than the previous one")

    # Total sequence length must cover everything
    sequence_end_ms = max(
        frame_times[-1] + max(camera_ttl_up, forced_aom_pulse_ms) + 100,
        imaging_AOM_end + 100,
        mot_coils_end + 100
    )

    # Camera trigger pulse: five triggers, one for each frame
    camera_trigger_pulse = [0] * sequence_end_ms
    for frame_time in frame_times:
        for offset in range(camera_ttl_up):
            pulse_idx = frame_time + offset
            if pulse_idx < sequence_end_ms:
                camera_trigger_pulse[pulse_idx] = 1

    # Imaging AOM digital + analog pulses:
    # keep the user-defined AOM window, and also ensure the AOM is on
    # for the camera exposure duration whenever a camera frame is triggered.
    imaging_AOM_pulse = [0] * sequence_end_ms
    for pulse_idx in range(imaging_AOM_start, min(imaging_AOM_end, sequence_end_ms)):
        imaging_AOM_pulse[pulse_idx] = 1

    for frame_idx, frame_time in enumerate(frame_times, start=1):
        if frame_idx == 3:
            continue
        for offset in range(forced_aom_pulse_ms):
            pulse_idx = frame_time + offset
            if pulse_idx < sequence_end_ms:
                imaging_AOM_pulse[pulse_idx] = 1

    # MOT coils logic:
    # TTL HIGH = coils OFF
    # TTL LOW  = coils ON
    #
    # Desired behavior:
    # before mot_coils_start -> OFF
    # mot_coils_start to mot_coils_end -> ON
    # after mot_coils_end -> OFF
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

    experiment_length_ms = int(sequence_end_ms * 1.01)
    buffer = 0

    while thread.running:
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

        N = experiment_length_ms
        with OPX_client.for_("i", 0, N, 1):
            OPX_client.set_digital_voltage(element=clock_elem)
            OPX_client.delay(999500)

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
            frames = []
            for i in range(5):
                logger.info(f"Requesting frame {i + 1}")
                timeout_ms = 10000 if i == 0 else 100000
                frame = get_frame(timeout_ms=timeout_ms)
                frames.append(frame)
                logger.info(
                    f"Got frame{i + 1}: shape={frame.shape}, dtype={frame.dtype}, "
                    f"min={frame.min()}, max={frame.max()}"
                )
        finally:
            logger.info("Stopping acquisition")
            dataset.camera_client.stop_acquisition()

        for i, frame in enumerate(frames, start=1):
            frame_ds = dataset.children[f"Frame {i}"]
            frame_ds.set_data(frame)
            frame_ds.update()

        dataset.children["Frame 1"].graph.setLevels(*FRAME1_LEVELS)
        dataset.children["Frame 2"].graph.setLevels(*FRAME2_LEVELS)
        dataset.children["Frame 3"].graph.setLevels(*FRAME3_LEVELS)
        dataset.children["Frame 4"].graph.setLevels(*FRAME4_LEVELS)
        dataset.children["Frame 5"].graph.setLevels(*FRAME5_LEVELS)

        time.sleep(wait_time)

        # thread.running = False
