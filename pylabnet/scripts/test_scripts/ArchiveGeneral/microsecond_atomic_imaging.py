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
# Timing values are in microseconds.
# -----------------------------

INIT_DICT = {
    'imaging_AOM_start': {'Imaging AOM Start Time (us)': '1'},
    'imaging_AOM_end': {'Imaging AOM End Time (us)': '200000'},
    'frame_1': {'Camera Frame 1 Time (us)': '100000'},
    'frame_2': {'Camera Frame 2 Time (us)': '200000'},
    'wait_time': {'Wait Time Between Cycles (s)': '3'},
    'camera_trigger_do': {'Camera Trigger DO Channel': '0'},
    'imaging_AOM_do': {'Imaging AOM DO Channel': '1'},
    'imaging_AOM_ao': {'Imaging AOM AO Channel': '1'},
    'opx_trigger_do': {'OPX Trigging DO Channel': '1'},
}


def define_dataset():
    """Specifies the type of plot to use for the data."""
    return 'Dataset'


def configure(**kwargs):
    """Sets up the hardware and the plot before the experiment runs."""
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")
    logger = dataset.log

    # Get device clients:
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
    frame_1 = int(dataset.get_input_parameter("frame_1"))
    frame_2 = int(dataset.get_input_parameter("frame_2"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    opx_trigger_do = int(dataset.get_input_parameter("opx_trigger_do"))
    camera_trigger_do = "dio" + str(int(dataset.get_input_parameter("camera_trigger_do")))
    imaging_AOM_do = "dio" + str(int(dataset.get_input_parameter("imaging_AOM_do")))
    imaging_AOM_ao = "ao" + str(int(dataset.get_input_parameter("imaging_AOM_ao")))

    # 1 sample = 1 us
    ni_sample_rate = 1000000
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"

    # Camera trigger width in us
    camera_ttl_up = 50

    if frame_2 < frame_1 + camera_ttl_up:
        raise ValueError("frame_2 must be at least frame_1 + camera_ttl_up")

    if imaging_AOM_end < imaging_AOM_start:
        raise ValueError("imaging_AOM_end must be >= imaging_AOM_start")

    down_time = frame_2 - frame_1 - camera_ttl_up
    camera_trigger_pulse = [0] * frame_1 + [1] * camera_ttl_up + [0] * down_time + [1] * camera_ttl_up
    imaging_AOM_pulse = [0] * imaging_AOM_start + [1] * (imaging_AOM_end - imaging_AOM_start) + [0] * 100

    experiment_lenght_us = max(
        int(len(camera_trigger_pulse) * 1.01),
        int(len(imaging_AOM_pulse) * 1.01)
    )
    buffer = 0

    while thread.running:
        camera_client.set_hardware_trigger(
            line=trigger_line,
            activation=trigger_edge,
            selector="FrameStart",
            overlap="ReadOut",
            acquisition_mode="Continuous",
        )
        camera_client.set_exposure(500)
        camera_client.try_set_float("Gain", 50.0)
        logger.info("Starting acquisition")
        dataset.camera_client.start_acquisition()

        # NI card 1 needs to be used to get the clock from OPX
        NI_card_1.arm_clock(length=experiment_lenght_us + buffer, sample_rate=ni_sample_rate)
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

        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(
            ao_channel=imaging_AOM_ao,
            voltages=imaging_AOM_pulse,
            sample_rate=ni_sample_rate
        )

        # OPX sends digital pulses to NI to set NI's clock
        OPX_client.build_stack()

        clock_elem = OPX_client.create_new_do_elem(
            do_channel=opx_trigger_do,
            length=500
        )
        N = experiment_lenght_us
        with OPX_client.for_("i", 0, N, 1):
            OPX_client.set_digital_voltage(
                element=clock_elem
            )
            OPX_client.delay(500)

        # Starts NI and then OPX
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

        # Update individual frame plots
        frame1_ds = dataset.children["Frame 1"]
        frame2_ds = dataset.children["Frame 2"]
        diff_ds = dataset.children["Image Difference"]

        frame1_ds.set_data(frame1)
        frame1_ds.update()

        frame2_ds.set_data(frame2)
        frame2_ds.update()

        diff_ds.set_data(diff)
        diff_ds.update()

        # Keep plot ranges fixed
        diff_ds.graph.setLevels(*DIFF_LEVELS)
        frame1_ds.graph.setLevels(*FRAME1_LEVELS)
        frame2_ds.graph.setLevels(*FRAME2_LEVELS)

        avg = diff_ds.children["Image Differencecurrentavg"]
        avg.graph.setLevels(*AVG_DIFF_LEVELS)
        avg.graph.ui.histogram.autoHistogramRange = False

        time.sleep(wait_time)

        # thread.running = False
