import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg  # noqa: F401
import time
from qt_plotting import QtMatplotlibFrameViewer


if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

# -----------------------------
# Experiment script settings
# Same structure as original, but timing values are now in ms.
# -----------------------------

INIT_DICT = {
    'imaging_AOM_start': {'Imaging AOM Start Time (ms)': '1'},
    'imaging_AOM_end': {'Imaging AOM End Time (ms)': '500'},
    'frame_1': {'Camera Frame 1 Time (ms)': '400'},
    'frame_1_scan_start': {'Camera Frame 1 Scan Start (ms)': '-1'},
    'frame_1_scan_stop': {'Camera Frame 1 Scan Stop (ms)': '-1'},
    'frame_1_scan_step': {'Camera Frame 1 Scan Step (ms)': '-1'},
    'frame_2': {'Camera Frame 2 Time (ms)': '1000'},
    'wait_time': {'Wait Time Between Cycles (s)': '3'},
    'camera_trigger_do': {'Camera Trigger DO Channel': '0'},
    'imaging_AOM_do': {'Imaging AOM DO Channel': '1'},
    'imaging_AOM_ao': {'Imaging AOM AO Channel': '1'},
    'opx_trigger_do': {'OPX Trigging DO Channel': '1'},
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
    logger = dataset.log

    #Get deivce clients:
    camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.camera_client = camera_client

    NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_1 = NI_card_1
    NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_2 = NI_card_2
    NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    dataset.NI_card_3 = NI_card_3

    dataset.add_child(
        name="Image Average",
        data_type=Plot2DWithAvg,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    avgview = dataset.children["Image Average"].graph
    avgview.setLevels(-1.5, 1.5)
    dataset.children["Image Average"].children["Image Averagecurrentavg"].graph.setLevels(-1.5, 1.5)

    #Hide the "dataset" graph
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
    frame_1_values = _get_frame_1_values(dataset)
    frame_2 = int(dataset.get_input_parameter("frame_2"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    opx_trigger_do = int(dataset.get_input_parameter("opx_trigger_do"))
    camera_trigger_do = "dio" + str(int(dataset.get_input_parameter("camera_trigger_do")))
    imaging_AOM_do = "dio" + str(int(dataset.get_input_parameter("imaging_AOM_do")))
    imaging_AOM_ao = "ao" + str(int(dataset.get_input_parameter("imaging_AOM_ao")))

    # CHANGED: 1 sample = 1 ms instead of 1 us
    ni_sample_rate = 1000
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"

    # CHANGED: pulse width now expressed in ms
    camera_ttl_up = 1
    if imaging_AOM_end < imaging_AOM_start:
        raise ValueError("imaging_AOM_end must be greater than or equal to imaging_AOM_start")

    for frame_1 in frame_1_values:
        if frame_1 < 0:
            raise ValueError("frame_1 values must be non-negative")
        if frame_2 <= frame_1 + camera_ttl_up:
            raise ValueError("frame_2 must be greater than each frame_1 value + camera_ttl_up")

    imaging_AOM_pulse = [0] * imaging_AOM_start + [1] * (imaging_AOM_end - imaging_AOM_start) + [0] * 100
    buffer = 0
    frame_1_idx = 0

    #Experimental Sequence:
    while thread.running:
        frame_1 = frame_1_values[frame_1_idx % len(frame_1_values)]
        frame_1_idx += 1
        down_time = frame_2 - frame_1 - camera_ttl_up
        camera_trigger_pulse = (
            [0] * frame_1 +
            [1] * camera_ttl_up +
            [0] * down_time +
            [1] * camera_ttl_up
        )

        # Same structure as original, just now in ms
        experiment_lenght_ms = int(len(camera_trigger_pulse) * 1.01)
        logger.info(f"Running sequence with frame_1={frame_1} ms and frame_2={frame_2} ms")

        camera_client.set_hardware_trigger(
            line=trigger_line,
            activation=trigger_edge,
            selector="FrameStart",
            overlap="ReadOut",
            acquisition_mode="Continuous",
        )
        camera_client.set_exposure(7)
        camera_client.try_set_float("Gain", 10.0)
        logger.info("Starting acquisition")
        dataset.camera_client.start_acquisition()

        #NI card 1 needs to be used to get the clock from OPX
        NI_card_1.arm_clock(length=experiment_lenght_ms + buffer, sample_rate=ni_sample_rate)
        logger.info("Clock configured")

        NI_card_2.build_stack()
        #Camera trigger pulse
        NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=camera_trigger_pulse, sample_rate=ni_sample_rate)
        #Imaging AOM pulse
        NI_card_2.set_do_voltage(do_channel=imaging_AOM_do, value=imaging_AOM_pulse, sample_rate=ni_sample_rate)

        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel=imaging_AOM_ao, voltages=imaging_AOM_pulse, sample_rate=ni_sample_rate)

        #OPX sends digital pulses to NI to set NI's clock
        OPX_client.build_stack()

        clock_elem = OPX_client.create_new_do_elem(
            do_channel=opx_trigger_do,
            length=500
        )
        N = experiment_lenght_ms
        with OPX_client.for_("i", 0, N, 1):
            OPX_client.set_digital_voltage(
                element=clock_elem
            )
            OPX_client.delay(999500)

        #Starts NI and then OPX
        h1 = NI_card_2.arm()
        h2 = NI_card_3.arm()
        OPX_client.execute()
        NI_card_2.finalize(h1, timeout=120.0)
        NI_card_3.finalize(h2, timeout=120.0)
        NI_card_1.finalize_clock()

        def get_frame(timeout_ms=1000):
            # get_frame_bytes() should return: (bytes, shape, dtype_str)
            b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
            logger.info(f"{shape}")
            return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        # Always stop acquisition even if something fails
        try:
            logger.info("Requesting one frame")
            frame1 = get_frame(timeout_ms=10000)
            frame2 = get_frame(timeout_ms=100000)
            logger.info(f"Got frame: shape={frame1.shape}, dtype={frame1.dtype}, min={frame1.min()}, max={frame1.max()}")
            logger.info(f"Got frame: shape={frame2.shape}, dtype={frame2.dtype}, min={frame2.min()}, max={frame2.max()}")

        finally:
            logger.info("Stopping acquisition")
            dataset.camera_client.stop_acquisition()

        img_ds = dataset.children["Image Average"]

        # Averaged child plot
        avg = img_ds.children["Image Averagecurrentavg"]

        # Data + update
        diff = frame2.astype(np.int32) - frame1.astype(np.int32)
        img_ds.set_data(diff)
        img_ds.update()        # updates both current + avg children

        # Lock color scale on AVG image
        img_ds.graph.setLevels(-1.5, 1.5)

        avg.graph.setLevels(-1.5, 1.5)
        avg.graph.ui.histogram.autoHistogramRange = False
        time.sleep(wait_time)

        # thread.running = False
