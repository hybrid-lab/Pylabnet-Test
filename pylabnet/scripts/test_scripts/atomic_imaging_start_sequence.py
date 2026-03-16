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
# -----------------------------

INIT_DICT = {
    'imaging_AOM_start': {'Imaging AOM Start Time (us)': '1000'},
    'imaging_AOM_end': {'Imaging AOM End Time (us)': '2000'},
    'frame_1': {'Camera Frame 1 Time (us)': '1000'},
    'frame_2': {'Camera Frame 2 Time (us)': '1000'},
    'wait_time': {'Wait Time Between Cycles (s)': '3'},
    'camera_trigger_do': {'Camera Trigger DO Channel': '0'},
    'imaging_AOM_do': {'Imaging AOM DO Channel': '0'},
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

    #Get deivce clients:
    camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m']
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
    avgview.setLevels(-255, 255)
    dataset.children["Image Average"].children["Image Averagecurrentavg"].graph.setLevels(0, 300)

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
    dataset.camera_client = kwargs["fluorescence_imaging_camera_bfs_u3_51s5m"]

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
    wait_time = int(dataset.get_input_parameter("wait_time"))

    opx_trigger_do = int(dataset.get_input_parameter("opx_trigger_do"))
    camera_trigger_do = "dio" + str(int(dataset.get_input_parameter("camera_trigger_do")))
    imaging_AOM_do = "dio" + str(int(dataset.get_input_parameter("imaging_AOM_do")))

    ni_sample_rate = 1000000
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"

    #Defines how long OPX will be producing a clock signal
    experiment_lenght_us = 10000

    camera_ttl_up = 50
    camera_trigger_pulse_1 = [0] * frame_1 + [1] * camera_ttl_up
    camera_trigger_pulse_2 = [0] * frame_2 + [1] * camera_ttl_up
    imaging_AOM_pulse = [0] * imaging_AOM_start + [1] * (imaging_AOM_end - imaging_AOM_start)

    #Experimental Sequence:
    while thread.running:
        camera_client.set_hardware_trigger(
            line=trigger_line,
            activation=trigger_edge,
            selector="FrameStart",
            overlap="ReadOut",
            acquisition_mode="Continuous",
        )
        logger.info("Starting acquisition")
        dataset.camera_client.start_acquisition()

        NI_card_2.build_stack()
        #Camera trigger 1
        NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=camera_trigger_pulse_1, sample_rate=ni_sample_rate)
        #Camera trigger 2
        NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=camera_trigger_pulse_2, sample_rate=ni_sample_rate)
        #Imaging AOM pulse
        NI_card_2.set_do_voltage(do_channel=imaging_AOM_do, value=imaging_AOM_pulse, sample_rate=ni_sample_rate)

        #OPX sends digital pulses to NI to set NI's clock
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

        #Starts NI and then OPX
        h1 = NI_card_2.arm()
        OPX_client.execute()
        NI_card_2.finalize(h1, timeout=120.0)

        def get_frame(timeout_ms=1000):
            # get_frame_bytes() should return: (bytes, shape, dtype_str)
            b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
            logger.info(f"{shape}")
            return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        # Always stop acquisition even if something fails
        try:
            logger.info("Requesting one frame")
            frame_1 = get_frame(timeout_ms=1000)
            frame_2 = get_frame(timeout_ms=1000)
            logger.info(f"Got frame: shape={frame_1.shape}, dtype={frame_1.dtype}, min={frame_1.min()}, max={frame_1.max()}")
            logger.info(f"Got frame: shape={frame_2.shape}, dtype={frame_2.dtype}, min={frame_2.min()}, max={frame_2.max()}")

        finally:
            logger.info("Stopping acquisition")
            dataset.camera_client.stop_acquisition()

        img_ds = dataset.children["Image Average"]

        # Averaged child plot
        avg = img_ds.children["Image Averagecurrentavg"]

        # Data + update
        diff = frame_2.astype(np.int32) - frame_1.astype(np.int32)
        img_ds.set_data(diff)
        img_ds.update()        # updates both current + avg children

        # Lock color scale on AVG image
        avg.graph.setLevels(-255, 255)
        avg.graph.ui.histogram.autoHistogramRange = False
        time.sleep(wait_time)
