import numpy as np

# pylabnet imports (kept to match pylabnet script expectations)
from pylabnet.scripts.data_center.take_data import ExperimentThread  # noqa: F401
from pylabnet.scripts.data_center.datasets import Dataset, Plot2D  # noqa: F401


if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

INIT_DICT = {
    "timeout_ms": {"GetNextImage timeout (ms)": "25000"},
    "blank1": {"filler": "0"},
    "blank2": {"filler": "0"},
    "blank3": {"filler": "0"},
    "blank4": {"filler": "0"},
}


def define_dataset():
    """Specifies the type of plot to use for the data."""
    return "Dataset"


def configure(**kwargs):
    """Sets up the hardware and the plot before the experiment runs."""
    dataset = kwargs["dataset"]
    logger = dataset.log

    camera_client = kwargs["quest_camera_orca_quest"]

    dataset.camera_client = camera_client
    ni = kwargs["nidaqmx_ni_daq_2"]
    dataset.ni = ni
    logger.info("Orca Quest camera client attached to dataset.")

    # Single image plot (no averaging)
    dataset.add_child(
        name="Image",
        data_type=Plot2D,
        min_x=0, max_x=4096, pts_x=4096,
        min_y=0, max_y=2304, pts_y=2304,
        new_plot=True,
    )

    imgview = dataset.children["Image"].graph  # pg.ImageView
    imgview.setLevels(0, 300)

    dataset.graph.hide()


def experiment(**kwargs):
    """Main experiment entrypoint called by DataTaker."""
    thread = kwargs["thread"]   # noqa: F841
    dataset = kwargs["dataset"]
    logger = dataset.log

    do_channel = "dio1"
    sample_rate = 100000
    low1_s = 0.1
    high_s = 0.010
    low2_s = 0.050

    low1_n = int(low1_s * sample_rate)
    high_n = int(high_s * sample_rate)
    low2_n = int(low2_s * sample_rate)

    waveform = ([0] * low1_n) + ([1] * high_n) + ([0] * low2_n)

    # Read trigger settings

    timeout_ms = int(float(dataset.get_input_parameter("timeout_ms")))

    cam = dataset.camera_client
    ni = dataset.ni

    # 1) Configure trigger
    cam.set_hardware_trigger(
        connector="BNC",
        polarity="RisingEdge",
        active="EDGE",
        trigger_mode="NORMAL",
    )

    cam.set_output_trigger_exposure(
        kind="EXPOSURE",
        source="EXPOSURE",
        polarity="POSITIVE",
    )

    # 2) Arm camera (start acquisition). This returns immediately.
    logger.info("Starting acquisition (armed; waiting for TTL pulse)")
    cam.start_acquisition()

    ni.build_stack()
    ni.set_do_voltage(
        do_channel=do_channel,
        value=waveform,
        sample_rate=sample_rate,
    )
    ni.not_use_OPX_clock()
    h = ni.arm()
    ni.finalize(h, 30)

    try:
        # 3) Get ONE frame via bytes (RPyC-safe)
        logger.info("Requesting one triggered frame (bytes transport)")
        b, shape, dtype = cam.get_frame_bytes(timeout_ms=timeout_ms)

        frame = np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        logger.info(
            f"Got frame: shape={frame.shape}, dtype={frame.dtype}, "
            f"min={frame.min()}, max={frame.max()}"
        )
    finally:
        logger.info("Stopping acquisition")
        cam.stop_acquisition()

    # 4) Push frame to Plot2D
    img_ds = dataset.children["Image"]
    img_ds.set_data(frame)
    img_ds.update()

    # Optional: lock color scale
    img_ds.graph.setLevels(0, 255)
    img_ds.graph.ui.histogram.autoHistogramRange = False

    #thread.running = False
