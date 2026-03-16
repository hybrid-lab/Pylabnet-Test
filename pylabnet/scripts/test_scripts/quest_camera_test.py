import numpy as np
from PyQt5 import QtCore

# pylabnet imports (kept to match pylabnet script expectations)
from pylabnet.scripts.data_center.take_data import ExperimentThread  # noqa: F401
from pylabnet.scripts.data_center.datasets import (
    Dataset,
    Plot2D,
    Plot2DWithAvg,
)  # noqa: F401

# Optional helpers (kept because your template imports them)
from pylabnet.launchers.siv_py_functions import upload_sequence, load_config  # noqa: F401

# IMPORTANT:
# Matplotlib GUI calls must run on the MAIN Qt thread.
import matplotlib.pyplot as plt


if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


# -----------------------------
# Qt-safe plotting infrastructure
# -----------------------------


class _FramePlotter(QtCore.QObject):
    """QObject to ferry frames from worker thread -> Qt main thread."""
    frame_ready = QtCore.pyqtSignal(object)


_plotter = _FramePlotter()
_fig = None
_im = None
_cb = None


def _plot_frame_mainthread(frame, vmin=0, vmax=4095):
    """Plot/update the most recent frame. MUST run on Qt main thread."""
    global _fig, _im, _cb

    if frame is None:
        return

    plt.ion()

    if _fig is None:
        _fig = plt.figure("Most recent Hamamatsu frame")
        _im = plt.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
        _cb = plt.colorbar()
        plt.axis("off")
    else:
        _im.set_data(frame)
        _im.set_clim(vmin, vmax)

    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()


_plotter.frame_ready.connect(_plot_frame_mainthread)


# -----------------------------
# Experiment script settings
# -----------------------------

INIT_DICT = {
    "timeout_ms": {"Get Frame Timeout (ms)": "5000"},
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

    try:
        # Replace this key if your launcher uses a different device name
        camera_client = kwargs["fluorescence_imaging_camera_orca_quest"]
        dataset.camera_client = camera_client
        logger.info("Hamamatsu camera client attached to dataset.")

    except KeyError:
        logger.error(
            "Could not find camera client in kwargs under "
            "'fluorescence_imaging_camera_orca_quest'. "
            "Check your launcher/device name."
        )
        raise
    except Exception as e:
        logger.error(f"An error occurred in CONFIGURE: {e}")
        raise

    # Optional: if your camera driver supports a known fixed exposure, set it here
    # try:
    #     dataset.camera_client.set_exposure_time(0.01)  # 10 ms
    # except Exception as e:
    #     logger.warning(f"Could not set exposure time: {e}")

    # Build a 2D image dataset.
    # These dimensions are placeholders; the actual image will still be written
    # as long as the plot object accepts the incoming frame shape.
    dataset.add_child(
        name="Image Average",
        data_type=Plot2DWithAvg,
        min_x=0,
        max_x=4096,
        pts_x=4096,
        min_y=0,
        max_y=2304,
        pts_y=2304,
        new_plot=True,
    )

    avgview = dataset.children["Image Average"].graph
    logger.info(f'CHILDREN: {dataset.children["Image Average"].children.keys()}')

    try:
        avgview.setLevels(0, 300)
        dataset.children["Image Average"].children["Image Averagecurrentavg"].graph.setLevels(0, 300)
    except Exception as e:
        logger.warning(f"Could not preset image display levels: {e}")

    dataset.graph.hide()


def experiment(**kwargs):
    """Main experiment entrypoint called by DataTaker."""
    thread = kwargs["thread"]   # noqa: F841 (kept for compatibility)
    dataset = kwargs["dataset"]
    logger = dataset.log

    timeout_ms = int(dataset.get_input_parameter("timeout_ms"))

    def get_frame(timeout_ms=1000):
        """
        Camera client should return:
            (bytes, shape, dtype_str)
        """
        b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms=timeout_ms)
        logger.info(f"Frame shape reported by camera: {shape}, dtype={dtype}")
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    frame = None

    logger.info("Starting acquisition")
    dataset.camera_client.start_acquisition()

    try:
        logger.info("Requesting one frame")
        frame = get_frame(timeout_ms=timeout_ms)

        logger.info(
            f"Got frame: shape={frame.shape}, dtype={frame.dtype}, "
            f"min={frame.min()}, max={frame.max()}"
        )

    finally:
        logger.info("Stopping acquisition")
        dataset.camera_client.stop_acquisition()

    if frame is None:
        raise RuntimeError("No frame was acquired.")

    measurements = frame

    # Parent Plot2DWithAvg
    img_ds = dataset.children["Image Average"]

    # Averaged child plot
    avg = img_ds.children["Image Averagecurrentavg"]

    # Write data and update plots
    img_ds.set_data(measurements)
    img_ds.update()

    # Lock color scale on average image
    try:
        avg.graph.setLevels(int(frame.min()), int(frame.max()))
        avg.graph.ui.histogram.autoHistogramRange = False
    except Exception as e:
        logger.warning(f"Could not lock histogram levels: {e}")

    # Optional matplotlib display in main Qt thread
    # _plotter.frame_ready.emit(frame)

    logger.info("Single-image acquisition complete.")
