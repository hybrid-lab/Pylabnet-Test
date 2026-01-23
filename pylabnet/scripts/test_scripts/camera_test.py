import numpy as np
from PyQt5 import QtCore

# pylabnet imports (kept to match pylabnet script expectations)
from pylabnet.scripts.data_center.take_data import ExperimentThread  # noqa: F401
from pylabnet.scripts.data_center.datasets import (
    SawtoothScan1D, ErrorBarGraph, InfiniteRollingLine, Dataset, SawtoothScan1D_array_update, Plot2D, Plot2DWithAvg  # noqa: F401
)

# Optional helpers (kept because your template imports them)
from pylabnet.launchers.siv_py_functions import upload_sequence, load_config  # noqa: F401

# IMPORTANT:
# Matplotlib GUI calls must run on the MAIN Qt thread.
import matplotlib.pyplot as plt


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

    # Ensure interactive mode
    plt.ion()

    if _fig is None:
        _fig = plt.figure("Most recent camera frame")
        _im = plt.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
        _cb = plt.colorbar()
        plt.axis("off")
    else:
        _im.set_data(frame)
        _im.set_clim(vmin, vmax)

    # Non-blocking redraw
    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()


# Connect signal to plotting slot; Qt will execute slot in main thread context
_plotter.frame_ready.connect(_plot_frame_mainthread)


# -----------------------------
# Experiment script settings
# -----------------------------

INIT_DICT = {
    'readout_len': {'Readout Length (ns)': '1000'},
    'blank1': {'filler': '0'},
    'blank2': {'filler': '0'},
    'blank3': {'filler': '0'},
    'blank4': {'filler': '0'},
}


def define_dataset():
    """Specifies the type of plot to use for the data."""
    return 'Dataset'


def configure(**kwargs):
    """Sets up the hardware and the plot before the experiment runs."""
    dataset = kwargs['dataset']
    logger = dataset.log

    try:
        # Pull the camera client from kwargs and stash it on the dataset for later use
        camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m']
        dataset.camera_client = camera_client

        logger.info("Camera client attached to dataset.")

    except Exception as e:
        logger.error(f"An error occurred in CONFIGURE: {e}")
        raise

    #measure_length = int(dataset.get_input_parameter('readout_len'))

    # dataset.add_child(
    # name="Image",
    # data_type=Plot2D,
    # min_x=0, max_x=2448, pts_x=2448,
    # min_y=0, max_y=2048, pts_y=2048,
    # new_plot=True
    # )

    dataset.add_child(
        name="Image Average",
        data_type=Plot2DWithAvg,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    # imgview = dataset.children["Image"].graph  # this is pg.ImageView
    avgview = dataset.children["Image Average"].graph  # this is pg.ImageView
    logger.info(f'CHILDREN: {dataset.children["Image Average"].children.keys()}')
    avgview.setLevels(0, 300)
    dataset.children["Image Average"].children["Image Averagecurrentavg"].graph.setLevels(0, 300)
    # imgview.setLevels(0, 300)

    #dataset.children["Image"].setLevels(min=0, max=300)

    # dataset.add_child(
    #     name='Image',
    #     data_type= Plot2D,
    #     data_length=measure_length,
    #     new_plot=True
    # )
    dataset.graph.hide()


def experiment(**kwargs):
    """Main experiment entrypoint called by DataTaker."""
    thread = kwargs['thread']   # noqa: F841 (kept for compatibility)
    dataset = kwargs['dataset']
    logger = dataset.log

    def get_frame(timeout_ms=1000):
        # get_frame_bytes() should return: (bytes, shape, dtype_str)
        b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
        logger.info(f"{shape}")
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    # Always stop acquisition even if something fails
    logger.info("Starting acquisition")
    dataset.camera_client.start_acquisition()
    try:
        logger.info("Requesting one frame")
        frame = get_frame(timeout_ms=1000)
        logger.info(f"Got frame: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
    finally:
        logger.info("Stopping acquisition")
        dataset.camera_client.stop_acquisition()

    measurements = frame
    # logger.info(f"DATA: {frame}")
    # Parent Plot2DWithAvg
    img_ds = dataset.children["Image Average"]

    # Averaged child plot
    avg = img_ds.children["Image Averagecurrentavg"]

    # -----------------------------
    # Data + update
    # -----------------------------

    img_ds.set_data(measurements)
    img_ds.update()        # updates both current + avg children

    # -----------------------------
    # Lock color scale on AVG image
    # -----------------------------

    avg.graph.setLevels(0, 255)
    avg.graph.ui.histogram.autoHistogramRange = False
    # time.sleep(3)

    #dataset.children["Image"].update()
    #Plot in the Qt main thread (prevents UI freezes / deadlocks)
    # logger.info("Sending frame to UI thread for plotting")
    # _plotter.frame_ready.emit(frame)
    # logger.info("Experiment complete")
