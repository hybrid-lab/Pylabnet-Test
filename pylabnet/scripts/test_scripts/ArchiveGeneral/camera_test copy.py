import numpy as np
from PyQt5 import QtCore

from pylabnet.scripts.data_center.take_data import ExperimentThread  # noqa: F401
from pylabnet.scripts.data_center.datasets import Dataset, Plot2D  # noqa: F401

from pylabnet.launchers.siv_py_functions import upload_sequence, load_config  # noqa: F401


INIT_DICT = {
    'readout_len': {'Readout Length (ns)': '1000'},  # not used for camera, kept for consistency
    'blank1': {'filler': '0'},
    'blank2': {'filler': '0'},
    'blank3': {'filler': '0'},
    'blank4': {'filler': '0'},
}


def define_dataset():
    return 'Dataset'


# -----------------------------
# Qt-safe plotting infrastructure
# -----------------------------

class _FramePlotter(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(object)


_plotter = _FramePlotter()
_plot2d_child = None  # set in configure()


@QtCore.pyqtSlot(object)
def _update_plot2d_mainthread(frame):
    """Runs on Qt main thread."""
    global _plot2d_child
    if frame is None or _plot2d_child is None:
        return

    # Feed full matrix; Plot2D accepts (pts_y, pts_x)
    _plot2d_child.set_data(frame)
    _plot2d_child.update()


_plotter.frame_ready.connect(_update_plot2d_mainthread)


def configure(**kwargs):
    """Attach camera and create Plot2D with correct dimensions."""
    global _plot2d_child

    dataset = kwargs['dataset']
    logger = dataset.log

    # Attach camera client
    camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.camera_client = camera_client
    logger.info("Camera client attached to dataset.")

    # Helper: bytes -> ndarray
    def get_frame(timeout_ms=1000):
        b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    # Get one frame to learn image size for Plot2D axes
    logger.info("Starting acquisition (configure) to get frame shape")
    dataset.camera_client.start_acquisition()
    try:
        frame0 = get_frame(timeout_ms=1000)
    finally:
        dataset.camera_client.stop_acquisition()

    if frame0.ndim != 2:
        raise ValueError(f"Expected 2D grayscale frame, got shape {frame0.shape}")

    h, w = frame0.shape
    logger.info(f"Detected camera frame shape: (H,W)=({h},{w}), dtype={frame0.dtype}")

    # Create Plot2D child with pixel-coordinate axes
    dataset.add_child(
        name='Image',
        data_type=Plot2D,
        window='camera_frame',
        window_title='Camera Frame',
        min_x=0,
        max_x=w,
        pts_x=w,
        min_y=0,
        max_y=h,
        pts_y=h,
        new_plot=True
    )

    _plot2d_child = dataset.children['Image']

    # Show the initial frame immediately (optional)
    _plotter.frame_ready.emit(frame0)


def experiment(**kwargs):
    """Acquire one frame and update Plot2D via the Qt main thread."""
    thread = kwargs['thread']  # noqa: F841
    dataset = kwargs['dataset']
    logger = dataset.log

    def get_frame(timeout_ms=1000):
        b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    logger.info("Starting acquisition (experiment)")
    dataset.camera_client.start_acquisition()
    try:
        frame = get_frame(timeout_ms=1000)
        logger.info(
            f"Got frame: shape={frame.shape}, dtype={frame.dtype}, "
            f"min={frame.min()}, max={frame.max()}"
        )
    finally:
        dataset.camera_client.stop_acquisition()

    # IMPORTANT: update GUI on Qt main thread
    _plotter.frame_ready.emit(frame)
