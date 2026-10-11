# qt_plotting.py
from PyQt5 import QtCore
import matplotlib.pyplot as plt


class QtMatplotlibFrameViewer(QtCore.QObject):
    """
    Thread-safe Qt->main-thread matplotlib image viewer.

    Call viewer.show(frame) from any thread.
    """
    frame_ready = QtCore.pyqtSignal(object)  # payload: (frame, vmin, vmax)

    def __init__(self, title: str = "Most recent camera frame", parent=None):
        super().__init__(parent)
        self._title = title
        self._fig = None
        self._im = None
        self._cb = None

        # Force queued connection (safe when emitter is in another thread)
        self.frame_ready.connect(self._plot_frame_mainthread, type=QtCore.Qt.QueuedConnection)

    def show(self, frame, vmin: float = 0, vmax: float = 4095) -> None:
        if frame is None:
            return
        self.frame_ready.emit((frame, vmin, vmax))

    @QtCore.pyqtSlot(object)
    def _plot_frame_mainthread(self, payload) -> None:
        frame, vmin, vmax = payload

        plt.ion()

        if self._fig is None:
            self._fig = plt.figure(self._title)
            self._im = plt.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
            self._cb = plt.colorbar()
            plt.axis("off")
        else:
            self._im.set_data(frame)
            self._im.set_clim(vmin, vmax)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
