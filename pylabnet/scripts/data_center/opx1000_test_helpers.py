"""Numeric DataTaker parameters and interruptible waits for OPX1000 tests."""

import math
import time

from PyQt5 import QtCore


class _PlotUpdate(QtCore.QObject):
    requested = QtCore.pyqtSignal()

    def __init__(self, dataset):
        super().__init__()
        # Configure creates this object in the GUI thread; its slot stays there.
        self._dataset = dataset
        self.requested.connect(self._refresh, QtCore.Qt.QueuedConnection)

    @QtCore.pyqtSlot()
    def _refresh(self):
        self._dataset.update()


def prepare_plot_updates(dataset):
    dataset._opx1000_plot_update = _PlotUpdate(dataset)


def request_plot_update(dataset):
    """Refresh in the GUI thread even when the one-shot worker has finished."""
    dataset._opx1000_plot_update.requested.emit()


def integer(dataset, name, minimum=1):
    value = float(dataset.get_input_parameter(name))
    if not math.isfinite(value) or not value.is_integer() or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def flag(dataset, name):
    value = integer(dataset, name, minimum=0)
    if value not in (0, 1):
        raise ValueError(name + " must be 0 or 1")
    return bool(value)


def wait_for_job(opx, thread, timeout_s):
    """Poll short RPCs so DataTaker's Stop flag can cancel a hardware job."""
    deadline = time.monotonic() + timeout_s
    while thread.running:
        status = opx.get_status()["status"]
        if status == "Done":
            return True
        if status in ("Canceled", "Error"):
            raise RuntimeError("OPX1000 job ended with status " + status)
        if time.monotonic() > deadline:
            opx.close()
            raise TimeoutError("OPX1000 test exceeded its completion timeout")
        time.sleep(0.05)
    opx.close()
    return False


def report_error(kwargs, error):
    kwargs["dataset"].log.error("OPX1000 test failed: " + str(error))
    signal = kwargs.get("status_flag")
    if signal is not None:
        signal.emit("BAD")
