import sys


from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from opx_config_scope import *

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5 import uic

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtCore import QObject, pyqtSignal as Signal, pyqtSlot as Slot, QThread, QTimer
import pyqtgraph as pg

QM_host = qop_ip


class OpxWorker(QObject):
    started = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def play_once(self):
        self.started.emit("Connecting to OPX...")
        qmm = QuantumMachinesManager(host=QM_host)
        qm = qmm.open_qm(config)

        with program() as prog:
            play("ON", "scope_dig")
            play("const", "scope") #Can adjust this to make it arbitrary waveforms by defining attributes

        self.started.emit("Executing...")
        job = qm.execute(prog)
        self.finished.emit("Pulse played.")


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        #load ui file
        self.ui = uic.loadUi("OPXGUI.ui", self)
        #grab widgets
        self.btnPlay: QPushButton = self.findChild(QPushButton, "btnPlay")
        self.plot: pg.PlotWidget = self.findChild(pg.PlotWidget, "plot")

        #plot setup
        self.curve = self.plot.plot()
        self.samples = np.full(const_len, const_amp)
        self.i = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)

        #Creating a separate worker thread
        self.thread = QThread(self)
        self.worker = OpxWorker()
        self.worker.moveToThread(self.thread)
        self.thread.start()

        #connect signals
        self.btnPlay.clicked.connect(self.worker.play_once)
        self.worker.started.connect(self.setWindowTitle)
        self.worker.finished.connect(self.setWindowTitle)
        self.worker.error.connect(lambda msg: self.setWindowTitle("ERR: " + msg))

    def update_plot(self):
        self.i = (self.i + 1)
        y = np.roll(self.samples, -self.i)
        self.curve.setData(np.arange(len(y)), y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Main()
    w.resize(900, 600)
    w.show()
    sys.exit(app.exec_())
