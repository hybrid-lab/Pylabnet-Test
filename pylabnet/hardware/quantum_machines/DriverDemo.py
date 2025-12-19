import qm
import numpy as np

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager

#from pylabnet.qua_practice.configuration import *

from pylabnet.utils.logging.logger import LogHandler

from pylabnet.hardware.quantum_machines.OPX import Driver as OPX

wf = (np.asarray(list(range(1, 41))) / 100).tolist()
opx_1 = OPX("OPX_1")
#opx_1.play_ao_voltage(pulse = "const", ao_channel = 1, length = 200, amplitude = 0.2,frequency = 10, simulate = True, wave_function = None, element = None, gauss_sd = 0)
#opx_1.play_ao_voltage(pulse = "arbitrary", ao_channel = 1, length = 40, amplitude = 0,frequency = 0, simulate = True, wave_function = wf, element = None, gauss_sd = 0)
#opx_1.set_digital_voltage(pulse = "ON", do_channel = 1, simulate = True)
