import numpy as np
import qm

from qm import SimulationConfig
from qm.qua import infinite_loop_, play, program, wait
from qm import LoopbackInterface
from qm import QuantumMachinesManager

from pylabnet.scripts.data_center.datasets import Dataset, InfiniteRollingLine
from pylabnet.hardware.ni_daqs.nidaqmx_card import Driver
from pylabnet.hardware.quantum_machines.OPXdriverConfig import config

ni = Driver(device_name="PXI1Slot2_1")
ni_2 = Driver(device_name="PXI1Slot6_1")

ni_ao = "ao0"
ni_2_ao = "ao15"
qop_ip = "192.168.88.253"
cluster_name = "Cluster_1"
#OPX timing frequency is 10 kHz
ramp_min = 0
ramp_max = 3
N = 10000
#ramp = [1] * N
ramp = np.linspace(ramp_min, ramp_max, N, dtype=np.float64)
voltages = ramp
fs = 100000

with program() as pulse_train:
    with infinite_loop_():
        play("ON", "generic_di_elem_ch1")
        wait(22500)

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config)
job = qm.execute(pulse_train)

ni.build_stack()
ni.set_ao_voltage(ao_channel=ni_ao, voltages=ramp, sample_rate=fs) #For future, make a set_timing function that will set the timing of a card to be an external clock
ni.execute()

ni_2.build_stack()
ni_2.set_ao_voltage(ao_channel=ni_2_ao, voltages=ramp, sample_rate=fs)
ni_2.execute()
