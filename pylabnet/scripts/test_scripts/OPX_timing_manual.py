import numpy as np
import qm

from qm.qua import infinite_loop_, play, program, wait
from qm import QuantumMachinesManager

from pylabnet.hardware.ni_daqs.nidaqmx_card import Driver
from pylabnet.hardware.quantum_machines.OPXdriverConfig import config

# ----------------------------
# Hardware / experiment params
# ----------------------------
ni = Driver(device_name="PXI1Slot2_1")
ni_2 = Driver(device_name="PXI1Slot6_1")

ni_ao = "ao0"
ni_2_ao = "ao15"

qop_ip = "192.168.88.253"
cluster_name = "Cluster_1"

ramp_min = 0
ramp_max = 3
N = 10000
ramp = np.linspace(ramp_min, ramp_max, N, dtype=np.float64)

ni_2.build_stack()
ni_2.set_ao_voltage(ao_channel=ni_2_ao, voltages=ramp, sample_rate=100)
ni_2.not_use_OPX_clock()


h2 = ni_2.arm()
out2 = ni_2.finalize(h2, timeout=60.0)
