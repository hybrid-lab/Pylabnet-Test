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
ni_3 = Driver(device_name="PXI1Slot4_1")

ni_ao = "ao0"
ni_2_ao = "ao15"

qop_ip = "192.168.88.253"
cluster_name = "Cluster_1"

ramp_min = 0
ramp_max = 3
N = 10000
ramp = np.linspace(ramp_min, ramp_max, N, dtype=np.float64)
digital_pulse = [1] * N

# IMPORTANT: set fs to the actual clock rate you feed the NI tasks with.
# If your OPX pulse train is ~10 kHz, use 10_000 here.
fs = 100000

ni.build_stack()
ni.set_ao_voltage(ao_channel=ni_ao, voltages=ramp, sample_rate=fs)
#ni.not_use_OPX_clock()
ni_2.build_stack()
ni_2.set_ao_voltage(ao_channel=ni_2_ao, voltages=ramp, sample_rate=1000)
#ni_2.not_use_OPX_clock()

ni_3.build_stack()
ni_3.set_do_voltage(do_channel="dio0", value=digital_pulse, sample_rate=1000)

with program() as pulse_train:
    with infinite_loop_():
        play("ON", "generic_di_elem_ch1")
        wait(225000)
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config)
job = qm.execute(pulse_train)

h2 = ni_2.arm()
h1 = ni.arm()
h3 = ni_3.arm()


out1 = ni.finalize(h1, timeout=60.0)
out2 = ni_2.finalize(h2, timeout=60.0)
out3 = ni_3.finalize(h3, timeout=60)

#print("Slot2 meta:", out1.get("_meta", {}))
# print("Slot6 meta:", out2.get("_meta", {}))
print("Finished Running")
