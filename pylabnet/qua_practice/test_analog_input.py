"""
A script used for playing with QUA
"""

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from configuration import *
from matplotlib import *


###################
# The QUA program #
###################


Redout_len = 2000

with program() as hello_qua:

    adc_stream = declare_stream(adc_trace=True)
    measure("const", "generic_output_elem", adc_stream)
    wait(124)
    with infinite_loop_():
        play("const", "AOM_2")

    with stream_processing():
        adc_stream.input1().save("raw_adc")


qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config)
job = qm.execute(hello_qua)  # execute QUA program

rh = job.result_handles
rh.wait_for_all_values()

raw = rh.get("raw_adc_input1").fetch_all()

plt.figure()
plt.plot(raw)
plt.show
