from qm.qua import *
from qm import QuantumMachinesManager
from config1 import *

qop_ip = "192.168.88.252"
cluster_name = "Cluster_1"

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)  # creates a manager instance
qm = qmm.open_qm(config)  # opens a quantum machine with the specified configuration

with program() as powerRabiProg:

    I = declare(fixed)
    Q = declare(fixed)
    a = declare(fixed)
    Nrep = declare(int)

    with for_(Nrep, 0, Nrep < 100, Nrep + 1):
        with for_(a, 0.00, a < 1.0 + 0.01 / 2, a + 0.01):
            play('gauss_pulse' * amp(a), 'qubit')
            align("qubit", "resonator")
            measure("readout", "resonator", None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q))
            save(I, 'I')
            save(Q, 'Q')
            save(a, 'a')


my_job = qm.execute(powerRabiProg)

res_handle = my_job.result_handles
res_handle.wait_for_all_values()
I = res_handle.get('I').fetch_all()
Q = res_handle.get('Q').fetch_all()
print(I)
print(Q)
