"""
A script used for playing with QUA
"""

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from opx_config_scope import *

with program() as scope_test:
    with infinite_loop_():
        play("const", "scope")


qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)


simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulate_config = SimulationConfig(duration=1000) # duration is in clock cycles (1 clock cycles is 4ns)
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, scope_test, simulate_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))

else:
    qm = qmm.open_qm(config)
    job = qm.execute(scope_test)  # execute QUA program
