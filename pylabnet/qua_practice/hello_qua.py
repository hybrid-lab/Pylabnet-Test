# """
# A script used for playing with QUA
# """

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from pylabnet.hardware.quantum_machines.OPXdriverConfigOld import *
from pylabnet.hardware.quantum_machines.OPX import Driver

#d = Driver("OPX")
#d.set_ao_voltage(pulse="const", ao_channel=1, voltage=0.5)

# ###################
# # The QUA program #
# ###################
with program() as hello_qua:

    play("const", "generic_ai_elem_ch2")
    play("const", "generic_ai_elem_ch1")

    adc_stream = declare_stream(adc_trace=True)
    measure("readout", "generic_output_elem_ch1_to_ch1", adc_stream=adc_stream)

    #adc_stream = declare_stream(adc_trace = True)
    #measure("readout", "generic_output_elem_ch1_to_ch1", adc_stream)
    #wait(124)
    #play("const", "generic_ai_elem_ch1")

    with stream_processing():
        adc_stream.input1().with_timestamps().save("raw_adc")

# ################################
# # Open quantum machine manager #
# ################################

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

# #######################
# # Simulate or execute #
# #######################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulate_config = SimulationConfig(duration=1000) # duration is in clock cycles (1 clock cycles is 4ns)
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, hello_qua, simulate_config)
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
    job = qm.execute(hello_qua)  # execute QUA program

res = job.result_handles
res.wait_for_all_values()
raw_adc = res.get('raw_adc').fetch_all()

print(raw_adc)
