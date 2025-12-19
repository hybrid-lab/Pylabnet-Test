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
    wait_elems = ["generic_ai_elem_ch1", "generic_ai_elem_ch2"]
    play("const", "generic_ai_elem_ch1")
    wait(100, *["generic_ai_elem_ch2", "generic_ai_elem_ch1"])
    play("const", "generic_ai_elem_ch1")
    play("const", "generic_ai_elem_ch2")


#     #print(st)
    # play("const", "AOM")
#     #wait(500)


#         #play("ON", "trigger")
#         #wait(1000,"trigger")
#     #align("AOM","AOM_1")
#     #wait(10, "AOM_1")
#     #play("const", "AOM_1")
#  #   update_frequency("AOM", 10e6)
#   #  play("gauss","AOM")
#    # wait(100,"AOM","trigger")
#  #   a = declare(fixed)

#     #play("ON", "trigger")
#   #  align()
#    # with infinite_loop_():
#     #    with for_(a, 0, a < 1.1, a + 0.05):
#      #       play("const" * amp(a), "AOM")
#       #      play("const" * amp(a), "AOM", duration = 100)
#        # wait(25, "AOM")

# with program() as external_trigger:
#     play("ON", "trigger")
#     wait_for_trigger("AOM")
#     play("const", "AOM")
# ################################
# # Open quantum machine manager #
# ################################

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

# #######################
# # Simulate or execute #
# #######################

simulate = True

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
    qmm.perform_healthcheck(strict=True)
    qmm.reset_data_processing()  # careful in shared environments
    qm = qmm.open_qm(config)
    job = qm.execute(hello_qua)  # execute QUA program

#res = job.result_handles
#res.wait_for_all_values()
#raw_adc = res.get('raw_adc').fetch_all()

#print(raw_adc)
