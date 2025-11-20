##############################################################################################################################################################################################################
import qm
import numpy as np

from collections import defaultdict
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel import *
import pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel as opx_cfg

#from pylabnet.qua_practice.configuration import *

from pylabnet.utils.logging.logger import LogHandler


# Right now the driver is only set up for outputting out of digital and analog channels 1
# Would be nice to be able to specify channel for output in set voltage functions but may be a little tricky
# Copy config dictionary into constructor

class Driver:
    def __init__(self, device_name, logger=None, dummy=None):

        self.dev = device_name
        self.log = LogHandler(logger=logger)
        self.dummy = dummy
        #below are lists for generating elements
        self.elem_counters = defaultdict(int) #can this list be in the config?
        self.elems = [] #can this list be in the config?
        self.qop_ip = "192.168.88.252"
        self.cluster_name = "Cluster_1"

        self.qmm = QuantumMachinesManager(host=self.qop_ip, cluster_name=self.cluster_name)

        self.adding_to_stack = False
        self.stack = []
        self.log.error("THIS should also RUNS")

    def build_stack(self):
        self.adding_to_stack = True
        self.elem_counters = defaultdict(int)
        opx_cfg.config = opx_cfg.hard_coded_config.copy()

    def set_ao_voltage(self, pulse, ao_channel, element, amplitude, length, wave_function, simulate, frequency): #removed pulse in driver #if we only specify ao_channel, I want pulse to play to first element for that channel
        ao_channel = int(float(ao_channel))

        if element == None:
            idx = self.elem_counters[ao_channel]

            self.create_new_ao_elem(
                ao_channel=ao_channel,
                pulse=pulse,
                amplitude=amplitude,
                length=length,
                wave_function=wave_function,
                frequency=frequency,
            )
            element = f"generic_ai_elem_ch{ao_channel}_{idx}"

        if not self.adding_to_stack:
            with program() as set_voltage:
                with infinite_loop_():
                    play(pulse, element) #right now only defined pulse is const (Modulated with IF frequency)

            # execute QUA program
            if simulate:
                # Simulates the QUA program for the specified duration
                simulate_config = SimulationConfig(duration=1000) # duration is in clock cycles (1 clock cycles is 4ns)
                # Simulate blocks python until the simulation is done
                job = self.qmm.simulate(config, set_voltage, simulate_config)
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
                qm = self.qmm.open_qm(config)
                job = qm.execute(set_voltage)  # execute QUA program
        else:
            #True of False is infinite or not infinite
            self.stack.append(["play", element, pulse, True])

    def set_digital_voltage(self, pulse, do_channel, simulate):
        element = DO_CHANNEL_TO_GEN_EL[do_channel]

        if not self.adding_to_stack:
            with program() as set_voltage:
                with infinite_loop_():
                    play(pulse, element) #right now only defined pulse is ON pulse
                    wait(20)

            # execute QUA program
            if simulate:
                # Simulates the QUA program for the specified duration
                simulate_config = SimulationConfig(duration=1000) # duration is in clock cycles (1 clock cycles is 4ns)
                # Simulate blocks python until the simulation is done
                job = self.qmm.simulate(config, set_voltage, simulate_config)
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
                qm = self.qmm.open_qm(config)
                job = qm.execute(set_voltage)  # execute QUA program
        else:
            #True of False is infinite or not infinite
            self.stack.append(["play", element, pulse, True])

    def get_ai_voltage(self, ai_channel, length):
        config["pulses"][f"meas_pulse_in_ch{ai_channel}"]["length"] = length

        element = AO_CHANNEL_TO_GEN_EL[ai_channel]

        if not self.adding_to_stack:
            with program() as get_voltage:
                adc_stream = declare_stream(adc_trace=True)

                measure("readout", element, adc_stream=adc_stream)

                if ai_channel == 1:
                    with stream_processing():
                        adc_stream.input1().with_timestamps().save('raw_adc')

                elif ai_channel == 2:
                    with stream_processing():
                        adc_stream.input2().with_timestamps().save('raw_adc')

            qm = self.qmm.open_qm(config)
            job = qm.execute(get_voltage)
            res = job.result_handles
            res.wait_for_all_values()
            raw = res.get("raw_adc").fetch_all()

            out = [(int(raw[i]["timestamp"]), float(raw[i]["value"]))
                   for i in range(raw.shape[0])]

            return out

        else:
            self.stack.append(["measure", element, ai_channel])

    def execute(self):
        self.adding_to_stack = False
        measuring = False
        streams = []

        with program() as curr_prog:
            for idx, op in enumerate(self.stack):
                measure_idx = 0
                if op[0] == "measure":
                    measuring = True
                    element = op[1]
                    input_idx = op[2]  # 1 or 2
                    measure_idx += 1

                    s = declare_stream(adc_trace=True)
                    measure("readout", element, adc_stream=s)

                    # Create a unique, descriptive result name per measure
                    # (QM handle names: letters/numbers/underscores are safe)
                    label = f"raw_adc_{measure_idx}_in{input_idx}"
                    streams.append((s, input_idx, label))
                elif op[0] == 'play':
                    if op[3]:
                        play(op[2], op[1])
                    else:
                        play(op[2], op[1])

                elif op[0] == "wait":
                    if op[2]:
                        wait(op[1], *op[2])
                    else:
                        wait(op[1])

            # ---- SINGLE stream-processing block for ALL streams ----
            with stream_processing():
                for s, input_idx, label in streams:
                    if input_idx == 1:
                        s.input1().with_timestamps().save_all(label)
                    elif input_idx == 2:
                        s.input2().with_timestamps().save_all(label)
                    else:
                        # Optional: if input_idx is unexpected, save both or raise
                        s.input1().with_timestamps().save_all(label)

        self.stack = []
        if measuring:
            qm = self.qmm.open_qm(config)
            job = qm.execute(curr_prog)
            res = job.result_handles
            res.wait_for_all_values()

            out = {}
            for _, _, label in streams:

                raw = res.get(label).fetch_all()
                raw = raw[0][0]
                out[label] = [(int(raw[i]["timestamp"]), float(raw[i]["value"]))
                              for i in range(raw.shape[0])]

            return out
        else:
            qm = self.qmm.open_qm(config)
            job = qm.execute(curr_prog)

    def create_new_ao_elem(self, pulse, ao_channel, amplitude, length, wave_function, frequency):
        ao_channel = int(float(ao_channel))
        self.elem_counters[ao_channel] += 1
        idx = self.elem_counters[ao_channel]
        elem_name = f"generic_ai_elem_ch{ao_channel}_{idx}"
        self.elems.append(elem_name)

        config["elements"][elem_name] = {
            "singleInput": {"port": ("con1", ao_channel)},
            "intermediate_frequency": frequency,
            "operations": {}
        }
        self.log.error("THIS RUNS")
        if pulse == "const":
            self.log.error("THIS shoud RUNS")

            # Element operation name
            config["elements"][elem_name]["operations"]["const"] = f"square_pulse_ch{ao_channel}_{idx}"

            # Pulse block
            config["pulses"][f"square_pulse_ch{ao_channel}_{idx}"] = {
                "operation": "control",
                "length": length,
                "waveforms": {"single": f"square_wf_ch{ao_channel}_{idx}"},
            }

            # Waveform block
            config["waveforms"][f"square_wf_ch{ao_channel}_{idx}"] = {
                "type": "constant",
                "sample": amplitude,
            }

        elif pulse == "gauss":
            config["elements"][elem_name]["operations"]["gauss"] = f"gaussian_pulse_ch{ao_channel}_{idx}"

            config["pulses"][f"gaussian_pulse_ch{ao_channel}_{idx}"] = {
                "operation": "control",
                "length": length,
                "waveforms": {"single": f"gaussian_wf_ch{ao_channel}_{idx}"},
            }

            config["waveforms"][f"gaussian_wf_ch{ao_channel}_{idx}"] = {
                "type": "gaussian",
                "samples": default_gauss_wf,   # you should define this earlier
            }

        elif pulse == "arbitrary":
            self.log.error("THIS ASLO RUNS")

            config["elements"][elem_name]["operations"]["arbitrary"] = f"arbitrary_pulse_ch{ao_channel}_{idx}"

            config["pulses"][f"arbitrary_pulse_ch{ao_channel}_{idx}"] = {
                "operation": "control",
                "length": length,
                "waveforms": {"single": f"arbitrary_wf_ch{ao_channel}_{idx}"},
            }

            config["waveforms"][f"arbitrary_wf_ch{ao_channel}_{idx}"] = {
                "type": "arbitrary",
                "samples": wave_function,  # also define this earlier
            }

        return idx

    def delay(self, length, elements, simulate):
        cycles = int(length / 4)
        if length % 4 != 0:
            self.log.error("ERROR. Input to length of delay function must be a multiple of 4ns.")
            raise ValueError("Length in delay function must be a multiple of 4")
        if not self.adding_to_stack:
            with program() as wait_prog:
                if elements:
                    wait(length, *elements)
                else:
                    wait(length)
            # execute QUA program
            qm = self.qmm.open_qm(config)
            job = qm.execute(wait_prog)  # execute QUA program
        else:
            #True of False is infinite or not infinite
            self.stack.append(["wait", length, elements])
