
##############################################################################################################################################################################################################
import qm
import numpy as np
import json

from collections import defaultdict
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
#from pylabnet.hardware.quantum_machines.OPXdriverConfig import *
from pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel import *
import pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel as opx_cfg


#from pylabnet.qua_practice.configuration import *

from pylabnet.utils.logging.logger import LogHandler

######################################################CHANGE ALL CONFIG TO OPX_CFG.CONFIG AND SEE IF IT WORKS

# Right now the driver is only set up for outputting out of digital and analog channels 1
# Would be nice to be able to specify channel for output in set voltage functions but may be a little tricky
# Copy config dictionary into constructor


class Driver:
    def __init__(self, device_name, logger=None, dummy=False):

        self.dev = device_name
        self.log = LogHandler(logger=logger)
        self.dummy = dummy
        self.elem_counters = defaultdict(int) #can this list be in the config?
        self.elems = [] #can this list be in the config?
        self.qop_ip = "192.168.88.251"
        self.cluster_name = "Cluster_1"

        self.qmm = QuantumMachinesManager(host=self.qop_ip, cluster_name=self.cluster_name)

        self.adding_to_stack = False
        self.stack = []
        self.simulating = False

    def _simulate(self, program, duration=4000):
        # Simulates the QUA program for the specified duration
        simulate_config = SimulationConfig(duration=duration // 4) # duration is in clock cycles (1 clock cycles is 4ns)
        # Simulate blocks python until the simulation is done
        job = self.qmm.simulate(config, program, simulate_config)
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

    def build_stack(self):
        self.adding_to_stack = True
        self.simulating = False
        self.elem_counters = defaultdict(int)
        opx_cfg.config = opx_cfg.hard_coded_config.copy() #In general, this is wrong

    def set_ao_voltage(self, pulse=None, ao_channel=None, element=None, amplitude=default_amplitude, length=default_length, wave_function=None, simulate=False, frequency=default_frequency, gauss_sd=default_length / 5,): #removed pulse in driver #if we only specify ao_channel, I want pulse to play to first element for that channel
        if not pulse and element:
            pulse = next(iter(config["elements"][element]["operations"]))
            self.log.error(f"THIS IS THE ELEMENT: {element}")

        if element == None:
            ao_channel = int(float(ao_channel))

            element = self.create_new_ao_elem(
                ao_channel=ao_channel,
                pulse=pulse,
                amplitude=amplitude,
                length=length,
                wave_function=wave_function,
                frequency=frequency,
                gauss_sd=gauss_sd,
            )

        if not self.adding_to_stack:
            with program() as set_voltage:
                with infinite_loop_():
                    play(pulse, element) #right now only defined pulse is const (Modulated with IF frequency)

            # execute QUA program
            if simulate:
                self._simulate(set_voltage)

            else:

                qm = self.qmm.open_qm(config)
                job = qm.execute(set_voltage)  # execute QUA program
        else:
            #True of False is infinite or not infinite
            self.stack.append(["play", element, pulse, True, simulate])

        return element

    def play_ao_voltage(self, pulse, ao_channel, element, amplitude, length, gauss_sd, wave_function, simulate, frequency): #removed pulse in driver #if we only specify ao_channel, I want pulse to play to first element for that channel
        if not pulse and element:
            pulse = next(iter(config["elements"][element]["operations"]))
            self.log.error(f"THIS IS THE ELEMENT: {element}")

        if element == None:
            ao_channel = int(float(ao_channel))

            element = self.create_new_ao_elem(
                ao_channel=ao_channel,
                pulse=pulse,
                amplitude=amplitude,
                length=length,
                wave_function=wave_function,
                frequency=frequency,
                gauss_sd=gauss_sd,
            )

        if not self.adding_to_stack:
            with program() as set_voltage:
                play(pulse, element) #right now only defined pulse is const (Modulated with IF frequency)

            # execute QUA program
            if simulate:
                self._simulate(set_voltage)

            else:

                qm = self.qmm.open_qm(config)
                job = qm.execute(set_voltage)  # execute QUA program
        else:
            #True of False is infinite or not infinite
            self.stack.append(["play", element, pulse, True, simulate])

        return element

    def set_digital_voltage(self, element, pulse, do_channel, length, delay, buffer, simulate):

        if element is None:
            element = self.create_new_do_elem(
                do_channel=do_channel,
                length=length,
                delay=delay,
                buffer=buffer,
            )
            if not pulse:
                pulse = "ON"

        if not self.adding_to_stack:
            with program() as set_voltage:
                play(pulse, element) #right now only defined pulse is ON pulse

            # execute QUA program
            if simulate:
                self._simulate(set_voltage)

            else:
                qm = self.qmm.open_qm(config)
                job = qm.execute(set_voltage)  # execute QUA program
        else:
            #True of False is infinite or not infinite
            self.stack.append(["play", element, pulse, False, simulate])

        return element

    def get_ai_voltage(self, pulse=default_measure_pulse, ai_channel=None, element=None, ao_channel=None, amplitude=default_amplitude, length=default_length, wave_function=None, frequency=default_frequency, time_of_flight=defaut_time_of_flight, smearing=default_smearing, gauss_sd=default_length / 5,):
        if element:
            ai_channel = config["elements"][element]["singleInput"]["port"][1]

        if not element:
            element = self.create_new_ai_elem(
                pulse=pulse,
                ai_channel=ai_channel,
                ao_channel=ao_channel,
                amplitude=amplitude,
                length=length,
                wave_function=wave_function,
                frequency=frequency,
                time_of_flight=time_of_flight,
                smearing=smearing,
                gauss_sd=gauss_sd,
            )

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

        return element

    def execute(self):
        self.log.error(f"EXECUTE STARTS. STACK: {self.stack}")
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

                    self.log.error(f"ELEMENT IN THE CURR PROG {element}")

                    s = declare_stream(adc_trace=True)
                    measure("readout", element, adc_stream=s)

                    # Create a unique, descriptive result name per measure
                    # (QM handle names: letters/numbers/underscores are safe)
                    label = f"raw_adc_{measure_idx}_in{input_idx}"
                    streams.append((s, input_idx, label))
                elif op[0] == 'play':
                    if op[4]:
                        self.simulating = True
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

        if self.simulating:
            self._simulate(curr_prog)
        elif measuring:
            self.log.error(f"CONFIG CURRENT {config}")

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
            self.log.error("EXECUTE ENDS")

    def create_new_ao_elem(self, pulse, ao_channel=None, amplitude=default_amplitude, length=default_length, wave_function=None, frequency=default_frequency, gauss_sd=default_length / 5,):
        ao_channel = int(float(ao_channel))
        self.elem_counters[f"ao_{ao_channel}"] += 1
        idx = self.elem_counters[f"ao_{ao_channel}"]
        elem_name = f"generic_ai_elem_ch{ao_channel}_{idx}"
        self.elems.append(elem_name)

        self.log.error(f"frequecy {frequency}")

        config["elements"][elem_name] = {
            "singleInput": {"port": ("con1", ao_channel)},
            "intermediate_frequency": frequency * u.MHz,
            "operations": {}
        }
        if pulse == "const":

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
            gauss_wf, default = drag_gaussian_pulse_waveforms(
                amplitude, length, gauss_sd,
                alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
            )

            config["elements"][elem_name]["operations"]["gauss"] = f"gaussian_pulse_ch{ao_channel}_{idx}"

            config["pulses"][f"gaussian_pulse_ch{ao_channel}_{idx}"] = {
                "operation": "control",
                "length": length,
                "waveforms": {"single": f"gaussian_wf_ch{ao_channel}_{idx}"},
            }

            config["waveforms"][f"gaussian_wf_ch{ao_channel}_{idx}"] = {
                "type": "gaussian",
                "samples": gauss_wf,   # you should define this earlier
            }

        elif pulse == "arbitrary":

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
        self.log.info(f"Element created: {elem_name}")

        return elem_name

    def create_new_do_elem(
        self,
        do_channel,
        length,
        delay,
        buffer,
    ):
        """Create a new digital-output element and its ON pulse."""

        # Normalize type
        do_channel = int(float(do_channel))

        # Counter/indexing scheme per DO channel
        self.elem_counters[f"do_{do_channel}"] += 1
        idx = self.elem_counters[f"do_{do_channel}"]

        # Names
        elem_name = f"generic_di_elem_ch{do_channel}_{idx}"
        pulse_name = f"ON_pulse_ch{do_channel}_{idx}"

        self.elems.append(elem_name)
        self.log.info(
            f"[create_new_do_elem] do_channel={do_channel}, idx={idx}, "
            f"elem={elem_name}, pulse={pulse_name}"
        )

        # Ensure digital_waveforms / ON exist
        if "digital_waveforms" not in config:
            config["digital_waveforms"] = {}
        if "ON" not in config["digital_waveforms"]:
            config["digital_waveforms"]["ON"] = {"samples": [(1, 0)]}

        # Element block
        config["elements"][elem_name] = {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", do_channel),
                    "delay": int(delay),
                    "buffer": int(buffer),
                }
            },
            "operations": {
                # Operation name -> pulse name
                "ON": pulse_name
            },
        }

        # Pulse block
        config["pulses"][pulse_name] = {
            "operation": "control",
            "length": int(length),   # ns
            "digital_marker": "ON",  # refers to digital_waveforms["ON"]
        }

        self.log.info(f"Digital element created: {elem_name}")
        return elem_name

    def create_new_ai_elem(self, pulse=default_measure_pulse, ai_channel=None, ao_channel=None, amplitude=default_amplitude, length=default_length, wave_function=None, frequency=default_frequency, time_of_flight=defaut_time_of_flight, smearing=default_smearing, gauss_sd=default_length / 5,):

        # Normalize types
        ai_channel = int(float(ai_channel))
        ao_channel = int(float(ao_channel))

        # Counter/indexing scheme (mirror AO creator: per AO channel)
        self.elem_counters[f"ai_{ao_channel}_{ai_channel}"] += 1
        idx = self.elem_counters[f"ai_{ao_channel}_{ai_channel}"]

        # Element & pulse names
        elem_name = f"generic_output_elem_ch{ao_channel}_ch{ai_channel}_{idx}"
        meas_pulse_name = f"meas_pulse_out_ch{ao_channel}_in_ch{ai_channel}_{idx}"

        self.elems.append(elem_name)
        self.log.info(f"[create_new_ai_elem] freq {frequency}, ao={ao_channel}, ai={ai_channel}, idx={idx}")

        # Element block (measurement element that drives AO and reads from AI)
        config["elements"][elem_name] = {
            "singleInput": {"port": ("con1", ao_channel)},           # drive out on AO
            "intermediate_frequency": frequency * u.MHz,
            "operations": {"readout": meas_pulse_name},
            "time_of_flight": int(time_of_flight),
            "smearing": int(smearing),
            "outputs": {"out1": ("con1", ai_channel)},               # read back on AI
        }

        # Choose / create waveform per requested pulse type
        if pulse == "const":
            self.log.info("[create_new_ai_elem] using CONST measurement pulse")
            wf_name = f"square_measure_wf_ch{ao_channel}_{idx}"
            config["waveforms"][wf_name] = {
                "type": "constant",
                "sample": float(amplitude),
            }

        elif pulse == "gauss":
            gauss_wf, default = drag_gaussian_pulse_waveforms(
                amplitude, length, gauss_sd,
                alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
            )

            self.log.info("[create_new_ai_elem] using GAUSS measurement pulse")
            wf_name = f"gaussian_measure_wf_ch{ao_channel}_{idx}"

            config["waveforms"][wf_name] = {
                "type": "arbitrary",
                "samples": gauss_wf,
            }

        elif pulse == "arbitrary":
            self.log.info("[create_new_ai_elem] using ARBITRARY measurement pulse")
            wf_name = f"arbitrary_measure_wf_ch{ao_channel}_{idx}"
            config["waveforms"][wf_name] = {
                "type": "arbitrary",
                "samples": wave_function,   # caller supplies list of samples
            }

        else:
            raise ValueError(f"Unknown pulse type for measurement: {pulse!r}")

        # Measurement pulse block
        config["pulses"][meas_pulse_name] = {
            "operation": "measurement",
            "length": int(length),
            "waveforms": {"single": wf_name},   # <-- reference by STRING name
            # These weights exist in your config; harmless if not demodulating
            "integration_weights": {
                "cos": "cos",
                "sin": "sin",
                "minus_sin": "minus_sin",
            },
            "digital_marker": "ON",
        }
        with open("C:/Users/User/pylabnet/config_dump.txt", "w") as f:
            f.write(json.dumps(config, indent=2, default=str))

        return elem_name

    def delay(self, length, elements, simulate=False):
        cycles = int(length / 4)
        if length % 4 != 0:
            self.log.error("ERROR. Input to length of delay function must be a multiple of 4ns.")
            raise ValueError("Length in delay function must be a multiple of 4")
        if not self.adding_to_stack:
            with program() as wait_prog:
                if elements:
                    wait(cycles, *elements)
                else:
                    wait(cycles)
            # execute QUA program
            qm = self.qmm.open_qm(config)
            job = qm.execute(wait_prog)  # execute QUA program
        else:
            #True of False is infinite or not infinite
            self.stack.append(["wait", cycles, elements])
