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
        # For nested control flow: a stack-of-stacks and open IFs
        self._current_stack = [self.stack]  # where _append() writes
        self._if_stack = []                # open IF nodes for else_()
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

        self.stack = []
        self._current_stack = [self.stack]
        self._if_stack = []

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
            self._append(["play", element, pulse, True, simulate])

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
            self._append(["play", element, pulse, True, simulate])

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
            self._append(["play", element, pulse, False, simulate])

        return element

    def get_ai_voltage(self, pulse=default_measure_pulse, ai_channel=None, element=None, ao_channel=None, amplitude=default_amplitude, length=default_length, wave_function=None, frequency=default_frequency, time_of_flight=defaut_time_of_flight, smearing=default_smearing, gauss_sd=default_length / 5, variable_name=None):
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
            if variable_name:
                self._append(["measure_demod", element, variable_name])
            else:
                self._append(["measure", element, ai_channel])

        return element

    def execute(self):
        self.log.error(f"EXECUTE STARTS. STACK: {self.stack}")
        self.adding_to_stack = False
        self.simulating = False

        measuring_flag = {"value": False}
        measure_counter = {"value": 0}
        streams = []

        with program() as curr_prog:
            # -----------------------------------------------------------------
            # 1) Declare any QUA scalar variables you need for conditionals
            #    For now, assume one logical variable "I0" as an example:
            #    Later you can auto-detect needed names from self.stack.
            # -----------------------------------------------------------------
            var_names = set()

            def collect_vars(stack):
                for op in stack:
                    if op[0] == "measure_demod":
                        var_names.add(op[2])
                    elif op[0].startswith("if_"):
                        var_names.add(op[1])
                        collect_vars(op[3])   # then branch
                        collect_vars(op[4])   # else branch
                    # Play/wait/measure ops do not have vars

            collect_vars(self.stack)

            # Declare all required QUA variables
            self._qua_vars = {name: declare(fixed) for name in var_names}            # Example: a single fixed var named "I0"
            # (You would extend this to scan stack for all var_names you use.)
            # self._qua_vars["I0"] = declare(fixed)

            # -----------------------------------------------------------------
            # 2) Emit all operations recursively (including any conditionals)
            # -----------------------------------------------------------------
            self._emit_stack(self.stack, streams, measuring_flag, measure_counter)

            # -----------------------------------------------------------------
            # 3) SINGLE stream-processing block for ALL streams
            # -----------------------------------------------------------------
            if streams:
                with stream_processing():
                    for s, input_idx, label in streams:
                        if input_idx == 1:
                            s.input1().with_timestamps().save_all(label)
                        elif input_idx == 2:
                            s.input2().with_timestamps().save_all(label)
                        else:
                            # fallback: save input1 if index is weird
                            s.input1().with_timestamps().save_all(label)

        # Clear stack after building
        self.stack = []

        # ---------------------------------------------------------------------
        # 4) Run or simulate
        # ---------------------------------------------------------------------
        if self.simulating:
            self._simulate(curr_prog)

        elif measuring_flag["value"]:
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

    # -----------------------------------------------------------------
    # Internal helpers for hierarchical stack building
    # -----------------------------------------------------------------

    def _emit_stack(self, stack, streams, measuring_flag, measure_counter):
        """
        Recursively emit QUA code for all ops in `stack`.

        `streams`         : list that collects (stream, input_idx, label)
        `measuring_flag`  : dict with {"value": bool}, so we can mutate it
        `measure_counter` : dict with {"value": int}, so we can give each
                            measurement a unique label.
        """
        for op in stack:
            kind = op[0]

            if kind == "measure":
                # Existing flat behavior, just moved here
                measuring_flag["value"] = True
                element = op[1]
                input_idx = op[2]   # 1 or 2

                # bump global measurement counter
                measure_counter["value"] += 1
                m_idx = measure_counter["value"]

                s = declare_stream(adc_trace=True)
                measure("readout", element, adc_stream=s)

                label = f"raw_adc_{m_idx}_in{input_idx}"
                streams.append((s, input_idx, label))

            elif kind == "measure_demod":
                _, element, var_name = op

                # Ensure this variable exists
                I = self._qua_vars[var_name]

                measure("readout", element, None,
                        dual_demod.full("cos", "sin", I))

            elif kind == "play":
                # op layout: ["play", element, pulse, is_infinite, simulate]
                element = op[1]
                pulse = op[2]
                simulate_flag = op[4]

                if simulate_flag:
                    self.simulating = True

                # right now you ignore op[3] (infinite) at QUA level, same as before
                play(pulse, element)

            elif kind == "wait":
                # op layout: ["wait", cycles, elements]
                cycles = op[1]
                elements = op[2]
                if elements:
                    wait(cycles, *elements)
                else:
                    wait(cycles)

            elif kind == "if_gt":
                # Conditional node: ["if_gt", var_name, threshold, then_stack, else_stack]
                _, var_name, threshold, then_stack, else_stack = op

                # For now: assume you have a QUA variable dict self._qua_vars
                # created in execute(), e.g. I0 = declare(fixed)
                v = self._qua_vars[var_name]

                with if_(v > threshold):
                    self._emit_stack(then_stack, streams, measuring_flag, measure_counter)

                if else_stack:
                    with else_():
                        self._emit_stack(else_stack, streams, measuring_flag, measure_counter)

            else:
                # Optional: debug unknown ops
                self.log.error(f"Unknown operation kind in stack: {kind}")

    def _append(self, op):
        """Append an operation node to the *current* active sub-stack."""
        self._current_stack[-1].append(op)

    def _stack_push(self, new_stack):
        """Descend into a nested sub-stack (e.g. THEN or ELSE)."""
        self._current_stack.append(new_stack)

    def _stack_pop(self):
        """Ascend back out of a nested sub-stack."""
        if len(self._current_stack) == 1:
            raise RuntimeError("Attempted to pop the root stack.")
        self._current_stack.pop()

    # -----------------------------------------------------------------
    # Public API: context-managed conditionals
    # -----------------------------------------------------------------
    def if_gt(self, var_name, threshold):
        """
        Start an IF block that will later be interpreted as:

            if <var_name> > threshold:
                ...

        Example usage:

            with driver.if_gt("I0", 0.1):
                driver.set_ao_voltage(...)
        """
        return _IfContext(self, "gt", var_name, threshold)

    def elif_gt(self, var_name, threshold):
        """
        Start an ELSE-IF block that will later be interpreted as:

            elif <var_name> > threshold:
                ...

        Usage:

            with driver.if_gt("I0", 0.5):
                ...
            with driver.elif_gt("I0", 0.2):
                ...
            with driver.else_():
                ...
        """
        return _ElifContext(self, "gt", var_name, threshold)

    def else_(self):
        """
        Start an ELSE block corresponding to the most recent open IF.

        Example usage:

            with driver.if_gt("I0", 0.1):
                ...
            with driver.else_():
                ...
        """
        return _ElseContext(self)

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
            self._append(["wait", cycles, elements])


class _IfContext:
    """
    Context manager for:

        with driver.if_gt("I0", 0.1):
            driver.set_ao_voltage(...)

    It creates an IF node in driver.stack and routes all stack
    appends inside the block into its THEN branch.
    """

    def __init__(self, driver, op_name, var_name, threshold):
        self.driver = driver
        self.op_name = op_name      # "gt", "lt", etc.
        self.var_name = var_name
        self.threshold = threshold

        # Node format: ["if_gt", "I0", 0.1, then_stack, else_stack]
        self.node = [f"if_{op_name}", var_name, threshold, [], []]

    def __enter__(self):
        # Attach IF node to current stack
        self.driver._append(self.node)

        # Register this IF as "open" (for else_)
        self.driver._if_stack.append(self.node)

        # Route subsequent appends into the THEN branch
        then_stack = self.node[3]
        self.driver._stack_push(then_stack)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Leave THEN branch (but keep IF on _if_stack so else_ can see it)
        self.driver._stack_pop()


class _ElifContext:
    """
    Context manager for:

        with driver.elif_gt("I0", 0.2):
            driver.set_ao_voltage(...)

    This is implemented as "else: if ..." on top of the previous IF.
    It creates a *nested* IF node inside the ELSE branch of the
    most recent open IF in driver._if_stack.
    """

    def __init__(self, driver, op_name, var_name, threshold):
        self.driver = driver
        self.op_name = op_name      # "gt", "lt", etc.
        self.var_name = var_name
        self.threshold = threshold
        self.node = None            # the nested IF node we create

    def __enter__(self):
        # Must have an open IF to attach to
        if not self.driver._if_stack:
            raise RuntimeError("elif_() called without a matching if_().")

        # The "current" IF in the chain (top of the stack)
        parent_if = self.driver._if_stack[-1]

        # parent_if has format: ["if_gt", var, thr, then_stack, else_stack]
        else_stack = parent_if[4]

        # Create a new nested IF node that will live inside parent's ELSE
        then_stack = []
        self.node = [f"if_{self.op_name}", self.var_name, self.threshold,
                     then_stack, []]  # empty nested-else for now

        # Attach nested IF to parent's ELSE branch
        else_stack.append(self.node)

        # Now treat this nested IF as the "current" IF for any further
        # elif/else in this chain
        self.driver._if_stack[-1] = self.node

        # Route subsequent stack appends into this nested IF's THEN branch
        self.driver._stack_push(then_stack)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Done with this clause's THEN body
        self.driver._stack_pop()


class _ElseContext:
    """
    Context manager for:

        with driver.else_():
            driver.set_ao_voltage(...)

    It attaches to the most recent open IF and routes stack appends
    into its ELSE branch.
    """

    def __init__(self, driver):
        self.driver = driver
        self.node = None  # the IF node this ELSE is attached to

    def __enter__(self):
        if not self.driver._if_stack:
            raise RuntimeError("else_() called without a matching if_().")

        # The most recent IF node
        self.node = self.driver._if_stack[-1]
        else_stack = self.node[4]

        # Route subsequent appends into ELSE branch
        self.driver._stack_push(else_stack)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Leave ELSE branch
        self.driver._stack_pop()
        # IF is now fully closed (THEN + ELSE done)
        self.driver._if_stack.pop()
