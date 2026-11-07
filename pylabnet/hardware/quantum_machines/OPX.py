"""pylabnet driver for the Quantum Machines OPX1000 (the OPX+ is not supported).

All channels are on one LF-FEM: analog outputs 1-8, digital outputs 1-8, analog
inputs 1-2. Each set_/play_/get_ call creates a QUA element for its channel and
either runs a one-off program at once, or, between build_stack() and execute(),
is recorded on a stack that execute() turns into a single QUA program.
"""
import time
import warnings
from collections import defaultdict

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *

# config (the QUA config dict), opx_port, u, drag_gaussian_pulse_waveforms and the default_* values
from pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel import *
import pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel as opx_cfg
from pylabnet.utils.logging.logger import LogHandler


class Driver:
    def __init__(self, device_name, logger=None, dummy=False, host=None, cluster_name=None, lf_fem_slot=None):

        self.dev = device_name
        self.log = LogHandler(logger=logger)
        self.dummy = dummy    # part of pylabnet's driver signature; there is no dummy mode
        self.elem_counters = defaultdict(int)    # per channel, numbers the elements created on it
        self.elems = []
        # Defaults live in the config module; the device config can override them.
        self.qop_ip = host or opx_cfg.qop_ip
        self.cluster_name = cluster_name or opx_cfg.cluster_name
        if lf_fem_slot is not None:
            opx_cfg.set_lf_fem_slot(lf_fem_slot)

        if self.cluster_name:
            self.qmm = QuantumMachinesManager(host=self.qop_ip, cluster_name=self.cluster_name)
        else:
            self.qmm = QuantumMachinesManager(host=self.qop_ip)
        self.log.info(f"Connected to OPX at {self.qop_ip}: {self.qmm.version()}")
        self._check_opx1000()

        self.adding_to_stack = False
        self.stack = []
        # For nested control flow: a stack-of-stacks and open IFs
        self._current_stack = [self.stack]  # where _append() writes
        self._if_stack = []                # open IF nodes for else_()
        self.simulating = False

    def _check_opx1000(self):
        """Refuse anything but an OPX1000 with an LF-FEM in the configured slot."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    # get_controllers is deprecated, but the only call that reports FEMs
            controllers = {c.name: c for c in self.qmm.get_controllers()}
        con = controllers.get(opx_cfg.CONTROLLER)
        if con is None:
            raise RuntimeError(f"No controller {opx_cfg.CONTROLLER} at {self.qop_ip}; found {sorted(controllers)}")
        if con.controller_type != "OPX1000":
            raise RuntimeError(f"{opx_cfg.CONTROLLER} is an {con.controller_type}; this driver supports only the OPX1000")
        fem = con.fems.get(opx_cfg.LF_FEM_SLOT)
        if fem != "LF":
            raise RuntimeError(f"Slot {opx_cfg.LF_FEM_SLOT} of {opx_cfg.CONTROLLER} holds {fem or 'no FEM'}, not an LF-FEM; "
                               f"FEMs present: {con.fems}. Set lf_fem_slot in configs/devices/OPX/OPX.json")
        self.log.info(f"{opx_cfg.CONTROLLER}: OPX1000, using LF-FEM in slot {opx_cfg.LF_FEM_SLOT}")

    def _simulate(self, program, duration=4000):
        """Simulate `program` for `duration` ns on the OPX's simulator, plot it and save the waveform report."""
        job = self.qmm.simulate(config, program, SimulationConfig(duration=duration // 4))    # in 4 ns clock cycles
        job.wait_until("Done", timeout=60)    # samples are only available once the simulation is done
        samples = job.get_simulated_samples()
        samples.con1.plot()
        # create_plot saves into dirname(save_path), so name a file inside save_dir (git-ignored)
        job.get_simulated_waveform_report().create_plot(
            samples, plot=True, save_path=str(opx_cfg.save_dir / "waveform_report"))

    def build_stack(self):
        """Start recording: later set_/play_/get_/delay/loop calls go on the stack until execute()."""
        self.adding_to_stack = True
        self.simulating = False
        # Elements from earlier programs stay in config; restarting the counters
        # reuses their names, and each new element overwrites the old entry.
        self.elem_counters = defaultdict(int)

        self.stack = []
        self._current_stack = [self.stack]
        self._if_stack = []

    def set_ao_voltage(self, pulse=None, ao_channel=None, element=None, amplitude=default_amplitude, length=default_length, wave_function=None, simulate=False, frequency=default_frequency, gauss_sd=default_length / 5,):
        """Play `pulse` ('const', 'gauss' or 'arbitrary') on an analog output in an infinite loop.

        Inside build_stack()/execute() it is played once at that point of the program instead.
        Pass `element` to reuse one (its first operation is played if `pulse` is not given);
        otherwise a new element is created on `ao_channel`. Returns the element name.
        """
        if not pulse and element:
            pulse = next(iter(config["elements"][element]["operations"]))

        if element is None:
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
                    play(pulse, element)

            if simulate:
                self._simulate(set_voltage)
            else:
                self.qmm.open_qm(config).execute(set_voltage)
        else:
            # Stack node: ["play", element, pulse, infinite (not used by _emit_stack), simulate]
            self._append(["play", element, pulse, True, simulate])

        return element

    def play_ao_voltage(self, pulse, ao_channel, element, amplitude, length, gauss_sd, wave_function, simulate, frequency):
        """Like set_ao_voltage, but plays the pulse once instead of in a loop."""
        if not pulse and element:
            pulse = next(iter(config["elements"][element]["operations"]))

        if element is None:
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
            with program() as play_voltage:
                play(pulse, element)

            if simulate:
                self._simulate(play_voltage)
            else:
                self.qmm.open_qm(config).execute(play_voltage)
        else:
            self._append(["play", element, pulse, True, simulate])

        return element

    def set_digital_voltage(self, element, pulse, do_channel, length, delay, buffer, simulate):
        """Play a digital output's ON pulse (`length` ns high) in an infinite loop.

        Inside build_stack()/execute() it is played once at that point of the program instead.
        Pass `element` to reuse one; otherwise a new element is created on `do_channel`.
        """
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
                with infinite_loop_():
                    play(pulse, element)

            if simulate:
                self._simulate(set_voltage)
            else:
                self.qmm.open_qm(config).execute(set_voltage)
        else:
            self._append(["play", element, pulse, False, simulate])

        return element

    def get_ai_voltage(self, pulse=default_measure_pulse, ai_channel=None, element=None, ao_channel=None, amplitude=default_amplitude, length=default_length, wave_function=None, frequency=default_frequency, time_of_flight=default_time_of_flight, smearing=default_smearing, gauss_sd=default_length / 5, variable_name=None):
        """Drive `ao_channel` with a measurement pulse and record the raw trace on `ai_channel`.

        Run at once, returns [(timestamp, value), ...]. Inside build_stack()/execute() it is
        recorded instead; with `variable_name` the trace is also demodulated into that QUA
        variable, for use in if_gt/while_. Pass `element` to reuse a measurement element.
        """
        if element:
            ai_channel = config["elements"][element]["outputs"]["out1"][-1]    # the element's AI port

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
                with stream_processing():
                    trace = adc_stream.input1() if ai_channel == 1 else adc_stream.input2()
                    trace.with_timestamps().save('raw_adc')

            res = self.qmm.open_qm(config).execute(get_voltage).result_handles
            res.wait_for_all_values()
            raw = res.get("raw_adc").fetch_all()

            out = [(int(raw[i]["timestamp"]), float(raw[i]["value"]))
                   for i in range(raw.shape[0])]

            return out

        else:
            if variable_name:
                self._append(["measure_demod", element, ai_channel, variable_name])
            else:
                self._append(["measure", element, ai_channel])

        return element

    def execute(self, wait=False, timeout=None):
        """Turn the recorded stack into one QUA program and run (or simulate) it.

        With measurements: waits for them and returns {stream label: data}. Otherwise
        returns the job, after it is done if `wait` (`timeout` in s).
        """
        self.log.info(f"Time OPX execute starts {time.perf_counter_ns()}")

        self.adding_to_stack = False
        self.simulating = False

        measuring_flag = {"value": False}
        measure_counter = {"value": 0}
        streams = []

        with program() as curr_prog:
            # -----------------------------------------------------------------
            # 1) Auto-detect variables and their types (int vs fixed)
            # -----------------------------------------------------------------
            var_declarations = {}

            def collect_vars(stack):
                for op in stack:
                    kind = op[0]

                    if kind == "measure_demod":
                        # measure_demod stores result in a fixed variable
                        var_name = op[3]
                        var_declarations[var_name] = fixed

                    elif kind.startswith("if_"):
                        # if_gt compares a fixed variable
                        var_name = op[1]
                        if var_name not in var_declarations:
                            var_declarations[var_name] = fixed
                        collect_vars(op[3]) # then
                        collect_vars(op[4]) # else

                    elif kind == "for":
                        # for loops need an int variable
                        var_name = op[1]
                        var_declarations[var_name] = int
                        collect_vars(op[5]) # body

                    elif kind == "while":
                        # while comparisons usually use fixed
                        var_name = op[1]
                        if var_name not in var_declarations:
                            var_declarations[var_name] = fixed
                        collect_vars(op[4]) # body
                    elif kind == "measure_demod_iq":
                        var_declarations[op[2]] = fixed
                        var_declarations[op[3]] = fixed

            collect_vars(self.stack)

            # Declare the variables in QUA
            self._qua_vars = {name: declare(dtype) for name, dtype in var_declarations.items()}

            # -----------------------------------------------------------------
            # 2) Emit all operations
            # -----------------------------------------------------------------
            self._emit_stack(self.stack, streams, measuring_flag, measure_counter)

            # -----------------------------------------------------------------
            # 3) Stream processing
            # -----------------------------------------------------------------
            if streams:
                with stream_processing():
                    for s, input_idx, label in streams:
                        if label in average_streams:
                            s.buffer(average_streams[label]).average().save(label)
                        elif input_idx == 1:
                            s.input1().with_timestamps().save_all(label)
                        elif input_idx == 2:
                            s.input2().with_timestamps().save_all(label)
                        elif input_idx == -1:
                            # Variables (like I0) are saved simply
                            s.save_all(label)

        # Clear stack
        self.stack = []

        # ---------------------------------------------------------------------
        # 4) Run or simulate
        # ---------------------------------------------------------------------
        if self.simulating:
            self._simulate(curr_prog)
        elif measuring_flag["value"]:
            res = self.qmm.open_qm(config).execute(curr_prog).result_handles
            res.wait_for_all_values()

            out = {}
            # We now unpack input_idx to know how to parse the data
            for _, input_idx, label in streams:
                raw = res.get(label).fetch_all()
                if label in average_streams:
                    out[label] = np.asarray(raw)
                    continue

                # Case A: Variables (input_idx == -1) -> Needs flattening
                if input_idx == -1:
                    # Flatten tuples: [(-0.04,), (-0.05,)] -> [-0.04, -0.05]
                    processed_data = []
                    for item in raw:
                        # Check if it is a tuple/void and extract value
                        if hasattr(item, "__getitem__") and len(item) == 1:
                            processed_data.append(item[0])
                        else:
                            processed_data.append(item)

                    # If it's a single measurement, return just the float (SCALAR)
                    if len(processed_data) == 1:
                        out[label] = float(processed_data[0])
                    else:
                        out[label] = processed_data

                # Case B: ADC Traces (input_idx == 1 or 2) -> Structured data with timestamps
                else:
                    if isinstance(raw, (int, float)):
                        out[label] = float(raw)
                    elif len(raw) == 0:
                        out[label] = []
                    else:
                        raw = raw[0][0]
                        if hasattr(raw[0], 'dtype') and raw[0].dtype.names and 'timestamp' in raw[0].dtype.names:
                            out[label] = [(int(x["timestamp"]), float(x["value"])) for x in raw]
                        else:
                            # Fallback if structure is unexpected
                            out[label] = raw
            return out
        else:
            self.log.info(f"Time OPX compilation local done {time.perf_counter_ns()}")

            job = self.qmm.open_qm(config).execute(curr_prog)
            self.log.info(f"Time OPX job submitted to hardware {time.perf_counter_ns()}")

            if wait:
                job.wait_until("Done", timeout=timeout)

            return job

    def _emit_stack(self, stack, streams, measuring_flag, measure_counter):
        """Recursively emit QUA code."""
        for op in stack:
            kind = op[0]

            if kind == "measure":
                measuring_flag["value"] = True
                element, input_idx = op[1], op[2]
                measure_counter["value"] += 1
                s = declare_stream(adc_trace=True)
                measure("readout", element, adc_stream=s)
                # Input indices 1 or 2 imply ADC traces (timestamps)
                streams.append((s, input_idx, f"raw_adc_{measure_counter['value']}_in{input_idx}"))

            elif kind == "measure_demod":
                _, element, input_idx, var_name = op
                measuring_flag["value"] = True
                measure_counter["value"] += 1

                s = declare_stream(adc_trace=True)
                var_stream = declare_stream()

                # Retrieve the declared QUA variable
                I = self._qua_vars[var_name]

                measure("readout", element, s, demod.full("cos", I))
                save(I, var_stream)

                streams.append((s, input_idx, f"raw_adc_{measure_counter['value']}_in{input_idx}"))
                # Input index -1 implies a variable (NO timestamps)
                streams.append((var_stream, -1, var_name))

            elif kind == "play":
                element, pulse, simulate_flag = op[1], op[2], op[4]
                if simulate_flag:
                    self.simulating = True
                play(pulse, element)

            elif kind == "wait":
                cycles, elements = op[1], op[2]
                if elements:
                    wait(cycles, *elements)
                else:
                    wait(cycles)

            elif kind == "if_gt":
                _, var_name, threshold, then_stack, else_stack = op
                v = self._qua_vars[var_name]
                with if_(v > threshold):
                    self._emit_stack(then_stack, streams, measuring_flag, measure_counter)
                if else_stack:
                    with else_():
                        self._emit_stack(else_stack, streams, measuring_flag, measure_counter)

            elif kind == "for":
                # ["for", var_name, start, stop, step, body]
                _, var_name, start, stop, step, body = op
                i = self._qua_vars[var_name]

                # Handle positive vs negative steps
                if step > 0:
                    cond = (i < stop)
                    update = (i + step)
                else:
                    cond = (i > stop)
                    update = (i + step)

                with for_(i, start, cond, update):
                    self._emit_stack(body, streams, measuring_flag, measure_counter)

            elif kind == "while":
                # ["while", var_name, operator, threshold, body]
                _, var_name, operator, threshold, body = op
                v = self._qua_vars[var_name]

                if operator == "<":
                    cond = (v < threshold)
                elif operator == ">":
                    cond = (v > threshold)
                elif operator == ">=":
                    cond = (v >= threshold)
                elif operator == "<=":
                    cond = (v <= threshold)
                else:
                    self.log.error(f"Unknown operator {operator}")
                    continue

                with while_(cond):
                    self._emit_stack(body, streams, measuring_flag, measure_counter)

            elif kind == "update_freq":
                update_frequency(op[1], self._qua_vars[op[2]])

            elif kind == "align":
                align(*op[1])

            elif kind == "measure_demod_iq":
                _, element, var_I, var_Q = op
                measuring_flag["value"] = True
                measure_counter["value"] += 1
                I = self._qua_vars[var_I]
                Q = self._qua_vars[var_Q]
                I_stream = declare_stream()
                Q_stream = declare_stream()
                measure("readout", element, None,
                        demod.full("cos", I),
                        demod.full("sin", Q))
                save(I, I_stream)
                save(Q, Q_stream)
                streams.append((I_stream, -1, var_I))
                streams.append((Q_stream, -1, var_Q))

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

    def _get_reachable_if_node(self):
        """
        Finds the active IF node for the current scope.
        Returns the 'tip' node (the one we should attach elif/else to).
        """
        current_block = self._current_stack[-1]

        while self._if_stack:
            # We now store tuples: (anchor_node, tip_node)
            # anchor_node: The IF node that sits physically in the current block
            # tip_node: The active node in the chain (could be deep in elifs)
            anchor, tip = self._if_stack[-1]

            # Check if anchor is still in the current scope
            found = False
            for instr in reversed(current_block):
                if instr is anchor:
                    found = True
                    break

            if found:
                return tip
            else:
                # Stale nested IF from a closed scope
                self._if_stack.pop()

        raise RuntimeError("else_() or elif_() called, but no matching open IF block found in the current scope.")

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

    def for_(self, var_name, start, stop, step):
        """
        Creates a QUA for_ loop: for var_name in range(start, stop, step)
        The variable 'var_name' will be automatically declared as an 'int'.
        """
        return _ForContext(self, var_name, start, stop, step)

    def while_(self, var_name, operator, threshold):
        """
        Creates a QUA while loop.
        Usage: with c.while_("I0", "<", 0.5):
        """
        return _WhileContext(self, var_name, operator, threshold)

    def create_new_ao_elem(self, pulse, ao_channel=None, amplitude=default_amplitude, length=default_length, wave_function=None, frequency=default_frequency, gauss_sd=default_length / 5,):
        """Add an analog-output element with one pulse: 'const' (`amplitude` V), 'gauss'
        (sigma `gauss_sd` ns) or 'arbitrary' (`wave_function` samples), `length` ns long,
        modulated at `frequency` MHz. Returns the element name.
        """
        ao_channel = int(float(ao_channel))
        ao_port = opx_port(ao_channel, "ao")    # raises before anything is registered
        self.elem_counters[f"ao_{ao_channel}"] += 1
        idx = self.elem_counters[f"ao_{ao_channel}"]
        elem_name = f"generic_ai_elem_ch{ao_channel}_{idx}"    # "ai" is historical; this is an output
        self.elems.append(elem_name)

        config["elements"][elem_name] = {
            "singleInput": {"port": ao_port},
            "intermediate_frequency": frequency * u.MHz,
            "operations": {}
        }
        if pulse == "const":
            config["elements"][elem_name]["operations"]["const"] = f"square_pulse_ch{ao_channel}_{idx}"
            config["pulses"][f"square_pulse_ch{ao_channel}_{idx}"] = {
                "operation": "control",
                "length": length,
                "waveforms": {"single": f"square_wf_ch{ao_channel}_{idx}"},
            }
            config["waveforms"][f"square_wf_ch{ao_channel}_{idx}"] = {
                "type": "constant",
                "sample": amplitude,
            }

        elif pulse == "gauss":
            # A DRAG pulse with alpha=0 is a plain Gaussian; the derivative waveform is not used
            gauss_wf, _ = drag_gaussian_pulse_waveforms(
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
                "type": "arbitrary",    # QUA waveforms are "constant" or "arbitrary"
                "samples": gauss_wf,
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
                "samples": wave_function,
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
        """Add a digital-output element whose "ON" pulse is high for `length` ns.

        `delay` and `buffer` (ns) shift and widen the marker, as in QUA's digitalInputs.
        Returns the element name.
        """
        do_channel = int(float(do_channel))
        do_port = opx_port(do_channel, "do")    # raises before anything is registered
        self.elem_counters[f"do_{do_channel}"] += 1
        idx = self.elem_counters[f"do_{do_channel}"]
        elem_name = f"generic_di_elem_ch{do_channel}_{idx}"    # "di" is historical; this is an output
        pulse_name = f"ON_pulse_ch{do_channel}_{idx}"
        self.elems.append(elem_name)

        config["elements"][elem_name] = {
            "digitalInputs": {
                "activate": {
                    "port": do_port,
                    "delay": int(delay),
                    "buffer": int(buffer),
                }
            },
            "operations": {"ON": pulse_name},
        }
        config["pulses"][pulse_name] = {
            "operation": "control",
            "length": int(length),   # ns
            "digital_marker": "ON",  # refers to digital_waveforms["ON"]
        }

        self.log.info(f"Digital element created: {elem_name}")
        return elem_name

    def create_new_ai_elem(self, pulse=default_measure_pulse, ai_channel=None, ao_channel=None, amplitude=default_amplitude, length=default_length, wave_function=None, frequency=default_frequency, time_of_flight=default_time_of_flight, smearing=default_smearing, gauss_sd=default_length / 5,):
        """Add a measurement element: plays a "readout" pulse on `ao_channel` and records `ai_channel`.

        Pulse shapes as in create_new_ao_elem. `time_of_flight` and `smearing` (ns) are QUA's
        readout timing settings. Returns the element name.
        """
        ai_channel = int(float(ai_channel))
        ao_channel = int(float(ao_channel))
        ao_port, ai_port = opx_port(ao_channel, "ao"), opx_port(ai_channel, "ai")
        self.elem_counters[f"ai_{ao_channel}_{ai_channel}"] += 1
        idx = self.elem_counters[f"ai_{ao_channel}_{ai_channel}"]
        elem_name = f"generic_output_elem_ch{ao_channel}_ch{ai_channel}_{idx}"
        meas_pulse_name = f"meas_pulse_out_ch{ao_channel}_in_ch{ai_channel}_{idx}"
        self.elems.append(elem_name)

        config["elements"][elem_name] = {
            "singleInput": {"port": ao_port},
            "intermediate_frequency": frequency * u.MHz,
            "operations": {"readout": meas_pulse_name},
            "time_of_flight": int(time_of_flight),
            "smearing": int(smearing),
            "outputs": {"out1": ai_port},
        }

        if pulse == "const":
            wf_name = f"square_measure_wf_ch{ao_channel}_{idx}"
            config["waveforms"][wf_name] = {
                "type": "constant",
                "sample": float(amplitude),
            }

        elif pulse == "gauss":
            gauss_wf, _ = drag_gaussian_pulse_waveforms(
                amplitude, length, gauss_sd,
                alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
            )
            wf_name = f"gaussian_measure_wf_ch{ao_channel}_{idx}"
            config["waveforms"][wf_name] = {
                "type": "arbitrary",
                "samples": gauss_wf,
            }

        elif pulse == "arbitrary":
            wf_name = f"arbitrary_measure_wf_ch{ao_channel}_{idx}"
            config["waveforms"][wf_name] = {
                "type": "arbitrary",
                "samples": wave_function,
            }

        else:
            raise ValueError(f"Unknown pulse type for measurement: {pulse!r}")

        config["pulses"][meas_pulse_name] = {
            "operation": "measurement",
            "length": int(length),
            "waveforms": {"single": wf_name},
            # Defined in the config module; "cos" is the one measure_demod uses
            "integration_weights": {
                "cos": "cos",
                "sin": "sin",
                "minus_sin": "minus_sin",
            },
            "digital_marker": "ON",
        }

        return elem_name

    def delay(self, length, elements, simulate=False):
        """Wait `length` ns (a multiple of 4) on `elements`, or on all elements if none are given.

        Only useful inside build_stack()/execute(); on its own it runs a program that just
        waits. `simulate` is not used.
        """
        if length % 4 != 0:
            self.log.error(f"delay length must be a multiple of 4 ns, got {length}")
            raise ValueError("Length in delay function must be a multiple of 4")
        cycles = int(length / 4)
        if not self.adding_to_stack:
            with program() as wait_prog:
                if elements:
                    wait(cycles, *elements)
                else:
                    wait(cycles)
            self.qmm.open_qm(config).execute(wait_prog)
        else:
            self._append(["wait", cycles, elements])

    #Dynamically update the frequency of the oscillator associated with a given element. (Ex: for readout/qubit spectroscopy)
    def update_frequency(self, element, var_name):
        """Schedule update_frequency(element, var) inside the stack."""
        if not self.adding_to_stack:
            raise RuntimeError("update_frequency must be inside build_stack/execute")
        self._append(["update_freq", element, var_name])

    #For synchronizing operations (Ex: First do saturation pulse, then align the readout with the end of the saturation pulse, then do the measurement)
    def align(self, *elements):
        """Schedule align(...) inside the stack."""
        if not self.adding_to_stack:
            raise RuntimeError("align must be inside build_stack/execute")
        self._append(["align", list(elements)])

    #For simple demodulated readout.
    def measure_demod_iq(self, element, var_I, var_Q):
        """Dual-quadrature demodulated readout. Stores I and Q into named QUA vars."""
        if not self.adding_to_stack:
            raise RuntimeError("measure_demod_iq must be inside build_stack/execute")
        self._append(["measure_demod_iq", element, var_I, var_Q])


class _IfContext:
    def __init__(self, driver, op_name, var_name, threshold):
        self.driver = driver
        # Node format: ["if_gt", "I0", 0.1, then_stack, else_stack]
        self.node = [f"if_{op_name}", var_name, threshold, [], []]

    def __enter__(self):
        # Attach IF node to current stack
        self.driver._append(self.node)

        # Register this IF as "open".
        # tuple format: (ANCHOR, TIP)
        # For a fresh IF, it acts as both the anchor and the tip.
        self.driver._if_stack.append((self.node, self.node))

        # Route subsequent appends into the THEN branch
        then_stack = self.node[3]
        self.driver._stack_push(then_stack)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Leave THEN branch
        self.driver._stack_pop()


class _ElifContext:
    def __init__(self, driver, op_name, var_name, threshold):
        self.driver = driver
        self.op_name = op_name
        self.var_name = var_name
        self.threshold = threshold
        self.node = None

    def __enter__(self):
        # 1. Get the current tip of the chain
        # (This will clean up any stale nested IFs from the stack first)
        parent_tip = self.driver._get_reachable_if_node()

        # 2. Attach new node to the ELSE branch of the parent tip
        else_stack = parent_tip[4]
        then_stack = []
        self.node = [f"if_{self.op_name}", self.var_name, self.threshold, then_stack, []]
        else_stack.append(self.node)

        # 3. Update the stack to point to this new node as the tip
        # We MUST preserve the original anchor so reachability checks still pass!
        anchor, _ = self.driver._if_stack[-1]
        self.driver._if_stack[-1] = (anchor, self.node)

        # Route subsequent appends into the THEN branch
        self.driver._stack_push(then_stack)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.driver._stack_pop()


class _ElseContext:
    def __init__(self, driver):
        self.driver = driver
        self.node = None

    def __enter__(self):
        # 1. Get the current tip
        self.node = self.driver._get_reachable_if_node()

        # 2. Route subsequent appends into its ELSE branch
        else_stack = self.node[4]
        self.driver._stack_push(else_stack)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Leave ELSE branch
        self.driver._stack_pop()
        # IF chain is now fully closed, pop from tracking stack
        self.driver._if_stack.pop()


class _ForContext:
    def __init__(self, driver, var_name, start, stop, step):
        self.driver = driver
        # Node: ["for", var_name, start, stop, step, body_stack]
        self.node = ["for", var_name, start, stop, step, []]

    def __enter__(self):
        self.driver._append(self.node)
        # Push the body stack so future commands go inside the loop
        self.driver._stack_push(self.node[5])
        return self

    def __exit__(self, exc_type, exc, tb):
        self.driver._stack_pop()


class _WhileContext:
    def __init__(self, driver, var_name, operator, threshold):
        self.driver = driver
        # Node: ["while", var_name, operator, threshold, body_stack]
        self.node = ["while", var_name, operator, threshold, []]

    def __enter__(self):
        self.driver._append(self.node)
        self.driver._stack_push(self.node[4])
        return self

    def __exit__(self, exc_type, exc, tb):
        self.driver._stack_pop()
