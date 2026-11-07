"""Finite LF-FEM analog I/O driver derived from the sequence structure in OPX.py.

Config/stack creation needs only Python's standard library. QUA program building
and execution import qm-qua lazily. No connection is made by the constructor.
"""

from collections import defaultdict
from copy import deepcopy
from math import exp, isfinite
from numbers import Integral

from . import OPX1000Config as opx_cfg


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError("{} must be a finite number".format(name))
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a finite number".format(name)) from None
    if not isfinite(value):
        raise ValueError("{} must be a finite number".format(name))
    return value


def _duration_ns(value):
    value = _integer(value, "duration in ns", minimum=16)
    if value % 4:
        raise ValueError("Duration must be a multiple of 4 ns")
    return value


def _qua():
    try:
        from qm import qua
    except ImportError as exc:
        raise ImportError(
            "QUA program building requires qm-qua. Config/stack creation works "
            "without it; see OPX1000_README.md for environment setup."
        ) from exc
    return qua


class Driver:
    def __init__(
        self, device_name="OPX1000", logger=None, dummy=False, *,
        qop_ip=None, cluster_name=None, hardware_config=None,
    ):
        self.dev = device_name
        self.log = None
        if logger is not None:
            from pylabnet.utils.logging.logger import LogHandler
            self.log = LogHandler(logger=logger)
        self.dummy = dummy
        self.qop_ip = opx_cfg.qop_ip if qop_ip is None else qop_ip
        self.cluster_name = opx_cfg.cluster_name if cluster_name is None else cluster_name
        self._base_config = deepcopy(
            opx_cfg.config if hardware_config is None else hardware_config
        )
        self.config = deepcopy(self._base_config)
        self.elem_counters = defaultdict(int)
        self.elems = []
        self.stack = []
        self.adding_to_stack = False
        self.qmm = None
        self.qm = None
        self.job = None
        self._result_specs = {}
        self._last_result_specs = {}
        self._result_job = None

    def build_stack(self):
        """Start a fresh sequence, resetting generated definitions and counters.

        Does not stop an already submitted hardware job. Any custom persistent
        definitions must be supplied in hardware_config at construction time.
        """
        self.config = deepcopy(self._base_config)
        self.elem_counters.clear()
        self.elems = []
        self.stack = []
        self._result_specs = {}
        self.adding_to_stack = True

    def _require_stack(self):
        if not self.adding_to_stack:
            raise RuntimeError("Call build_stack() before queuing operations")

    def create_new_ao_elem(
        self, pulse="const", ao_channel=None, amplitude=opx_cfg.default_amplitude,
        length=opx_cfg.default_length, wave_function=None,
        frequency=opx_cfg.default_frequency, gauss_sd=None, *,
        fem_slot=opx_cfg.default_fem_slot, controller=opx_cfg.default_controller,
    ):
        """Create an LF AO element plus pulse/waveform; return its unique name.

        ao_channel: physical LF output port (1-8); fem_slot: chassis slot (3/4).
        length and gauss_sd: ns. frequency: MHz, converted to integer Hz in QUA.
        amplitude: volts for const/gauss. For arbitrary, wave_function contains
        absolute voltage samples (amplitude is unused), one sample/ns at 1 GS/s.
        Gaussian endpoints are zero and its sampled peak equals amplitude.
        This first skeleton supports 1 GS/s LF ports; MW output is separate.
        """
        channel = _integer(ao_channel, "ao_channel")
        slot = _integer(fem_slot, "fem_slot")
        try:
            controller_config = self.config["controllers"][controller]
            fem = controller_config["fems"][slot]
        except KeyError:
            raise ValueError("Unknown controller/FEM: {} slot {}".format(controller, slot)) from None
        if controller_config.get("type") != "opx1000":
            raise ValueError("Controller must be configured as opx1000")
        if fem["type"] != "LF":
            raise ValueError("DC/AO creation requires an LF-FEM; MWInput is not supported")
        if channel not in range(1, 9) or channel not in fem.get("analog_outputs", {}):
            raise ValueError("ao_channel must select a configured LF output port 1-8")
        port_config = fem["analog_outputs"][channel]
        if port_config.get("sampling_rate", 1e9) != 1e9:
            raise ValueError("This skeleton supports 1 GS/s LF ports only")
        mode = port_config.get("output_mode", "direct")
        if mode not in ("direct", "amplified"):
            raise ValueError("LF output_mode must be direct or amplified")
        length = _duration_ns(length)
        frequency_hz = _finite(frequency, "frequency in MHz") * 1e6
        if not frequency_hz.is_integer() or abs(frequency_hz) >= 500e6:
            raise ValueError("Frequency must resolve to integer Hz with magnitude < 500 MHz")
        if mode == "amplified" and abs(frequency_hz) > 330e6:
            raise ValueError("Amplified LF output bandwidth is limited to 330 MHz")

        if pulse == "const":
            samples = [_finite(amplitude, "amplitude")]
            waveform = {"type": "constant", "sample": samples[0]}
        elif pulse == "gauss":
            amplitude = _finite(amplitude, "amplitude")
            sigma = length / 5 if gauss_sd is None else _finite(gauss_sd, "gauss_sd")
            if sigma <= 0:
                raise ValueError("gauss_sd must be positive")
            center = (length - 1) / 2
            envelope = [exp(-0.5 * ((i - center) / sigma) ** 2) for i in range(length)]
            edge = envelope[0]
            span = max(envelope) - edge
            if span <= 0:
                raise ValueError("gauss_sd cannot be represented at this pulse length")
            samples = [amplitude * ((value - edge) / span) for value in envelope]
            waveform = {"type": "arbitrary", "samples": samples}
        elif pulse == "arbitrary":
            if wave_function is None:
                raise ValueError("Arbitrary pulses require wave_function voltage samples")
            samples = [_finite(value, "wave_function sample") for value in wave_function]
            if len(samples) != length:
                raise ValueError("At 1 GS/s, wave_function needs exactly length samples")
            waveform = {"type": "arbitrary", "samples": samples}
        else:
            raise ValueError("pulse must be const, gauss, or arbitrary")

        limit = 2.5 if mode == "amplified" else 0.5
        offset = _finite(port_config.get("offset", 0.0), "port offset")
        low, high = min(samples), max(samples)
        if frequency_hz:
            # Modulation can swing either way, even for a positive envelope.
            high = max(abs(value) for value in samples)
            low = -high
        if (min(samples) < -limit or max(samples) >= limit
                or low + offset < -limit or high + offset >= limit):
            raise ValueError("Waveform plus offset exceeds LF {} range [-{}, {}) V".format(
                mode, limit, limit
            ))

        key = (controller, slot, channel)
        while True:
            self.elem_counters[key] += 1
            name = "ao_{}_fem{}_ch{}_{}".format(
                controller, slot, channel, self.elem_counters[key]
            )
            pulse_name, waveform_name = name + "_pulse", name + "_wf"
            if (name not in self.config["elements"]
                    and pulse_name not in self.config["pulses"]
                    and waveform_name not in self.config["waveforms"]):
                break
        self.config["elements"][name] = {
            "singleInput": {"port": (controller, slot, channel)},
            "intermediate_frequency": int(frequency_hz),
            "operations": {pulse: pulse_name},
        }
        self.config["pulses"][pulse_name] = {
            "operation": "control", "length": length,
            "waveforms": {"single": waveform_name},
        }
        self.config["waveforms"][waveform_name] = waveform
        self.elems.append(name)
        return name

    def set_ao_voltage(
        self, pulse=None, ao_channel=None, element=None,
        amplitude=opx_cfg.default_amplitude, length=opx_cfg.default_length,
        wave_function=None, frequency=opx_cfg.default_frequency, gauss_sd=None, *,
        fem_slot=opx_cfg.default_fem_slot, controller=opx_cfg.default_controller,
    ):
        """Queue ONE finite pulse; return its element name.

        An existing element uses its configured operation/waveform/duration.
        Creation parameters apply only when element is None. Calls on different
        elements may overlap; use align()/delay() for explicit synchronization.
        """
        self._require_stack()
        if element is None:
            pulse = "const" if pulse is None else pulse
            element = self.create_new_ao_elem(
                pulse=pulse, ao_channel=ao_channel, amplitude=amplitude,
                length=length, wave_function=wave_function, frequency=frequency,
                gauss_sd=gauss_sd, fem_slot=fem_slot, controller=controller,
            )
        if element not in self.config["elements"]:
            raise ValueError("Unknown element: {}".format(element))
        operations = self.config["elements"][element]["operations"]
        if pulse is None:
            if len(operations) != 1:
                raise ValueError("Specify pulse when an element has multiple operations")
            pulse = next(iter(operations))
        if pulse not in operations:
            raise ValueError("Element has no operation named {}".format(pulse))
        self.stack.append(("play", element, pulse))
        return element

    def pulse_train(
        self, ao_channel, pulse_count, period_ns, amplitude=opx_cfg.default_amplitude,
        length=opx_cfg.default_length, *, fem_slot=opx_cfg.default_fem_slot,
        controller=opx_cfg.default_controller,
    ):
        """Queue a finite constant-pulse train in QUA strict timing.

        Period and high time are ns. The low interval must be zero or >=16 ns,
        and a multiple of 4 ns. Includes a final low interval. Repeats happen
        on hardware; no Python/RPC loop determines the pulse spacing.
        """
        self._require_stack()
        pulse_count = _integer(pulse_count, "pulse_count")
        period_ns, length = _duration_ns(period_ns), _duration_ns(length)
        gap = period_ns - length
        if pulse_count > 2**31 - 1 or period_ns // 4 > 2**31 - 1:
            raise ValueError("Train exceeds the QUA integer range")
        if gap < 0 or (gap and gap < 16):
            raise ValueError("period_ns must exceed length by >=16 ns, or equal length")
        element = self.create_new_ao_elem(
            "const", ao_channel=ao_channel, amplitude=amplitude, length=length,
            frequency=0, fem_slot=fem_slot, controller=controller,
        )
        self.stack.append(("pulse_train", element, pulse_count, gap // 4))
        return element

    def _new_ai_element(self, ai_channel, length, fem_slot, controller, time_of_flight,
                        threshold_adc=None, polarity="Below"):
        """Create an input-only measurement element: no AO or digital port."""
        channel = _integer(ai_channel, "ai_channel")
        slot = _integer(fem_slot, "fem_slot")
        length = _duration_ns(length)
        time_of_flight = _integer(time_of_flight, "time_of_flight", minimum=24)
        if time_of_flight % 4:
            raise ValueError("time_of_flight must be divisible by 4 ns")
        try:
            control = self.config["controllers"][controller]
            fem = control["fems"][slot]
        except KeyError:
            raise ValueError("Unknown controller/FEM") from None
        if control.get("type") != "opx1000" or fem["type"] != "LF":
            raise ValueError("Acquisition requires an OPX1000 LF-FEM")
        if channel not in (1, 2) or channel not in fem.get("analog_inputs", {}):
            raise ValueError("Select a configured LF analog input 1 or 2")
        if fem["analog_inputs"][channel].get("sampling_rate", 1e9) != 1e9:
            raise ValueError("Acquisition currently supports 1 GS/s LF inputs only")
        if threshold_adc is not None:
            threshold_adc = _integer(threshold_adc, "threshold_adc", minimum=-2048)
            if threshold_adc > 2047 or polarity not in ("Above", "Below"):
                raise ValueError("Threshold must be -2048..2047; polarity Above or Below")

        marker = "opx1000_adc_gate"
        marker_value = {"samples": [(1, 0)]}
        markers = self.config.get("digital_waveforms", {})
        if marker in markers and markers[marker] != marker_value:
            raise ValueError("Reserved ADC gate waveform has conflicting contents")
        key = ("ai", controller, slot, channel)
        while True:
            self.elem_counters[key] += 1
            name = "ai_{}_fem{}_ch{}_{}".format(controller, slot, channel, self.elem_counters[key])
            if name not in self.config["elements"] and name + "_pulse" not in self.config["pulses"]:
                break
        output = "out{}".format(channel)
        element = {
            "outputs": {output: (controller, slot, channel)},
            "time_of_flight": time_of_flight, "smearing": 0,
            "operations": {"readout": name + "_pulse"},
        }
        if threshold_adc is not None:
            element["timeTaggingParameters"] = {
                "signalThreshold": threshold_adc, "signalPolarity": polarity,
                "derivativeThreshold": -2048, "derivativePolarity": "Above",
            }
        self.config["elements"][name] = element
        self.config["pulses"][name + "_pulse"] = {
            "operation": "measurement", "length": length, "digital_marker": marker,
        }
        self.config.setdefault("digital_waveforms", {})[marker] = marker_value
        self.elems.append(name)
        return name, channel, output

    def _check_result_label(self, label):
        self._require_stack()
        if not isinstance(label, str) or not label.isidentifier() or "__" in label:
            raise ValueError("Result label must be an identifier without double underscores")
        if label in self._result_specs:
            raise ValueError("Duplicate result label: " + label)

    def get_ai_voltage(
        self, ai_channel, length=10000, *, fem_slot=opx_cfg.default_fem_slot,
        controller=opx_cfg.default_controller, time_of_flight=24, label="raw_adc",
    ):
        """Queue a short raw ADC capture (16..16000 ns, 4 ns increments).

        Results are ADC codes, not calibrated connector volts. No AO waveform
        or physical digital output is configured by this operation.
        """
        self._check_result_label(label)
        length = _duration_ns(length)
        if length > 16000:
            raise ValueError("Raw trace capture is limited to 16000 ns in this interface")
        element, channel, _ = self._new_ai_element(
            ai_channel, length, fem_slot, controller, time_of_flight
        )
        self._result_specs[label] = {
            "kind": "adc", "length_ns": length, "sample_rate_hz": 1e9,
        }
        self.stack.append(("measure_adc", element, channel, label))
        return element

    def count_ai_pulses(
        self, ai_channel, num_windows, window_ns=1000000, threshold_adc=-500,
        polarity="Below", max_tags=1024, *, fem_slot=opx_cfg.default_fem_slot,
        controller=opx_cfg.default_controller, time_of_flight=24, label="counts",
    ):
        """Queue repeated hardware threshold-counting gates on an LF input.

        Nominal live time is num_windows * window_ns, excluding gaps. Counts
        and start timestamps are saved per gate. Full tag arrays are flagged:
        such results cannot establish an accurate count rate/live time.
        Threshold/polarity are in internal signed ADC codes; verify with trace.
        """
        self._check_result_label(label)
        num_windows = _integer(num_windows, "num_windows")
        window_ns = _duration_ns(window_ns)
        max_tags = _integer(max_tags, "max_tags")
        if num_windows > 1000000 or window_ns > 1000000000 or max_tags > 65536:
            raise ValueError("Limits: 1e6 windows, 1 second/gate, 65536 tags/gate")
        element, _, output = self._new_ai_element(
            ai_channel, window_ns, fem_slot, controller, time_of_flight,
            threshold_adc, polarity,
        )
        self._result_specs[label] = {
            "kind": "counts", "num_windows": num_windows, "window_ns": window_ns,
            "max_tags": max_tags, "threshold_adc": int(threshold_adc),
            "polarity": polarity,
        }
        self.stack.append(("count_adc", element, output, label))
        return element

    def get_results(self, timeout=10):
        """Fetch completed acquisitions as plain values; no QM objects returned.

        Count timestamps are QUA clock cycles, converted to ns here. Rates use
        gate live time, not host wall time. Saturated runs return rate/live time
        as None rather than a misleading dark-count estimate.
        """
        if self.job is None or self.job is not self._result_job or not self._last_result_specs:
            raise RuntimeError("No submitted acquisition results are available")
        if self.job.get_status() != "Done":
            raise RuntimeError("Wait for the acquisition job to finish before fetching results")
        if not self.job.result_handles.wait_for_all_values(timeout=timeout):
            raise RuntimeError("Acquisition result streams did not complete")
        import numpy as np

        def values(label):
            handle = self.job.result_handles.get(label)
            if handle is None:
                raise RuntimeError("Missing result stream: " + label)
            data = handle.fetch_all(flat_struct=True)
            if data is None:
                raise RuntimeError("Empty result stream: " + label)
            if isinstance(data, dict):
                data = data["value"]
            data = np.asarray(data)
            if data.dtype.names and "value" in data.dtype.names:
                data = data["value"]
            return data.reshape(-1)

        results = {}
        for label, spec in self._last_result_specs.items():
            item = dict(spec)
            if spec["kind"] == "adc":
                raw = values(label)
                if raw.size != spec["length_ns"]:
                    raise RuntimeError("ADC sample count does not match the requested duration")
                item["raw_adc"] = raw.tolist()
            else:
                counts = values(label).astype(np.int64)
                starts = values(label + "__starts").astype(np.int64)
                expected, window = spec["num_windows"], spec["window_ns"]
                if counts.size != expected or starts.size != expected:
                    raise RuntimeError("Missing counting gates; refusing a partial rate estimate")
                if np.any(counts < 0) or np.any(np.diff(starts) < 0):
                    raise RuntimeError("Invalid count data or non-monotonic timestamps")
                start_ns = (starts - starts[0]) * 4
                gaps = np.diff(start_ns) - window
                saturated = counts >= spec["max_tags"]
                invalid = bool(np.any(saturated))
                live_s = expected * window * 1e-9
                item.update(
                    counts=counts.tolist(), window_start_ns=start_ns.tolist(),
                    saturated=saturated.tolist(), total_counts=int(counts.sum()),
                    requested_live_time_s=live_s,
                    live_time_s=None if invalid else live_s,
                    count_rate_hz=None if invalid else int(counts.sum()) / live_s,
                    measurement_span_s=None if invalid else float(start_ns[-1] + window) * 1e-9,
                    inter_window_gap_s=None if invalid else float(np.maximum(gaps, 0).sum()) * 1e-9,
                    timing_overlap=bool(np.any(gaps < 0)),
                    valid=not invalid and not bool(np.any(gaps < 0)),
                )
                if item["timing_overlap"]:
                    item["count_rate_hz"] = None
                    item["live_time_s"] = None
            results[label] = item
        return results

    def _elements(self, elements):
        # Accept a single name as well as an iterable of names.
        elements = (elements,) if isinstance(elements, str) else tuple(elements or ())
        for element in elements:
            if element not in self.config["elements"]:
                raise ValueError("Unknown element: {}".format(element))
        return elements

    def delay(self, length, elements=None):
        """Queue a wait in ns (>=16, multiple of 4); None waits on all elements."""
        self._require_stack()
        self.stack.append(("wait", _duration_ns(length) // 4, self._elements(elements)))

    def align(self, *elements):
        """Queue synchronization of named elements, or all elements if omitted."""
        self._require_stack()
        self.stack.append(("align", self._elements(elements)))

    def _emit_stack(self, qua):
        for op in self.stack:
            if op[0] == "play":
                qua.play(op[2], op[1])
            elif op[0] == "wait":
                qua.wait(op[1], *op[2])
            elif op[0] == "align":
                qua.align(*op[1])

            elif op[0] == "pulse_train":
                _, element, repeat_count, gap_cycles = op
                index = qua.declare(int)
                with qua.strict_timing_():
                    with qua.for_(index, 0, index < repeat_count, index + 1):
                        qua.play("const", element)
                        if gap_cycles:
                            qua.wait(gap_cycles, element)
            elif op[0] == "measure_adc":
                _, element, channel, label = op
                stream = qua.declare_stream(adc_trace=True)
                qua.measure("readout", element, adc_stream=stream)
                self._qua_streams[label] = (stream, channel)
            elif op[0] == "count_adc":
                _, element, output, label = op
                spec = self._result_specs[label]
                index = qua.declare(int)
                counts = qua.declare(int)
                tags = qua.declare(int, size=spec["max_tags"])
                stream, starts = qua.declare_stream(), qua.declare_stream()
                with qua.for_(index, 0, index < spec["num_windows"], index + 1):
                    qua.measure(
                        "readout", element,
                        qua.time_tagging.analog(tags, spec["window_ns"], counts, output),
                        timestamp_stream=starts,
                    )
                    qua.save(counts, stream)
                self._qua_streams[label] = (stream, 0)
                self._qua_streams[label + "__starts"] = (starts, 0)
            else:
                raise ValueError("Unknown stack operation: {}".format(op[0]))

    def build_program(self):
        """Build a QUA program locally without submitting or consuming the stack.

        Requires qm-qua. This builds the QUA representation; full compilation
        and hardware timing validation still require QOP/simulator access.
        """
        self._require_stack()
        if not any(op[0] in ("play", "pulse_train", "measure_adc", "count_adc") for op in self.stack):
            raise RuntimeError("Queue at least one pulse or acquisition before building a program")
        qua = _qua()
        self._qua_streams = {}
        with qua.program() as sequence:
            self._emit_stack(qua)
            if self._qua_streams:
                with qua.stream_processing():
                    for label, (stream, channel) in self._qua_streams.items():
                        if channel:
                            selected = stream.input1() if channel == 1 else stream.input2()
                            selected.save(label)
                        else:
                            stream.save_all(label)
        return sequence

    def _get_qmm(self):
        if self.dummy:
            raise RuntimeError("dummy=True allows offline building only")
        if not self.qop_ip:
            raise ValueError("Set qop_ip to the QM router address before connecting")
        if self.qmm is None:
            from qm import QuantumMachinesManager
            self.qmm = QuantumMachinesManager(
                host=self.qop_ip, cluster_name=self.cluster_name
            )
        return self.qmm

    def _simulate(self, sequence, duration=4000):
        """Return a simulator job; duration is ns, converted to 4 ns clock cycles.

        Uses QOP simulation through QMM; this is not an offline simulator.
        Does not open a hardware QM or plot/save files automatically.
        """
        duration = _duration_ns(duration)
        from qm import SimulationConfig
        return self._get_qmm().simulate(
            self.config, sequence, SimulationConfig(duration=duration // 4)
        )

    def execute(self, wait=True, timeout=None, *, simulate=False, duration=4000):
        """Submit the finite stack and return a job; timeout is seconds.

        A hardware submission replaces this driver's previous QM (and stops any
        job still running in it). Other clients' QMs are not closed.
        The stack is consumed after successful submission, even if waiting fails.
        Simulation consumes the stack too, but does not replace a hardware QM.
        """
        if self.dummy:
            raise RuntimeError("dummy=True allows offline building only")
        sequence = self.build_program()
        if simulate:
            job = self._simulate(sequence, duration=duration)
        else:
            qmm = self._get_qmm()
            if self.qm is not None:
                self.qm.close()
                self.qm = None
            self.qm = qmm.open_qm(self.config, close_other_machines=False)
            job = self.qm.execute(sequence)
        self.job = job
        self._result_job = job
        self._last_result_specs = deepcopy(self._result_specs)
        self.stack = []
        self.adding_to_stack = False
        if wait and not simulate:
            job.wait_until("Done", timeout=timeout)
        return job

    def close(self):
        """Close only this driver's QM and manager connection; stop its jobs."""
        if self.qm is not None:
            self.qm.close()
            self.qm = None
        if self.qmm is not None:
            # qmm.close() was introduced in qm-qua 1.3.1.
            if hasattr(self.qmm, "close"):
                self.qmm.close()
            self.qmm = None
        self.job = None
        self._result_job = None
        self._last_result_specs = {}
