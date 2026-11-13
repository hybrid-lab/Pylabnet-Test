"""Pylabnet RPC interface for the finite LF-FEM OPX1000 driver.

The server owns the driver, pending sequence, and QM jobs. Use one experiment
at a time per server: separate clients do not get independent sequence stacks.
Complex results travel as pickled values, following other Pylabnet wrappers;
QM job/program objects never leave the server.
"""

import pickle

from pylabnet.hardware.quantum_machines import OPX1000Config as defaults
from pylabnet.network.core.client_base import ClientBase
from pylabnet.network.core.service_base import ServiceBase


def _decode_waveform(kwargs):
    """Materialize waveform samples locally instead of retaining an RPC proxy."""
    if kwargs.get("wave_function") is not None:
        kwargs["wave_function"] = pickle.loads(kwargs["wave_function"])
    return kwargs


class Service(ServiceBase):
    def assign_module(self, module):
        super().assign_module(module)
        self._simulated_job = None

    def exposed_build_stack(self):
        return self._module.build_stack()

    def exposed_create_new_ao_elem(self, **kwargs):
        return self._module.create_new_ao_elem(**_decode_waveform(kwargs))

    def exposed_set_ao_voltage(self, **kwargs):
        return self._module.set_ao_voltage(**_decode_waveform(kwargs))

    def exposed_delay(self, length, elements=None):
        if elements is not None and not isinstance(elements, str):
            elements = tuple(elements)
        return self._module.delay(length=length, elements=elements)

    def exposed_align(self, *elements):
        return self._module.align(*elements)

    def exposed_execute(self, wait=True, timeout=None, simulate=False, duration=4000):
        job = self._module.execute(
            wait=wait, timeout=timeout, simulate=simulate, duration=duration
        )
        self._simulated_job = job if simulate else None
        return pickle.dumps({"job_id": str(job.id), "simulated": bool(simulate)})

    def exposed_pulse_train(self, **kwargs):
        return self._module.pulse_train(**kwargs)

    def exposed_get_ai_voltage(self, **kwargs):
        return self._module.get_ai_voltage(**kwargs)

    def exposed_count_ai_pulses(self, **kwargs):
        return self._module.count_ai_pulses(**kwargs)

    def exposed_get_results(self, timeout=10):
        if self._module.job is not None and self._module.job is self._simulated_job:
            raise RuntimeError("Acquisition results require a hardware run")
        return pickle.dumps(self._module.get_results(timeout=timeout))

    def exposed_get_status(self):
        job = self._module.job
        simulated = job is not None and job is self._simulated_job
        status = None
        if job is not None:
            status = job.get_status()
        return pickle.dumps({
            "job_id": None if job is None else str(job.id),
            "simulated": simulated,
            "status": status,
        })

    def exposed_get_config(self):
        """Return a snapshot for inspection; editing it cannot change the driver."""
        return pickle.dumps(self._module.config)

    def exposed_get_simulated_samples(self):
        job = self._module.job
        if job is None or job is not self._simulated_job:
            raise RuntimeError("Run execute(simulate=True) before fetching simulated samples")
        result = {}
        for controller, samples in job.get_simulated_samples().items():
            analog = {
                str(port): values.tolist() if hasattr(values, "tolist") else list(values)
                for port, values in samples.analog.items()
            }
            digital = {
                str(port): values.tolist() if hasattr(values, "tolist") else list(values)
                for port, values in samples.digital.items()
            }
            rates = samples.analog_sampling_rate
            result[str(controller)] = {
                "analog": analog,
                "digital": digital,
                "analog_sampling_rate": {
                    port: float(rates.get(port, 1e9)) for port in analog
                },
            }
        return pickle.dumps(result)

    def exposed_close(self):
        """Release this driver's QM/QMM; keep the Pylabnet server running."""
        self._module.close()
        self._simulated_job = None

    def close_server(self):
        """Release QM resources before the usual Pylabnet process shutdown."""
        try:
            self.exposed_close()
        finally:
            super().close_server()


class Client(ClientBase):
    """Experiment-facing methods; host/port identify the Pylabnet server.

    Constructor and SSL handling are inherited from ClientBase. Timing,
    amplitudes and frequencies use the driver units: ns, V and MHz.
    """

    def build_stack(self):
        return self._service.exposed_build_stack()

    def create_new_ao_elem(
        self, pulse="const", ao_channel=None, amplitude=defaults.default_amplitude,
        length=defaults.default_length, wave_function=None,
        frequency=defaults.default_frequency, gauss_sd=None, *,
        fem_slot=defaults.default_fem_slot, controller=defaults.default_controller,
    ):
        return self._service.exposed_create_new_ao_elem(
            pulse=pulse, ao_channel=ao_channel, amplitude=amplitude, length=length,
            wave_function=None if wave_function is None else pickle.dumps(list(wave_function)),
            frequency=frequency, gauss_sd=gauss_sd,
            fem_slot=fem_slot, controller=controller,
        )

    def set_ao_voltage(
        self, pulse=None, ao_channel=None, element=None,
        amplitude=defaults.default_amplitude, length=defaults.default_length,
        wave_function=None, frequency=defaults.default_frequency, gauss_sd=None, *,
        fem_slot=defaults.default_fem_slot, controller=defaults.default_controller,
    ):
        return self._service.exposed_set_ao_voltage(
            pulse=pulse, ao_channel=ao_channel, element=element,
            amplitude=amplitude, length=length,
            wave_function=None if wave_function is None else pickle.dumps(list(wave_function)),
            frequency=frequency, gauss_sd=gauss_sd,
            fem_slot=fem_slot, controller=controller,
        )

    def delay(self, length, elements=None):
        return self._service.exposed_delay(length=length, elements=elements)

    def align(self, *elements):
        return self._service.exposed_align(*elements)

    def execute(self, wait=True, timeout=None, *, simulate=False, duration=4000):
        """Return {'job_id': str, 'simulated': bool}; the job stays on the server.

        A successful submission consumes the stack. For wait=False, poll
        get_status(). timeout is the hardware completion wait in seconds;
        a timeout does not cancel the submitted job.
        """
        return pickle.loads(self._service.exposed_execute(
            wait=wait, timeout=timeout, simulate=simulate, duration=duration
        ))

    def pulse_train(
        self, ao_channel, pulse_count, period_ns, amplitude=defaults.default_amplitude,
        length=defaults.default_length, *, fem_slot=defaults.default_fem_slot,
        controller=defaults.default_controller,
    ):
        """Queue a finite constant-pulse train; width/period are ns."""
        return self._service.exposed_pulse_train(
            ao_channel=ao_channel, pulse_count=pulse_count, period_ns=period_ns,
            amplitude=amplitude, length=length, fem_slot=fem_slot, controller=controller,
        )

    def get_ai_voltage(
        self, ai_channel, length=10000, *, fem_slot=defaults.default_fem_slot,
        controller=defaults.default_controller, time_of_flight=28, label="raw_adc",
    ):
        """Queue an input-only raw ADC capture; length is ns (maximum 16000)."""
        return self._service.exposed_get_ai_voltage(
            ai_channel=ai_channel, length=length, fem_slot=fem_slot,
            controller=controller, time_of_flight=time_of_flight, label=label,
        )

    def count_ai_pulses(
        self, ai_channel, num_windows, window_ns=1000000, threshold_adc=-500,
        polarity="Below", max_tags=1024, *, fem_slot=defaults.default_fem_slot,
        controller=defaults.default_controller, time_of_flight=28, label="counts",
    ):
        """Queue hardware threshold counting; threshold/polarity use ADC codes."""
        return self._service.exposed_count_ai_pulses(
            ai_channel=ai_channel, num_windows=num_windows, window_ns=window_ns,
            threshold_adc=threshold_adc, polarity=polarity, max_tags=max_tags,
            fem_slot=fem_slot, controller=controller, time_of_flight=time_of_flight,
            label=label,
        )

    def get_results(self, timeout=10):
        """Fetch completed ADC/count streams as local dictionaries/lists."""
        return pickle.loads(self._service.exposed_get_results(timeout=timeout))

    def get_status(self):
        """Return job_id, simulated and status; status is None before execution."""
        return pickle.loads(self._service.exposed_get_status())

    def get_config(self):
        """Return an independent copy of the server's current QUA configuration."""
        return pickle.loads(self._service.exposed_get_config())

    def get_simulated_samples(self):
        """Return ordinary dictionaries/lists from the most recent simulation.

        Example: result["con1"]["analog"]["3-1"] is the slot-3/port-1 trace.
        analog_sampling_rate maps each analog port to its rate in Hz.
        """
        return pickle.loads(self._service.exposed_get_simulated_samples())

    def close(self):
        """Stop/release the driver's QM resources; leave the RPC server available.

        This affects the shared driver, including jobs submitted by other clients.
        It does not disconnect this client. close_server() stops the whole server.
        """
        return self._service.exposed_close()
