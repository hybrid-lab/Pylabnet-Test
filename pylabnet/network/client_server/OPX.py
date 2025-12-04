from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase
from pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel import default_amplitude, default_digital_pulse, default_frequency, default_length, default_measure_pulse, default_smearing, defaut_time_of_flight


class Service(ServiceBase):

    def exposed_set_digital_voltage(
        self,
        element,
        pulse,
        do_channel,
        simulate,
        length,
        delay,
        buffer,
    ):
        return self._module.set_digital_voltage(
            element=element,
            do_channel=do_channel,
            pulse=pulse,
            simulate=simulate,
            length=length,
            delay=delay,
            buffer=buffer,
        )

    def exposed_set_ao_voltage(self, pulse, ao_channel, element, amplitude, length, wave_function, frequency, simulate, gauss_sd):
        return self._module.set_ao_voltage(
            pulse=pulse,
            ao_channel=ao_channel,
            amplitude=amplitude,
            length=length,
            wave_function=wave_function,
            frequency=frequency,
            element=element,
            simulate=simulate,
            gauss_sd=gauss_sd,
        )

    def exposed_play_ao_voltage(self, pulse, ao_channel, element, amplitude, length, wave_function, frequency, simulate, gauss_sd):
        return self._module.play_ao_voltage(
            pulse=pulse,
            ao_channel=ao_channel,
            amplitude=amplitude,
            length=length,
            wave_function=wave_function,
            frequency=frequency,
            element=element,
            simulate=simulate,
            gauss_sd=gauss_sd,
        )

    def exposed_get_ai_voltage(
        self,
        pulse,
        ai_channel,
        element,
        ao_channel,
        amplitude,
        length,
        wave_function,
        frequency,
        time_of_flight,
        smearing,
        gauss_sd,
        variable_name,
    ):
        return self._module.get_ai_voltage(
            pulse=pulse,
            ai_channel=ai_channel,
            element=element,
            ao_channel=ao_channel,
            amplitude=amplitude,
            length=length,
            wave_function=wave_function,
            frequency=frequency,
            time_of_flight=time_of_flight,
            smearing=smearing,
            gauss_sd=gauss_sd,
            variable_name=variable_name,
        )

    def exposed_create_new_ao_elem(
        self,
        pulse,
        ao_channel=None,
        amplitude=default_amplitude,
        length=default_length,
        wave_function=None,
        frequency=default_frequency,
        gauss_sd=default_length / 5,
    ):
        return self._module.create_new_ao_elem(
            pulse=pulse,
            ao_channel=ao_channel,
            amplitude=amplitude,
            gauss_sd=gauss_sd,
            length=length,
            wave_function=wave_function,
            frequency=frequency,
        )

    def exposed_create_new_ai_elem(
        self,
        pulse=default_measure_pulse,
        ai_channel=None,
        ao_channel=None,
        amplitude=default_amplitude,
        length=default_length,
        wave_function=None,
        frequency=default_frequency,
        time_of_flight=defaut_time_of_flight,
        smearing=default_smearing,
        gauss_sd=default_length / 5,
    ):
        return self._module.create_new_ai_elem(
            pulse=pulse,
            ai_channel=ai_channel,
            ao_channel=ao_channel,
            amplitude=amplitude,
            length=length,
            gauss_sd=gauss_sd,
            wave_function=wave_function,
            frequency=frequency,
            time_of_flight=time_of_flight,
            smearing=smearing,
        )

    def exposed_create_new_do_elem(
        self,
        do_channel,
        length,
        delay,
        buffer,
    ):
        return self._module.create_new_do_elem(
            do_channel=do_channel,
            length=length,
            delay=delay,
            buffer=buffer,
        )

    def build_stack(self):
        return self._module.build_stack()

    def execute(self):
        return self._module.execute()

    def exposed_delay(self, length, elements=None):
        return self._module.delay(
            length=length,
            elements=elements,
        )

    def exposed_if_gt(self, var_name, threshold):
        return self._module.if_gt(var_name, threshold)

    def exposed_elif_gt(self, var_name, threshold):
        return self._module.elif_gt(var_name, threshold)

    def exposed_else(self):
        return self._module.else_()


class Client(ClientBase):

    def set_digital_voltage(
        self,
        element=None,
        pulse=default_digital_pulse,
        do_channel=None,
        simulate=False,
        length=default_length,
        delay=0,
        buffer=0,

    ):
        return self._service.exposed_set_digital_voltage(
            element=element,
            do_channel=do_channel,
            pulse=pulse,
            simulate=simulate,
            length=length,
            delay=delay,
            buffer=buffer,
        )

    def set_ao_voltage(
        self,
        pulse=None,
        ao_channel=None,
        element=None,
        amplitude=default_amplitude,
        length=default_length,
        wave_function=None,
        simulate=False,
        frequency=default_frequency,
        gauss_sd=default_length / 5,
    ):
        return self._service.exposed_set_ao_voltage(
            pulse=pulse,
            ao_channel=ao_channel,
            amplitude=amplitude,
            length=length,
            wave_function=wave_function,
            frequency=frequency,
            element=element,
            simulate=simulate,
            gauss_sd=gauss_sd,
        )

    def play_ao_voltage(
        self,
        pulse=None,
        ao_channel=None,
        element=None,
        amplitude=default_amplitude,
        length=default_length,
        wave_function=None,
        simulate=False,
        frequency=default_frequency,
        gauss_sd=default_length / 5,
    ):
        return self._service.exposed_play_ao_voltage(
            pulse=pulse,
            ao_channel=ao_channel,
            amplitude=amplitude,
            length=length,
            wave_function=wave_function,
            frequency=frequency,
            element=element,
            simulate=simulate,
            gauss_sd=gauss_sd,
        )

    def get_ai_voltage(
        self,
        pulse=default_measure_pulse,
        ai_channel=None,
        element=None,
        ao_channel=None,
        amplitude=default_amplitude,
        length=default_length,
        wave_function=None,
        frequency=default_frequency,
        time_of_flight=defaut_time_of_flight,
        smearing=default_smearing,
        gauss_sd=default_length / 5,
        variable_name=None,
    ):
        return self._service.exposed_get_ai_voltage(
            pulse=pulse,
            ai_channel=ai_channel,
            element=element,
            ao_channel=ao_channel,
            amplitude=amplitude,
            length=length,
            wave_function=wave_function,
            frequency=frequency,
            time_of_flight=time_of_flight,
            smearing=smearing,
            gauss_sd=gauss_sd,
            variable_name=variable_name,
        )

    def create_new_ao_elem(
        self,
        pulse,
        ao_channel=None,
        amplitude=default_amplitude,
        length=default_length,
        wave_function=None,
        frequency=default_frequency,
        gauss_sd=default_length / 5,
    ):
        return self._service.exposed_create_new_ao_elem(
            pulse=pulse,
            ao_channel=ao_channel,
            amplitude=amplitude,
            length=length,
            wave_function=wave_function,
            frequency=frequency,
            gauss_sd=gauss_sd,
        )

    def create_new_ai_elem(
        self,
        pulse=default_measure_pulse,
        ai_channel=None,
        ao_channel=None,
        amplitude=default_amplitude,
        length=default_length,
        wave_function=None,
        frequency=default_frequency,
        time_of_flight=defaut_time_of_flight,
        smearing=default_smearing,
        gauss_sd=default_length / 5,
    ):
        return self._service.exposed_create_new_ai_elem(
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

    def create_new_do_elem(
        self,
        do_channel=None,
        length=default_length,
        delay=0,
        buffer=0,
    ):
        return self._service.exposed_create_new_do_elem(
            do_channel=do_channel,
            length=length,
            delay=delay,
            buffer=buffer,
        )

    def build_stack(self):
        return self._service.build_stack()

    def execute(self):
        return self._service.execute()

    def delay(self, length, elements=None):
        return self._service.exposed_delay(
            length=length,
            elements=elements,
        )

    def if_gt(self, var_name, threshold):
        return self._service.exposed_if_gt(var_name, threshold)

    def elif_gt(self, var_name, threshold):
        return self._service.exposed_elif_gt(var_name, threshold)

    def else_(self):
        return self._service.exposed_else()
