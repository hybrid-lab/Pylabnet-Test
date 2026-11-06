from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


class Service(ServiceBase):

    def exposed_set_current(self, channel, current):
        return self._module.set_current(channel, current)

    def exposed_set_voltage(self, channel, voltage):
        return self._module.set_voltage(channel, voltage)

    def exposed_get_current_setpoint(self, channel):
        return self._module.get_current_setpoint(channel)

    def exposed_get_voltage_setpoint(self, channel):
        return self._module.get_voltage_setpoint(channel)

    def exposed_output_on(self, channel):
        return self._module.output_on(channel)

    def exposed_output_off(self, channel):
        return self._module.output_off(channel)

    def exposed_is_output_on(self, channel):
        return self._module.is_output_on(channel)

    def exposed_is_ocp_tripped(self, channel):
        return self._module.is_ocp_tripped(channel)

    def exposed_measure(self, channel):
        return self._module.measure(channel)


class Client(ClientBase):

    def set_current(self, channel, current):
        return self._service.exposed_set_current(channel, current)

    def set_voltage(self, channel, voltage):
        return self._service.exposed_set_voltage(channel, voltage)

    def get_current_setpoint(self, channel):
        return self._service.exposed_get_current_setpoint(channel)

    def get_voltage_setpoint(self, channel):
        return self._service.exposed_get_voltage_setpoint(channel)

    def output_on(self, channel):
        return self._service.exposed_output_on(channel)

    def output_off(self, channel):
        return self._service.exposed_output_off(channel)

    def is_output_on(self, channel):
        return self._service.exposed_is_output_on(channel)

    def is_ocp_tripped(self, channel):
        return self._service.exposed_is_ocp_tripped(channel)

    def measure(self, channel):
        # rpyc returns a remote tuple; make it local
        return tuple(self._service.exposed_measure(channel))
