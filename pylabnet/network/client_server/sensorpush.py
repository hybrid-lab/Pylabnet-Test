from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


class Service(ServiceBase):
    def exposed_get_data(self, num_points=1):
        return self._module.get_data(num_points)


class Client(ClientBase):
    def get_data(self, num_points=1):
        return self._service.exposed_get_data(num_points)

    def get_time(self):
        data = self.get_data()
        return data['datetime'][0]

    def get_temperature(self):
        data = self.get_data()
        return data['temperature'][0]

    def get_humidity(self):
        data = self.get_data()
        return data['humidity'][0]
