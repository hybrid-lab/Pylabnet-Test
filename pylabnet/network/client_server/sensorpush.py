from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


class Service(ServiceBase):
    def exposed_get_data(self):
        return self._module.get_data()


class Client(ClientBase):
    def get_data(self):
        return self._service.exposed_get_data()
