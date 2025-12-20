from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


class Service(ServiceBase):
    def exposed_connect(self):
        return(self._module.connect())

    def exposed_disconnect(self):
        return(self._module.disconnect())

    def exposed_set_exposure(self, exposure_us):
        return(self._module.set_exposure(exposure_us))

    def exposed_start_acquisition(self):
        return(self._module.start_acquisition())

    def exposed_stop_acquisition(self):
        return(self._module.stop_acquisition())

    def exposed_get_frame(self, timeout_ms=1000):
        return(self._module.get_frame(timeout_ms))


class Client(ClientBase):
    def connect(self):
        return(self._service.exposed_connect())

    def disconnect(self):
        return(self._service.exposed_disconnect())

    def set_exposure(self, exposure_us):
        return(self._service.exposed_set_exposure(exposure_us))

    def start_acquisition(self):
        return(self._service.exposed_start_acquisition())

    def stop_acquisition(self):
        return(self._service.exposed_stop_acquisition())

    def get_frame(self, timeout_ms=1000):
        return(self._service.exposed_get_frame(timeout_ms))
