from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


class Service(ServiceBase):
    def exposed_connect_camera(self):
        return self._module.connect()

    def exposed_disconnect(self):
        return self._module.disconnect()

    def exposed_set_exposure(self, exposure_us):
        if self._module.cam is None or not self._module.initialized:
            raise RuntimeError("Camera not initialized")

        nm = self._module.cam.GetNodeMap()
        self._module._set_enum(nm, "ExposureAuto", "Off")

        ok = self._module._try_set_float("ExposureTime", exposure_us)
        if not ok:
            raise RuntimeError(f"Failed to set ExposureTime to {exposure_us}")

        return True

    def exposed_try_set_float(self, node_name, value):
        return self._module._try_set_float(node_name, value)

    def exposed_try_set_enum(self, node_name, entry_name):
        return self._module._try_set_enum(node_name, entry_name)

    def exposed_start_acquisition(self):
        return self._module.start_acquisition()

    def exposed_stop_acquisition(self):
        return self._module.stop_acquisition()

    def exposed_get_frame(self, timeout_ms=1000):
        return self._module.get_frame(timeout_ms)

    def exposed_display_most_recent(self):
        return self._module.display_most_recent()

    def exposed_get_frame_bytes(self, timeout_ms=1000):
        return self._module.get_frame_bytes(timeout_ms)

    def exposed_disable_trigger(self):
        return self._module.disable_trigger()

    def exposed_set_hardware_trigger(
        self,
        line="Line0",
        activation="RisingEdge",
        selector="FrameStart",
        overlap="ReadOut",
        acquisition_mode="Continuous",
    ):
        return self._module.set_hardware_trigger(
            line=line,
            activation=activation,
            selector=selector,
            overlap=overlap,
            acquisition_mode=acquisition_mode,
        )


class Client(ClientBase):
    def connect_camera(self):
        return self._service.exposed_connect_camera()

    def disconnect(self):
        return self._service.exposed_disconnect()

    def set_exposure(self, exposure_us):
        return self._service.exposed_set_exposure(exposure_us)

    def try_set_float(self, node_name, value):
        return self._service.exposed_try_set_float(node_name, value)

    def try_set_enum(self, node_name, entry_name):
        return self._service.exposed_try_set_enum(node_name, entry_name)

    def start_acquisition(self):
        return self._service.exposed_start_acquisition()

    def stop_acquisition(self):
        return self._service.exposed_stop_acquisition()

    def get_frame(self, timeout_ms=1000):
        return self._service.exposed_get_frame(timeout_ms)

    def display_most_recent(self):
        return self._service.exposed_display_most_recent()

    def get_frame_bytes(self, timeout_ms=1000):
        return self._service.exposed_get_frame_bytes(timeout_ms)

    def disable_trigger(self):
        return self._service.exposed_disable_trigger()

    def set_hardware_trigger(
        self,
        line="Line0",
        activation="RisingEdge",
        selector="FrameStart",
        overlap="ReadOut",
        acquisition_mode="Continuous",
    ):
        return self._service.exposed_set_hardware_trigger(
            line=line,
            activation=activation,
            selector=selector,
            overlap=overlap,
            acquisition_mode=acquisition_mode,
        )
