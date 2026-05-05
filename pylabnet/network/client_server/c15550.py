from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


class Service(ServiceBase):

    def exposed_connect_camera(self):
        return self._module.connect()

    def exposed_disconnect(self):
        return self._module.disconnect()

    def exposed_set_exposure_time(self, exposure_s):
        return self._module.set_exposure_time(exposure_s)

    def exposed_get_exposure_time(self):
        return self._module.get_exposure_time()

    def exposed_start_acquisition(self, framecount=16, sequence=True):
        return self._module.start_acquisition(framecount=framecount, sequence=sequence)

    def exposed_stop_acquisition(self):
        return self._module.stop_acquisition()

    def exposed_get_frame(self, timeout_ms=1000):
        return self._module.get_frame(timeout_ms=timeout_ms)

    def exposed_get_frame_bytes(self, timeout_ms=1000):
        return self._module.get_frame_bytes(timeout_ms=timeout_ms)

    def exposed_get_frame_view(self, timeout_ms=1000):
        return self._module.get_frame_view(timeout_ms=timeout_ms)

    def exposed_snap(self, timeout_ms=1000):
        return self._module.snap(timeout_ms=timeout_ms)

    def exposed_display_most_recent(self, vmin=None, vmax=None, show=False):
        return self._module.display_most_recent(vmin=vmin, vmax=vmax, show=show)

    def exposed_get_transfer_info(self):
        return self._module.get_transfer_info()

    def exposed_get_model(self):
        return self._module.get_model()

    def exposed_get_camera_id(self):
        return self._module.get_camera_id()

    def exposed_get_bus(self):
        return self._module.get_bus()

    def exposed_list_properties(self):
        return self._module.list_properties()

    # -----------------------------
    # Trigger controls
    # -----------------------------
    def exposed_disable_trigger(self):
        return self._module.disable_trigger()

    def exposed_set_software_trigger(self):
        return self._module.set_software_trigger()

    def exposed_fire_software_trigger(self):
        return self._module.fire_software_trigger()

    def exposed_set_hardware_trigger(
        self,
        connector="BNC",
        polarity="RisingEdge",
        active="EDGE",
        trigger_mode="NORMAL",
        first_exposure=None,
        global_exposure=None,
    ):
        return self._module.set_hardware_trigger(
            connector=connector,
            polarity=polarity,
            active=active,
            trigger_mode=trigger_mode,
            first_exposure=first_exposure,
            global_exposure=global_exposure,
        )

    # -----------------------------
    # Output trigger / strobe
    # -----------------------------
    def exposed_set_output_trigger_exposure(
        self,
        kind="EXPOSURE",
        source="EXPOSURE",
        polarity="POSITIVE",
    ):
        return self._module.set_output_trigger_exposure(
            kind=kind,
            source=source,
            polarity=polarity,
        )

    # -----------------------------
    # ROI / subarray
    # -----------------------------
    def exposed_set_subarray(self, width, height, x=0, y=0, enable=True):
        return self._module.set_subarray(
            width=width,
            height=height,
            x=x,
            y=y,
            enable=enable,
        )

    def exposed_disable_subarray(self):
        return self._module.disable_subarray()

    def exposed_get_subarray(self):
        return self._module.get_subarray()

    def exposed_get_image_shape(self):
        return self._module.get_image_shape()

    # -----------------------------
    # Readout helpers
    # -----------------------------
    def exposed_set_readout_slowest(self):
        return self._module.set_readout_slowest()

    def exposed_set_readout_fastest(self):
        return self._module.set_readout_fastest()


class Client(ClientBase):

    def connect_camera(self):
        return self._service.exposed_connect_camera()

    def disconnect(self):
        return self._service.exposed_disconnect()

    def set_exposure_time(self, exposure_s):
        return self._service.exposed_set_exposure_time(exposure_s)

    def get_exposure_time(self):
        return self._service.exposed_get_exposure_time()

    def start_acquisition(self, framecount=16, sequence=True):
        return self._service.exposed_start_acquisition(
            framecount=framecount,
            sequence=sequence,
        )

    def stop_acquisition(self):
        return self._service.exposed_stop_acquisition()

    def get_frame(self, timeout_ms=1000):
        return self._service.exposed_get_frame(timeout_ms=timeout_ms)

    def get_frame_bytes(self, timeout_ms=1000):
        return self._service.exposed_get_frame_bytes(timeout_ms=timeout_ms)

    def get_frame_view(self, timeout_ms=1000):
        return self._service.exposed_get_frame_view(timeout_ms=timeout_ms)

    def snap(self, timeout_ms=1000):
        return self._service.exposed_snap(timeout_ms=timeout_ms)

    def display_most_recent(self, vmin=None, vmax=None, show=False):
        return self._service.exposed_display_most_recent(
            vmin=vmin,
            vmax=vmax,
            show=show,
        )

    def get_transfer_info(self):
        return self._service.exposed_get_transfer_info()

    def get_model(self):
        return self._service.exposed_get_model()

    def get_camera_id(self):
        return self._service.exposed_get_camera_id()

    def get_bus(self):
        return self._service.exposed_get_bus()

    def list_properties(self):
        return self._service.exposed_list_properties()

    # -----------------------------
    # Trigger controls
    # -----------------------------
    def disable_trigger(self):
        return self._service.exposed_disable_trigger()

    def set_software_trigger(self):
        return self._service.exposed_set_software_trigger()

    def fire_software_trigger(self):
        return self._service.exposed_fire_software_trigger()

    def set_hardware_trigger(
        self,
        connector="BNC",
        polarity="RisingEdge",
        active="EDGE",
        trigger_mode="NORMAL",
        first_exposure=None,
        global_exposure=None,
    ):
        return self._service.exposed_set_hardware_trigger(
            connector=connector,
            polarity=polarity,
            active=active,
            trigger_mode=trigger_mode,
            first_exposure=first_exposure,
            global_exposure=global_exposure,
        )

    # -----------------------------
    # Output trigger / strobe
    # -----------------------------
    def set_output_trigger_exposure(
        self,
        kind="EXPOSURE",
        source="EXPOSURE",
        polarity="POSITIVE",
    ):
        return self._service.exposed_set_output_trigger_exposure(
            kind=kind,
            source=source,
            polarity=polarity,
        )

    # -----------------------------
    # ROI / subarray
    # -----------------------------
    def set_subarray(self, width, height, x=0, y=0, enable=True):
        return self._service.exposed_set_subarray(
            width=width,
            height=height,
            x=x,
            y=y,
            enable=enable,
        )

    def disable_subarray(self):
        return self._service.exposed_disable_subarray()

    def get_subarray(self):
        return self._service.exposed_get_subarray()

    def get_image_shape(self):
        return self._service.exposed_get_image_shape()

    # -----------------------------
    # Readout helpers
    # -----------------------------
    def set_readout_slowest(self):
        return self._service.exposed_set_readout_slowest()

    def set_readout_fastest(self):
        return self._service.exposed_set_readout_fastest()
