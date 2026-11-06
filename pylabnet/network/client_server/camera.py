"""Client and server for any camera driver in hardware/cameras (FLIR, USB, network).

The drivers share one set of calls, so experiment scripts can use any camera the same
way. Calls a camera can't do raise NotImplementedError; get_info() says what it can do.
"""

from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


class Service(ServiceBase):

    def exposed_get_frame_bytes(self, timeout_ms=1000, fresh=True):
        return self._module.get_frame_bytes(timeout_ms, fresh)

    def exposed_set_exposure(self, exposure_us):
        return self._module.set_exposure(exposure_us)

    def exposed_get_exposure(self):
        return self._module.get_exposure()

    def exposed_set_gain(self, gain_db):
        return self._module.set_gain(gain_db)

    def exposed_get_gain(self):
        return self._module.get_gain()

    def exposed_set_free_run(self):
        return self._module.set_free_run()

    def exposed_set_hardware_trigger(self, line='Line0', edge='RisingEdge'):
        return self._module.set_hardware_trigger(line, edge)

    def exposed_flush(self):
        return self._module.flush()

    def exposed_get_info(self):
        return self._module.get_info()


class Client(ClientBase):

    def get_frame_bytes(self, timeout_ms=1000, fresh=True):
        """Next frame as (bytes, shape, dtype string); rebuild it with
        np.frombuffer(raw, dtype=dtype).reshape(shape). fresh=False: the next of a run
        of consecutive frames (see the driver)."""
        # Send fresh only when False: camera servers started before it existed take
        # only timeout_ms, and fresh=True is what they do anyway
        if fresh:
            raw, shape, dtype = self._service.exposed_get_frame_bytes(timeout_ms)
        else:
            raw, shape, dtype = self._service.exposed_get_frame_bytes(timeout_ms, False)
        # rpyc hands back remote references; copy them to local values
        return bytes(raw), tuple(shape), str(dtype)

    def set_exposure(self, exposure_us):
        """Set the exposure in us; returns the value the camera applied."""
        return self._service.exposed_set_exposure(exposure_us)

    def get_exposure(self):
        return self._service.exposed_get_exposure()

    def set_gain(self, gain_db):
        """Set the gain in dB; returns the value the camera applied."""
        return self._service.exposed_set_gain(gain_db)

    def get_gain(self):
        return self._service.exposed_get_gain()

    def set_free_run(self):
        return self._service.exposed_set_free_run()

    def set_hardware_trigger(self, line='Line0', edge='RisingEdge'):
        return self._service.exposed_set_hardware_trigger(line, edge)

    def flush(self):
        """Drop frames waiting in the camera's buffer; returns how many."""
        return self._service.exposed_flush()

    def get_info(self):
        info = dict(self._service.exposed_get_info())
        if info.get('shape') is not None:
            info['shape'] = tuple(info['shape'])
        return info
