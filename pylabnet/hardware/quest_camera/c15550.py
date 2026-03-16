import numpy as np
import matplotlib.pyplot as plt

from pyDCAM import (
    HDCAM,
    dcamapi_init,
    dcamapi_uninit,
    DCAMError,
    DCAMCAP_START,
    DCAMWAIT_EVENT,
    DCAMIDPROP,
    DCAMPROPMODEVALUE,
)


class Driver:
    """
    Hamamatsu ORCA-Quest / DCAM driver using pyDCAM.

    Written in a style similar to a FLIR PySpin driver:
      - connect / disconnect
      - start_acquisition / stop_acquisition
      - get_frame / get_frame_bytes
      - trigger helpers
      - ROI / exposure / readout helpers
      - output trigger helpers for strobe/exposure monitoring

    Notes
    -----
    1. pyDCAM uses DCAM property IDs + enum values, not a GenICam nodemap.
    2. Some properties differ by interface / firmware / camera model.
       Therefore many setters below are "best effort".
    3. For the ORCA-Quest, external trigger and output trigger behavior may
       depend on the physical interface/connector actually in use.
    """

    def __init__(self, device_name=None, logger=None, dummy=False, serial=None, index=0):
        self.device_name = device_name
        self.logger = logger
        self.dummy = dummy

        # pyDCAM exposes camera_id/model/bus. There is not a FLIR-style
        # "DeviceSerialNumber" node in the documented API, so we use `serial`
        # as a requested camera_id substring / exact-match hint.
        self.serial = str(serial) if serial is not None else None
        self.index = int(index)

        self.cam = None
        self.wait_handle = None

        self.initialized = False
        self.acquiring = False
        self.buffer_allocated = False
        self.buffer_framecount = 0
        self.device_count = 0

        self.image_list = []
        self._last_frame = None

        self.connect()

    # -------------------------------------------------------------------------
    # Logging helper
    # -------------------------------------------------------------------------
    def _log(self, msg):
        if self.logger is not None:
            try:
                self.logger.info(msg)
                return
            except Exception:
                pass
        print(msg)

    # -------------------------------------------------------------------------
    # Low-level property helpers
    # -------------------------------------------------------------------------
    def _require_cam(self):
        if self.cam is None or not self.initialized:
            raise RuntimeError("Camera not initialized")

    def _prop_supported(self, prop_id) -> bool:
        self._require_cam()
        try:
            self.cam.dcamprop_getvalue(prop_id)
            return True
        except Exception:
            return False

    def _get_prop(self, prop_id):
        self._require_cam()
        return self.cam.dcamprop_getvalue(prop_id)

    def _set_prop(self, prop_id, value):
        self._require_cam()
        self.cam.dcamprop_setvalue(prop_id, value)

    def _setget_prop(self, prop_id, value):
        self._require_cam()
        return self.cam.dcamprop_setgetvalue(prop_id, value)

    def _try_set_prop(self, prop_id, value) -> bool:
        try:
            self._set_prop(prop_id, value)
            return True
        except Exception:
            return False

    def _try_setget_prop(self, prop_id, value):
        try:
            return self._setget_prop(prop_id, value)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Camera discovery / connection
    # -------------------------------------------------------------------------
    def _find_camera_index(self):
        """
        If self.serial is provided, try to match it against camera_id.
        Otherwise use self.index.
        """
        if self.serial is None:
            if self.index < 0 or self.index >= self.device_count:
                raise RuntimeError(
                    f"Requested camera index {self.index} out of range "
                    f"(found {self.device_count} camera(s))"
                )
            return self.index

        # Search by camera_id string
        for i in range(self.device_count):
            cam = HDCAM(i)
            try:
                cam_id = getattr(cam, "camera_id", "")
                model = getattr(cam, "model", "")
                if cam_id == self.serial or self.serial in str(cam_id):
                    try:
                        cam.dcamdev_close()
                    except Exception:
                        pass
                    return i
                # also allow model substring if user is sloppy with "serial"
                if self.serial in str(model):
                    try:
                        cam.dcamdev_close()
                    except Exception:
                        pass
                    return i
            finally:
                try:
                    cam.dcamdev_close()
                except Exception:
                    pass

        raise RuntimeError(f"Requested camera not found: {self.serial}")

    def connect(self):
        if self.initialized:
            return

        self.device_count = dcamapi_init()
        if self.device_count <= 0:
            dcamapi_uninit()
            raise RuntimeError("No Hamamatsu DCAM cameras detected")

        cam_index = self._find_camera_index()
        self.cam = HDCAM(cam_index)

        # Best-effort initial configuration
        try:
            # Conservative / quiet readout, useful for Quest-style imaging
            self._try_set_prop(
                DCAMIDPROP.DCAM_IDPROP_READOUTSPEED,
                DCAMPROPMODEVALUE.DCAMPROP_READOUTSPEED__SLOWEST,
            )
        except Exception:
            pass

        self.initialized = True
        self._log(
            f"Connected to Hamamatsu camera: "
            f"model={self.get_model()} camera_id={self.get_camera_id()} bus={self.get_bus()}"
        )

    def disconnect(self):
        # Stop acquisition and release buffers/wait handle first
        try:
            self.stop_acquisition()
        except Exception:
            pass

        if self.wait_handle is not None:
            try:
                self.wait_handle.dcamwait_close()
            except Exception:
                pass
            self.wait_handle = None

        if self.cam is not None:
            try:
                self.cam.dcamdev_close()
            except Exception:
                pass
            self.cam = None

        self.initialized = False
        self.acquiring = False
        self.buffer_allocated = False
        self.buffer_framecount = 0

        try:
            dcamapi_uninit()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Info helpers
    # -------------------------------------------------------------------------
    def get_model(self):
        self._require_cam()
        try:
            return self.cam.model
        except Exception:
            return "Unknown"

    def get_camera_id(self):
        self._require_cam()
        try:
            return self.cam.camera_id
        except Exception:
            return "Unknown"

    def get_bus(self):
        self._require_cam()
        try:
            return self.cam.bus
        except Exception:
            return "Unknown"

    def list_properties(self):
        self._require_cam()
        out = []
        for prop_id in self.cam.dcamprop_ids():
            try:
                name = self.cam.dcamprop_getname(prop_id)
            except Exception:
                name = str(prop_id)
            try:
                value = self.cam.dcamprop_getvalue(prop_id)
            except Exception:
                value = None
            out.append((prop_id, name, value))
        return out

    # -------------------------------------------------------------------------
    # Exposure / readout / image geometry
    # -------------------------------------------------------------------------
    def set_exposure_time(self, exposure_s: float):
        """
        Set exposure time in seconds.
        DCAM exposure property is documented as EXPOSURETIME.
        """
        self._require_cam()
        self.cam.exposure_time = float(exposure_s)

    def get_exposure_time(self) -> float:
        self._require_cam()
        return float(self.cam.exposure_time)

    def set_readout_slowest(self):
        self._require_cam()
        self._set_prop(
            DCAMIDPROP.DCAM_IDPROP_READOUTSPEED,
            DCAMPROPMODEVALUE.DCAMPROP_READOUTSPEED__SLOWEST,
        )

    def set_readout_fastest(self):
        self._require_cam()
        self._set_prop(
            DCAMIDPROP.DCAM_IDPROP_READOUTSPEED,
            DCAMPROPMODEVALUE.DCAMPROP_READOUTSPEED__FASTEST,
        )

    def set_subarray(self, width: int, height: int, x: int = 0, y: int = 0, enable=True):
        """
        Configure ROI / subarray.
        pyDCAM documents subarray_mode, subarray_size, and subarray_pos.
        """
        self._require_cam()
        if self.acquiring:
            self.stop_acquisition()

        self.cam.subarray_mode = bool(enable)
        self.cam.subarray_size = (int(width), int(height))
        self.cam.subarray_pos = (int(x), int(y))

    def disable_subarray(self):
        self._require_cam()
        if self.acquiring:
            self.stop_acquisition()
        self.cam.subarray_mode = False

    def get_subarray(self):
        self._require_cam()
        return {
            "enabled": bool(self.cam.subarray_mode),
            "size": tuple(self.cam.subarray_size),
            "pos": tuple(self.cam.subarray_pos),
        }

    def get_image_shape(self):
        """
        Returns (height, width) of the current acquisition geometry by grabbing
        a single frame if needed.
        """
        frame = self.get_frame(timeout_ms=1000, keep=False)
        return tuple(frame.shape)

    # -------------------------------------------------------------------------
    # Trigger helpers
    # -------------------------------------------------------------------------
    def disable_trigger(self):
        """
        Return to internal/free-run triggering.
        """
        self._require_cam()
        if self.acquiring:
            self.stop_acquisition()

        self._set_prop(
            DCAMIDPROP.DCAM_IDPROP_TRIGGERSOURCE,
            DCAMPROPMODEVALUE.DCAMPROP_TRIGGERSOURCE__INTERNAL,
        )

    def set_software_trigger(self):
        """
        Use software trigger mode.
        Then call start_acquisition(), and call fire_software_trigger() whenever
        you want one frame.
        """
        self._require_cam()
        if self.acquiring:
            self.stop_acquisition()

        self._set_prop(
            DCAMIDPROP.DCAM_IDPROP_TRIGGERSOURCE,
            DCAMPROPMODEVALUE.DCAMPROP_TRIGGERSOURCE__SOFTWARE,
        )

        # Best-effort defaults
        self._try_set_prop(
            DCAMIDPROP.DCAM_IDPROP_TRIGGERACTIVE,
            DCAMPROPMODEVALUE.DCAMPROP_TRIGGERACTIVE__EDGE,
        )
        self._try_set_prop(
            DCAMIDPROP.DCAM_IDPROP_TRIGGER_MODE,
            DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_MODE__NORMAL,
        )

    def fire_software_trigger(self):
        self._require_cam()
        self.cam.dcamcap_firetrigger()

    def set_hardware_trigger(
        self,
        connector: str = "BNC",
        polarity: str = "RisingEdge",
        active: str = "EDGE",
        trigger_mode: str = "NORMAL",
        first_exposure: str = None,
        global_exposure: str = None,
    ):
        """
        Configure external hardware trigger.

        Parameters
        ----------
        connector:
            "BNC", "INTERFACE", or "MULTI" (if supported by camera/interface)
        polarity:
            "RisingEdge" -> POSITIVE
            "FallingEdge" -> NEGATIVE
        active:
            "EDGE", "LEVEL", "PULSE", "POINT"
        trigger_mode:
            "NORMAL", "START", "PIV", "MULTIGATE", "MULTIFRAME"
        first_exposure:
            optional "NEW" or "CURRENT"
        global_exposure:
            optional "NONE", "ALWAYS", "DELAYED", "EMULATE", "GLOBALRESET"

        After calling this, call start_acquisition() and then get_frame();
        each external trigger should produce a frame according to the camera settings.
        """
        self._require_cam()
        if self.acquiring:
            self.stop_acquisition()

        connector_map = {
            "INTERFACE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_CONNECTOR__INTERFACE,
            "BNC": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_CONNECTOR__BNC,
            "MULTI": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_CONNECTOR__MULTI,
        }
        polarity_map = {
            "RISINGEDGE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGERPOLARITY__POSITIVE,
            "FALLINGEDGE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGERPOLARITY__NEGATIVE,
            "POSITIVE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGERPOLARITY__POSITIVE,
            "NEGATIVE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGERPOLARITY__NEGATIVE,
        }
        active_map = {
            "EDGE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGERACTIVE__EDGE,
            "LEVEL": DCAMPROPMODEVALUE.DCAMPROP_TRIGGERACTIVE__LEVEL,
            "PULSE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGERACTIVE__PULSE,
            "POINT": DCAMPROPMODEVALUE.DCAMPROP_TRIGGERACTIVE__POINT,
        }
        mode_map = {
            "NORMAL": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_MODE__NORMAL,
            "START": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_MODE__START,
            "PIV": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_MODE__PIV,
            "MULTIGATE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_MODE__MULTIGATE,
            "MULTIFRAME": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_MODE__MULTIFRAME,
        }
        first_exposure_map = {
            "NEW": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_FIRSTEXPOSURE__NEW,
            "CURRENT": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_FIRSTEXPOSURE__CURRENT,
        }
        global_exposure_map = {
            "NONE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_GLOBALEXPOSURE__NONE,
            "ALWAYS": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_GLOBALEXPOSURE__ALWAYS,
            "DELAYED": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_GLOBALEXPOSURE__DELAYED,
            "EMULATE": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_GLOBALEXPOSURE__EMULATE,
            "GLOBALRESET": DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_GLOBALEXPOSURE__GLOBALRESET,
        }

        connector_key = connector.strip().upper()
        polarity_key = polarity.strip().upper()
        active_key = active.strip().upper()
        mode_key = trigger_mode.strip().upper()

        # Set external triggering first
        self._set_prop(
            DCAMIDPROP.DCAM_IDPROP_TRIGGERSOURCE,
            DCAMPROPMODEVALUE.DCAMPROP_TRIGGERSOURCE__EXTERNAL,
        )

        # Best-effort: not every interface/firmware exposes all of these
        self._try_set_prop(
            DCAMIDPROP.DCAM_IDPROP_TRIGGER_CONNECTOR,
            connector_map[connector_key],
        )
        self._try_set_prop(
            DCAMIDPROP.DCAM_IDPROP_TRIGGERPOLARITY,
            polarity_map[polarity_key],
        )
        self._try_set_prop(
            DCAMIDPROP.DCAM_IDPROP_TRIGGERACTIVE,
            active_map[active_key],
        )
        self._try_set_prop(
            DCAMIDPROP.DCAM_IDPROP_TRIGGER_MODE,
            mode_map[mode_key],
        )

        if first_exposure is not None:
            fe_key = first_exposure.strip().upper()
            self._try_set_prop(
                DCAMIDPROP.DCAM_IDPROP_TRIGGER_FIRSTEXPOSURE,
                first_exposure_map[fe_key],
            )

        if global_exposure is not None:
            ge_key = global_exposure.strip().upper()
            self._try_set_prop(
                DCAMIDPROP.DCAM_IDPROP_TRIGGER_GLOBALEXPOSURE,
                global_exposure_map[ge_key],
            )

    # -------------------------------------------------------------------------
    # Output trigger / strobe helpers
    # -------------------------------------------------------------------------
    def set_output_trigger_exposure(
        self,
        kind: str = "EXPOSURE",
        source: str = "EXPOSURE",
        polarity: str = "POSITIVE",
    ):
        """
        Configure the camera's output trigger so you can monitor acquisition
        timing (for example, exposure-active or readout-end) on external hardware.

        Useful when you want the camera to tell NI/OPX exactly when it is exposing.

        Parameters
        ----------
        kind:
            "EXPOSURE", "PROGRAMABLE", "TRIGGERREADY", "HIGH", "LOW"
        source:
            "EXPOSURE", "READOUTEND", "HSYNC", "VSYNC", "TRIGGER"
        polarity:
            "POSITIVE" or "NEGATIVE"
        """
        self._require_cam()
        if self.acquiring:
            self.stop_acquisition()

        kind_map = {
            "EXPOSURE": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_KIND__EXPOSURE,
            "PROGRAMABLE": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_KIND__PROGRAMABLE,
            "TRIGGERREADY": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_KIND__TRIGGERREADY,
            "HIGH": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_KIND__HIGH,
            "LOW": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_KIND__LOW,
        }
        source_map = {
            "EXPOSURE": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_SOURCE__EXPOSURE,
            "READOUTEND": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_SOURCE__READOUTEND,
            "HSYNC": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_SOURCE__HSYNC,
            "VSYNC": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_SOURCE__VSYNC,
            "TRIGGER": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_SOURCE__TRIGGER,
        }
        polarity_map = {
            "POSITIVE": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_POLARITY__POSITIVE,
            "NEGATIVE": DCAMPROPMODEVALUE.DCAMPROP_OUTPUTTRIGGER_POLARITY__NEGATIVE,
        }

        self._try_set_prop(
            DCAMIDPROP.DCAM_IDPROP_OUTPUTTRIGGER_KIND,
            kind_map[kind.strip().upper()],
        )
        self._try_set_prop(
            DCAMIDPROP.DCAM_IDPROP_OUTPUTTRIGGER_SOURCE,
            source_map[source.strip().upper()],
        )
        self._try_set_prop(
            DCAMIDPROP.DCAM_IDPROP_OUTPUTTRIGGER_POLARITY,
            polarity_map[polarity.strip().upper()],
        )

    # -------------------------------------------------------------------------
    # Buffer / acquisition helpers
    # -------------------------------------------------------------------------
    def _ensure_buffers(self, framecount=16):
        self._require_cam()
        if self.buffer_allocated and self.buffer_framecount == framecount:
            return

        if self.acquiring:
            self.stop_acquisition()

        if self.buffer_allocated:
            try:
                self.cam.dcambuf_release()
            except Exception:
                pass
            self.buffer_allocated = False
            self.buffer_framecount = 0

        self.cam.dcambuf_alloc(int(framecount))
        self.buffer_allocated = True
        self.buffer_framecount = int(framecount)

    def _ensure_wait_handle(self):
        self._require_cam()
        if self.wait_handle is None:
            self.wait_handle = self.cam.dcamwait_open()

    def start_acquisition(self, framecount=16, sequence=True):
        """
        Start capture into internal DCAM buffers.

        Parameters
        ----------
        framecount:
            number of internal DCAM frame buffers
        sequence:
            True  -> continuous sequence capture
            False -> snap mode
        """
        self._require_cam()

        if self.acquiring:
            return

        self._ensure_buffers(framecount=framecount)
        self._ensure_wait_handle()

        mode = (
            DCAMCAP_START.DCAMCAP_START_SEQUENCE
            if sequence
            else DCAMCAP_START.DCAMCAP_START_SNAP
        )
        self.cam.dcamcap_start(mode)
        self.acquiring = True

    def stop_acquisition(self):
        self._require_cam()

        if self.acquiring:
            try:
                self.cam.dcamcap_stop()
            finally:
                self.acquiring = False

        if self.buffer_allocated:
            try:
                self.cam.dcambuf_release()
            except Exception:
                pass
            self.buffer_allocated = False
            self.buffer_framecount = 0

    # -------------------------------------------------------------------------
    # Frame acquisition
    # -------------------------------------------------------------------------
    def get_frame(self, timeout_ms=1000, keep=True):
        """
        Wait for the next frame-ready event and return a NumPy array copy.

        This is the safest default because dcambuf_copyframe() returns an array
        copied from the internal DCAM buffer.
        """
        self._require_cam()
        if not self.acquiring:
            self.start_acquisition(framecount=16, sequence=True)

        self._ensure_wait_handle()

        try:
            self.wait_handle.dcamwait_start(
                eventmask=DCAMWAIT_EVENT.DCAMWAIT_CAPEVENT_FRAMEREADY,
                timeout=int(timeout_ms),
            )
        except Exception as exc:
            raise RuntimeError(f"Timed out waiting for frame ({timeout_ms} ms)") from exc

        arr = self.cam.dcambuf_copyframe()

        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        arr = arr.copy()
        self._last_frame = arr

        if keep:
            self.image_list.append(arr)

        return arr

    def get_frame_view(self, timeout_ms=1000, keep=False):
        """
        Return an array view pointing into the internal DCAM buffer.

        Use with caution: for long-lived use or post-processing, prefer get_frame().
        """
        self._require_cam()
        if not self.acquiring:
            self.start_acquisition(framecount=16, sequence=True)

        self._ensure_wait_handle()

        try:
            self.wait_handle.dcamwait_start(
                eventmask=DCAMWAIT_EVENT.DCAMWAIT_CAPEVENT_FRAMEREADY,
                timeout=int(timeout_ms),
            )
        except Exception as exc:
            raise RuntimeError(f"Timed out waiting for frame ({timeout_ms} ms)") from exc

        arr = self.cam.dcambuf_lockframe()
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        self._last_frame = arr
        if keep:
            self.image_list.append(np.array(arr, copy=True))
        return arr

    def get_frame_bytes(self, timeout_ms=1000):
        arr = self.get_frame(timeout_ms=timeout_ms, keep=False)
        return (arr.tobytes(), arr.shape, str(arr.dtype))

    def get_transfer_info(self):
        """
        Returns (newest_frame_index, transferred_frame_count)
        """
        self._require_cam()
        return self.cam.dcamcap_transferinfo()

    # -------------------------------------------------------------------------
    # Display / convenience
    # -------------------------------------------------------------------------
    def display_most_recent(self, vmin=None, vmax=None, show=False):
        if self._last_frame is None:
            return None

        arr = self._last_frame
        if show:
            plt.imshow(arr, vmin=vmin, vmax=vmax, cmap="gray")
            plt.colorbar()
            plt.show()
        return arr

    def snap(self, timeout_ms=1000):
        """
        Single-frame convenience method.
        Uses one-frame snap mode.
        """
        self._require_cam()

        if self.acquiring:
            self.stop_acquisition()

        self._ensure_buffers(framecount=1)
        self._ensure_wait_handle()

        self.cam.dcamcap_start(DCAMCAP_START.DCAMCAP_START_SNAP)
        self.acquiring = True
        try:
            self.wait_handle.dcamwait_start(
                eventmask=DCAMWAIT_EVENT.DCAMWAIT_CAPEVENT_FRAMEREADY,
                timeout=int(timeout_ms),
            )
            arr = self.cam.dcambuf_copyframe()
            arr = np.asarray(arr).copy()
            self._last_frame = arr
            self.image_list.append(arr)
            return arr
        finally:
            try:
                self.cam.dcamcap_stop()
            except Exception:
                pass
            self.acquiring = False
            try:
                self.cam.dcambuf_release()
            except Exception:
                pass
            self.buffer_allocated = False
            self.buffer_framecount = 0

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass
