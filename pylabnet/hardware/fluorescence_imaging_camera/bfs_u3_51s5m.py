import PySpin
import matplotlib.pyplot as plt


class Driver:
    def __init__(self, device_name, logger=None, dummy=False, serial=None):
        self.logger = logger
        self.serial = str(serial) if serial is not None else None
        self.system = None
        self.cam_list = None
        self.cam = None
        self.initialized = False
        self.connect()
        self.image_list = []

    def _is_streaming(self):
        """Streaming state, or 'unknown' if even asking fails."""
        try:
            return bool(self.cam.IsStreaming())
        except Exception:
            return "unknown"

    def _node_state(self, node):
        """(exists, available, writable) for a node, never raising."""
        state = []
        for probe in (lambda: node.IsValid(),
                      lambda: PySpin.IsAvailable(node),
                      lambda: PySpin.IsWritable(node)):
            try:
                state.append(bool(probe()))
            except Exception:
                state.append(False)
            if not state[-1]:
                return tuple(state) + (False,) * (3 - len(state))
        return tuple(state)

    def _set_enum(self, nodemap, node_name: str, entry_name: str):
        node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))

        # These three conditions have completely different causes and fixes, so
        # say which one it is. Collapsing them into one "not available/writable"
        # message turns every camera problem into a guessing game.
        exists, available, writable = self._node_state(node)
        if not exists:
            raise RuntimeError(
                f"Node {node_name} does not exist on this camera model")
        if not available:
            raise RuntimeError(
                f"Node {node_name} is not available (streaming="
                f"{self._is_streaming()}). The camera has usually dropped off "
                f"the bus -- a USB3 re-enumeration invalidates the node map "
                f"while the handle still looks alive. Reconnect the camera "
                f"server.")
        if not writable:
            raise RuntimeError(
                f"Node {node_name} is read-only right now (streaming="
                f"{self._is_streaming()}). Most Blackfly S features lock while "
                f"acquisition is running, so stop acquisition first. If it is "
                f"not streaming, then something else owns the camera -- "
                f"SpinView, or a second pylabnet camera server -- and this "
                f"connection only has read access.")

        entry = node.GetEntryByName(entry_name)
        if not (PySpin.IsAvailable(entry) and PySpin.IsReadable(entry)):
            raise RuntimeError(
                f"Enum entry {entry_name} for {node_name} not readable")
        node.SetIntValue(entry.GetValue())

    def _try_set_enum(self, node_name: str, entry_name: str):
        """Best-effort enum set on the camera nodemap."""
        try:
            if self.cam is None or not self.initialized:
                raise RuntimeError("Camera not initialized")
            nm = self.cam.GetNodeMap()
            self._set_enum(nm, node_name, entry_name)
            return True
        except Exception:
            return False

    def _try_get_float(self, node_name: str):
        """Read back a float node, or None if it cannot be read.

        Needed because _try_set_float is best-effort: a value outside the
        node's range raises inside SetValue, the exception is swallowed, and
        the camera silently keeps its old setting. For Gain that is invisible
        in the image but scales every photometric result by 10^(dG/20), so the
        only safe thing is to ask the camera what it actually did.
        """
        try:
            if self.cam is None or not self.initialized:
                raise RuntimeError("Camera not initialized")
            nm = self.cam.GetNodeMap()
            node = PySpin.CFloatPtr(nm.GetNode(node_name))
            if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                return float(node.GetValue())
        except Exception:
            pass
        return None

    def _try_set_float(self, node_name: str, value: float):
        """Best-effort float set on the camera nodemap."""
        try:
            if self.cam is None or not self.initialized:
                raise RuntimeError("Camera not initialized")
            nm = self.cam.GetNodeMap()
            node = PySpin.CFloatPtr(nm.GetNode(node_name))
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                node.SetValue(float(value))
                return True
        except Exception:
            pass
        return False

    def _force_stop_acquisition(self):
        """Stop acquisition whether or not IsStreaming() admits it is running.

        A run that dies between BeginAcquisition() and its cleanup leaves the
        camera streaming; the next run then finds every feature node read-only.
        Asking EndAcquisition() unconditionally recovers that, and costs only a
        swallowed exception when the camera really was idle.
        """
        try:
            self.cam.EndAcquisition()
            return True
        except Exception:
            return False

    def disable_trigger(self):
        """Return camera to non-triggered (free-run / normal) mode."""
        if self.cam is None or not self.initialized:
            raise RuntimeError("Camera not initialized")

        self._force_stop_acquisition()

        nm = self.cam.GetNodeMap()
        self._set_enum(nm, "TriggerMode", "Off")

    def set_hardware_trigger(
        self,
        line: str = "Line0",
        activation: str = "RisingEdge",
        selector: str = "FrameStart",
        overlap: str = "ReadOut",   # often "Off" or "ReadOut"
        acquisition_mode: str = "SingleFrame",
    ):
        """
        Configure external hardware trigger.

        After calling this, you must call arm() / start_acquisition().
        Each TTL pulse on `line` will cause one frame, and GetNextImage()
        will block until a trigger arrives (or timeout).

        Common values:
          line: "Line0", "Line1", ...
          activation: "RisingEdge" or "FallingEdge"
          selector: "FrameStart"
          overlap: "Off" or "ReadOut" (if available)
          acquisition_mode: "Continuous"
        """
        if self.cam is None or not self.initialized:
            raise RuntimeError("Camera not initialized")

        # TriggerMode and friends are locked while the camera streams, so
        # acquisition has to stop first. Don't gate this on IsStreaming(): if a
        # previous run died mid-acquisition the flag and the camera can
        # disagree, and the symptom is an unwritable TriggerMode here rather
        # than anything pointing at the leaked acquisition. EndAcquisition() on
        # an idle camera just raises, which is why it is swallowed.
        self._force_stop_acquisition()

        nm = self.cam.GetNodeMap()

        # Best practice: disable trigger before changing trigger settings
        self._set_enum(nm, "TriggerMode", "Off")

        # Set acquisition mode (continuous stream, but frames only happen on triggers)
        self._try_set_enum("AcquisitionMode", acquisition_mode)

        # Choose what the trigger starts
        self._set_enum(nm, "TriggerSelector", selector)

        # Source is the hardware line
        self._set_enum(nm, "TriggerSource", line)

        # Edge polarity
        self._set_enum(nm, "TriggerActivation", activation)

        # Optional performance-related setting (not present on all models)
        self._try_set_enum("TriggerOverlap", overlap)

        # Enable trigger
        self._set_enum(nm, "TriggerMode", "On")

    def connect(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            self.disconnect()
            raise RuntimeError("No cameras detected")

        if self.logger:
            self.logger.info(
                f"Fluorescence camera connect requested: serial={self.serial}, "
                f"detected_cameras={self.cam_list.GetSize()}"
            )

        if self.serial is None:
            self.cam = self.cam_list.GetByIndex(0)
        else:
            self.cam = None
            for i in range(self.cam_list.GetSize()):
                cam = self.cam_list.GetByIndex(i)
                tl = cam.GetTLDeviceNodeMap()
                sn_node = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
                sn = sn_node.GetValue() if PySpin.IsReadable(sn_node) else None
                if sn == self.serial:
                    self.cam = cam
                    break

            if self.cam is None:
                self.disconnect()
                raise RuntimeError(f"Requested camera not found: {self.serial}")

        if self.logger:
            try:
                tl = self.cam.GetTLDeviceNodeMap()
                sn_node = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
                actual_serial = sn_node.GetValue() if PySpin.IsReadable(sn_node) else "unknown"
                self.logger.info(f"Fluorescence camera attaching to serial={actual_serial}")
            except Exception:
                self.logger.info("Fluorescence camera attached, but serial could not be read")

        self.cam.Init()
        self.initialized = True

        nm = self.cam.GetNodeMap()

        # Turn off autos
        self._set_enum(nm, "ExposureAuto", "Off")
        self._set_enum(nm, "GainAuto", "Off")

        # Optional fixed exposure time + gain after turning autos off
        self._try_set_float("ExposureTime", 5000.0)  # microseconds, example
        self._try_set_float("Gain", 0.0)             # dB, example

    def disconnect(self):
        # Stop/DeInit first
        if self.cam is not None:
            try:
                if self.cam.IsInitialized():
                    try:
                        if self.cam.IsStreaming():
                            self.cam.EndAcquisition()
                    except Exception:
                        pass
                    self.cam.DeInit()
            except Exception:
                pass

        # Drop cam ref before clearing list
        self.cam = None
        self.initialized = False

        if self.cam_list is not None:
            try:
                self.cam_list.Clear()
            except Exception:
                pass
        self.cam_list = None

        if self.system is not None:
            try:
                self.system.ReleaseInstance()
            except Exception:
                pass
        self.system = None

    def start_acquisition(self):
        # Don't crash if already streaming
        if self.cam.IsStreaming():
            return
        self.cam.BeginAcquisition()

    def stop_acquisition(self):
        # Don't crash if not streaming
        if not self.cam.IsStreaming():
            return
        self.cam.EndAcquisition()

    def get_frame(self, timeout_ms=1000):
        image = self.cam.GetNextImage(timeout_ms)
        self.image_list.append(image)
        try:
            if image.IsIncomplete():
                raise RuntimeError("Incomplete image")
            return image.GetNDArray()
        finally:
            image.Release()

    def display_most_recent(self, vmin=0, vmax=4095):
        if not self.image_list:
            return

        image = self.image_list[-1]
        array = image.GetNDArray()
        return array

    def get_frame_bytes(self, timeout_ms=1000):
        image = self.cam.GetNextImage(timeout_ms)
        try:
            if image.IsIncomplete():
                raise RuntimeError("Incomplete image")
            arr = image.GetNDArray().copy()   # local numpy array
        finally:
            image.Release()

        return (arr.tobytes(), arr.shape, str(arr.dtype))
