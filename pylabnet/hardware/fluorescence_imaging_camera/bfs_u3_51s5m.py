import PySpin
import matplotlib.pyplot as plt


class Driver:
    def __init__(self, device_name, logger=None, dummy=False, serial=None):
        self.serial = str(serial) if serial is not None else None
        self.system = None
        self.cam_list = None
        self.cam = None
        self.initialized = False
        self.connect()
        self.image_list = []

    def _set_enum(self, nodemap, node_name: str, entry_name: str):
        node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
        if not (PySpin.IsAvailable(node) and PySpin.IsWritable(node)):
            raise RuntimeError(f"Node {node_name} not available/writable")
        entry = node.GetEntryByName(entry_name)
        if not (PySpin.IsAvailable(entry) and PySpin.IsReadable(entry)):
            raise RuntimeError(f"Enum entry {entry_name} for {node_name} not readable")
        node.SetIntValue(entry.GetValue())

    def _try_set_enum(self, nodemap, node_name: str, entry_name: str):
        """Best-effort enum set (some cameras don't expose all nodes)."""
        try:
            self._set_enum(nodemap, node_name, entry_name)
            return True
        except Exception:
            return False

    def _try_set_float(self, nodemap, node_name: str, value: float):
        try:
            node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                node.SetValue(float(value))
                return True
        except Exception:
            pass
        return False

    def disable_trigger(self):
        """Return camera to non-triggered (free-run / normal) mode."""
        if self.cam is None or not self.initialized:
            raise RuntimeError("Camera not initialized")

        if self.cam.IsStreaming():
            self.cam.EndAcquisition()

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

        # safest to configure when not streaming
        if self.cam.IsStreaming():
            self.cam.EndAcquisition()

        nm = self.cam.GetNodeMap()

        # Best practice: disable trigger before changing trigger settings
        self._set_enum(nm, "TriggerMode", "Off")

        # Set acquisition mode (continuous stream, but frames only happen on triggers)
        self._try_set_enum(nm, "AcquisitionMode", acquisition_mode)

        # Choose what the trigger starts
        self._set_enum(nm, "TriggerSelector", selector)

        # Source is the hardware line
        self._set_enum(nm, "TriggerSource", line)

        # Edge polarity
        self._set_enum(nm, "TriggerActivation", activation)

        # Optional performance-related setting (not present on all models)
        self._try_set_enum(nm, "TriggerOverlap", overlap)

        # Enable trigger
        self._set_enum(nm, "TriggerMode", "On")

    def connect(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            self.disconnect()
            raise RuntimeError("No cameras detected")

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

        self.cam.Init()

        nm = self.cam.GetNodeMap()

        # ExposureAuto -> Off
        exp_auto = PySpin.CEnumerationPtr(nm.GetNode("ExposureAuto"))
        if PySpin.IsAvailable(exp_auto) and PySpin.IsWritable(exp_auto):
            exp_off = exp_auto.GetEntryByName("Off")
            exp_auto.SetIntValue(exp_off.GetValue())

        # GainAuto -> Off
        gain_auto = PySpin.CEnumerationPtr(nm.GetNode("GainAuto"))
        if PySpin.IsAvailable(gain_auto) and PySpin.IsWritable(gain_auto):
            gain_off = gain_auto.GetEntryByName("Off")
            gain_auto.SetIntValue(gain_off.GetValue())

        # (Optional) set fixed exposure time + gain after turning autos off
        exp_time = PySpin.CFloatPtr(nm.GetNode("ExposureTime"))
        if PySpin.IsAvailable(exp_time) and PySpin.IsWritable(exp_time):
            exp_time.SetValue(5000.0)  # microseconds, example

        gain = PySpin.CFloatPtr(nm.GetNode("Gain"))
        if PySpin.IsAvailable(gain) and PySpin.IsWritable(gain):
            gain.SetValue(0.0)         # dB, example

        self.initialized = True

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
