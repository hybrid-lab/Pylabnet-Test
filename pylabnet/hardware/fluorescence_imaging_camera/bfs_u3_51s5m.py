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
