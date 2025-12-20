import PySpin


class BlackflySU3:
    def __init__(self, serial=None):
        self.serial = serial
        self.system = None
        self.cam = None
        self.initialized = False

    def connect(self):
        self.system = PySpin.System.GetInstance()
        cam_list = self.system.GetCameras()

        if cam_list.GetSize() == 0:
            raise RuntimeError("No cameras detected")

        if self.serial is None:
            self.cam = cam_list[0]
        else:
            for cam in cam_list:
                cam.Init()
                if cam.TLDevice.DeviceSerialNumber.GetValue() == self.serial:
                    self.cam = cam
                    break
                cam.DeInit()

        if self.cam is None:
            raise RuntimeError("Requested camera not found")

        self.cam.Init()
        self.initialized = True

    def disconnect(self):
        if self.cam:
            self.cam.DeInit()
        if self.system:
            self.system.ReleaseInstance()

    def set_exposure(self, exposure_us):
        node = self.cam.ExposureTime
        node.SetValue(float(exposure_us))

    def start_acquisition(self):
        self.cam.BeginAcquisition()

    def stop_acquisition(self):
        self.cam.EndAcquisition()

    def get_frame(self, timeout_ms=1000):
        image = self.cam.GetNextImage(timeout_ms)
        if image.IsIncomplete():
            image.Release()
            raise RuntimeError("Incomplete image")
        data = image.GetNDArray()
        image.Release()
        return data
