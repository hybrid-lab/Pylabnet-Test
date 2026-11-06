"""Driver for FLIR / Teledyne machine-vision cameras (e.g. Blackfly S BFS-U3-51S5M), via PySpin.

Same calls as the OpenCV camera driver (opencv_camera.py), so scripts can use any camera:
get_frame_bytes, set_exposure / get_exposure, set_gain / get_gain, get_info. On top of
those it has the two trigger modes:

- Free run (the default): the camera streams continuously. get_frame_bytes() throws
  away the frame already waiting and returns the next one, so it is always fresh.
- Hardware trigger: one frame per edge on an input line (set_hardware_trigger). The
  camera queues triggered frames, oldest first, and get_frame_bytes() returns them in
  order. Call flush() before a sequence to drop frames left from earlier triggers.

The camera keeps acquiring between calls, so no frame waits for acquisition to start.
Settings that can't change while acquiring (trigger mode, pixel format) are handled
here: the driver stops acquisition, changes them and starts it again.
"""

import threading

import PySpin

from pylabnet.utils.logging.logger import LogHandler

# Frames the camera can queue in hardware-trigger mode before it drops new ones
TRIGGERED_BUFFER_COUNT = 20
# Frames to throw away after an exposure or gain change in free run: the frame being
# read out, and the one already being exposed, still use the old setting
FRAMES_TO_SKIP_AFTER_CHANGE = 2


class Driver:

    def __init__(self, serial=None, logger=None, exposure_us=5000, gain_db=0, pixel_format='Mono16'):
        """Connect to the camera, put it in free run with fixed exposure and gain, and
        start acquiring.

        :param serial: (str) camera serial number; None takes the only camera found
        :param logger: instance of LogClient
        :param exposure_us: (float) exposure at start
        :param gain_db: (float) gain at start
        :param pixel_format: (str) 'Mono8', or 'Mono16' for the full 12-bit ADC range
            (values then go up to 65535: the 12-bit value times 16)
        """
        self.log = LogHandler(logger=logger)
        self._lock = threading.RLock()
        self._system = None
        self._cam_list = None
        self.cam = None
        self.mode = 'free_run'
        self._skip = 0

        self._system = PySpin.System.GetInstance()
        self._cam_list = self._system.GetCameras()
        self.cam = self._pick_camera(None if serial is None else str(serial))
        self.cam.Init()
        self.serial = self._get_string('DeviceSerialNumber', self.cam.GetTLDeviceNodeMap())
        self.model = self._get_string('DeviceModelName', self.cam.GetTLDeviceNodeMap())

        # A run that crashed can leave the camera acquiring, which locks most settings
        self._end_acquisition()
        self._set_enum('TriggerMode', 'Off')
        self._set_enum('ExposureAuto', 'Off')
        self._set_enum('GainAuto', 'Off')
        self._set_enum('PixelFormat', pixel_format)
        self._set_enum('AcquisitionMode', 'Continuous')
        self._set_free_run_buffers()
        self._set_float('ExposureTime', exposure_us)
        self._set_float('Gain', gain_db)
        self.cam.BeginAcquisition()

        # After a process that was killed mid-acquisition, the first acquisition here
        # delivers only incomplete frames (measured: 4 of 4); restarting it once fixes
        # that. Check with one frame.
        try:
            image = self.cam.GetNextImage(int(self.get_exposure() / 1000) + 3000)
            stale = image.IsIncomplete()
            image.Release()
        except PySpin.SpinnakerException:
            stale = True
        if stale:
            self.log.info(f'FLIR {self.serial}: first frame incomplete (camera left acquiring by an '
                          f'earlier process?); restarting acquisition')
            self._end_acquisition()
            self.cam.BeginAcquisition()

        self.log.info(f'Connected to {self.model} serial {self.serial}: {self._width()} x {self._height()} '
                      f'{pixel_format}, exposure {self.get_exposure():.0f} us, gain {self.get_gain():.1f} dB')

        # On USB 2 ("HighSpeed") a full 5 MP frame takes ~0.25 s in Mono16 (~4 fps) instead of ~0.03 s
        speed = self._get_enum_symbol('DeviceCurrentSpeed', self.cam.GetTLDeviceNodeMap())
        if speed not in ('SuperSpeed', 'unknown'):
            fps = self._node('AcquisitionResultingFrameRate', PySpin.CFloatPtr).GetValue()
            self.log.warn(f'FLIR {self.serial} is connected at USB {speed} (not USB 3 SuperSpeed): at most '
                          f'{fps:.1f} fps. Plug it into a USB 3 port with a USB 3 cable.')

    #########################
    # Node access           #
    #########################

    def _pick_camera(self, serial):
        found = {}
        for i in range(self._cam_list.GetSize()):
            cam = self._cam_list.GetByIndex(i)
            found[self._get_string('DeviceSerialNumber', cam.GetTLDeviceNodeMap())] = cam
        if serial is None and len(found) == 1:
            return next(iter(found.values()))
        if serial in found:
            return found[serial]
        self.close()
        msg = f'FLIR camera {serial or ""} not found; cameras present: {sorted(found) or "none"}'
        self.log.error(msg)
        raise RuntimeError(msg)

    def _node(self, name, ptr_type, nodemap=None):
        nodemap = self.cam.GetNodeMap() if nodemap is None else nodemap
        node = ptr_type(nodemap.GetNode(name))
        if not PySpin.IsAvailable(node):
            raise RuntimeError(f'Camera has no setting {name} (or it is unavailable right now)')
        return node

    def _get_string(self, name, nodemap):
        node = PySpin.CStringPtr(nodemap.GetNode(name))
        return node.GetValue() if PySpin.IsReadable(node) else 'unknown'

    def _get_enum_symbol(self, name, nodemap):
        node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
        return node.GetCurrentEntry().GetSymbolic() if PySpin.IsReadable(node) else 'unknown'

    def _set_enum(self, name, entry, nodemap=None):
        node = self._node(name, PySpin.CEnumerationPtr, nodemap)
        if not PySpin.IsWritable(node):
            raise RuntimeError(f'Camera setting {name} is read-only now. Is acquisition running, '
                               f'or does another program (SpinView, camera_viewer) hold the camera?')
        value = node.GetEntryByName(entry)
        if not PySpin.IsReadable(value):
            raise RuntimeError(f'{name} has no option {entry}')
        node.SetIntValue(value.GetValue())

    def _set_float(self, name, value):
        """Set a float setting, clamped to its allowed range; returns the value the camera took."""
        node = self._node(name, PySpin.CFloatPtr)
        value = min(max(float(value), node.GetMin()), node.GetMax())
        node.SetValue(value)
        return node.GetValue()

    def _get_float(self, name):
        return self._node(name, PySpin.CFloatPtr).GetValue()

    def _width(self):
        return self._node('Width', PySpin.CIntegerPtr).GetValue()

    def _height(self):
        return self._node('Height', PySpin.CIntegerPtr).GetValue()

    #########################
    # Acquisition           #
    #########################

    def _end_acquisition(self):
        # Raises if the camera isn't acquiring, which is fine
        try:
            self.cam.EndAcquisition()
        except PySpin.SpinnakerException:
            pass
        # A camera left acquiring by a process that was killed (e.g. its server window
        # closed) keeps acquiring, and EndAcquisition here doesn't know about it; its
        # settings then stay read-only. AcquisitionStop stops it on the camera itself.
        try:
            stop = PySpin.CCommandPtr(self.cam.GetNodeMap().GetNode('AcquisitionStop'))
            if PySpin.IsWritable(stop):
                stop.Execute()
        except PySpin.SpinnakerException:
            pass

    def _set_free_run_buffers(self):
        # Keep only the newest frame: anything older is stale in free run
        s = self.cam.GetTLStreamNodeMap()
        self._set_enum('StreamBufferHandlingMode', 'NewestOnly', s)

    def _set_triggered_buffers(self):
        # Keep every triggered frame, oldest first, so none are lost in a sequence
        s = self.cam.GetTLStreamNodeMap()
        self._set_enum('StreamBufferCountMode', 'Manual', s)
        self._node('StreamBufferCountManual', PySpin.CIntegerPtr, s).SetValue(TRIGGERED_BUFFER_COUNT)
        self._set_enum('StreamBufferHandlingMode', 'OldestFirst', s)

    def set_free_run(self):
        """Stream continuously on the camera's own clock (no trigger)."""
        with self._lock:
            self._end_acquisition()
            self._set_enum('TriggerMode', 'Off')
            self._set_free_run_buffers()
            self.cam.BeginAcquisition()
            self.mode = 'free_run'
        self.log.info(f'FLIR {self.serial}: free run')

    def set_hardware_trigger(self, line='Line0', edge='RisingEdge'):
        """Take one frame per edge on an input line.

        :param line: (str) input line, e.g. 'Line0' (opto-isolated input) or 'Line3' (GPIO)
        :param edge: (str) 'RisingEdge' or 'FallingEdge'
        """
        with self._lock:
            self._end_acquisition()
            # Trigger settings can only change while the trigger is off
            self._set_enum('TriggerMode', 'Off')
            self._set_enum('TriggerSelector', 'FrameStart')
            self._set_enum('TriggerSource', line)
            self._set_enum('TriggerActivation', edge)
            self._set_enum('TriggerMode', 'On')
            self._set_triggered_buffers()
            self.cam.BeginAcquisition()
            self.mode = 'hardware_trigger'
        self.log.info(f'FLIR {self.serial}: hardware trigger on {line}, {edge}')

    def flush(self):
        """Drop every frame waiting in the buffer. Returns how many were dropped."""
        dropped = 0
        with self._lock:
            while True:
                try:
                    self.cam.GetNextImage(1).Release()
                    dropped += 1
                except PySpin.SpinnakerException:
                    return dropped

    def get_frame_bytes(self, timeout_ms=1000, fresh=True):
        """Next frame as (bytes, shape, dtype string).

        Free run: with fresh=True, a frame taken after this call (the frame already
        waiting is dropped). fresh=False takes the waiting frame instead, for the 2nd,
        3rd, ... frame of a run of consecutive frames (averaging); dropping it there
        would use only every other frame.
        Hardware trigger: the oldest triggered frame not yet read; waits up to
        timeout_ms for a trigger.
        """
        with self._lock:
            if self.mode == 'free_run' and fresh:
                # Drop the frame already waiting, plus any still exposed with old settings
                self.flush()
                for _ in range(self._skip):
                    self._next_image(timeout_ms).Release()
                self._skip = 0
            image = self._next_image(timeout_ms)
            try:
                if image.IsIncomplete():
                    raise RuntimeError(f'Incomplete frame (status {image.GetImageStatus()}); '
                                       f'USB bandwidth too low? Use a USB 3 port.')
                frame = image.GetNDArray().copy()
            finally:
                image.Release()
        return frame.tobytes(), frame.shape, str(frame.dtype)

    def _next_image(self, timeout_ms):
        try:
            return self.cam.GetNextImage(int(timeout_ms))
        except PySpin.SpinnakerException as e:
            what = 'trigger' if self.mode == 'hardware_trigger' else 'frame'
            raise TimeoutError(f'No {what} from FLIR {self.serial} within {timeout_ms:.0f} ms ({e})')

    def get_frame(self, timeout_ms=1000):
        """Next frame as a numpy array (for use without the network)."""
        import numpy as np
        raw, shape, dtype = self.get_frame_bytes(timeout_ms)
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    #########################
    # Exposure and gain     #
    #########################

    def set_exposure(self, exposure_us):
        """Set the exposure in us; returns the value the camera applied (clamped to its range)."""
        with self._lock:
            actual = self._set_float('ExposureTime', exposure_us)
            self._skip = FRAMES_TO_SKIP_AFTER_CHANGE
        self.log.info(f'FLIR {self.serial}: exposure {actual:.0f} us')
        return actual

    def get_exposure(self):
        return self._get_float('ExposureTime')

    def set_gain(self, gain_db):
        """Set the gain in dB; returns the value the camera applied (clamped to its range)."""
        with self._lock:
            actual = self._set_float('Gain', gain_db)
            self._skip = FRAMES_TO_SKIP_AFTER_CHANGE
        self.log.info(f'FLIR {self.serial}: gain {actual:.2f} dB')
        return actual

    def get_gain(self):
        return self._get_float('Gain')

    def get_info(self):
        """State of the camera, as a dict of plain values."""
        with self._lock:
            return {
                'name': f'FLIR {self.model} {self.serial}',
                'kind': 'flir',
                'mode': self.mode,
                'shape': (self._height(), self._width()),
                'pixel_format': self._node('PixelFormat', PySpin.CEnumerationPtr).GetCurrentEntry().GetSymbolic(),
                'exposure_us': self.get_exposure(),
                'gain_db': self.get_gain(),
                'exposure_settable': True,
                'gain_settable': True,
                'triggerable': True,
            }

    def close(self):
        """Stop acquiring and release the camera, so other programs can use it."""
        with self._lock:
            if self.cam is not None:
                try:
                    self._end_acquisition()
                    if self.cam.IsInitialized():
                        self.cam.DeInit()
                except PySpin.SpinnakerException:
                    pass
                # PySpin needs every camera reference gone before the system is released
                del self.cam
                self.cam = None
            if self._cam_list is not None:
                self._cam_list.Clear()
                self._cam_list = None
            if self._system is not None:
                self._system.ReleaseInstance()
                self._system = None
