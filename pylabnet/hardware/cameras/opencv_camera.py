"""Driver for cameras that OpenCV reads: a local USB camera (DirectShow) or a network
camera streaming RTSP.

A background thread reads the stream continuously and keeps only the newest frame.
get_frame_bytes() then returns the first frame that arrives after the call. That frame
is fresh, not one left waiting in a buffer, and the stream doesn't have to be
reopened, which takes seconds for RTSP.

Frames are converted to grayscale uint8. These cameras free-run on their own clock;
they can't be hardware-triggered.

Exposure:
- USB (DirectShow): settable in steps of powers of 2 s (2^-13 s = 122 us up to
  2^-1 s = 0.5 s). set_exposure() rounds to the nearest step and returns the applied
  value. A frame takes at least the exposure time, so above 2^-6 s (15.6 ms) the camera
  drops below 30 fps.
- Network (RTSP): not settable over RTSP; the camera uses its own settings.
"""

import math
import sys
import threading
import time

import cv2

from pylabnet.utils.logging.logger import LogHandler

# DirectShow can set the exposure of a USB camera on Windows; Media Foundation ignores it.
USB_BACKEND = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
# DirectShow exposure range, in log2 seconds
USB_EXPOSURE_RANGE = (-13, -1)
# Frames to throw away after an exposure change: the camera has a few frames in its
# pipeline already exposed with the old setting
FRAMES_TO_SKIP_AFTER_EXPOSURE = 3
# Wait this long before reopening a stream that failed or dropped, in s
RECONNECT_DELAY_S = 2


class Driver:

    def __init__(self, source, logger=None, usb_exposure_us=None):
        """Open the camera and start reading frames.

        :param source: (int or str) USB device index (e.g. 0), or an RTSP URL
        :param logger: instance of LogClient
        :param usb_exposure_us: (float) exposure to apply at start (USB only); None keeps
            the camera's setting
        """
        self.log = LogHandler(logger=logger)
        self.is_usb = isinstance(source, int) or str(source).isdigit()
        self.source = int(source) if self.is_usb else str(source)
        # Name for messages; RTSP URLs can contain the login, so show only the host
        self.name = f'USB camera {self.source}' if self.is_usb else f'network camera {self._host()}'

        # Shared between the reading thread and the caller, guarded by _lock
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frame = None
        self._count = 0              # frames read since start
        self._min_count = 0          # first frame count that may be returned (after an exposure change)
        self._connected = False
        self._exposure_log2 = None   # exposure the camera is set to (USB)
        self._pending_exposure = None if usb_exposure_us is None else self._to_log2(usb_exposure_us)
        self._fps = 0.0
        self._stop = False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # Wait up to 10 s for the first frame so a camera that's missing shows up as an
        # error now rather than at the first measurement
        with self._lock:
            self._new_frame.wait_for(lambda: self._count > 0, timeout=10)
            ok = self._count > 0
        if ok:
            h, w = self._frame.shape
            self.log.info(f'Connected to {self.name}: {w} x {h}')
        else:
            self.log.warn(f'No frames from {self.name} yet; still trying in the background')

    def _host(self):
        """Host part of an RTSP URL, without the login."""
        rest = self.source.split('://', 1)[-1]
        return rest.split('/', 1)[0].split('@')[-1]

    #########################
    # Reading thread        #
    #########################

    def _open(self):
        if self.is_usb:
            cap = cv2.VideoCapture(self.source, USB_BACKEND)
            # MJPG keeps the USB bandwidth low enough for the full frame rate
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        return cap

    def _run(self):
        while not self._stop:
            cap = self._open()
            if not cap.isOpened():
                self.log.warn(f'Could not open {self.name}; retrying in {RECONNECT_DELAY_S} s')
                time.sleep(RECONNECT_DELAY_S)
                continue
            # A reopened USB camera needs its exposure set again
            if self.is_usb and self._exposure_log2 is not None and self._pending_exposure is None:
                self._pending_exposure = self._exposure_log2
            fps_t0, fps_n = time.time(), 0

            while not self._stop:
                if self._pending_exposure is not None:
                    self._apply_exposure(cap)
                ok, frame = cap.read()
                if not ok:
                    self.log.warn(f'Lost {self.name}; reconnecting')
                    break
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                fps_n += 1
                now = time.time()
                with self._lock:
                    self._frame = gray
                    self._count += 1
                    self._connected = True
                    if now - fps_t0 >= 1:
                        self._fps = fps_n / (now - fps_t0)
                        fps_t0, fps_n = now, 0
                    self._new_frame.notify_all()

            with self._lock:
                self._connected = False
            cap.release()
            if not self._stop:
                time.sleep(RECONNECT_DELAY_S)

    def _apply_exposure(self, cap):
        """Runs on the reading thread: OpenCV captures aren't safe to use from two threads."""
        e = self._pending_exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, e)
        with self._lock:
            self._exposure_log2 = e
            self._pending_exposure = None
            self._min_count = self._count + FRAMES_TO_SKIP_AFTER_EXPOSURE + 1
            self._new_frame.notify_all()

    #########################
    # Public methods        #
    #########################

    def get_frame_bytes(self, timeout_ms=1000, fresh=True):
        """Wait for the next new frame and return it as (bytes, shape, dtype string).

        :param timeout_ms: (float) how long to wait for a frame
        :param fresh: accepted for the same calls as the FLIR driver; every frame
            returned here is new, consecutive calls give consecutive frames
        """
        with self._lock:
            wanted = max(self._count + 1, self._min_count)
            if not self._new_frame.wait_for(lambda: self._count >= wanted, timeout=timeout_ms / 1000):
                state = 'connected' if self._connected else 'not connected'
                raise TimeoutError(f'No frame from {self.name} within {timeout_ms:.0f} ms ({state})')
            frame = self._frame
        return frame.tobytes(), frame.shape, str(frame.dtype)

    def get_frame(self, timeout_ms=1000):
        """Next new frame as a numpy array (for use without the network)."""
        import numpy as np
        raw, shape, dtype = self.get_frame_bytes(timeout_ms)
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def _to_log2(self, exposure_us):
        e = round(math.log2(float(exposure_us) * 1e-6))
        return min(max(e, USB_EXPOSURE_RANGE[0]), USB_EXPOSURE_RANGE[1])

    def set_exposure(self, exposure_us):
        """Set the exposure (USB only), rounded to the nearest power of 2 s.
        Returns the applied exposure in us. Frames returned afterwards all use it."""
        if not self.is_usb:
            raise NotImplementedError(f'Exposure of the {self.name} cannot be set over RTSP')
        e = self._to_log2(exposure_us)
        with self._lock:
            self._pending_exposure = e
            # Wait for the reading thread to apply it, so the next frame uses it
            applied = self._new_frame.wait_for(lambda: self._pending_exposure is None, timeout=5)
        if not applied:
            raise TimeoutError(f'{self.name} did not take the exposure setting (not connected?)')
        actual = 2 ** e * 1e6
        self.log.info(f'{self.name}: exposure {actual:.0f} us (2^{e} s; asked for {float(exposure_us):.0f} us)')
        return actual

    def get_exposure(self):
        """Exposure in us, or None when unknown (network camera, or not set yet)."""
        e = self._exposure_log2
        return None if e is None else 2 ** e * 1e6

    def set_gain(self, gain_db):
        raise NotImplementedError(f'Gain of the {self.name} is not settable')

    def get_gain(self):
        return None

    def set_free_run(self):
        """These cameras always free-run; nothing to change."""

    def set_hardware_trigger(self, line=None, edge=None):
        raise NotImplementedError(f'The {self.name} has no trigger input')

    def flush(self):
        """Nothing is queued: get_frame_bytes() always waits for a new frame."""
        return 0

    def get_info(self):
        """State of the camera, as a dict of plain values."""
        with self._lock:
            shape = None if self._frame is None else tuple(self._frame.shape)
            return {
                'name': self.name,
                'kind': 'usb' if self.is_usb else 'network',
                'mode': 'free_run',
                'connected': self._connected,
                'frames': self._count,
                'fps': round(self._fps, 1),
                'shape': shape,
                'exposure_us': self.get_exposure(),
                'gain_db': None,
                'exposure_settable': self.is_usb,
                'gain_settable': False,
                'triggerable': False,
            }

    def close(self):
        """Stop reading and release the camera."""
        self._stop = True
        self._thread.join(timeout=5)
