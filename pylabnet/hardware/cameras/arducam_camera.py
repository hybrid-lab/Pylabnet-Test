"""Driver for the Arducam OV9281 USB camera (1 MP monochrome global shutter, UVC).

Same calls as the other camera drivers (flir_camera.py, opencv_camera.py), so scripts can
use any camera: get_frame_bytes, set_exposure / get_exposure, set_gain / get_gain,
set_free_run / set_hardware_trigger, flush, get_info.

Frames
- 1280 x 800, uint8: the camera's luma (Y) bytes as sent, uncompressed (YUY2), read
  through Media Foundation. Nothing is converted; the camera's colour channels are dropped.
- The camera's chip processes the image. The driver turns sharpening and backlight
  compensation off and sets gamma to its lowest (most linear) value, but the response is
  still not linear: measured 2026-11-10, the signal roughly doubles per doubling of
  exposure up to ~100 counts above the dark level (~18), then grows more slowly; 235 is
  the top of the range. For absorption imaging keep the probe image well below that, and
  calibrate the response if the optical density has to be accurate.
- The camera sometimes stops sending frames: with some contrast and gamma values (e.g.
  contrast 64 with a 2 ms exposure, gamma 150; contrast therefore stays at its default),
  and once, 2026-11-11, with the driver's own settings after it was plugged back in.
  Setting the image settings to their defaults and then to the wanted values brings it
  back, so the driver always applies them that way, and does it again (and reopens the
  stream) when no frame comes for 10 s in free run.

Modes
- Free run (the default): ~20 fps on the camera's own clock. get_frame_bytes() returns a
  frame that arrives after the call.
- Hardware trigger: one frame per rising edge on the trigger input (pin F of the board,
  ground on pin G; 3.3 V logic, at least 2 us long). Frames are queued, oldest first; call
  flush() before a sequence. The camera switches mode through its UVC "auto-exposure
  priority" setting ("low light compensation" in Windows' camera settings).

Exposure is set in powers of 2 seconds (2^-13 s = 122 us to 2^-1 s), as for any USB
camera on Windows; set_exposure() rounds to the nearest and returns the value applied.
Gain is the camera's 0-100 scale, not dB.

Settings are sent through DirectShow (uvc_controls.py) from one thread of their own:
COM objects only work on the thread that made them.
"""

import collections
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import comtypes
import cv2

from pylabnet.hardware.cameras.uvc_controls import Controls, media_foundation_index
from pylabnet.utils.logging.logger import LogHandler

DEFAULT_NAME = 'Arducam OV9281 USB Camera'
WIDTH, HEIGHT = 1280, 800
# Asking for 20 fps makes Media Foundation pick the camera's uncompressed YUY2 mode; at 30
# fps it picks MJPG, which is lossy
FPS = 20
# Image processing fixed at start (camera units); see the module docstring
IMAGE_SETTINGS = {'sharpness': 0, 'backlight_compensation': 0, 'gamma': 72, 'contrast': 32,
                  'brightness': 0}
# Frames kept for reading: in hardware-trigger mode, the triggered frames not yet read
BUFFER_FRAMES = 20
# Frames to throw away after an exposure or gain change in free run: the camera has a
# few frames in its pipeline already exposed with the old setting
FRAMES_TO_SKIP_AFTER_CHANGE = 3
# After a mode change, frames still on their way from the old mode are dropped for this long
MODE_CHANGE_SETTLE_S = 0.3
# Wait this long before reopening a camera that failed
RECONNECT_DELAY_S = 2


class Driver:

    def __init__(self, name=DEFAULT_NAME, logger=None, exposure_us=1000, gain=0):
        """Open the camera in free run and start reading frames.

        :param name: (str) the camera's name in Windows (Device Manager, Cameras)
        :param logger: instance of LogClient
        :param exposure_us: (float) exposure at start, rounded to a power of 2 s
        :param gain: (int) gain at start, 0-100 (camera units)
        """
        self.log = LogHandler(logger=logger)
        self.name = name
        self.mode = 'free_run'

        # Settings go through one thread that owns the COM objects
        self._com = ThreadPoolExecutor(max_workers=1, initializer=comtypes.CoInitialize)
        self._call(self._open_controls)
        self._index = self._call(media_foundation_index, name)
        self._exposure_range = self._call(lambda: self._controls.get_range('exposure'))
        self._gain_range = self._call(lambda: self._controls.get_range('gain'))

        self._exposure_log2 = None
        self._gain = None

        # Shared between the reading thread and the caller, guarded by _lock
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frames = collections.deque(maxlen=BUFFER_FRAMES)  # (count, frame)
        self._count = 0          # frames read since start
        self._min_count = 0      # first frame count that may be returned (after a change)
        self._last_read = 0      # count of the last frame returned
        self._drop_until = 0.0   # time.time() before which arriving frames are dropped
        self._connected = False
        self._fps = 0.0
        self._stop = False

        # Free run (a process that ended in trigger mode leaves the camera in it), the
        # image settings from their defaults, then exposure and gain
        self._call(self._apply_settings)
        self.set_exposure(exposure_us)
        self.set_gain(gain)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # Wait up to 10 s for the first frame, so a camera that's missing or held by
        # another program shows up as an error now rather than at the first measurement
        with self._lock:
            ok = self._new_frame.wait_for(lambda: self._count > 0, timeout=10)
        if ok:
            self.log.info(f'Connected to {self.name}: {WIDTH} x {HEIGHT}, exposure {self.get_exposure():.0f} us, '
                          f'gain {self._gain}')
        else:
            self.log.warn(f'No frames from {self.name} yet (held by another program, e.g. camera_viewer '
                          f'or OBS?); still trying in the background')

    #########################
    # Settings thread       #
    #########################

    def _call(self, fn, *args):
        """Run fn on the settings thread and return its result."""
        return self._com.submit(fn, *args).result(timeout=10)

    def _open_controls(self):
        self._controls = Controls(self.name)

    def _apply_settings(self):
        """All settings, starting from the defaults (see the module docstring: this also
        gets a camera going again that stopped sending frames). Runs on the settings thread."""
        c = self._controls
        c.set('auto_exposure_priority', int(self.mode == 'hardware_trigger'))
        c.reset_image_settings()
        for setting, value in IMAGE_SETTINGS.items():
            c.set(setting, value)
        if self._exposure_log2 is not None:
            c.set('exposure', self._exposure_log2)
        if self._gain is not None:
            c.set('gain', self._gain)

    #########################
    # Reading thread        #
    #########################

    def _open(self):
        return cv2.VideoCapture(self._index, cv2.CAP_MSMF, [
            cv2.CAP_PROP_CONVERT_RGB, 0,  # the camera's own bytes, not BGR
            cv2.CAP_PROP_FRAME_WIDTH, WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT,
            cv2.CAP_PROP_FPS, FPS,
        ])

    def _run(self):
        while not self._stop:
            cap = self._open()
            if not cap.isOpened():
                self.log.warn(f'Could not open {self.name}; retrying in {RECONNECT_DELAY_S} s')
                time.sleep(RECONNECT_DELAY_S)
                continue
            fps_t0, fps_n = time.time(), 0

            while not self._stop:
                # Waits up to 10 s for a frame (Media Foundation's limit)
                ok, raw = cap.read()
                if not ok:
                    if self.mode == 'hardware_trigger':
                        continue  # no trigger in the last 10 s: normal
                    self.log.warn(f'No frame from {self.name} for 10 s in free run; resetting its '
                                  f'settings and reopening it')
                    try:
                        self._call(self._apply_settings)
                    except Exception as e:
                        self.log.warn(f'{self.name}: could not reset its settings ({e})')
                    break
                if raw.size != WIDTH * HEIGHT * 2:
                    self.log.error(f'{self.name} sent {raw.size} bytes per frame, not {WIDTH * HEIGHT * 2} '
                                   f'(uncompressed {WIDTH} x {HEIGHT}); is camera {self._index} the Arducam?')
                    break
                # YUY2: luma and colour bytes alternate; keep the luma
                frame = raw.reshape(HEIGHT, WIDTH, 2)[..., 0].copy()

                fps_n += 1
                now = time.time()
                with self._lock:
                    self._connected = True
                    if now - fps_t0 >= 1:
                        self._fps = fps_n / (now - fps_t0)
                        fps_t0, fps_n = now, 0
                    if now < self._drop_until:
                        continue
                    self._count += 1
                    self._frames.append((self._count, frame))
                    self._new_frame.notify_all()

            with self._lock:
                self._connected = False
            cap.release()
            if not self._stop:
                time.sleep(RECONNECT_DELAY_S)

    #########################
    # Frames                #
    #########################

    def get_frame_bytes(self, timeout_ms=1000, fresh=True):
        """Next frame as (bytes, shape, dtype string).

        Free run: with fresh=True, a frame that arrives after this call. fresh=False takes
        the frame after the one returned last instead, for the 2nd, 3rd, ... frame of a
        run of consecutive frames (averaging).
        Hardware trigger: the oldest triggered frame not yet read; waits up to timeout_ms
        for a trigger.
        """
        with self._lock:
            if self.mode == 'hardware_trigger':
                if not self._new_frame.wait_for(lambda: self._frames, timeout=timeout_ms / 1000):
                    raise TimeoutError(f'No trigger for {self.name} within {timeout_ms:.0f} ms')
                count, frame = self._frames.popleft()
            else:
                after = self._count if fresh else self._last_read
                wanted = max(after + 1, self._min_count)
                if not self._new_frame.wait_for(lambda: self._count >= wanted, timeout=timeout_ms / 1000):
                    state = 'connected' if self._connected else 'not connected'
                    raise TimeoutError(f'No frame from {self.name} within {timeout_ms:.0f} ms ({state})')
                # The wanted frame, or the oldest one still kept if it was pushed out
                count, frame = next(((c, f) for c, f in self._frames if c >= wanted), self._frames[-1])
            self._last_read = count
        return frame.tobytes(), frame.shape, str(frame.dtype)

    def get_frame(self, timeout_ms=1000):
        """Next frame as a numpy array (for use without the network)."""
        import numpy as np
        raw, shape, dtype = self.get_frame_bytes(timeout_ms)
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def flush(self):
        """Drop every frame waiting to be read. Returns how many were dropped."""
        with self._lock:
            dropped = len(self._frames) if self.mode == 'hardware_trigger' else 0
            self._frames.clear()
        return dropped

    #########################
    # Modes                 #
    #########################

    def _set_mode(self, mode):
        self._call(lambda: self._controls.set('auto_exposure_priority', int(mode == 'hardware_trigger')))
        with self._lock:
            self.mode = mode
            # Frames from the old mode may still be on their way through USB and Windows
            self._drop_until = time.time() + MODE_CHANGE_SETTLE_S
            self._frames.clear()
        time.sleep(MODE_CHANGE_SETTLE_S)

    def set_free_run(self):
        """Stream continuously on the camera's own clock (no trigger)."""
        self._set_mode('free_run')
        self.log.info(f'{self.name}: free run')

    def set_hardware_trigger(self, line=None, edge='RisingEdge'):
        """Take one frame per rising edge on the trigger input (pin F).

        :param line: ignored: the camera has one trigger input. Accepted for the same calls
            as the FLIR driver.
        :param edge: only 'RisingEdge' (the camera can't trigger on falling edges)
        """
        if edge not in (None, 'RisingEdge'):
            raise ValueError(f'{self.name} triggers on rising edges only, not {edge}')
        self._set_mode('hardware_trigger')
        self.log.info(f'{self.name}: hardware trigger on the rising edge of pin F')

    #########################
    # Exposure and gain     #
    #########################

    def _skip_after_change(self):
        with self._lock:
            self._min_count = self._count + FRAMES_TO_SKIP_AFTER_CHANGE + 1

    def set_exposure(self, exposure_us):
        """Set the exposure, rounded to the nearest power of 2 s. Returns the value applied
        in us. In free run, frames returned afterwards all use it."""
        lo, hi = self._exposure_range[:2]
        e = min(max(round(math.log2(float(exposure_us) * 1e-6)), lo), hi)
        self._call(lambda: self._controls.set('exposure', e))
        self._exposure_log2 = e
        self._skip_after_change()
        actual = 2 ** e * 1e6
        self.log.info(f'{self.name}: exposure {actual:.0f} us (2^{e} s; asked for {float(exposure_us):.0f} us)')
        return actual

    def get_exposure(self):
        """Exposure in us."""
        return 2 ** self._exposure_log2 * 1e6

    def set_gain(self, gain):
        """Set the gain, 0-100 (camera units, not dB). Returns the value applied."""
        lo, hi = self._gain_range[:2]
        g = int(min(max(round(float(gain)), lo), hi))
        self._call(lambda: self._controls.set('gain', g))
        self._gain = g
        self._skip_after_change()
        self.log.info(f'{self.name}: gain {g}')
        return g

    def get_gain(self):
        """Gain, 0-100 (camera units, not dB)."""
        return self._gain

    #########################
    # State                 #
    #########################

    def get_info(self):
        """State of the camera, as a dict of plain values."""
        with self._lock:
            return {
                'name': self.name,
                'kind': 'arducam',
                'mode': self.mode,
                'connected': self._connected,
                'frames': self._count,
                'fps': round(self._fps, 1),
                'shape': (HEIGHT, WIDTH),
                'exposure_us': self.get_exposure(),
                'gain_db': None,
                'gain': self._gain,
                'exposure_settable': True,
                'gain_settable': True,
                'triggerable': True,
            }

    def close(self):
        """Back to free run (so other programs get frames), stop reading and release the camera."""
        try:
            # Also ends a read waiting for a trigger
            self._call(lambda: self._controls.set('auto_exposure_priority', 0))
        except Exception:
            pass
        self._stop = True
        self._thread.join(timeout=15)
        self._com.shutdown(wait=False)
