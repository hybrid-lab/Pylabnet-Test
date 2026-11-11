"""Settings of a USB (UVC) camera through DirectShow, on Windows.

OpenCV can set only some of a USB camera's settings, and only through its DirectShow
backend. This module reaches all of them, whichever backend reads the frames: exposure,
gain, gamma, sharpness, ... and "auto-exposure priority", which some cameras (e.g. the
Arducam OV9281) use as their external-trigger switch.

Units are DirectShow's: exposure in log2 seconds (-10 = 2^-10 s = 0.98 ms); the others
in the camera's own units (see get_range).

COM objects belong to the thread that made them: make a Controls and use it on one
thread only, after calling comtypes.CoInitialize() there (the importing thread already is).
"""

from ctypes import POINTER, c_long, c_ulong, c_void_p

import comtypes
import comtypes.client
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
from comtypes.automation import VARIANT
from comtypes.persist import IPropertyBag

CLSID_SYSTEM_DEVICE_ENUM = GUID('{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}')
CLSID_VIDEO_INPUT_DEVICE_CATEGORY = GUID('{860BB310-5D01-11d0-BD3B-00A0C911CE86}')

# Camera settings (IAMCameraControl), by DirectShow / kernel-streaming property number
CAMERA_CONTROLS = {'exposure': 4, 'auto_exposure_priority': 19}
# Image settings (IAMVideoProcAmp)
IMAGE_CONTROLS = {'brightness': 0, 'contrast': 1, 'hue': 2, 'saturation': 3, 'sharpness': 4,
                  'gamma': 5, 'white_balance': 7, 'backlight_compensation': 8, 'gain': 9}
# Flags: the camera's automatic control, or the value given
AUTO, MANUAL = 1, 2


#########################
# COM interfaces        #
#########################

class _IPersistStream(comtypes.IPersist):
    _iid_ = GUID('{00000109-0000-0000-C000-000000000046}')
    _methods_ = [
        COMMETHOD([], HRESULT, 'IsDirty'),
        COMMETHOD([], HRESULT, 'Load', (['in'], c_void_p, 'pstm')),
        COMMETHOD([], HRESULT, 'Save', (['in'], c_void_p, 'pstm'), (['in'], c_long, 'fClearDirty')),
        COMMETHOD([], HRESULT, 'GetSizeMax', (['out'], POINTER(c_void_p), 'pcbSize')),
    ]


class _IMoniker(_IPersistStream):
    # Only the first two of IMoniker's own methods are declared; the vtable needs no more
    _iid_ = GUID('{0000000F-0000-0000-C000-000000000046}')
    _methods_ = [
        COMMETHOD([], HRESULT, 'BindToObject', (['in'], c_void_p, 'pbc'), (['in'], c_void_p, 'pmkToLeft'),
                  (['in'], POINTER(GUID), 'riidResult'), (['out'], POINTER(POINTER(IUnknown)), 'ppvResult')),
        COMMETHOD([], HRESULT, 'BindToStorage', (['in'], c_void_p, 'pbc'), (['in'], c_void_p, 'pmkToLeft'),
                  (['in'], POINTER(GUID), 'riid'), (['out'], POINTER(POINTER(IUnknown)), 'ppvObj')),
    ]


class _IEnumMoniker(IUnknown):
    _iid_ = GUID('{00000102-0000-0000-C000-000000000046}')
    _methods_ = [
        COMMETHOD([], HRESULT, 'Next', (['in'], c_ulong, 'celt'),
                  (['out'], POINTER(POINTER(_IMoniker)), 'rgelt'), (['out'], POINTER(c_ulong), 'pceltFetched')),
    ]


class _ICreateDevEnum(IUnknown):
    _iid_ = GUID('{29840822-5B84-11D0-BD3B-00A0C911CE86}')
    _methods_ = [
        COMMETHOD([], HRESULT, 'CreateClassEnumerator', (['in'], POINTER(GUID), 'clsidDeviceClass'),
                  (['out'], POINTER(POINTER(_IEnumMoniker)), 'ppEnumMoniker'), (['in'], c_ulong, 'dwFlags')),
    ]


# IAMCameraControl and IAMVideoProcAmp have the same three methods
_RANGE_GET_SET = [
    COMMETHOD([], HRESULT, 'GetRange', (['in'], c_long, 'Property'), (['out'], POINTER(c_long), 'pMin'),
              (['out'], POINTER(c_long), 'pMax'), (['out'], POINTER(c_long), 'pSteppingDelta'),
              (['out'], POINTER(c_long), 'pDefault'), (['out'], POINTER(c_long), 'pCapsFlags')),
    COMMETHOD([], HRESULT, 'Set', (['in'], c_long, 'Property'), (['in'], c_long, 'lValue'),
              (['in'], c_long, 'Flags')),
    COMMETHOD([], HRESULT, 'Get', (['in'], c_long, 'Property'), (['out'], POINTER(c_long), 'lValue'),
              (['out'], POINTER(c_long), 'Flags')),
]


class _IAMCameraControl(IUnknown):
    _iid_ = GUID('{C6E13370-30AC-11d0-A18C-00A0C9118956}')
    _methods_ = _RANGE_GET_SET


class _IAMVideoProcAmp(IUnknown):
    _iid_ = GUID('{C6E13360-30AC-11d0-A18C-00A0C9118956}')
    _methods_ = _RANGE_GET_SET


#########################
# Camera list           #
#########################

def _read(bag, name):
    try:
        return bag.Read(name, VARIANT(), None)
    except comtypes.COMError:
        return None


def _cameras():
    """(moniker, friendly name, device path) for each DirectShow camera, in DirectShow's
    order. Virtual cameras (OBS, ...) have no device path."""
    dev_enum = comtypes.client.CreateObject(CLSID_SYSTEM_DEVICE_ENUM, interface=_ICreateDevEnum)
    monikers = dev_enum.CreateClassEnumerator(CLSID_VIDEO_INPUT_DEVICE_CATEGORY, 0)
    if not monikers:  # no cameras at all
        return []
    found = []
    while True:
        moniker, fetched = monikers.Next(1)
        if not fetched:
            return found
        bag = moniker.BindToStorage(None, None, IPropertyBag._iid_).QueryInterface(IPropertyBag)
        found.append((moniker, _read(bag, 'FriendlyName'), _read(bag, 'DevicePath')))


def list_cameras():
    """Names of the cameras, in DirectShow's order (OpenCV's DirectShow index)."""
    return [name for _, name, _ in _cameras()]


def media_foundation_index(name):
    """OpenCV's Media Foundation index of the camera called name. Media Foundation lists
    the same hardware cameras in the same order as DirectShow, without the virtual ones."""
    hardware = [n for _, n, path in _cameras() if path]
    if name not in hardware:
        raise RuntimeError(f'No camera called "{name}"; cameras present: {list_cameras()}')
    return hardware.index(name)


class Controls:
    """Settings of the camera called name (the first one, if several have that name)."""

    def __init__(self, name):
        for moniker, n, _ in _cameras():
            if n == name:
                device = moniker.BindToObject(None, None, IUnknown._iid_)
                self._camera = device.QueryInterface(_IAMCameraControl)
                self._image = device.QueryInterface(_IAMVideoProcAmp)
                self.name = name
                return
        raise RuntimeError(f'No camera called "{name}"; cameras present: {list_cameras()}')

    def _target(self, setting):
        if setting in CAMERA_CONTROLS:
            return self._camera, CAMERA_CONTROLS[setting]
        if setting in IMAGE_CONTROLS:
            return self._image, IMAGE_CONTROLS[setting]
        raise KeyError(f'Unknown setting {setting}; known: {sorted({**CAMERA_CONTROLS, **IMAGE_CONTROLS})}')

    def get(self, setting):
        """Current value of a setting."""
        iface, prop = self._target(setting)
        return iface.Get(prop)[0]

    def set(self, setting, value, auto=False):
        """Set a setting (fixed value, or the camera's automatic control if auto)."""
        iface, prop = self._target(setting)
        iface.Set(prop, int(value), AUTO if auto else MANUAL)

    def get_range(self, setting):
        """(min, max, step, default) of a setting."""
        iface, prop = self._target(setting)
        return iface.GetRange(prop)[:4]

    def reset_image_settings(self):
        """Every image setting (IMAGE_CONTROLS) to the camera's default. Settings the
        camera doesn't have are skipped."""
        for setting in IMAGE_CONTROLS:
            try:
                self.set(setting, self.get_range(setting)[3])
            except comtypes.COMError:
                pass
