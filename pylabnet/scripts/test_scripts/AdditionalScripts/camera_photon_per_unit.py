"""
Quantitative frame capture for FLIR Blackfly S (BFS-U3-51S5M).

Configures the camera for linear photometry (Mono16, no gamma, no auto
anything), grabs N frames, and saves the per-pixel average as float64.

Usage:
    python grab_frames.py light 100
    python grab_frames.py dark  100     # with the beam blocked

Then:
    light = np.load("light.npy")
    dark  = np.load("dark.npy")
    signal = light - dark               # per-pixel, float
    total  = signal[y0:y1, x0:x1].sum() # DN16 over an ROI
"""

import sys
import numpy as np
import PySpin

# ---- acquisition settings -------------------------------------------------
SERIAL = "25380235"  # two cameras on this rig; pick one explicitly
EXPOSURE_US = 10000.0     # must be IDENTICAL for light and dark
GAIN_DB = 0.0
BLACK_LEVEL = 2.0         # percent; keeps the low tail off the 0 rail
# ---------------------------------------------------------------------------


def configure(cam):
    cam.Init()
    nm = cam.GetNodeMap()

    # Stop any streaming left over from SpinView before touching PixelFormat.
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

    # 12-bit sensor data, left-shifted into a 16-bit container.
    cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
    adc = PySpin.CEnumerationPtr(nm.GetNode("AdcBitDepth"))
    if PySpin.IsWritable(adc):
        adc.SetIntValue(adc.GetEntryByName("Bit12").GetValue())

    # Kill everything nonlinear or self-adjusting.
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    gamma_en = PySpin.CBooleanPtr(nm.GetNode("GammaEnable"))
    if PySpin.IsWritable(gamma_en):
        gamma_en.SetValue(False)
    sharp_en = PySpin.CBooleanPtr(nm.GetNode("SharpeningEnable"))
    if PySpin.IsWritable(sharp_en):
        sharp_en.SetValue(False)

    cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
    cam.ExposureTime.SetValue(EXPOSURE_US)
    cam.Gain.SetValue(GAIN_DB)

    bl = PySpin.CFloatPtr(nm.GetNode("BlackLevel"))
    if PySpin.IsWritable(bl):
        bl.SetValue(BLACK_LEVEL)

    # Take every frame in order; don't let the driver drop any.
    snm = cam.GetTLStreamNodeMap()
    mode = PySpin.CEnumerationPtr(snm.GetNode("StreamBufferHandlingMode"))
    mode.SetIntValue(mode.GetEntryByName("OldestFirst").GetValue())

    # Echo back what the camera actually accepted, not what we asked for.
    print(f"  exposure : {cam.ExposureTime.GetValue():.1f} us")
    print(f"  gain     : {cam.Gain.GetValue():.2f} dB")
    print(f"  format   : {cam.PixelFormat.GetCurrentEntry().GetSymbolic()}")


def grab_average(cam, n):
    cam.BeginAcquisition()
    acc = None
    kept = 0
    clipped = 0
    peak = 0
    try:
        while kept < n:
            img = cam.GetNextImage(5000)
            try:
                if img.IsIncomplete():
                    print(f"  dropped incomplete frame ({img.GetImageStatus()})")
                    continue
                a = img.GetNDArray()          # ndarray, not ImagePtr
                if acc is None:
                    acc = np.zeros(a.shape, dtype=np.float64)
                    print(f"  shape {a.shape}, dtype {a.dtype}")
                clipped += int(np.count_nonzero(a >= 65520))
                peak = max(peak, int(a.max()))
                acc += a
                kept += 1
            finally:
                img.Release()                 # also runs if we raise
    finally:
        cam.EndAcquisition()

    print(f"  peak (single frame) : {peak} DN16")
    if clipped:
        print(f"  *** CLIPPED: {clipped} pixel-samples at the 65520 ceiling ***")
    return acc / kept


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "frames"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    if cams.GetSize() == 0:
        cams.Clear()
        system.ReleaseInstance()
        sys.exit("No camera found.")

    cam = cams.GetBySerial(SERIAL)
    if cam is None or not cam.IsValid():
        found = [c.TLDevice.DeviceSerialNumber.GetValue() for c in cams]
        del cam
        cams.Clear()
        system.ReleaseInstance()
        sys.exit(f"Serial {SERIAL} not found. Cameras present: {found}")
    print(f"camera {SERIAL}")

    try:
        configure(cam)
        avg = grab_average(cam, n)

        np.save(f"{tag}.npy", avg)
        print(f"saved {tag}.npy  mean={avg.mean():.2f}  max={avg.max():.0f} DN16")

        # Single raw frame too, for an independent check of the encoding.
        cam.BeginAcquisition()
        one = cam.GetNextImage(5000)
        try:
            one.GetNDArray().tofile(f"{tag}_single.raw")
        finally:
            one.Release()
            del one
        cam.EndAcquisition()
    finally:
        # Cleanup must not raise, or it masks whatever actually went wrong.
        try:
            cam.DeInit()
        except Exception as e:
            print(f"  DeInit: {e}")
        del cam
        cams.Clear()
        try:
            system.ReleaseInstance()
        except Exception as e:
            print(f"  ReleaseInstance: {e}")


if __name__ == "__main__":
    main()
