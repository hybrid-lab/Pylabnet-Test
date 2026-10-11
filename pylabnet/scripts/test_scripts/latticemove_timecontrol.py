# Lattice TRANSPORT: OUT AND BACK, SPECIFIED BY DETUNING AND TIME.
#
# Duplicate of latticemovedouble_fast.py with ONE difference: what you
# type in defines the move. That file takes a DISTANCE and derives the
# ramp and plateau durations from a fixed acceleration and velocity cap.
# This one takes the three numbers the DIM-3000 chain is actually built
# out of and derives everything else from them:
#
#   Max RF Change per Channel (kHz)  -- how far each driver walks from
#                                       base at the top of the ramp
#   Ramp Time to Max (ms)            -- T_r, how long it takes to get
#                                       there (and to come back down)
#   Hold Time at Max (ms)            -- T_p, how long it stays there
#
# Peak velocity, acceleration and transport distance are then REPORTED,
# not requested -- the log prints all three at startup and the panel
# header carries them every shot. Nothing else about the script changes:
# same segment chain, same trigger train, same imaging, same display.
#
# WHEN TO USE THIS INSTEAD. The distance-driven file is the right one for
# asking how far atoms survive being moved, because it holds acceleration
# fixed so a distance sweep is a measurement of distance. Use this one
# when the RF itself is the thing under test -- when you want a known
# excursion on a known ramp because you are watching the AOM efficiency,
# the driver's segment playback or the beam pointing, rather than the
# atoms. Here acceleration is a CONSEQUENCE of what you typed:
#
#     a = v_peak / T_r,   and v_peak = (lambda/2) * beat
#
# so halving T_r at the same detuning doubles the acceleration. Nothing
# stops you, and the log says what you got.
#
# BOTH DIM-3000 DRIVERS MOVE, IN OPPOSITE DIRECTIONS. This is the one thing
# that separates this script from latticemove&back_single.py, which walks the
# up beam alone and leaves the down beam parked at the base frequency.
#
# Only the DIFFERENCE between the two beams moves the standing wave:
#
#     v = (lambda/2) * Df,   Df = f_up - f_down
#
# so it does not matter which beam carries the detuning, only that the beat is
# right. Sweeping one beam by Df and sweeping both by Df/2 in opposite
# directions produce the SAME transport. What changes is how far either AOM has
# to walk from the frequency it was aligned and peaked at:
#
#     one beam moving :  f_up = base +/- Df,    f_down = base
#     both beams      :  f_up = base +/- Df/2,  f_down = base -/+ Df/2
#
# The AOM is what runs out first -- diffraction efficiency falls away from the
# centre frequency and the deflected beam walks with it -- so halving the
# excursion is what buys the speed. For a given tolerable per-AOM detuning the
# top transport velocity DOUBLES, which is the whole point of this file.
#
# WHAT 'Max RF Change per Channel' MEANS, EXACTLY. It is the excursion ONE
# driver makes from its base frequency -- the quantity the AOM's diffraction
# efficiency limits, and the one worth thinking in when the RF is what you
# are testing. The BEAT is what moves the atoms, and with both AOMs sweeping
# the beat is TWICE what you type, because the two drivers walk opposite ways:
#
#     both beams  :  COM4 = base + df,  COM5 = base - df,  beat = 2*df
#     COM4 only   :  COM4 = base + df,  COM5 = base,       beat =   df
#
# So turning 'Both AOMs Sweep' off HALVES the transport velocity and distance
# at the same typed detuning, where in the distance-driven file it changed
# neither. That is the honest behaviour for a per-channel parameter -- you
# asked for a given walk per crystal, and one crystal walking produces half
# the beat of two -- but it is the opposite of what the other file does, so
# the startup log prints the beat and the velocity both ways round.
#
# WIRING -- THE ONE THING THAT WILL BITE. Both chains are advanced by the SAME
# NI line: dio6 must reach the seg input of BOTH drivers (tee the pulse). The
# two chains are built from one leg list, so their segment durations are
# identical by construction and one trigger train drives both in lockstep --
# but only if both actually receive it. A COM5 whose seg input is not connected
# sits at its segment-1 start frequency, which is base: the beat is then Df/2
# instead of Df, and EVERY MOVE RUNS AT HALF THE REQUESTED VELOCITY AND HALF
# THE DISTANCE, with nothing in the images or the log to say so. Nothing here
# can detect that over serial, so it is logged as a warning on every run.
#
# Set 'Both AOMs Sweep' to 0 to fall back to the single-beam behaviour exactly:
# COM4 carries the whole detuning, COM5 is parked at base and never put into
# segment mode, and the dio6 tee is then irrelevant.
#
# The panel is otherwise the single-point display: the picture below, every
# number this run produces above it, single shot with no averaging anywhere.
#
# THE MOVE IS SPECIFIED BY DETUNING AND DURATIONS. Everything else follows,
# in schedule_from_timing():
#     beat    = df_per_channel * (2 if both AOMs sweep else 1)
#     v_peak  = (lambda/2) * beat
#     a       = v_peak / T_r
#     d       = v_peak * (T_r + T_p)      <- one way, to the far point
# T_p = 0 makes it a TRIANGLE (ramp straight into the turnaround), which is
# the same shape the distance-driven file produces below its velocity-cap
# breakpoint; the header still prints which shape it came out as and how
# many DIM segments that needs.
#
# Full sequence (Turnaround Hold > 0, the default):
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> pre-move hold
#   -> ACCELERATE out      (seg 1: base_freq -> out_freq,  T_r)
#   -> CONSTANT VELOCITY   (seg 2: hold out_freq,          T_p)
#   -> DECELERATE to rest  (seg 3: out_freq -> base_freq,  T_r)
#   -> HOLD AT FAR POINT   (seg 4: hold base_freq,         turn_hold_ms)
#   -> ACCELERATE back     (seg 5: base_freq -> back_freq, T_r)
#   -> CONSTANT VELOCITY   (seg 6: hold back_freq,         T_p)
#   -> DECELERATE home     (seg 7: back_freq -> base_freq, T_r)
#   -> post-move hold
#   -> release -> short TOF -> image
#
# The lattice is STATIONARY during seg 4 and the atoms sit still, displaced by
# the full transport distance. Segs 2 and 6 disappear entirely when the profile
# is a triangle. Set Turnaround Hold to 0 to remove seg 4: segs 3-5 then
# collapse into ONE continuous ramp through zero velocity (5 segments), which
# is the smoother reversal but gives the cloud no time at the far point.
#
# Lattice TTL & AM held constant for the entire cycle (AOM thermal stability).
# The MOVE is done by detuning the two lattice beams AWAY FROM EACH OTHER, both
# via DIM-3000 segment mode -- up on COM4, down on COM5 -- each walking half the
# beat in the opposite direction. A detuning Df between the beams makes the
# standing wave move at v = (lambda/2) * Df. With lambda=795nm:
# v = 3.975 mm/s @ 10 kHz of BEAT, which is 5 kHz per driver.
#
# THE LAST LEG IS FOLLOWED BY A TERMINATOR SEGMENT, AND GETS ITS OWN TRIGGER.
# The DIM-3000 manual (v9, "Frequency segments play", Wseg attribute A):
#     1 - segment active
#     8 - last segment, WILL NOT BE PLAYED, by next pulse segment 1 will be
#         played
# Attribute 8 marks the END of the chain; the segment carrying it is never
# played. Every script in this repo used to put 8 on the final real leg --
# here, "decelerate home" -- so that leg was silently skipped: the trigger
# meant to start it landed on an unplayable segment and the output fell
# straight back to base, a velocity step of the full v_peak. On a signal
# generator that is constant velocity followed by an abrupt jump to zero,
# with no ramp between.
#
# So every real leg is written with attribute 1, and one extra base->base
# segment carrying the 8 is appended after them. Reaching it takes a trigger
# of its own, so each move fires (legs + 1) pulses; that last pulse is placed
# just AFTER the final deceleration has finished, because the manual also says
# "if next pulse comes before segment play is done, the next segment will
# start" -- fired any earlier, it would cut the deceleration short instead.
#
# Chain length is therefore legs + 1: 8 segments with a far-point dwell, 6
# without, 4 for a triangle. The DIM-3000 accepts 20. The startup Rseg
# readback checks every segment, and now also checks each one's ATTRIBUTE, so
# an 8 on a real leg is caught as an ERROR rather than seen on the analyser.
#
# DISPLACEMENT. out_freq is base + direction*Df and back_freq is its mirror,
# base - direction*Df, so the velocity profile is ANTISYMMETRIC about the
# turning point. Two consequences, both exact rather than approximate:
#   far point   = v_peak * (T_r + T_p) = the requested distance
#   final point = 0                                          <- always
# The image is therefore taken at the STARTING position, and a cloud that comes
# back displaced or heated is telling you about the transport rather than about
# where it was dropped.

from pyqtgraph.Qt import QtCore
import numpy as np
import dim_state
import serial
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
import time
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg

try:
    from scipy.optimize import curve_fit
except ImportError:      # fall back to moments; see fit_gaussian_1d
    curve_fit = None

# A 1/e^2 radius is 2 sigma, and integrating a 1/e^2-parameterised Gaussian
# gives w*sqrt(pi/2) -- both used to turn the two 1D fits back into a 2D peak.
ROOT_HALF_PI = np.sqrt(np.pi / 2.0)

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

# Brightest a single pixel can report on this camera. Mono16 is 12-bit data
# left-shifted by four plus a 12 count pedestal, so the ceiling is
# 16*4095 + 12, NOT the 65535 the container suggests. A typed display range
# is clamped to +/- this, so the scale can never open on an impossible value.
SENSOR_MAX_MONO16 = 16 * 4095 + 12   # 65532


# ---------------------------------------------------------------------------
# Atom number and density
# ---------------------------------------------------------------------------
# Same apparatus and same camera as motAtomNumber.py, so the calibration
# constants are carried over unchanged. Only two things differ here and both
# matter: this script runs the camera at 50 dB of gain, and its exposure comes
# from the GUI rather than a module constant.
#
#   N_atoms = sum(S - B) / (Gamma_DN * (Omega/4pi) * T_opt * R_sc * t_exp)
#
GAMMA_HZ = 6.0666e6                  # Rb-87 D2 linewidth Gamma/2pi [Hz]
GAMMA = 2 * np.pi * GAMMA_HZ
I_SAT = 3.5779                       # [mW/cm^2] isotropic polarisation, D2

DN_PER_PHOTON_16BIT = 1.84           # from camera_photon_per_unit.py, at 0 dB
DN_PER_PHOTON_12BIT = 0.115
CALIB_GAIN_DB = 0.0                  # gain the calibration above was taken at

WINDOW_DIAMETER_MM = 19.3            # viewport clear aperture
APERTURE_DISTANCE_MM = 100.0         # chamber centre -> window
OPTICS_TRANSMISSION = 1.0

BEAM_POWER_MW = 50.0                 # per beam, at the atoms
BEAM_DIAMETER_MM = 30.0
N_BEAMS = 6
BEAM_PROFILE = 'flat'
IMAGING_DETUNING_MHZ = 0.0

# CHANGED from motAtomNumber's default of 0 dB: this script sets Gain to 50 dB
# in the camera setup below. FLIR gain is 10^(G/20), so 50 dB is a factor of
# 316 -- getting this wrong is a two-and-a-half order of magnitude error in N.
CAMERA_GAIN_DB = 50.0

# ---- Imaging geometry ----
# Sensor pitch divided by the magnification gives the size of one pixel
# projected back into the cloud.
PIXEL_PITCH_UM = 3.45
MAGNIFICATION = 0.59
PIXEL_UM = PIXEL_PITCH_UM / MAGNIFICATION      # ~5.847 um per pixel at the atoms

# The line-of-sight axis is not imaged, so its extent has to be assumed. Taking
# it equal to the axis along the 2448-pixel direction (x) is the usual choice
# for a MOT that is round in the plane perpendicular to the camera.
DEPTH_FOLLOWS_X = True

# Fraction of the 1/e^2 radius the brightness is integrated over. At 2w an
# elliptical aperture already encloses 1 - exp(-8) = 99.97% of a Gaussian, so
# this is the total in all but name, while still being a measurement of real
# pixels rather than an extrapolation of the fit.
INTEGRATION_RADII = 2.0


def collection_efficiency(aperture_diameter_mm, distance_mm):
    """Fraction of isotropic 4pi emission through a circular aperture.

    Exact (1 - cos(theta))/2 rather than the small-angle a^2/4L^2; they differ
    by ~1% at this 5.5 degree half-angle.
    """
    theta = np.arctan((aperture_diameter_mm / 2.0) / distance_mm)
    return 0.5 * (1.0 - np.cos(theta))


def beam_intensity(power_mw, diameter_mm, profile='flat'):
    """Intensity in mW/cm^2. Gaussian returns the on-axis peak (d = 1/e^2)."""
    radius_cm = diameter_mm / 2.0 / 10.0
    area_cm2 = np.pi * radius_cm ** 2
    if str(profile).lower().startswith('gauss'):
        return 2.0 * power_mw / area_cm2
    return power_mw / area_cm2


def scattering_rate(s_total, detuning_hz):
    """Two-level steady-state scattering rate [photons/s/atom]."""
    return (GAMMA / 2.0) * s_total / (
        1.0 + s_total + (2.0 * detuning_hz / GAMMA_HZ) ** 2)


def dn_per_photon_effective(dtype, gain_db=CAMERA_GAIN_DB,
                            calib_gain_db=CALIB_GAIN_DB):
    """DN per photon for this container, rescaled to the gain in use.

    Mono16 is 12-bit data left-shifted by four, so it reads 16x higher per
    photon than raw 12-bit; Mono8 is the same data shifted the other way and
    reads 16x lower. Picking the wrong one is a factor-of-16 error in N.
    """
    dt = np.dtype(dtype)
    if dt == np.uint16:
        base = DN_PER_PHOTON_16BIT
    elif dt == np.uint8:
        base = DN_PER_PHOTON_12BIT / 16.0
    else:
        base = DN_PER_PHOTON_12BIT
    return base * 10.0 ** ((gain_db - calib_gain_db) / 20.0)


def integrated_brightness(diff, fx, fy, n_radii=INTEGRATION_RADII):
    """Background-subtracted counts inside n_radii x the 1/e^2 radii.

    The aperture is the ellipse the fit describes, not a box, so it follows a
    cloud that is wider in one axis than the other. The pedestal is taken as
    the median OUTSIDE a 1.5x larger ellipse -- median rather than mean so
    stray light and hot pixels cannot drag it, and outside a margin so the
    cloud's own wings are not counted as background.

    :return: (dict) counts, offset, pixels, area_px
    """
    h, w = diff.shape
    yy, xx = np.ogrid[0:h, 0:w]
    rx, ry = n_radii * fx['w'], n_radii * fy['w']
    if rx <= 0 or ry <= 0:
        return None
    norm = (((xx - fx['center']) / rx) ** 2 + ((yy - fy['center']) / ry) ** 2)
    inside = norm <= 1.0
    if not inside.any():
        return None

    # Background estimate. The 1.5x margin is right for a compact cloud, but a
    # cloud that has expanded for milliseconds fills most of the sensor and the
    # annulus can fall entirely OFF it -- at sigma ~ 320 px the 2w ellipse
    # already covers 70% of the frame and nothing at all is left outside 3w.
    # The old code then used offset = 0.0, so no pedestal was removed and the
    # raw sum over most of the sensor was reported as cloud counts. That is
    # invisible in the picture and inflates the atom number by whatever the
    # pedestal integrates to.
    #
    # So step the margin in, and if even the 2w boundary leaves too little,
    # fall back to the median of the whole frame: still dominated by background
    # as long as the cloud is not most of the pixels, and far better than
    # assuming zero. Which estimator was used is returned, so a caller can say
    # when the number is not to be trusted.
    MIN_BG_PX = 20000
    offset, n_bg, bg_source = 0.0, 0, 'none'
    for margin, tag in ((1.5, '3w annulus'), (1.25, '2.5w annulus'),
                        (1.0, '2w boundary')):
        outside = norm > margin ** 2
        n_bg = int(outside.sum())
        if n_bg >= MIN_BG_PX:
            offset = float(np.median(diff[outside]))
            bg_source = tag
            break
    else:
        offset = float(np.median(diff))
        n_bg = int(diff.size)
        bg_source = 'whole-frame median (cloud fills the sensor)'

    n_px = int(inside.sum())
    return {
        # dtype=float64 is load-bearing, not decoration. diff is int32 and
        # numpy sums it with an int32 accumulator on Windows (np.dtype(int) is
        # int32 there), which wraps silently past 2.15e9. A bright cloud filling
        # a 2w ellipse reaches that -- 100k pixels near the 65532 ceiling is
        # 6.5e9 -- and the sum comes back NEGATIVE, taking the atom number and
        # the density negative with it. No error, no warning, just a sign flip.
        'counts': float(diff[inside].sum(dtype=np.float64) - offset * n_px),
        'offset': offset,
        'pixels': n_px,
        'rx': rx,
        'ry': ry,
        'inside': inside,
        'bg_pixels': n_bg,
        'bg_source': bg_source,
        # Fraction of the sensor the integration ellipse covers. Past ~0.5 the
        # background estimate is being squeezed and the counts are suspect.
        'fill_fraction': n_px / float(diff.size),
    }


def capture_fraction(shape, fx, fy, inside):
    """Fraction of a perfect 2D Gaussian that actually lands in the measured region.

    The 2w ellipse is 4 sigma across, which encloses 1 - exp(-8) = 99.97% of a
    Gaussian -- IF it fits on the sensor. After a few milliseconds of expansion
    it does not: at sigma ~ 300 px the 4 sigma semi-axis is 1200 px against a
    1024 px half-height, so the ellipse is clipped by the chip and `inside` is
    the ellipse intersected with the frame.

    Integrating the FITTED Gaussian over exactly that mask, against its
    analytic whole-plane integral 2*pi*sx*sy, gives the fraction collected.
    Dividing the measured atom number by it extrapolates to the whole cloud --
    on the explicit assumption that the cloud really is Gaussian out in the
    wings nobody measured, which is the whole point of the extrapolation and
    also its main weakness.

    :return: (float) 0..1, or nan if the fit is degenerate
    """
    h, w = shape
    yy, xx = np.ogrid[0:h, 0:w]
    sx = float(fx['sigma'])
    sy = float(fy['sigma'])
    if not (sx > 0 and sy > 0):
        return float('nan')
    g = np.exp(-((xx - fx['center']) ** 2) / (2.0 * sx ** 2)
               - ((yy - fy['center']) ** 2) / (2.0 * sy ** 2))
    total = 2.0 * np.pi * sx * sy      # exact integral over the whole plane
    return float(g[inside].sum() / total)


def atom_number(sum_dn, dn_per_photon, eta_collect, optics_transmission,
                r_sc, exposure_s):
    """N = sum(S - B) / (DN_per_photon * Omega/4pi * T * R_sc * t_exp)."""
    photons_sensor = sum_dn / dn_per_photon
    photons_emitted = photons_sensor / (eta_collect * optics_transmission)
    return photons_emitted / (r_sc * exposure_s)


def peak_density(n_atoms, wx_px, wy_px):
    """Peak number density [cm^-3] of a 3D Gaussian with 1/e^2 radii w.

    Integrating exp(-2r^2/w^2) over each axis gives w*sqrt(pi/2), so
    N = n0 (pi/2)^{3/2} wx wy wz. The depth wz is not measured; see
    DEPTH_FOLLOWS_X.
    """
    wx_cm = wx_px * PIXEL_UM * 1e-4
    wy_cm = wy_px * PIXEL_UM * 1e-4
    wz_cm = wx_cm if DEPTH_FOLLOWS_X else wy_cm
    volume = (np.pi / 2.0) ** 1.5 * wx_cm * wy_cm * wz_cm
    return n_atoms / volume if volume > 0 else float('nan')


def cloud_metrics(diff, fx, fy, dtype, exposure_us, gain_db=CAMERA_GAIN_DB):
    """Brightness, atom number and peak density for one difference frame.

    :param diff: (ndarray) frame1 - frame2, in camera counts
    :param fx, fy: (dict) fit_gaussian_1d results for the two axes
    :param dtype: raw frame dtype, which fixes the DN/photon scale
    :param exposure_us: (float) camera exposure actually used
    :return: (dict) or None if the region could not be formed
    """
    band = integrated_brightness(diff, fx, fy)
    if band is None:
        return None

    # Total brightness straight from the fitted centre and radii, out to
    # infinity: integrating amplitude*exp(-2r^2/w^2) along one axis gives
    # amplitude*w*sqrt(pi/2). Each marginal was already summed over the other
    # axis, so each integral is the WHOLE cloud on its own -- two independent
    # estimates of one number, not two factors to multiply.
    #
    # This is the extrapolated counterpart to band['counts']: same quantity,
    # one from the model and one from the pixels. Their ratio is the useful
    # part. It should sit at 1 - exp(-8) = 99.97% for a real Gaussian; well
    # under that means the fit is inventing wings the sensor never saw, and
    # over 100% means the aperture is collecting something the fit does not
    # describe -- stray light, a second spot, a badly placed centre.
    total_x = fx['amplitude'] * fx['w'] * ROOT_HALF_PI
    total_y = fy['amplitude'] * fy['w'] * ROOT_HALF_PI
    total_fit = 0.5 * (total_x + total_y)

    dn_ph = dn_per_photon_effective(dtype, gain_db=gain_db)
    eta = collection_efficiency(WINDOW_DIAMETER_MM, APERTURE_DISTANCE_MM)
    s_total = N_BEAMS * beam_intensity(
        BEAM_POWER_MW, BEAM_DIAMETER_MM, BEAM_PROFILE) / I_SAT
    r_sc = scattering_rate(s_total, IMAGING_DETUNING_MHZ * 1e6)
    n_atoms = atom_number(band['counts'], dn_ph, eta, OPTICS_TRANSMISSION,
                          r_sc, exposure_us * 1e-6)
    # What fraction of the cloud the measured region actually held, and the
    # whole-cloud number that implies. For a compact in-situ cloud this is
    # ~1.000 and n_atoms_total == n_atoms; for an expanded one it is the
    # correction that matters most.
    frac = capture_fraction(diff.shape, fx, fy, band['inside'])
    n_total = n_atoms / frac if (np.isfinite(frac) and frac > 0) else float('nan')
    return {
        'counts': band['counts'],
        'offset': band['offset'],
        'pixels': band['pixels'],
        'bg_source': band['bg_source'],
        'fill_fraction': band['fill_fraction'],
        'total_fit': total_fit,
        'total_from_x': total_x,
        'total_from_y': total_y,
        'frac_2w': band['counts'] / total_fit if total_fit > 0 else float('nan'),
        'n_atoms': n_atoms,
        'capture_fraction': frac,
        'n_atoms_total': n_total,
        # Peak density of the WHOLE cloud, so it is not depressed by the part
        # of the cloud that fell off the sensor.
        'density': peak_density(n_total, fx['w'], fy['w']),
        'dn_per_photon': dn_ph,
        'gain_db': gain_db,
        'eta': eta,
        's_total': s_total,
        'r_sc': r_sc,
        'wx_um': fx['w'] * PIXEL_UM,
        'wy_um': fy['w'] * PIXEL_UM,
    }


def _gaussian(x, offset, amplitude, center, sigma):
    return offset + amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def fit_gaussian_1d(profile):
    """Least-squares Gaussian fit of one summed profile.

    The profile is a SUM along a whole axis, not a single row. That is what
    makes a fit possible at all on this data: one row through the cloud carries
    a couple of counts, but adding 2048 of them lifts the cloud by 2048x while
    the noise only grows as sqrt(2048), a ~45x gain in signal-to-noise.

    A flat pedestal between the two frames survives that summation too, so the
    fit carries an explicit offset term rather than assuming the baseline is
    zero -- otherwise the offset is absorbed into the width and the radius
    comes out far too large.

    :param profile: (ndarray) 1D summed profile
    :return: (dict) offset, amplitude, center, sigma, w (1/e^2 radius), curve;
        or None if it could not be fitted
    """
    y = np.asarray(profile, dtype=float)
    n = y.size
    if n < 8 or not np.all(np.isfinite(y)):
        return None
    x = np.arange(n, dtype=float)

    # Seed from moments. Baseline comes from the outer 10% at each end, which
    # is cloud-free for any sensibly framed image.
    edge = max(1, n // 10)
    offset0 = float(np.median(np.concatenate([y[:edge], y[-edge:]])))
    weights = np.clip(y - offset0, 0, None)
    total = weights.sum()
    if total <= 0:
        return None
    center0 = float((weights * x).sum() / total)
    var = float((weights * (x - center0) ** 2).sum() / total)
    sigma0 = float(np.sqrt(var)) if var > 0 else n / 10.0
    amp0 = float(y.max() - offset0)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        return None

    p = [offset0, amp0, center0, sigma0]
    if curve_fit is not None:
        try:
            p, _ = curve_fit(_gaussian, x, y, p0=p, maxfev=5000)
        except Exception:
            pass        # keep the moment estimate rather than losing the frame
    offset, amplitude, center, sigma = (float(v) for v in p)
    sigma = abs(sigma)
    if sigma <= 0 or not np.isfinite(sigma):
        return None

    curve = _gaussian(x, offset, amplitude, center, sigma)
    # Least squares always returns numbers. On a frame with no cloud it returns
    # a broad, shallow bump fitted to noise, and a 1/e^2 radius quoted off that
    # is meaningless -- so judge the fit against the residual scatter and say
    # whether it can be trusted, rather than printing it as though it can.
    noise = float(np.std(y - curve))
    # Significance of the whole peak, not of one sample. The amplitude is
    # constrained by every point across the width, so a per-point ratio badly
    # understates a fit built from thousands of them: at 3x per-point SNR this
    # still recovers the radius to a few percent.
    # Capped at the profile length: a runaway width would otherwise inflate the
    # significance of a fit that is really just a broad bump across the frame.
    n_eff = min(max(1.0, 2.0 * sigma), float(n))
    snr = abs(amplitude) * np.sqrt(n_eff) / noise if noise > 0 else float('inf')
    # 25 is calibrated, not guessed: on synthetic clouds everything above it
    # recovers the radius and peak to better than 5%, while 20 already lets
    # through fits that are 10-16% out.
    reliable = bool(amplitude > 0 and snr >= 25.0
                    and 2.0 * sigma < 0.4 * n
                    and 0 <= center < n)
    return {
        'offset': offset,
        'amplitude': amplitude,
        'center': center,
        'sigma': sigma,
        'w': 2.0 * sigma,          # 1/e^2 radius
        'snr': snr,
        'reliable': reliable,
        'curve': curve,
        'x': x,
    }


# Percentile the autoscaled top follows -- identical to temp_PGC_aziza and to
# the sweep script, so all three render the same cloud the same way. Not the
# max: one hot pixel would otherwise set the scale for the whole frame.
AUTOSCALE_PCT = 99.95
AUTOSCALE_STRIDE = 4


def inferno_lut():
    """The inferno lookup table, whichever pyqtgraph is installed.

    pg.colormap.get() only arrived in pyqtgraph 0.12, and 0.11.1 is what this
    machine has: calling it there raises AttributeError from inside
    configure(), before a single frame exists, and the run dies with no panel
    at all. The fallback builds the same 256-entry table from matplotlib, which
    qt_plotting already depends on, so the picture is identical either way.
    """
    try:
        return pg.colormap.get('inferno').getLookupTable()
    except Exception:
        pass
    try:
        from matplotlib import cm
        lut = cm.get_cmap('inferno')(np.linspace(0, 1, 256))[:, :3] * 255
        return lut.astype(np.ubyte)
    except Exception:
        # Greyscale. Not the map anyone wants, but a picture beats no window.
        grey = np.linspace(0, 255, 256).astype(np.ubyte)
        return np.stack([grey, grey, grey], axis=1)


def autoscale_top(image, lo):
    """Top of the colour range for this image, from its own brightness.

    A percentile and not the max: one hot pixel would otherwise set the scale
    for the whole frame and push the cloud back down into the dark end.
    Sampled every 4th row and column, because a percentile over all 5.0 M
    pixels is a full sort on every frame pair while 313 k of them give the same
    number to well under a count.
    """
    sample = np.asarray(image)[::AUTOSCALE_STRIDE, ::AUTOSCALE_STRIDE]
    hi = float(np.percentile(sample, AUTOSCALE_PCT))
    if hi <= lo + 1.0:
        hi = lo + 10.0        # empty frame: keep a usable, non-degenerate range
    return hi


def profiles_from_fits(x_proj, y_proj, fx, fy):
    """Margin-strip traces for one image, with the FITTED offset removed.

    Data and curve are both shifted down by the fitted offset so they sit on
    the same baseline, instead of the fit floating above a pedestal the data
    still carries.

    Returns plain numpy arrays and touches no Qt: this runs on the acquisition
    thread, and the drawing happens later on the GUI thread.

    :return: (dict) 'x' and/or 'y' -> (axis, data, fit), the key missing when
        that fit failed outright.
    """
    out = {}
    if fx:
        ax = np.arange(x_proj.size, dtype=np.float64)
        out['x'] = (ax,
                    np.clip(x_proj - fx['offset'], 0, None),
                    np.clip(fx['curve'] - fx['offset'], 0, None))
    if fy:
        ay = np.arange(y_proj.size, dtype=np.float64)
        out['y'] = (ay,
                    np.clip(y_proj - fy['offset'], 0, None),
                    np.clip(fy['curve'] - fy['offset'], 0, None))
    return out


def results_header_html(move, fx, fy, met, reliable, started, levels,
                        autoscale):
    """Everything this run knows, as one block to sit ABOVE the image.

    Two halves. The MOVE half is fixed for the whole run and is filled in
    before the first frame arrives, so the transport can be checked while the
    MOT is still loading; the CLOUD half is replaced by every frame pair, and
    describes THAT pair alone -- nothing here is accumulated. Both are laid out
    wide rather than tall, in two label/value column pairs, so the strip stays
    short and never crowds the picture.

    Monospace throughout: proportional digits make a column of numbers that is
    meant to be compared between shots much harder to read.

    :param move: (dict) the whole-move quantities, from experiment()
    :param started: (bool) whether a frame pair has arrived yet, which is what
        tells "nothing has happened" apart from "the fit failed"
    :param levels: (lo, hi) colour range actually in use
    """
    lo, hi = levels
    m = move or {}
    d = float(m.get('d_mm', float('nan')))
    move_rows = [
        ("transport (one way)", f"{d:.3f} mm",
         "total path (out+back)", f"{2 * d:.3f} mm"),
        ("acceleration", f"{m.get('accel_mps2', float('nan')):.2f} m/s&sup2;",
         "peak velocity", f"{m.get('v_mmps', float('nan')):.1f} mm/s"),
        ("accel / decel time T_r", f"{m.get('ramp_ms', float('nan')):.2f} ms",
         "constant velocity T_p", f"{m.get('plat_ms', float('nan')):.2f} ms"),
        ("hold at far point", f"{float(m.get('turn_hold_ms', 0)):.0f} ms",
         "TOTAL MOVE TIME", f"{m.get('total_ms', float('nan')):.1f} ms"),
        ("beat detuning &Delta;f",
         f"&plusmn;{m.get('df_khz', float('nan')):.1f} kHz",
         "profile",
         f"{m.get('shape', '--')}, {m.get('n_legs', 0)} DIM segments"),
        ("COM4 / COM5 outward",
         (f"{m.get('df_up_khz', float('nan')):+.1f} / "
          f"{m.get('df_dn_khz', float('nan')):+.1f} kHz"),
         "sweep mode",
         "both AOMs" if m.get('dual') else "COM4 only, COM5 at base"),
        ("pre / post-move hold",
         f"{float(m.get('pre_hold_ms', 0)):.0f} / "
         f"{float(m.get('post_hold_ms', 0)):.0f} ms",
         "TOF / full cycle",
         f"{float(m.get('tof_us', 0)):.0f} &micro;s / "
         f"{m.get('cycle_ms', float('nan')):.0f} ms"),
    ]
    if met is None or not fx or not fy:
        cloud_rows = [
            ("cloud",
             "waiting for the first frame pair" if not started
             else "no cloud region -- fit unusable", "", ""),
        ]
    else:
        cloud_rows = [
            ("1/e&sup2; radius x",
             f"{fx['w']:.1f} px = {met['wx_um']:.0f} &micro;m",
             "brightness (2w)", f"{met['counts']:.3e} counts"),
            ("1/e&sup2; radius y",
             f"{fy['w']:.1f} px = {met['wy_um']:.0f} &micro;m",
             "ATOM NUMBER N(seen)", f"{met['n_atoms']:.3e}"),
            ("sigma x / y", f"{fx['sigma']:.1f} / {fy['sigma']:.1f} px",
             "N(total)",
             f"{met['n_atoms_total']:.3e} "
             f"({met['capture_fraction'] * 100:.1f}% on sensor)"),
            ("centre", f"({fx['center']:.0f}, {fy['center']:.0f}) px",
             "total brightness (fit)", f"{met['total_fit']:.3e} counts"),
        ]
    cloud_rows.append(
        ("display",
         f"({lo:.0f}, {hi:.0f}) {'auto' if autoscale else 'manual'}",
         "source", "single shot (latest pair)"))

    def _rows(rows, colour):
        return "".join(
            f"<tr><td>{a}</td>"
            f"<td><b><span style='color:{colour}'>{b}</span></b></td>"
            f"<td>&nbsp;&nbsp;&nbsp;{c}</td>"
            f"<td><b><span style='color:{colour}'>{e}</span></b></td></tr>"
            for a, b, c, e in rows)

    rule = "<tr><td colspan=4>&nbsp;</td></tr>"
    warn = "" if (reliable or met is None) else (
        "<tr><td colspan=4><span style='color:#ff8a80'>"
        "fit unreliable -- the cloud numbers above are meaningless"
        "</span></td></tr>")
    return (f"<div style='font-family:monospace'>"
            f"<table cellspacing=2 cellpadding=1>"
            f"{_rows(move_rows, '#4DD0E1')}{rule}"
            f"{_rows(cloud_rows, '#FFB74D')}{warn}"
            f"</table></div>")


class MoveLiveView:
    """One window: every number above, one picture below.

    pyqtgraph is not thread safe. Calling setImage/setText straight from the
    acquisition thread appears to work and then takes the WHOLE PROCESS down
    with an access violation -- no Python traceback, every RPyC client
    disconnecting at once. Worse, the NI tasks are armed inside the server
    processes, which survive; nothing runs the finally block, so those servers
    keep holding dio0..dio6 and every later run dies at arm() with -200597.

    So experiment() only ever calls set_header() and submit(), which copy plain
    data and touch no Qt at all, and a QTimer owned by the GUI thread does the
    drawing. Same split fastLattice and the scan-grid scripts already use.

    The header is a SEPARATE slot from the image on purpose: the move
    parameters are worth reading while the MOT is still loading, which is
    minutes before the first frame pair exists.
    """

    POLL_MS = 200
    MARGIN_PX = 60
    HEADER_PX = 250

    def __init__(self, title, levels):
        self._title = title
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(880, 1020)
        # Squeeze the gaps: pyqtgraph leaves ~10 px around every cell, which
        # between the image and its two profile strips is more whitespace than
        # picture.
        try:
            self.win.ci.layout.setSpacing(2)
            self.win.ci.setContentsMargins(4, 4, 4, 4)
        except Exception:
            pass
        # Row 0: the numbers, ABOVE the image and spanning the full width.
        self.header_label = self.win.addLabel(
            'collecting...', row=0, col=0, colspan=2, justify='left',
            size='10pt')
        try:
            self.win.ci.layout.setRowFixedHeight(0, self.HEADER_PX)
        except Exception:
            pass
        # Row 1 col 0: the y profile, on its side beside the image.
        y_plot = self.win.addPlot(row=1, col=0)
        y_plot.setMaximumWidth(self.MARGIN_PX)
        y_plot.hideAxis('bottom')
        y_plot.hideAxis('left')
        y_plot.setMouseEnabled(x=False, y=False)
        y_plot.setMenuEnabled(False)
        y_plot.invertX(True)
        y_plot.invertY(True)
        self.y_data = y_plot.plot([], [], pen=pg.mkPen('w', width=1))
        self.y_fit = y_plot.plot([], [], pen=pg.mkPen('y', width=2))
        # Row 1 col 1: the image itself.
        vb = self.win.addViewBox(row=1, col=1, lockAspect=True)
        vb.invertY(True)
        self.img_item = pg.ImageItem(axisOrder='row-major')
        self.img_item.setLookupTable(inferno_lut())
        self.img_item.setLevels(levels)
        vb.addItem(self.img_item)
        y_plot.setYLink(vb)
        # Row 2 col 1: the x profile, under the image and sharing its axis.
        x_plot = self.win.addPlot(row=2, col=1)
        x_plot.setMaximumHeight(self.MARGIN_PX)
        x_plot.hideAxis('bottom')
        x_plot.hideAxis('left')
        x_plot.setMouseEnabled(x=False, y=False)
        x_plot.setMenuEnabled(False)
        self.x_data = x_plot.plot([], [], pen=pg.mkPen('w', width=1))
        self.x_fit = x_plot.plot([], [], pen=pg.mkPen('y', width=2))
        x_plot.setXLink(vb)

        self._pending_header = None
        self._rendered_header = 0
        self._pending_img = None
        self._rendered_img = 0
        self._pending_suffix = None
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(self.POLL_MS)

    def set_header(self, html):
        """Called from the acquisition thread. Stores a string, touches no Qt."""
        self._pending_header = (html, self._rendered_header + 1)

    def submit(self, image, levels, profiles=None):
        """Called from the acquisition thread. Copies data, touches no Qt.

        The copy is deliberate: the worker reuses and mutates its arrays, and
        handing the widget a live reference means Qt can read one mid-write.
        """
        self._pending_img = (np.asarray(image).copy(), tuple(levels),
                             profiles, self._rendered_img + 1)

    def set_title_suffix(self, suffix):
        """Atom number onto the window title, so it is readable when minimised."""
        self._pending_suffix = suffix

    def _poll(self):
        entry = self._pending_header
        if entry is not None and entry[1] > self._rendered_header:
            self._rendered_header = entry[1]
            try:
                self.header_label.setText(entry[0], size='10pt')
            except Exception:
                pass          # a closed window must not kill the run
        suffix = self._pending_suffix
        if suffix is not None:
            self._pending_suffix = None
            try:
                self.win.setWindowTitle(f"{self._title}   |   {suffix}")
            except Exception:
                pass
        entry = self._pending_img
        if entry is None:
            return
        image, levels, profiles, version = entry
        if version <= self._rendered_img:
            return
        self._rendered_img = version
        try:
            self.img_item.setImage(image, levels=levels, autoLevels=False)
            if profiles:
                if 'x' in profiles:
                    ax, data, fit = profiles['x']
                    self.x_data.setData(ax, data)
                    self.x_fit.setData(ax, fit)
                if 'y' in profiles:
                    ay, data, fit = profiles['y']
                    self.y_data.setData(data, ay)
                    self.y_fit.setData(fit, ay)
        except Exception:
            pass


# =============================================================================
INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '800'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '6'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0'},
    'Repump_VCO': {'Repump VCO (V)': '0'},

    # ===================== PHASE 2: CMOT =====================
    # MOT -> CMOT ramp length. Every quantity that differs between the two
    # phases (AOM powers, coil current, both VCOs) is swept linearly over this
    # window; 0 makes it an instant jump. Was hardcoded at 10 ms here.
    'mot_switch_time': {'MOT Switch Time (ms) [MOT->CMOT ramp]': '15'},
    'cmot_hold_time': {'CMOT Hold Time (ms)': '5'},
    'cmot_mot_power': {'CMOT MOT AOM Power (V)': '0.6'},
    'cmot_repump_power': {'CMOT Repump AOM Power (V)': '0.2'},
    'cmot_detuning_voltage': {'CMOT MOT VCO (V) [-40MHz]': '-0.6'},
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '8'},
    'cmot_repump_vco': {'CMOT Repump VCO (V) [-3MHz]': '-0.2'},

    # ===================== PHASE 3: DARK MOT =====================
    'dark_mot_enabled': {'Dark MOT Enabled (1=yes, 0=no)': '1'},
    'dark_mot_duration': {'Dark MOT Duration (ms)': '4'},
    'dark_mot_power': {'Dark MOT Cooling Power (V)': '0.2'},
    'dark_mot_detuning': {'Dark MOT Cooling VCO (V) [-55MHz]': '-0.6'},
    'dark_mot_repump_power': {'Dark MOT Repump Power (V)': '0.2'},
    'dark_mot_repump_vco': {'Dark MOT Repump VCO (V)': '0.0'},

    # ===================== PHASE 4a: PGC COOL =====================
    'pgc_cool_ms': {'PGC Cool Duration (ms)': '5'},
    'pgc_mot_power': {'PGC Cooling Power (V)': '0.2'},
    'pgc_detuning': {'PGC Cooling VCO (V) [-122MHz]': '-2.3'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.03'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '-0.05'},

    # ===================== PHASE 4b: PGC RAMP-DOWN (lattice loading) =====================
    # PGC cooling/repump ramp from full to 0. Lattice already at full power.
    # By end of rampdown, atoms only see the lattice.
    'pgc_rampdown_ms': {'PGC Rampdown Duration (ms)': '5'},

    # ===================== LATTICE AM =====================
    # DIM-3000 set to AMP=320 (+32dBm) via dim_rf_on.py; AM=1.0V => full max.
    # AOM thermal stability: lattice TTL stays ON & AM stays at lat_am_v
    # for the ENTIRE cycle.
    'lat_am_v': {'Lattice AM (V)': '0.6'},

    # ===================== FIXED LATTICE HOLDS (before + after move) =====================
    'pre_hold_ms': {'Pre-move Hold (ms)': '5'},
    'post_hold_ms': {'Post-move Hold (ms)': '5'},

    # ===================== THE MOVE: DETUNING AND TIMES =====================
    # These three numbers ARE the move. Peak velocity, acceleration and
    # transport distance are worked out from them and reported; none of the
    # three is an input here.
    #
    # PEAK RF CHANGE PER CHANNEL. How far ONE driver walks from its base
    # frequency at the top of the ramp -- COM4 up by this much, COM5 down by
    # the same, so the BEAT that actually moves the standing wave is TWICE
    # this when both AOMs sweep and equal to it when only COM4 does.
    #
    #     v_peak = (lambda/2) * beat = 3.975e-4 mm/s per Hz of beat
    #
    # so at 795 nm, with both AOMs sweeping:
    #
    #        per channel      beat        v_peak
    #           10 kHz       20 kHz      7.95 mm/s
    #           30 kHz       60 kHz     23.85 mm/s
    #           50 kHz      100 kHz     39.75 mm/s
    #
    # Per channel rather than beat because this is the number the hardware
    # cares about: diffraction efficiency falls off away from the AOM's centre
    # frequency and the deflected beam walks with it, so this is what you are
    # really trading against when you ask for more speed.
    'df_chan_khz': {'Max RF Change per Channel (kHz, from base)': '30.0'},

    # RAMP TIME. How long the frequency takes to get from base to the peak
    # above -- one DIM linear segment. The same duration is used for every
    # ramp in the chain (accel out, decel at the far point, accel back, decel
    # home), so the profile stays symmetric and the atoms come home to the
    # position they started from.
    #
    # This is what sets the acceleration, which is NOT an input here:
    # a = v_peak / T_r. At 30 kHz per channel (23.85 mm/s) a 10 ms ramp is
    # 2.4 m/s^2 and a 5 ms ramp is 4.8 m/s^2. Short ramps are where atoms get
    # left behind; the log prints the acceleration you actually asked for.
    'ramp_ms': {'Ramp Time to Max (ms) [T_r]': '10.0'},

    # HOLD TIME AT THE PEAK. How long the detuning sits at its maximum, i.e.
    # how long the lattice runs at constant velocity -- one DIM hold segment,
    # whose length is set by the trigger train rather than by the segment.
    #
    # Constant velocity is an inertial frame and should not heat, so this is
    # the cheap way to add distance: every extra ms here adds v_peak * 1 ms to
    # the transport, at no acceleration. 0 removes the plateau segments
    # entirely and the move becomes a triangle.
    'plateau_ms': {'Hold Time at Max (ms) [T_p, 0 = triangle]': '10.0'},
    'seg_lead_ms': {'DIM3000 Trigger Lead (ms)': '3'},
    # Dwell at REST at the far point, between the outward and return legs.
    # The atoms sit still in a stationary lattice, displaced by the full
    # transport distance, for this long. It splits the reversal into
    # decel / dwell / accel and takes the chain to 7 DIM segments.
    #
    # 0 removes the segment entirely and the reversal becomes one continuous
    # ramp through zero velocity (5 segments): smoother, but the cloud gets no
    # time at the far point. 10 ms is a starting value, not a calibrated one.
    #
    # It costs time twice over -- once in the move, once against the lattice
    # lifetime -- so total move = 4*ramp + 2*plateau + turn_hold.
    # Whole ms only -- the waveform is built in
    # whole-ms blocks -- and each leg needs its own trigger, so a dwell
    # shorter than the trigger pulse cannot be represented.
    'turn_hold_ms': {'Hold at Far Point Between Legs (ms) [0 = continuous]': '0'},
    # +1 or -1: which sign of detuning -> reverses lattice motion direction.
    # This picks which way the OUTWARD leg goes; the return is always its
    # mirror, so flipping it does not change where the atoms end up.
    'move_direction': {'Move Direction (+1 or -1)': '-1'},

    # ===================== BOTH AOMs SWEEPING =====================
    # 1 = share the beat detuning between the two drivers, each walking half of
    #     it in the opposite direction. Same transport, half the excursion per
    #     AOM, so the velocity cap above can go roughly twice as high.
    #     REQUIRES dio6 to reach the seg input of BOTH DIM-3000s. If COM5's
    #     seg input is not connected, it stays at base, the beat is halved and
    #     every move silently runs at half speed over half the distance.
    # 0 = the single-beam behaviour of latticemove&back_single.py: COM4 carries
    #     the whole detuning, COM5 is parked at base and never put into segment
    #     mode. Use this until the tee is in place.
    'dual_sweep_enabled': {'Both AOMs Sweep? (1=yes, 0=COM4 only)': '1'},

    # NOTE: there is no Move Distance field. The transport distance is
    # v_peak * (T_r + T_p), worked out from the three numbers above and
    # printed in the log and on the panel. To make a stationary reference
    # run, set Max RF Change per Channel to 0: no triggers fire, the holds
    # and the imaging run unchanged, and the atoms simply sit in the
    # lattice -- exactly what Move Distance = 0 did in the other file.

    # ===================== DIM-3000 AOM DRIVER =====================
    'dim_base_freq_mhz': {'DIM3000 Base Freq (MHz)': '100.800'},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)': '32.0'},

    # ===================== FIXED TOF after release =====================
    'tof_us': {'TOF (us)': '0'},

    # ===================== IMAGING =====================
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '200'},

    # ===================== CALIBRATION =====================
    # UNUSED. Every um figure on the panel comes from the module constant
    # PIXEL_UM = PIXEL_PITCH_UM / MAGNIFICATION, which is the pixel size
    # projected back into the cloud; changing this field changes nothing.
    'pixel_size_um': {'Pixel Size (um/px) [UNUSED -- see PIXEL_UM]': '3.45'},

    # ===================== BIAS COILS =====================
    # Held CONSTANT at these values for the ENTIRE sub-sequence
    # (no ramping, on through MOT loading too). Matches sweep scripts.
    'bias_x_v': {'Bias X (V)': '0.0'},
    'bias_y_v': {'Bias Y (V)': '0.0'},
    'bias_z_v': {'Bias Z (V)': '0.0'},

    # ===================== DISPLAY =====================
    # The colour range in absolute camera counts, as in temp_PGC_aziza.
    #
    # Autoscale (1, the default here) sizes the top of each panel to its own
    # 99.95th percentile, so the picture is right before anyone knows what the
    # correct number is -- which is the state this script was in with its old
    # hardcoded (0, 20). The header on each panel prints the range it used, so
    # once those numbers settle, put one in Display Max and set this to 0.
    #
    # Manual (0) is the mode to measure in: every panel then uses Display Min
    # .. Display Max exactly, so brightness is comparable BETWEEN transport
    # distances and between runs. Autoscale is not -- each panel gets its own
    # top, so a distance that kept half the atoms looks identical to one that
    # kept them all. Both are clamped to what a pixel can physically reach
    # (+/- 65532 counts on this camera).
    # ===================== WARM-UP =====================
    # Cycles run before any data is kept, at the FIRST scan point's settings.
    # Between one arming and the next the coils are off and the light is down
    # for seconds while the waveform is finalized, uploaded and re-armed, so
    # the MOT is completely gone. The first sub-sequence after that loads from
    # ZERO atoms, while every later one only tops up a MOT that never fully
    # decayed -- which is why the first point or two come out different. These
    # cycles absorb that transient. Set to 0 to keep every shot; each one costs
    # a full sub-sequence of run time.
    'warmup_cycles': {'Warm-up Cycles (0 = none)': '3'},

    'display_autoscale': {'Autoscale Display? (1=yes, 0=manual)': '1'},
    'display_min': {'Display Min / Background (counts)': '0'},
    'display_max': {'Display Max (counts) [used when autoscale=0]': '2000'},

    # ===================== GENERAL =====================
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},
}


# ---------------------------------------------------------------------------
# Safe state applied when the run stops.  (learned from jingfastMOT)
# ---------------------------------------------------------------------------
# Without this, "stop" leaves the rails wherever the waveform happened to be.
# The AO/DO tasks run with regeneration, looping the buffer forever, and
# force_finish=True stops them mid-loop at an arbitrary sample; DAQmx then
# holds every output at that last sample when the task closes. So the coils
# could park anywhere between 0 V and the CMOT value and the VCOs at any phase
# detuning, differently on every stop.
#
# These are single-sample immediate writes made AFTER the sequence tasks are
# closed -- they would collide with the running tasks otherwise.
SAFE_AO_BASE = {
    'ao1': 0.8,     # MOT AOM drive -- held up for AOM thermal stability
    'ao5': 0.8,     # repump AOM drive
    'ao2': 0.0,     # MOT coils -- current setpoint to zero
    'ao3': 0.0,     # MOT VCO
    'ao4': 0.0,     # repump VCO
    'ao8': 0.0,     # bias X
    'ao9': 0.0,     # bias Y
    'ao10': 0.0,    # bias Z
}

# THE LATTICE IS DELIBERATELY LEFT ON, which is where this differs from
# jingfastMOT (it parks dio4/dio5 HIGH and ao6/ao7 at 0 V, i.e. lattice off).
# These scripts drive the lattice AOMs at a constant RF level for the entire
# cycle precisely so the crystals stay in one thermal state and the fibre
# coupling does not drift. Parking the lattice off between runs throws that
# away: the crystal cools, and the next run starts with the coupling walking
# back to equilibrium for minutes. So ao6/ao7 stay at the AM voltage the run
# used and dio4/dio5 stay LOW (= beam on).
#
# dio2 = 1 is "coils off": the state the sequence drives from just before
# release through imaging. The coil analog goes to 0 V regardless, so the
# current setpoint is zero either way.
SAFE_DO_LATTICE_ON = {
    'dio0': 0,      # camera trigger idle
    'dio1': 0,      # MOT AOM enable -- light off
    'dio2': 1,      # coils TTL: the sequence's "coils off" state
    'dio3': 0,      # repump AOM enable -- light off
    'dio4': 0,      # lattice up   -- LOW = ON, left running on purpose
    'dio5': 0,      # lattice down -- LOW = ON, left running on purpose
    'dio6': 0,      # DIM-3000 segment trigger idle (no move)
}


def park_outputs(NI_do, NI_ao, logger, lattice_am=0.0, lattice_on=True):
    """Drive every line to its safe value, lattice left ON and stationary.

    Best effort and never raises: this runs from a finally block, where an
    exception would mask whatever actually ended the run and would also skip
    the channels after the one that failed.

    write_static() is preferred because it forces the driver out of stack mode
    first -- if the run died between build_stack() and arm(), a plain
    set_ao_voltage() would be silently queued into a stack nobody will arm and
    nothing would actually be parked. Older servers do not have it, so fall
    back to per-channel writes, each in its own try so one bad line cannot
    leave the rest at random.

    :param lattice_am: AM voltage the run used, held on ao6/ao7
    :param lattice_on: False parks the lattice off (ao6/ao7 = 0, TTL high),
        for a run that had the lattice disabled to begin with
    """
    safe_ao = dict(SAFE_AO_BASE)
    safe_do = dict(SAFE_DO_LATTICE_ON)
    if lattice_on:
        safe_ao['ao6'] = float(lattice_am)
        safe_ao['ao7'] = float(lattice_am)
    else:
        safe_ao['ao6'] = 0.0
        safe_ao['ao7'] = 0.0
        safe_do['dio4'] = 1
        safe_do['dio5'] = 1
    try:
        NI_ao.write_static(ao=safe_ao)
        NI_do.write_static(do=safe_do)
    except Exception as exc:
        logger.warn(f"write_static unavailable ({exc}); parking channel by channel")
        for channel, value in safe_ao.items():
            try:
                NI_ao.set_ao_voltage(ao_channel=channel, voltages=value)
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value} V: {e}")
        for channel, value in safe_do.items():
            try:
                NI_do.set_do_voltage(do_channel=channel, value=bool(value))
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value}: {e}")
    logger.info(
        "Outputs parked: VCOs 0 V, coils 0 V, bias 0 V, AOM drive 0.8 V, "
        "cooling/repump light off, lattice %s."
        % (f"ON at {float(lattice_am):.2f} V (stationary)" if lattice_on
           else "OFF")
    )


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")
    dataset.camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    df_chan_khz = float(dataset.get_input_parameter("df_chan_khz"))
    ramp_ms_cfg = float(dataset.get_input_parameter("ramp_ms"))
    plat_ms_cfg = float(dataset.get_input_parameter("plateau_ms"))
    # Absolute camera counts, clamped to what a pixel can physically reach.
    disp_min = float(dataset.get_input_parameter("display_min"))
    disp_max = float(dataset.get_input_parameter("display_max"))
    disp_min = max(disp_min, -float(SENSOR_MAX_MONO16))
    disp_max = min(disp_max, float(SENSOR_MAX_MONO16))
    if disp_max <= disp_min:
        disp_max = disp_min + 1.0    # a zero-width range renders as a flat block
    # Every Qt object lives in MoveLiveView, which redraws on its own timer in
    # the GUI thread. experiment() only calls set_header() and submit().
    dataset.live = MoveLiveView(
        title=(f"Lattice move out and back -- {df_chan_khz:.1f} kHz/channel, "
               f"T_r={ramp_ms_cfg:g} ms, T_p={plat_ms_cfg:g} ms"),
        levels=(disp_min, disp_max))
    dataset.img_win = dataset.live.win
    dataset.img_item = dataset.live.img_item
    dataset.graph.hide()


def experiment(**kwargs):
    """Run the sequence, then put the RF drivers back the way they were found.

    DataTaker calls this; the sequence itself is _run_experiment(). This
    wrapper exists for its `finally:` alone -- whatever ends the run (Stop, an
    exception, a device dropping out), every DIM-3000 driver the run opened is
    restored to the settings dim_state.snapshot() recorded before the run
    wrote anything to it, and left OUT of segment mode. See dim_state.py.
    """
    snaps = []
    try:
        _run_experiment(dim_snapshots=snaps, **kwargs)
    finally:
        dim_state.restore_all(snaps, kwargs['dataset'].log)


def _run_experiment(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    logger = dataset.log

    NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
    camera_client = kwargs["fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber"]
    OPX_client = kwargs["OPX_OPX"]

    # =========================================================================
    # READ GUI PARAMETERS
    # =========================================================================
    # Phase 1: MOT
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    # Phase 2: CMOT (always on)
    CMOT_RAMP_MS = int(dataset.get_input_parameter("mot_switch_time"))
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_v = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

    # Phase 3: Dark MOT (toggleable)
    dark_enabled = int(dataset.get_input_parameter("dark_mot_enabled")) != 0
    dark_ms = int(dataset.get_input_parameter("dark_mot_duration"))
    dark_power = float(dataset.get_input_parameter("dark_mot_power"))
    dark_detuning = float(dataset.get_input_parameter("dark_mot_detuning"))
    dark_rep_power = float(dataset.get_input_parameter("dark_mot_repump_power"))
    dark_rep_vco = float(dataset.get_input_parameter("dark_mot_repump_vco"))

    # Phase 4: PGC (two-stage: cool + rampdown). Always on for this experiment.
    pgc_cool_ms = int(dataset.get_input_parameter("pgc_cool_ms"))
    pgc_rampdown_ms = int(dataset.get_input_parameter("pgc_rampdown_ms"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))
    # Lattice
    lat_am_v = float(dataset.get_input_parameter("lat_am_v"))

    # ONE move. n_scan stays because every per-point structure below --
    # schedule, legs_per_point, build_sub_sequence, the accumulators -- indexes
    # by scan point; with a single entry the sweep machinery collapses to one
    # point without any of it needing to know, and the sequence built here is
    # identical to the sweep script's. dist_values_mm is filled in AFTER the
    # schedule is built, because here the distance is an output of the timing
    # rather than the input that produces it.
    n_scan = 1
    pre_hold_ms = int(dataset.get_input_parameter("pre_hold_ms"))
    post_hold_ms = int(dataset.get_input_parameter("post_hold_ms"))
    df_chan_hz = int(round(
        float(dataset.get_input_parameter("df_chan_khz")) * 1e3))
    ramp_ms_in = max(0.0, float(dataset.get_input_parameter("ramp_ms")))
    plat_ms_in = max(0.0, float(dataset.get_input_parameter("plateau_ms")))
    seg_lead_ms = int(dataset.get_input_parameter("seg_lead_ms"))
    turn_hold_ms = max(0, int(dataset.get_input_parameter("turn_hold_ms")))
    dim_base_freq_mhz = float(dataset.get_input_parameter("dim_base_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))
    move_direction = int(dataset.get_input_parameter("move_direction"))
    dual_sweep = bool(int(float(dataset.get_input_parameter("dual_sweep_enabled"))))
    tof_us = int(dataset.get_input_parameter("tof_us"))
    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))
    warmup_cycles = max(0, int(dataset.get_input_parameter("warmup_cycles")))

    # Display, read here rather than in the view: the worker decides the colour
    # range and does the fitting, and the view only draws what it is handed.
    disp_lo = max(float(dataset.get_input_parameter("display_min")),
                  -float(SENSOR_MAX_MONO16))
    disp_hi = min(float(dataset.get_input_parameter("display_max")),
                  float(SENSOR_MAX_MONO16))
    if disp_hi <= disp_lo:
        disp_hi = disp_lo + 1.0
    disp_auto = bool(int(float(dataset.get_input_parameter("display_autoscale"))))

    # ============ Move physics ============
    # v = lambda/2 * df. For 795nm: v[mm/s] = 0.0003975 * df[Hz]
    LATTICE_WAVELENGTH = 795e-9
    v_per_Hz_mmps = LATTICE_WAVELENGTH / 2 * 1e3  # mm/s per Hz of detuning
    base_freq_hz = int(round(dim_base_freq_mhz * 1e6))

    def schedule_from_timing():
        """The move, from the peak detuning and the two durations.

        The inverse of the distance-driven schedule_for_distance() this file
        was copied from: there the distance was given and the timings fell out
        of a fixed acceleration, here the detuning and the timings are given
        and the velocity, acceleration and distance fall out of them.

            beat   = df_per_channel * (2 if dual_sweep else 1)
            v_peak = (lambda/2) * beat
            a      = v_peak / T_r
            d      = v_peak * (T_r + T_p)     <- one way, to the far point

        That last one is exact for both turnaround shapes, not an
        approximation. With a dwell at the far point the outward half is
        accelerate (v*T_r/2) + plateau (v*T_p) + decelerate (v*T_r/2); with
        the continuous turnaround the far point is the midpoint of the
        reversal ramp and the area either side of it is the same.

        Returns the SAME keys schedule_for_distance() did, so the segment
        writer, the trigger builder, the panel header and every log line below
        are untouched.

        :return: (dict) the point's v, T_r, T_p, outward frequencies and the
            total move duration including the turnaround hold.
        """
        ramp_ms = float(ramp_ms_in)
        plat_ms = float(plat_ms_in)
        # Nothing to play: no detuning, or no time to apply it in. Returned as
        # a stationary point rather than as an error, because that is the
        # reference run -- the holds and the imaging go ahead unchanged and
        # d_mm == 0 is what tells the rest of the script not to fire triggers.
        if df_chan_hz <= 0 or ramp_ms <= 0:
            return {'d_mm': 0.0, 'v_mmps': 0.0, 'ramp_ms': 0.0,
                    'plat_ms': 0.0, 'freq_hz': base_freq_hz,
                    'freq_hz_dn': base_freq_hz, 'df_hz': 0,
                    'df_up_hz': 0, 'df_dn_hz': 0, 'total_ms': 0.0,
                    'accel_mps2': 0.0}
        # Per channel in, beat out. Both drivers walk df_chan_hz in OPPOSITE
        # directions, so their difference -- the only thing that moves the
        # standing wave -- is twice it. With one driver sweeping the other
        # stays at base and the beat is just the one excursion.
        df_up = df_chan_hz
        df_dn = df_chan_hz if dual_sweep else 0
        df_hz = df_up + df_dn
        v_mmps = df_hz * v_per_Hz_mmps
        ramp_s_ = ramp_ms * 1e-3
        # Acceleration is a RESULT here, not a setting. It is reported rather
        # than bounded: this script exists to put a known ramp on the RF, and
        # silently lengthening T_r to keep a from getting large would break
        # the one guarantee it offers.
        accel_mps2 = (v_mmps * 1e-3) / ramp_s_
        # One way, to the far point. See the docstring.
        d_mm = v_mmps * (ramp_ms + plat_ms) * 1e-3
        total_ms = 4 * ramp_ms + 2 * plat_ms + (turn_hold_ms if turn_hold_ms > 0 else 0)
        return {'d_mm': d_mm, 'v_mmps': v_mmps, 'ramp_ms': ramp_ms,
                'plat_ms': plat_ms, 'total_ms': total_ms,
                'df_hz': df_hz, 'df_up_hz': df_up, 'df_dn_hz': df_dn,
                # The two drivers walk OPPOSITE ways, which is what makes the
                # beat their sum rather than their difference.
                'freq_hz': base_freq_hz + move_direction * df_up,
                'freq_hz_dn': base_freq_hz - move_direction * df_dn,
                'accel_mps2': accel_mps2}

    schedule = [schedule_from_timing()]
    # Downstream -- the scan loop, the tile labels, the warm-up log -- still
    # talks in distance, so hand it the one this timing produced.
    dist_values_mm = [schedule[0]['d_mm']]
    v_peak_mmps_per_point = [sc['v_mmps'] for sc in schedule]

    logger.info(
        f"Move physics (OUT AND BACK), specified by DETUNING AND TIME: "
        f"{df_chan_hz / 1e3:.3f} kHz per channel, T_r={ramp_ms_in:g} ms, "
        f"T_p={plat_ms_in:g} ms, turn_hold={turn_hold_ms} ms, "
        f"{'BOTH AOMs sweeping' if dual_sweep else 'COM4 only'}.")
    for sc in schedule:
        logger.info(
            f"  beat Df={sc['df_hz'] / 1e3:.3f}kHz "
            f"({'2x' if dual_sweep else '1x'} the per-channel "
            f"{df_chan_hz / 1e3:.3f}kHz) -> v={sc['v_mmps']:.2f}mm/s, "
            f"a={sc['accel_mps2']:.2f}m/s^2 (DERIVED = v/T_r), "
            f"d={sc['d_mm']:.3f}mm one way (DERIVED = v*(T_r+T_p)), "
            f"{'TRIANGLE' if sc['plat_ms'] <= 0 else 'trapezoid'}, "
            f"move={sc['total_ms']:.1f}ms, "
            f"COM4 {(sc['freq_hz'] - base_freq_hz) / 1e3:+.3f}kHz, "
            f"COM5 {(sc['freq_hz_dn'] - base_freq_hz) / 1e3:+.3f}kHz on the "
            f"outward leg; both mirror about base on the way back")
    if not dual_sweep:
        logger.warn(
            f"'Both AOMs Sweep' is OFF, so COM5 stays at base and the beat is "
            f"the per-channel {df_chan_hz / 1e3:.3f} kHz rather than twice it. "
            f"This move therefore runs at HALF the velocity and HALF the "
            f"distance it would with both sweeping "
            f"({schedule[0]['v_mmps']:.2f} mm/s, {schedule[0]['d_mm']:.3f} mm "
            f"instead of {2 * schedule[0]['v_mmps']:.2f} mm/s, "
            f"{2 * schedule[0]['d_mm']:.3f} mm). That is what a PER-CHANNEL "
            f"detuning means and it is not an error -- but it is the opposite "
            f"of latticemovedouble_fast.py, where the same switch left the "
            f"velocity alone.")

    # =========================================================================
    # CONFIGURE DIM-3000 DRIVERS (called once per experiment start)
    # =========================================================================
    DIM_COM_UP = 'COM4'     # lattice up   -- always swept
    DIM_COM_DOWN = 'COM5'   # lattice down -- swept too when dual_sweep, and
    # made deaf when not, because its Seg input is
    # teed to dio6 either way

    # Longest chain anything in this family writes: out-and-back with a
    # far-point dwell is seven legs, plus the attribute-8 terminator after
    # them, so eight segments. The DIM-3000 allows twenty (manual, Wseg:
    # "N -- min 1, max 20"), so this is a fact about these scripts, not a
    # hardware limit. Used to scrub segments a PREVIOUS, longer run left
    # behind, which the terminator hides but does not erase.
    DIM_MAX_SEGMENTS = 8
    amplitude_x10 = int(round(dim_amplitude_dbm * 10))

    def _dim_open(port):
        try:
            return serial.Serial(port=port, baudrate=19200,
                                 bytesize=8, parity='N', stopbits=1, timeout=1)
        except Exception:
            return None

    def _dim_send(ser, cmd):
        ser.write((cmd + '\n').encode())
        time.sleep(0.05)

    def _dim_query(ser, cmd):
        ser.flushInput()
        _dim_send(ser, cmd)
        time.sleep(0.1)
        return ser.readline().decode().strip()

    def _dim_make_deaf(ser, port):
        """Park a driver at base and make its Seg input do nothing.

        WHY THIS EXISTS. dio6 is teed to the Seg input of BOTH drivers, so
        every trigger this run fires reaches the driver it is not steering as
        well. In single-beam mode that is the down beam -- and the dual-beam
        path leaves it in Mseg:1 holding a real chain, so a single-beam run
        following a dual-beam one would fire five triggers per cycle straight
        into that stale chain. The down beam would then walk off base on its
        own, the pair would be detuned by something nobody asked for, and the
        move would be wrong in a way no image can show.

        Chain first, then Mseg:0, so an edge landing during the rewrite plays
        a no-op leg rather than a stale one. Per the manual, Mseg:0 is "off;
        Connector inactive, single frequency operation".
        """
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_send(ser, f'AMP:{amplitude_x10}')
        _dim_send(ser, 'OUT_on')
        for i in range(1, DIM_MAX_SEGMENTS + 1):
            flag = 8 if i == DIM_MAX_SEGMENTS else 1
            _dim_send(ser, f'Wseg:{i};{flag};{base_freq_hz};{base_freq_hz};0;1;0')
        _dim_send(ser, 'Mseg:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        logger.info(
            f"DIM-3000 {port}: {base_freq_hz/1e6:.6f} MHz, "
            f"+{amplitude_x10/10:.1f} dBm, ON, segments neutralised and Seg "
            f"DISABLED (Mseg:0) -- this run does not sweep it, and dio6 "
            f"reaches it through the tee.")

    def _dim_scrub_tail(ser, port, n_active):
        """Blank segments above the chain this run uses.

        the attribute-8 terminator makes the chain wrap before reaching them,
        so they should never play -- but "should never" is doing a lot of
        work there, and a stale
        leg holding a previous run's real detuning is precisely the thing that
        would move the lattice without appearing anywhere in this log. Costs a
        few serial writes once, at startup, and nothing per cycle.
        """
        for i in range(n_active + 1, DIM_MAX_SEGMENTS + 1):
            _dim_send(ser, f'Wseg:{i};1;{base_freq_hz};{base_freq_hz};0;1;0')
        if n_active < DIM_MAX_SEGMENTS:
            logger.info(
                f"DIM-3000 {port}: segments {n_active + 1}-{DIM_MAX_SEGMENTS} "
                f"blanked (left over from a longer chain).")

    def move_legs(move_freq, ramp_ms, plat_ms):
        """The out-and-back move: one leg per DIM segment, in play order.

        ONE definition, two consumers -- _dim_program_segments() turns it into
        the segment chain and build_sub_sequence() turns it into the dio6
        trigger train. They MUST agree on how many legs there are and on their
        order: a segment with no trigger stalls the chain until the NEXT
        cycle's first trigger arrives, which then plays the whole move one
        segment out of phase for every cycle afterwards, and a trigger with no
        segment runs the chain off its end. Deriving both from this one list
        is what makes disagreement impossible rather than merely unlikely.

        back_freq is the mirror of move_freq about base, so the return leg has
        the same speed and the same ramp rate as the outward one and the
        displacement cancels exactly rather than approximately.

        :param move_freq: (int) outward plateau frequency [Hz]
        :param ramp_ms: (float) accel/decel duration for THIS scan point
        :param plat_ms: (float) constant-velocity duration, 0 for a triangle
        :return: (list of dict) name, dur_ms, f0, f1. f0 == f1 marks a hold
            leg, whose duration is set by the trigger train and not by the
            segment itself.
        """
        back_freq = 2 * base_freq_hz - move_freq
        legs = [
            {'name': 'accel out', 'dur_ms': ramp_ms,
             'f0': base_freq_hz, 'f1': move_freq},
            {'name': 'plateau out', 'dur_ms': plat_ms,
             'f0': move_freq, 'f1': move_freq},
        ]
        if turn_hold_ms > 0:
            # Split turnaround: come to a stop at the far point, sit there,
            # then set off back. Three legs instead of one.
            legs += [
                {'name': 'decel to rest', 'dur_ms': ramp_ms,
                 'f0': move_freq, 'f1': base_freq_hz},
                {'name': 'dwell at far point', 'dur_ms': turn_hold_ms,
                 'f0': base_freq_hz, 'f1': base_freq_hz},
                {'name': 'accel back', 'dur_ms': ramp_ms,
                 'f0': base_freq_hz, 'f1': back_freq},
            ]
        else:
            # Continuous reversal. At one ramp rate, decelerating to rest and
            # accelerating the other way is a single straight line through
            # base with no change of slope, so it is one segment -- twice as
            # long as a plain ramp because it covers twice the frequency span.
            # Velocity crosses zero at its midpoint: that instant is the far
            # turning point.
            legs += [
                {'name': 'turnaround', 'dur_ms': 2 * ramp_ms,
                 'f0': move_freq, 'f1': back_freq},
            ]
        legs += [
            {'name': 'plateau back', 'dur_ms': plat_ms,
             'f0': back_freq, 'f1': back_freq},
            {'name': 'decel home', 'dur_ms': ramp_ms,
             'f0': back_freq, 'f1': base_freq_hz},
        ]
        # A zero-length leg must not reach the chain. It would still be
        # written as a segment, but its trigger would land on the SAME sample
        # as the next leg's and the two pulses would merge into one -- the DIM
        # would then be one trigger short, stall part-way through the chain
        # with the lattice left translating, and every cycle after the first
        # would play the move one segment out of phase. Dropping the leg
        # instead is exact, not a workaround: a hold has f0 == f1, so removing
        # it leaves the frequency chain continuous. A distance below the
        # v_max^2/a breakpoint has plat_ms == 0 and so becomes a proper
        # TRIANGLE -- accel straight into the turnaround straight into decel --
        # which is the minimum-time profile for a given acceleration.
        return [leg for leg in legs if leg['dur_ms'] > 0]

    # One leg list per scan point. The COUNT can now differ between points:
    # a distance below the triangle breakpoint has no plateau legs, so it runs
    # a shorter chain than a long one does. Both the segment writer and the
    # trigger builder index this list by scan point, so they stay in step even
    # though the chain length changes underneath them.
    legs_per_point = [move_legs(sc['freq_hz'], sc['ramp_ms'], sc['plat_ms'])
                      for sc in schedule]
    # The down beam's chain, built by the SAME function from the SAME ramp and
    # plateau durations and only a different endpoint. That is what makes one
    # dio6 train able to drive both drivers: the two chains have the same
    # number of legs and the same length for each, leg for leg, by
    # construction rather than by anyone keeping two lists in agreement.
    # move_legs() mirrors its own out leg about base to get the return leg, so
    # the down beam comes back the other way exactly as the up beam does.
    legs_dn_per_point = [move_legs(sc['freq_hz_dn'], sc['ramp_ms'], sc['plat_ms'])
                         for sc in schedule]
    n_legs_max = max(len(lg) for lg in legs_per_point)
    # Cheap, and it catches the only way the construction above could fail.
    for _i, (_lu, _ld) in enumerate(zip(legs_per_point, legs_dn_per_point)):
        if len(_lu) != len(_ld):
            logger.error(
                f"Point {_i}: the two drivers were given chains of different "
                f"length ({len(_lu)} vs {len(_ld)}). One dio6 train cannot "
                f"advance both, and the beams would end the move detuned from "
                f"each other with the lattice still walking.")

    def _dim_program_segments(ser, idx, legs=None, freq_range=None):
        """Program the whole out-and-back segment chain for `move_freq`.

        One Wseg per leg of move_legs(), in play order, EVERY ONE with
        attribute 1, then one base->base TERMINATOR segment with attribute 8
        after them. The manual is explicit that attribute 8 means "last
        segment, will not be played, by next pulse segment 1 will be played"
        -- it ends the chain, it is not a leg. Putting it on the final real
        leg (as every script in this repo used to) skips that leg entirely,
        which here is the deceleration home.

        The terminator costs one extra trigger per move; build_sub_sequence
        places it after the last leg has finished playing.

        Ramp rate is the same for every ramp leg (step_freq per step_time), so
        the turnaround, spanning twice the frequency of a single ramp, takes
        twice as long -- exactly the duration move_legs() gives it. Hold legs
        get step 0 and sit at their frequency until the next trigger, so their
        length is set by the NI trigger train, not here.

        Both the ramp rate and the endpoints now change per scan point, since
        the timings are derived from the distance rather than typed in.

        Re-asserts Mseg:1 at the end so the DIM stays in segment-trigger mode.

        :param idx: (int) scan point index into schedule / legs_per_point
        :param legs: (list) chain to write; defaults to the UP beam's. The
            down beam passes its own, which has the same durations and the
            mirrored endpoints.
        :param freq_range: (int) that chain's excursion from base, used for the
            step size. It is HALF the beat when both beams sweep, so each
            driver steps half as far per step over the same ramp time and the
            two stay in step with each other.
        """
        sc = schedule[idx]
        if legs is None:
            legs = legs_per_point[idx]
        ramp_ms = sc['ramp_ms']
        if freq_range is None:
            freq_range = abs(sc['freq_hz'] - base_freq_hz)
        max_step_us = 256.0
        min_steps = max(1, int(np.ceil(ramp_ms * 1000.0 / max_step_us)))
        num_steps = max(min_steps, 10)
        step_freq = max(1, freq_range // num_steps) if freq_range > 0 else 1
        step_time_us = (ramp_ms * 1000.0) / num_steps if ramp_ms > 0 else 1.0
        step_time_arb = max(1, min(65535, int(round(step_time_us * 1000.0 / 3.90625))))
        for i, leg in enumerate(legs, start=1):
            # Attribute 1 on EVERY real leg, the last one included.
            if leg['f0'] == leg['f1']:
                _dim_send(ser, f"Wseg:{i};1;{leg['f0']};{leg['f1']};0;1;0")
            else:
                _dim_send(ser, f"Wseg:{i};1;{leg['f0']};{leg['f1']};"
                               f"{step_freq};{step_time_arb};0")
        # The terminator. Not played; consumes one trigger and sends the next
        # one back to segment 1. base->base so that whatever the driver does
        # with the output on reaching it, it is where the last leg left it.
        term = len(legs) + 1
        _dim_send(ser, f"Wseg:{term};8;{base_freq_hz};{base_freq_hz};0;1;0")
        # Neutralise any segment left over from a LONGER chain written for a
        # previous scan point. The terminator above already makes the chain
        # wrap before reaching them, so this is insurance rather than
        # necessity -- but a stale segment holding another point's detuning
        # is precisely the thing that would move the lattice without
        # appearing anywhere in this point's log.
        for i in range(term + 1, n_legs_max + 2):
            _dim_send(ser, f"Wseg:{i};1;{base_freq_hz};{base_freq_hz};0;1;0")
        _dim_send(ser, 'Mseg:1')
        return num_steps, step_freq, step_time_arb, len(legs)

    # Open both ports ONCE and keep them open across the sweep so reprogramming
    # between scan points only costs ~10-20ms of serial writes (not open/close).
    ser_up = _dim_open(DIM_COM_UP)
    ser_dn = _dim_open(DIM_COM_DOWN)
    # Record the drivers BEFORE anything is written to them, so the
    # wrapper around this function can put them back exactly as found
    # when the run stops, however it stops (dim_state.py).
    kwargs['dim_snapshots'].append(dim_state.snapshot(ser_up, DIM_COM_UP, logger))
    kwargs['dim_snapshots'].append(dim_state.snapshot(ser_dn, DIM_COM_DOWN, logger))
    # ---- From here to the finally below, BOTH PORTS ARE OPEN. ----
    # The close used to sit at the end of the function with nothing guarding
    # it, so it ran only when the run ended the way it was supposed to. Any
    # exception in between -- a failed arm, a lost camera, a stop that raised
    # -- walked straight past it, and the ports stayed claimed inside the
    # still-running DataTaker process. Windows then refuses the next open with
    # PermissionError(13) "Access is denied", which reads like a cable fault
    # and is not one: motdriverinit, and every later run, fail until that
    # process is killed.
    #
    # Worse than the inconvenience: a driver abandoned mid-run is left in
    # segment mode at whatever frequency its last segment reached, so the two
    # beams stay detuned and the lattice KEEPS WALKING. The finally parks both
    # at base before letting go, which is the only thing that stops it.
    try:
        if ser_up is not None and ser_dn is not None:
            id1 = _dim_query(ser_up, '*IDN?')
            id2 = _dim_query(ser_dn, '*IDN?')
            logger.info(f"DIM-3000 up ({DIM_COM_UP}): {id1}")
            logger.info(f"DIM-3000 down ({DIM_COM_DOWN}): {id2}")
            if dual_sweep:
                # Swept: base, amplitude, output on. Its chain is written
                # below, and Mseg:1 there puts it into segment mode.
                _dim_send(ser_dn, f'FRQ:{base_freq_hz}')
                _dim_send(ser_dn, f'AMP:{amplitude_x10}')
                _dim_send(ser_dn, 'OUT_on')
                logger.info(
                    f"DIM-3000 down ({DIM_COM_DOWN}): "
                    f"{base_freq_hz/1e6:.6f} MHz, "
                    f"+{amplitude_x10/10:.1f} dBm, ON, segment chain below")
            else:
                # NOT swept -- the case the tee made dangerous. See
                # _dim_make_deaf: this driver still hears every dio6 pulse.
                _dim_make_deaf(ser_dn, DIM_COM_DOWN)
            # Up driver: start at base, full amp, always on, segment mode enabled
            _dim_send(ser_up, f'FRQ:{base_freq_hz}')
            _dim_send(ser_up, f'AMP:{amplitude_x10}')
            _dim_send(ser_up, 'OUT_on')
            # Program initial segments using first scan point's frequency
            ns, sf, st, nl = _dim_program_segments(ser_up, 0)
            _dim_scrub_tail(ser_up, DIM_COM_UP, n_legs_max + 1)
            _dim_send(ser_up, 'Mseg:1')
            logger.info(
                f"DIM-3000 up ({DIM_COM_UP}): initialized with point 0, {nl} active "
                f"segments (chain padded to {n_legs_max}), {ns} steps, "
                f"step_freq={sf}, step_time_arb={st} "
                f"(~{schedule[0]['ramp_ms']:.2f}ms ramp)")
            if dual_sweep:
                # The mirrored chain, same durations, same step time, half the
                # excursion each. Written second so that if the port dies part way
                # the log shows the up beam already programmed.
                ns2, sf2, st2, nl2 = _dim_program_segments(
                    ser_dn, 0, legs=legs_dn_per_point[0],
                    freq_range=abs(schedule[0]['freq_hz_dn'] - base_freq_hz))
                _dim_scrub_tail(ser_dn, DIM_COM_DOWN, n_legs_max + 1)
                _dim_send(ser_dn, 'Mseg:1')
                logger.info(
                    f"DIM-3000 down ({DIM_COM_DOWN}): initialized with point 0, "
                    f"{nl2} active segments, {ns2} steps, step_freq={sf2}, "
                    f"step_time_arb={st2} -- mirrored about "
                    f"{base_freq_hz/1e6:.3f} MHz")
                if (nl2, ns2, st2) != (nl, ns, st):
                    logger.error(
                        f"The two chains do not match: up has {nl} legs / {ns} "
                        f"steps / step_time {st}, down has {nl2} / {ns2} / {st2}. "
                        f"One dio6 pulse advances both, so they must be identical "
                        f"in shape or the beams drift apart during the move.")
                logger.warn(
                    f"BOTH AOMs SWEEP: COM4 and COM5 are both in segment mode and "
                    f"both chains are advanced by the SAME dio6 pulse, so dio6 "
                    f"must reach the seg input of BOTH drivers. If COM5's seg "
                    f"input is not connected it stays at {base_freq_hz/1e6:.3f} "
                    f"MHz, the beat is "
                    f"{schedule[0]['df_up_hz']/1e3:.1f}kHz instead of "
                    f"{schedule[0]['df_hz']/1e3:.1f}kHz, and EVERY MOVE RUNS AT "
                    f"HALF THE REQUESTED VELOCITY OVER HALF THE DISTANCE. Nothing "
                    f"here can detect that over serial -- check the cable.")
            # Read every segment back before trusting the chain. This one is
            # longer than the three segments the one-way scripts write, and a
            # driver that silently ignored segments past its own limit would leave
            # the return trip unprogrammed while every other line in this log
            # looked healthy: the atoms would go out and simply stay there. An
            # empty reply to Rseg is what that failure looks like, so it gets said
            # here rather than left to be inferred from the images.
            missing = []
            _names0 = [lg['name'] for lg in legs_per_point[0]]
            _readback = [(ser_up, DIM_COM_UP)]
            if dual_sweep:
                _readback.append((ser_dn, DIM_COM_DOWN))
            wrong_attr = []
            for _ser, _port in _readback:
                for idx in range(1, n_legs_max + 2):
                    reply = _dim_query(_ser, f'Rseg:{idx}')
                    if idx <= len(_names0):
                        label, want = _names0[idx - 1], '1'
                    elif idx == len(_names0) + 1:
                        label, want = 'terminator', '8'
                    else:
                        label, want = 'padding', None
                    logger.info(f"  {_port} Rseg:{idx} ({label}) -> "
                                f"{reply or '<no reply>'}")
                    if not reply:
                        missing.append(f"{_port}:{idx}")
                        continue
                    # Rseg answers "Rseg:N;A;B;C;D;E;F". A is the attribute:
                    # an 8 on a real leg means that leg will NEVER play.
                    try:
                        got = reply.split(';')[1].strip()
                    except IndexError:
                        got = '?'
                    if want is not None and got != want:
                        wrong_attr.append(f"{_port}:{idx}({label}) A={got}, want {want}")
            if wrong_attr:
                logger.error(
                    f"DIM-3000 segment ATTRIBUTES are wrong: {wrong_attr}. "
                    f"Attribute 8 means 'last segment, will not be played' -- a "
                    f"real leg carrying it is skipped, and the lattice steps "
                    f"straight to base instead of decelerating.")
            if missing:
                logger.error(
                    f"DIM-3000 did not read back segment(s) {missing} of {n_legs_max}. "
                    f"This driver does not appear to accept a chain this long, so "
                    f"the move plays only as far as the last segment that took and "
                    f"THE ATOMS DO NOT COME BACK. Turnaround Hold = 0 needs 5 "
                    f"segments and is the shortest chain this sequence can use; if "
                    f"5 is still too many, this driver cannot do out-and-back in "
                    f"segment mode.")
        else:
            logger.info("DIM-3000: ports unavailable, will run without serial control")
            if ser_up is not None:
                ser_up.close()
                ser_up = None
            if ser_dn is not None:
                ser_dn.close()
                ser_dn = None

        # Imaging
        coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
        coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
        vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
        vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
        camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
        atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))
        wait_time = float(dataset.get_input_parameter("wait_time"))

        bias_x_v = float(dataset.get_input_parameter("bias_x_v"))
        bias_y_v = float(dataset.get_input_parameter("bias_y_v"))
        bias_z_v = float(dataset.get_input_parameter("bias_z_v"))

        # =========================================================================
        # CHANNELS
        # =========================================================================
        camera_trigger_do = "dio0"
        MOT_AOM_do = "dio1"
        mot_coils_do = "dio2"
        Repump_AOM_do = "dio3"
        MOT_AOM_ao = "ao1"
        mot_coils_ao = "ao2"
        MOT_freq_VCO_ao = "ao3"
        Repump_freq_VCO_ao = "ao4"
        Repump_AOM_ao = "ao5"
        opx_trigger_do = 1

        # =========================================================================
        # TIMING
        # =========================================================================
        ni_rate = 20000
        SPM = ni_rate // 1000
        SPU = ni_rate / 1e6
        delay_ns = int(round(1e9 / ni_rate)) - 500
        ttl_up = max(1, SPM // 20)
        # Rounds, because the derived ramp/plateau durations are floats (a
        # 14.14ms ramp is 283 samples). The 50us sample grid is 20x finer than
        # whole ms, so snapping here costs ~0.3% of a ramp instead of ~7%.
        def ms2s(t): return int(round(t * SPM))
        def us2s(t): return int(round(t * SPU))

        mot_s = ms2s(mot_loading_ms)
        cmot_ramp_s = ms2s(CMOT_RAMP_MS)
        cmot_hold_s = ms2s(cmot_hold_ms)
        dark_s = ms2s(dark_ms) if dark_enabled else 0
        pgc_cool_s = ms2s(pgc_cool_ms)
        pgc_rampdown_s = ms2s(pgc_rampdown_ms)
        pgc_s = pgc_cool_s + pgc_rampdown_s  # total PGC duration
        pre_hold_s = ms2s(pre_hold_ms)
        post_hold_s = ms2s(post_hold_ms)
        seg_lead_s = ms2s(seg_lead_ms)
        seg_pulse_s = max(1, SPM // 20)   # ~50us trigger pulse
        tof_s_fixed = us2s(tof_us)
        clear_s = ms2s(atoms_clear_ms)
        vco_lead_s = us2s(vco_lead_us)
        coils_lead_s = ms2s(coils_lead_ms)
        coils_restore_s = ms2s(coils_restore_ms)
        aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

        # =========================================================================
        # TIMELINE ANCHORS
        # MOT -> CMOT ramp -> CMOT hold -> [Dark MOT] -> [PGC] -> release
        # =========================================================================
        cmot_start = mot_s
        cmot_hold_start = cmot_start + cmot_ramp_s
        cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s

        dark_start = cmot_end
        dark_end = dark_start + dark_s

        pgc_start = dark_end
        pgc_cool_end = pgc_start + pgc_cool_s
        pgc_end = pgc_cool_end + pgc_rampdown_s
        # release time depends on hold (per scan point) - computed in build_sub_sequence

        # Coils OFF before Dark MOT (or PGC if no Dark MOT) -- field-free phases
        if dark_enabled:
            coils_off = cmot_end - coils_lead_s
        else:
            coils_off = pgc_start - coils_lead_s

        dark_str = "ON" if dark_enabled else "OFF"
        legs_desc = " -> ".join(
            "{} {:.2f}ms".format(leg['name'], leg['dur_ms'])
            for leg in legs_per_point[0])
        logger.info(f"Time at start {time.perf_counter_ns()}")
        logger.info(
            f"Timeline: MOT 0-{mot_loading_ms}ms, "
            f"CMOT {mot_loading_ms}-{cmot_end/SPM:.0f}ms, "
            f"Dark MOT {dark_str} ({dark_start/SPM:.0f}-{dark_end/SPM:.0f}ms), "
            f"PGC_cool ({pgc_start/SPM:.0f}-{pgc_cool_end/SPM:.0f}ms), "
            f"PGC_rampdown+lattice load ({pgc_cool_end/SPM:.0f}-{pgc_end/SPM:.0f}ms), "
            f"pre_hold={pre_hold_ms}ms, "
            f"move OUT AND BACK (point 0) {schedule[0]['total_ms']:.1f}ms in "
            f"{len(legs_per_point[0])} legs ({legs_desc}), "
            f"post_hold={post_hold_ms}ms, TOF={tof_us}us"
        )
        # The first trigger of the chain fires seg_lead_ms before the move starts,
        # and the move starts pre_hold_ms after PGC ends. A lead longer than the
        # pre-move hold would push it back into the PGC ramp-down, where it would
        # start the atoms moving while the cooling light is still coming down.
        if seg_lead_ms >= pre_hold_ms:
            logger.warn(
                f"Trigger Lead ({seg_lead_ms}ms) is not shorter than the pre-move "
                f"hold ({pre_hold_ms}ms), so the first DIM trigger lands during "
                f"PGC ramp-down instead of in the hold. Raise Pre-move Hold above "
                f"{seg_lead_ms}ms.")
        # Two triggers on the same sample merge into one pulse, leaving the DIM a
        # trigger short for the rest of the run. Zero-length legs are already
        # dropped by move_legs(), so this can only fire if a leg is shorter than
        # the trigger pulse itself -- but it is the failure that produces a
        # plausible-looking run rather than an error, so it is checked rather than
        # assumed away.
        for _i, _legs in enumerate(legs_per_point):
            _starts = np.cumsum([0] + [ms2s(leg['dur_ms']) for leg in _legs[:-1]])
            if len(set(_starts)) < len(_starts):
                logger.error(
                    f"Point {_i} (d={schedule[_i]['d_mm']}mm): two DIM segment "
                    f"triggers would land on the same sample (leg starts "
                    f"{list(_starts)}). The chain would stall part-way through and "
                    f"the lattice would be left moving. Lower the acceleration so "
                    f"the ramp is longer, or raise the NI sample rate.")
        # Legs shorter than the trigger lead are not fatal -- the lead shifts every
        # pulse equally, so their spacing is preserved -- but the DIM then receives
        # the next trigger before it has acted on the previous one, and nothing
        # here can say what it does with that.
        for _i, _legs in enumerate(legs_per_point):
            _short = [lg['name'] for lg in _legs if lg['dur_ms'] < seg_lead_ms]
            if _short:
                logger.warn(
                    f"Point {_i} (d={schedule[_i]['d_mm']}mm): leg(s) {_short} are "
                    f"shorter than the {seg_lead_ms}ms trigger lead, so a segment "
                    f"trigger arrives before the DIM has started the previous "
                    f"segment. Shorten Trigger Lead, or lower Move Acceleration so "
                    f"the ramps get longer.")

        # =========================================================================
        # WAVEFORM CONSTRUCTION
        # =========================================================================
        def build_sub_sequence(idx):
            sc = schedule[idx]
            do_move = (sc['d_mm'] > 0)
            # Sequence after PGC end:
            #   pre_hold -> [out-and-back move if moving] -> post_hold -> release
            # The move is laid out leg by leg from move_legs(), the same list the
            # DIM segment chain is written from, so the trigger train cannot drift
            # out of step with the segments it is advancing.
            # For 0mm reference (dist_mm==0) the move collapses to nothing AND no
            # DIM triggers fire => atoms just sit in the stationary lattice. That
            # cycle is correspondingly SHORTER than the moving ones; it shares the
            # holds and the imaging, not the transport.
            legs = legs_per_point[idx] if do_move else []
            move_start = pgc_end + pre_hold_s
            leg_starts = []
            t = move_start
            for leg in legs:
                leg_starts.append(t)
                t += ms2s(leg['dur_ms'])
            move_end = t
            post_hold_end = move_end + post_hold_s
            release = post_hold_end
            f1 = release + tof_s_fixed
            f2 = f1 + clear_s
            crt = f2 + aom_pulse_s + coils_restore_s
            N = max(crt + ttl_up + 1, f2 + ttl_up + 1)

            cam = [0] * N
            for s in range(ttl_up):
                cam[f1 + s] = 1
                cam[f2 + s] = 1

            ado = [0] * N
            aao = [mot_aom_power] * N
            rdo = [0] * N
            rao = [repump_aom_power] * N

            # Phase 1: MOT -- full power
            for s in range(cmot_start):
                ado[s] = 1
                aao[s] = mot_aom_power
                rdo[s] = 1
                rao[s] = repump_aom_power

            # Phase 2a: CMOT ramp
            for s in range(cmot_ramp_s):
                t = cmot_start + s
                frac = s / max(cmot_ramp_s - 1, 1)
                ado[t] = 1
                aao[t] = mot_aom_power + frac * (cmot_mot_power - mot_aom_power)
                rdo[t] = 1
                rao[t] = repump_aom_power + frac * (cmot_repump_power - repump_aom_power)

            # Phase 2b: CMOT hold
            for s in range(cmot_hold_s):
                t = cmot_hold_start + s
                ado[t] = 1
                aao[t] = cmot_mot_power
                rdo[t] = 1
                rao[t] = cmot_repump_power

            # Phase 3: Dark MOT (if enabled)
            if dark_enabled:
                for s in range(dark_s):
                    t = dark_start + s
                    ado[t] = 1
                    aao[t] = dark_power
                    rdo[t] = 1
                    rao[t] = dark_rep_power

            # Phase 4a: PGC cool (full PGC light, lattice already on)
            for s in range(pgc_cool_s):
                t = pgc_start + s
                ado[t] = 1
                aao[t] = pgc_mot_power
                rdo[t] = 1
                rao[t] = pgc_rep_power
            # Phase 4b: PGC rampdown (cooling & repump amplitude -> 0; lattice unchanged)
            for s in range(pgc_rampdown_s):
                t = pgc_cool_end + s
                frac = s / max(pgc_rampdown_s - 1, 1)
                ado[t] = 1
                aao[t] = pgc_mot_power * (1.0 - frac)
                rdo[t] = 1
                rao[t] = pgc_rep_power * (1.0 - frac)
            # Lattice HOLD: PGC TTL off (no cooling light), analog stays at full
            # for thermal stability. Lattice provides the trapping during this time.
            for s in range(pgc_end, release):
                ado[s] = 0
                rdo[s] = 0
            # TOF: same (digital off, analog stays at full)
            for s in range(release, min(f1, N)):
                ado[s] = 0
                rdo[s] = 0

            # Imaging pulses: full MOT power
            for s in range(aom_pulse_s):
                if f1 + s < N:
                    ado[f1 + s] = 1
                    aao[f1 + s] = mot_aom_power
                    rdo[f1 + s] = 1
                    rao[f1 + s] = repump_aom_power
                if f2 + s < N:
                    ado[f2 + s] = 1
                    aao[f2 + s] = mot_aom_power
                    rdo[f2 + s] = 1
                    rao[f2 + s] = repump_aom_power

            # Coils DO: 0=ON, 1=OFF
            cdo = [0] * N
            for s in range(coils_off, min(crt, N)):
                cdo[s] = 1

            # Coils AO
            cao = [mot_coils_v] * N
            for s in range(cmot_ramp_s):
                t = cmot_start + s
                frac = s / max(cmot_ramp_s - 1, 1)
                cao[t] = mot_coils_v + frac * (cmot_coils_v - mot_coils_v)
            for s in range(cmot_hold_start, N):
                cao[s] = cmot_coils_v

            # MOT VCO
            mvco = [vco_loading] * N
            # CMOT ramp
            for s in range(cmot_ramp_s):
                t = cmot_start + s
                frac = s / max(cmot_ramp_s - 1, 1)
                mvco[t] = vco_loading + frac * (cmot_detuning - vco_loading)
            # CMOT hold
            for s in range(cmot_hold_start, cmot_end):
                mvco[s] = cmot_detuning
            # Dark MOT
            if dark_enabled:
                for s in range(dark_s):
                    mvco[dark_start + s] = dark_detuning
            # PGC (both stages held at pgc_detuning)
            for s in range(pgc_s):
                mvco[pgc_start + s] = pgc_detuning
            for s in range(pgc_end, N):
                mvco[s] = pgc_detuning
            # Imaging VCO override
            vlo = max(0, f1 - vco_lead_s)
            vhi = min(N, f2 + aom_pulse_s)
            for s in range(vlo, vhi):
                mvco[s] = vco_imaging

            # Repump VCO
            rvco = [repump_vco] * N
            for s in range(cmot_ramp_s):
                t = cmot_start + s
                frac = s / max(cmot_ramp_s - 1, 1)
                rvco[t] = repump_vco + frac * (cmot_repump_vco - repump_vco)
            for s in range(cmot_hold_start, cmot_end):
                rvco[s] = cmot_repump_vco
            if dark_enabled:
                for s in range(dark_s):
                    rvco[dark_start + s] = dark_rep_vco
            for s in range(pgc_s):
                rvco[pgc_start + s] = pgc_rep_vco
            for s in range(pgc_end, N):
                rvco[s] = pgc_rep_vco

            # Bias coils: held CONSTANT at GUI values for the ENTIRE sub-sequence
            # (no ramping, on through MOT loading too). Matches sweep scripts.
            biasx = [bias_x_v] * N
            biasy = [bias_y_v] * N
            biasz = [bias_z_v] * N

            # Lattice: TTL ON and AM constant at lat_am_v for the entire sub-seq
            # (matches the fast lattice code for AOM thermal stability).
            ludo = [0] * N
            lddo = [0] * N
            luao = [lat_am_v] * N
            ldao = [lat_am_v] * N

            # DIM-3000 segment trigger (dio6): ONE pulse per leg, each seg_lead_ms
            # BEFORE that leg's waveform start (the manual's "approx. 3 ms"
            # between edge and segment start), PLUS ONE MORE for the
            # attribute-8 terminator that ends the chain.
            #
            # That last pulse must not arrive until the final deceleration has
            # finished: per the manual, "if next pulse comes before segment play
            # is done, the next segment will start", so a pulse at move_end
            # would clip the end of the ramp. It is placed so the driver acts on
            # it TERM_MARGIN_MS after move_end, which covers the one extra step
            # a truncated step_freq can add to a ramp (<= 256 us) with room to
            # spare, and still lands well inside the post-move hold.
            # For 0mm reference, no triggers fire and DIM stays at base_freq.
            TERM_MARGIN_MS = 1.0
            seg_do = [0] * N
            if do_move:
                trig_times = list(leg_starts) + [move_end + ms2s(TERM_MARGIN_MS)]
                for trig_t in trig_times:
                    t0 = max(0, trig_t - seg_lead_s)
                    for s in range(seg_pulse_s):
                        if t0 + s < N:
                            seg_do[t0 + s] = 1

            return cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, biasx, biasy, biasz, ludo, lddo, luao, ldao, seg_do, N

        # Concatenate all TOF sub-sequences
        cam_p = []
        ado_p = []
        aao_p = []
        rdo_p = []
        rao_p = []
        cdo_p = []
        cao_p = []
        mvco_p = []
        rvco_p = []
        biasx_p = []
        biasy_p = []
        biasz_p = []
        ludo_p = []
        lddo_p = []
        luao_p = []
        ldao_p = []
        seg_p = []
        # Warm-up first, at the first point's settings, then the real sweep. They
        # are part of the SAME regenerating buffer, so they run at the head of
        # every pass and the MOT is always in its cyclic steady state by the time
        # point 1 is recorded. Their frames are discarded in the acquisition loop.
        # Scan-point INDICES, not values: every per-point quantity (timings,
        # detuning, leg list) now lives in schedule/legs_per_point, so passing the
        # index is what keeps a sub-sequence from ever being built against another
        # point's numbers.
        warm_idx = [0] * warmup_cycles
        for idx in warm_idx + list(range(n_scan)):
            (cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco,
             bx_w, by_w, bz_w, ludo, lddo, luao, ldao, seg_w, N) = build_sub_sequence(idx)
            cam_p += cam
            ado_p += ado
            aao_p += aao
            rdo_p += rdo
            rao_p += rao
            cdo_p += cdo
            cao_p += cao
            mvco_p += mvco
            rvco_p += rvco
            biasx_p += bx_w
            biasy_p += by_w
            biasz_p += bz_w
            ludo_p += ludo
            lddo_p += lddo
            luao_p += luao
            ldao_p += ldao
            seg_p += seg_w

        total = len(cam_p)
        sc0_v = schedule[0]['v_mmps']
        sc0_a = schedule[0]['accel_mps2']
        logger.info(
            f"Lattice MOVE-AND-BACK, single point, "
            f"{'BOTH AOMs sweeping' if dual_sweep else 'COM4 only'}: "
            f"{df_chan_hz / 1e3:.3f}kHz/channel, T_r={ramp_ms_in:g}ms, "
            f"T_p={plat_ms_in:g}ms -> out distance={dist_values_mm[0]:.3f}mm "
            f"(net displacement 0, imaged at the starting position), "
            f"v={sc0_v:.2f}mm/s, a={sc0_a:.2f}m/s^2 (both DERIVED), "
            f"move={[round(sc['total_ms'], 1) for sc in schedule]}ms, "
            f"tof={tof_us}us, {warmup_cycles} warm-up cycle(s) at "
            f"{dist_values_mm[0]:.3f}mm, total={total} samples")

        # Everything the header says about the MOVE, worked out once, here, where
        # the schedule, the leg list and the sub-sequence length are all in scope.
        # The view never recomputes any of it: a second copy of this arithmetic is
        # a second place for it to disagree with what the hardware is actually
        # doing, and the panel is exactly where that would be least likely to be
        # noticed. N is the sub-sequence length left by the concatenation loop
        # above, so "full cycle" is the real one, holds and imaging included.
        sc0 = schedule[0]
        move_summary = {
            'd_mm': sc0['d_mm'],
            'v_mmps': sc0['v_mmps'],
            'accel_mps2': sc0['accel_mps2'],
            'ramp_ms': sc0['ramp_ms'],
            'plat_ms': sc0['plat_ms'],
            'total_ms': sc0['total_ms'],
            'turn_hold_ms': turn_hold_ms,
            'df_khz': sc0['df_hz'] / 1e3,
            'df_up_khz': (sc0['freq_hz'] - base_freq_hz) / 1e3,
            'df_dn_khz': (sc0['freq_hz_dn'] - base_freq_hz) / 1e3,
            'dual': dual_sweep,
            'n_legs': len(legs_per_point[0]),
            'shape': ('stationary' if sc0['d_mm'] <= 0 else
                      'triangle' if sc0['plat_ms'] <= 0 else 'trapezoid'),
            'pre_hold_ms': pre_hold_ms,
            'post_hold_ms': post_hold_ms,
            'tof_us': tof_us,
            'cycle_ms': N / SPM,
        }
        # Straight onto the panel, before a single frame exists: the transport can
        # then be checked while the MOT is still loading, which is minutes before
        # the first pair arrives.
        try:
            dataset.live.set_header(results_header_html(
                move_summary, None, None, None, False, False,
                (disp_lo, disp_hi), disp_auto))
        except Exception as exc:
            # Never let a display detail stop a run.
            logger.warn(f"Could not put the move parameters on the panel: {exc!r}")

        # =========================================================================
        # MAIN ACQUISITION LOOP
        # =========================================================================
        while thread.running:
            camera_client.set_hardware_trigger(
                line="Line0", activation="RisingEdge",
                selector="FrameStart", overlap="ReadOut",
                acquisition_mode="Continuous")
            camera_client.set_exposure(camera_exposure_us)
            # Request it, then READ IT BACK: try_set_float is best effort, and a
            # value past the node maximum is swallowed by the driver, leaving the
            # camera on its previous gain. N scales as 10^(-G/20), so assuming the
            # requested value when the camera refused it is a factor-316 error.
            camera_client.try_set_float("Gain", CAMERA_GAIN_DB)
            try:
                applied_gain = float(camera_client.get_float("Gain"))
            except Exception:
                applied_gain = CAMERA_GAIN_DB
            if abs(applied_gain - CAMERA_GAIN_DB) > 0.01:
                logger.error(
                    "Camera gain is %.2f dB, NOT the %.2f dB requested. Atom "
                    "numbers use the ACTUAL value so they stay correct, but set "
                    "CAMERA_GAIN_DB to something the camera accepts."
                    % (applied_gain, CAMERA_GAIN_DB))
            logger.info("Starting acquisition")
            dataset.camera_client.start_acquisition()

            # Clear stale clock task from previous cycle
            try:
                NI_card_1.finalize_clock()
            except Exception:
                pass

            # Load waveforms into data cards first
            NI_card_2.build_stack()
            NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=cam_p, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel=MOT_AOM_do, value=ado_p, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel=mot_coils_do, value=cdo_p, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel=Repump_AOM_do, value=rdo_p, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel="dio4", value=ludo_p, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel="dio5", value=lddo_p, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel="dio6", value=seg_p, sample_rate=ni_rate)
            NI_card_3.build_stack()
            NI_card_3.set_ao_voltage(ao_channel=MOT_AOM_ao, voltages=aao_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel=mot_coils_ao, voltages=cao_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel=MOT_freq_VCO_ao, voltages=mvco_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel=Repump_freq_VCO_ao, voltages=rvco_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel=Repump_AOM_ao, voltages=rao_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=luao_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=ldao_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel="ao8", voltages=biasx_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel="ao9", voltages=biasy_p, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel="ao10", voltages=biasz_p, sample_rate=ni_rate)

            # Arm data cards BEFORE clock
            h1 = NI_card_2.arm(regeneration=True)
            h2 = NI_card_3.arm(regeneration=True)

            # Arm clock AFTER data cards are ready
            NI_card_1.arm_clock(length=total, sample_rate=ni_rate)

            # OPX starts clocking -- all cards are ready
            OPX_client.build_stack()
            clk = OPX_client.create_new_do_elem(do_channel=opx_trigger_do, length=500)
            M = 4
            run_buffer = 10_000_000
            with OPX_client.for_("j", 0, M, 1):
                with OPX_client.for_("i", 0, total + 1, 1):
                    OPX_client.set_digital_voltage(element=clk)
                    OPX_client.delay(delay_ns)
                OPX_client.delay(run_buffer)

            OPX_client.execute(wait=False)
            logger.info("Image gathering starts")

            def get_frame(timeout_ms=1000):
                b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
                return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

            # Stop is only a flag on the worker thread, so it takes effect where
            # the script LOOKS at it. A single get_frame() blocking for 100 s made
            # the red Stop button appear dead for up to that long. Poll in short
            # slices instead and check the flag between them: worst case the button
            # now takes GRAB_SLICE_MS to respond, not the whole timeout.
            GRAB_SLICE_MS = 500

            def grab(timeout_ms):
                """One frame, or None if it timed out or Stop was pressed."""
                waited = 0
                while waited < timeout_ms:
                    if not thread.running:
                        return None
                    try:
                        return get_frame(timeout_ms=min(GRAB_SLICE_MS,
                                                        timeout_ms - waited))
                    except Exception:
                        waited += GRAB_SLICE_MS
                if thread.running:
                    logger.warn(
                        f"Frame timed out after {timeout_ms} ms; abandoning this "
                        f"pass and re-arming.")
                return None

            live = dataset.live

            try:
                # Initial reprogram for scan point 0 (in case it differs from startup)
                if ser_up is not None:
                    _dim_program_segments(ser_up, 0)
                if dual_sweep and ser_dn is not None:
                    _dim_program_segments(
                        ser_dn, 0, legs=legs_dn_per_point[0],
                        freq_range=abs(schedule[0]['freq_hz_dn'] - base_freq_hz))
                stopping = False
                for run_idx in range(M):
                    if not thread.running:
                        stopping = True
                        break
                    logger.info(f"Run {run_idx+1} of {M}")
                    # The warm-up cycles sit at the head of the buffer and fire
                    # their camera triggers like any other cycle, so their frames
                    # MUST be consumed here. Skip this and every pair afterwards
                    # is offset by one, so the panel shows a warm-up cycle's cloud
                    # -- taken while the MOT was still coming back from the re-arm
                    # -- as though it were data, with no error to say so.
                    for _ in range(2 * warmup_cycles):
                        if grab(100000) is None:
                            stopping = True
                            break
                    if stopping:
                        break
                    for scan_idx, d_v in enumerate(dist_values_mm):
                        if not thread.running:
                            stopping = True
                            break
                        is_first = (run_idx == 0 and scan_idx == 0)
                        frame1 = grab(10000 if is_first else 100000)
                        if frame1 is None:
                            stopping = True
                            break
                        frame2 = grab(100000)
                        if frame2 is None:
                            stopping = True
                            break
                        # No DIM reprogramming between cycles. The sweep
                        # script rewrites the segment chain after every pair
                        # because the next sub-sequence is a different distance;
                        # here every cycle is the same one, so the chain written at
                        # startup is already right and rewriting it would only put
                        # serial traffic on the driver in the middle of a run.
                        diff = frame1.astype(np.int32) - frame2.astype(np.int32)
                        # THIS PAIR AND NOTHING ELSE. Nothing is accumulated
                        # between cycles, so the panel is always showing the shot
                        # that just happened and answers immediately when a
                        # parameter is turned. float64 because every fit and every
                        # sum below runs on it.
                        img = diff.astype(np.float64)
                        # Fits run on the SUMMED profiles, not on single rows:
                        # collapsing 2048 rows lifts the cloud 2048x while the
                        # noise grows only as sqrt(2048), a ~45x gain in SNR. That
                        # is what makes a fit possible on a cloud too faint to see.
                        x_proj = img.sum(axis=0)
                        y_proj = img.sum(axis=1)
                        fx = fit_gaussian_1d(x_proj)
                        fy = fit_gaussian_1d(y_proj)
                        good = bool(fx and fy and fx['reliable'] and fy['reliable'])
                        sx = fx['sigma'] if fx else float('nan')
                        sy = fy['sigma'] if fy else float('nan')
                        # frame1.dtype, NOT img.dtype: DN/photon is a property of
                        # the raw ADC word, and img is float64, which would select
                        # the 12-bit calibration and put N out by 16x.
                        # applied_gain is what the camera reported back, not what
                        # was requested.
                        met = (cloud_metrics(img, fx, fy, frame1.dtype,
                                             camera_exposure_us,
                                             gain_db=applied_gain)
                               if (fx and fy) else None)
                        # Display Min always pins the bottom, so the background
                        # sits at the same colour in every shot and only the top
                        # moves. Manual keeps the pair exactly as typed, which is
                        # what makes brightness comparable between runs.
                        hi = autoscale_top(img, disp_lo) if disp_auto else disp_hi
                        live.set_header(results_header_html(
                            move_summary, fx, fy, met, good, True,
                            (disp_lo, hi), disp_auto))
                        live.submit(img, (disp_lo, hi),
                                    profiles_from_fits(x_proj, y_proj, fx, fy))
                        if met is not None:
                            live.set_title_suffix(
                                "N(total) = %.3e" % met['n_atoms_total'])
                        # N in the log too, so a run is reconstructable afterwards
                        # from the logfile alone rather than only from whoever was
                        # watching the panel.
                        logger.info(
                            f"  d={d_v:.2f}mm (v_peak={v_peak_mmps_per_point[scan_idx]:.1f}mm/s): "
                            f"sx={sx:.1f}px, sy={sy:.1f}px"
                            + (f", N(seen)={met['n_atoms']:.3e}, "
                               f"N(total)={met['n_atoms_total']:.3e} "
                               f"({met['capture_fraction'] * 100:.1f}% seen), "
                               f"n0={met['density']:.2e}/cm3"
                               if met is not None else ", no cloud region")
                            + ("" if good else "  [UNRELIABLE FIT]"))

                    if stopping:
                        break
                    time.sleep(wait_time)
                    logger.info(f"Updated plots for run {run_idx+1} of {M}")
            finally:
                logger.info("Stopping acquisition")
                dataset.camera_client.stop_acquisition()

            logger.info(f"Cycle over {time.perf_counter_ns()}")
            time.sleep(wait_time)
            try:
                NI_card_2.finalize(h1, timeout=120.0, force_finish=True)
            except Exception as e:
                logger.info(f"NI card 2 finalize: {e!r}")
            try:
                NI_card_3.finalize(h2, timeout=120.0, force_finish=True)
            except Exception as e:
                logger.info(f"NI card 3 finalize: {e!r}")
            try:
                NI_card_1.finalize_clock()
            except Exception as e:
                logger.info(f"Clock finalize: {e!r}")
            # Sequence tasks are closed now, so static writes can land.
            # The lattice is deliberately left ON and stationary; see
            # park_outputs() for why parking it off would cost coupling.
            park_outputs(NI_card_2, NI_card_3, logger, lattice_am=lat_am_v)
    finally:
        # After the while loop ends (thread stopped): close DIM serial ports
        # (release REMOTE CONTROL display, free the COM port for next run)
        # Park the DIMs before letting go of the ports. A driver left in segment
        # mode sits at whatever frequency the last move ended on; if that is not
        # the base frequency the two beams stay detuned and the standing wave KEEPS
        # WALKING after the run has stopped, dragging any atoms still in the trap.
        # Put both back on base so the lattice is on but stationary, which is the
        # state park_outputs() assumes on the NI side.
        # BOTH drivers, not just the up one. In dual mode the down beam is left
        # in segment mode sitting at whatever frequency the last move ended on, and
        # if that is not base then the two beams stay detuned and the standing wave
        # KEEPS WALKING after the run has stopped, dragging any atoms still in the
        # trap. Parking only the up beam would leave exactly that.
        # READ BACK BEFORE PARKING. Parking writes base to both, which
        # erases the one piece of evidence that says whether the two chains
        # stayed in step: where each driver actually finished. A completed
        # move ends both on base; a driver that missed a trigger finishes
        # somewhere else, and with the Seg pulse split two ways a marginal
        # amplitude is exactly how that would happen. Logged, not judged -- a
        # run stopped mid-move ends off base legitimately.
        for _ser, _name in ((ser_up, DIM_COM_UP), (ser_dn, DIM_COM_DOWN)):
            if _ser is None:
                continue
            try:
                _f = _dim_query(_ser, 'FRQ?')
                logger.info(f"DIM {_name} finished the run at "
                            f"{_f or '<no reply>'} (base is {base_freq_hz} Hz)")
            except Exception as e:
                logger.info(f"Could not read {_name} back: {e!r}")
        for _ser, _name in ((ser_up, DIM_COM_UP), (ser_dn, DIM_COM_DOWN)):
            if _ser is None:
                continue
            try:
                _dim_send(_ser, f'FRQ:{base_freq_hz}')
                logger.info(f"DIM {_name} parked at base {base_freq_hz/1e6:.3f} MHz "
                            f"(lattice on, stationary)")
            except Exception as e:
                logger.warn(f"Could not park {_name} at base frequency: {e!r}")
            try:
                _ser.close()
            except Exception:
                pass
