# Lattice TRANSPORT: OUT AND BACK -- BOTH AOMs, DIM-3000 SWEEP MODE, SINGLE PANEL.
#
# Same experiment as latticemovedouble_fast.py -- same MOT/CMOT/Dark MOT/PGC,
# same single-shot panel, same move specified by acceleration and peak
# velocity, but WITHOUT its PGC thermometry block -- and the frequency ramps are made by the
# DIM-3000's built-in FREQUENCY SWEEP (manual v9, "Frequency sweep", SWPm)
# instead of its segment chain (Wseg/Mseg). Nothing is written to the segment
# table and segment mode is switched OFF on both drivers.
#
# THE MODE. SWPm:2, "triangle with external trigger". The manual's timing
# figure (p. 9, "Sync In/Out and output frequency relationship") shows that in
# this mode the Sync I/O input is a LEVEL, not an edge:
#
#     Sync in LOW   -> output ramps towards Fstart and sits there
#     Sync in HIGH  -> output ramps towards Fstop  and sits there
#     edge mid-ramp -> the ramp simply reverses from where it is
#
# at a fixed rate of SWPf Hz per SWPt ns. That is exactly a trapezoidal
# velocity profile. ONE gate pulse of length T_gate on a driver's Sync input
# gives accelerate (T_r) -> constant velocity -> decelerate (T_r), and the
# distance covered is v * T_gate EXACTLY, whatever T_r is, provided
# T_gate >= T_r. There is no chain, no terminator segment and no pulse count
# to keep in step: a driver whose gate is low is at Fstart, full stop.
#
# WHY BOTH DRIVERS, AND WHY EACH HAS ITS OWN GATE. A sweep runs one way only,
# between Fstart and Fstop, and rests at Fstart. The lattice velocity is
#
#     v = (lambda/2) * Df,   Df = f_up - f_down
#
# and out-and-back needs Df = 0, +Df and -Df. One driver sweeping against a
# fixed one only ever gives two beat values, so it cannot do the return leg.
# With BOTH drivers given the SAME sweep and gated SEPARATELY:
#
#     both gates low          f_up = f_down = Fstart      Df = 0  (at rest)
#     up gate high            f_up = Fstart + Df          Df = +Df (outward)
#     down gate high          f_down = Fstart + Df        Df = -Df (return)
#
# so COM4 carries the outward leg and COM5 the return (swapped by Move
# Direction = -1). The two sweeps are programmed identically, so the lattice
# comes back to rest at EXACTLY zero beat and the two legs cover exactly the
# same distance.
#
# WIRING -- DIFFERENT FROM THE SEGMENT SCRIPT, AND IT WILL BITE:
#     dio6 -> Sync I/O SMA of COM4 (lattice up)   -- ONLY that one
#     dio7 -> Sync I/O SMA of COM5 (lattice down) -- ONLY that one
# NOT the rear "Seg" input, and NOT teed. If dio6 is still teed to both
# drivers the way the segment script needed, both ramp together, the beat
# stays at zero and THE LATTICE NEVER MOVES -- with nothing in the log to say
# so. dio7 has been used as a second camera trigger in other scripts; it must
# not be connected to one here. Both lines are named at GATE_UP_DO/GATE_DN_DO.
#
# PER-AOM EXCURSION. Each driver sweeps the FULL beat Df (only one moves at a
# time). 'Sweep Range Centred' chooses where that range sits:
#     1 : Fstart = base - Df/2, Fstop = base + Df/2. Neither AOM ever goes
#         more than Df/2 from base -- the same peak excursion as the segment
#         script's dual mode -- but at REST both beams sit Df/2 BELOW base, for
#         the whole cycle (loading and imaging included).
#     0 : Fstart = base, Fstop = base + Df. At rest both beams are on base, but
#         the moving AOM goes the full Df off it (the single-beam excursion).
# Half the beat off-centre AND at base when at rest is not available: that
# needs each driver to go both ways from base, which a sweep cannot.
#
# QUANTISATION. The DIM ramps in steps of SWPf (whole Hz) every SWPt (whole
# 4 ns). choose_sweep() picks the pair whose rate is closest to the requested
# acceleration and rounds the beat to a whole number of steps; the ACHIEVED
# velocity, acceleration and ramp time are what get used for the gate timing
# and what the panel prints. The error is normally well under 0.1%.
#
# Full sequence (Move Direction = +1):
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> pre-move hold
#   -> dio6 HIGH for T_gate : COM4 ramps up, holds, and on the falling edge
#                             ramps back down to Fstart   (outward trapezoid)
#   -> T_r + Turnaround Hold after dio6 falls:
#      dio7 HIGH for T_gate : COM5 does the same           (return trapezoid)
#   -> COM5 back at Fstart after its final T_r ramp
#   -> post-move hold -> release -> short TOF -> image
#
# dio7 rises T_r after dio6 falls: that is the instant COM4 has finished
# decelerating. With Turnaround Hold = 0 the beat therefore passes straight
# through zero at the same acceleration -- one continuous ramp from +Df to
# -Df taking 2*T_r, the same smooth reversal the segment script's single
# "turnaround" segment made. A hold > 0 parks the atoms at the far point for
# that long. Total move = 4*T_r + 2*T_p + hold, as before.
#
# DISPLACEMENT. Both legs have identical gate length and identical sweeps, so
#   far point   = v * T_gate = the requested distance (to one 50 us sample)
#   final point = 0                                          <- always
# The image is taken at the STARTING position.

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
         f"&plusmn;{m.get('df_khz', float('nan')):.3f} kHz",
         "profile",
         f"{m.get('shape', '--')}, gate {m.get('gate_ms', float('nan')):.2f} ms"),
        ("sweep Fstart &rarr; Fstop",
         (f"{m.get('fstart_off_khz', float('nan')):+.1f} &rarr; "
          f"{m.get('fstop_off_khz', float('nan')):+.1f} kHz from base"),
         "sweep step",
         f"{m.get('step_hz', 0)} Hz / {m.get('step_ns', 0)} ns"),
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
    keep holding dio0..dio7 and every later run dies at arm() with -200597.

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

    # ===================== MOVE PARAMETERS (fixed across scan) =====================
    # The move is specified by ACCELERATION and PEAK VELOCITY. The ramp and
    # plateau DURATIONS are then derived per scan point from the distance:
    #
    #   d <= v_max^2/a :  triangle    v = sqrt(a*d),  T_r = sqrt(d/a),  T_p = 0
    #   otherwise      :  trapezoid   v = v_max,      T_r = v/a,  T_p = d/v-T_r
    #
    # WHY THIS WAY ROUND. Fixing the durations instead makes acceleration a
    # CONSEQUENCE of distance: at a fixed 10ms/10ms, a 1.5mm point runs at
    # 7.5 m/s^2 while a 0.1mm point runs at 0.5 m/s^2, fifteen times gentler.
    # A distance sweep is then an acceleration sweep wearing a disguise, and
    # "atoms die further out" cannot be told apart from "atoms die under
    # harder acceleration" in the resulting data. Holding acceleration fixed
    # and letting the timings follow is what makes the distance sweep a
    # measurement of distance.
    #
    # It also fixes the profile shape for free: short distances come out as
    # TRIANGLES (no plateau) rather than being stretched to
    # fill a duration someone typed in, and a triangle is the minimum-time
    # move for a given acceleration. The plateau only appears once the
    # velocity cap bites, which is the only reason to stop accelerating.
    #
    # Move time is therefore NOT fixed across the scan: it is
    # 4*T_r + 2*T_p + turn_hold, so shorter distances spend LESS time in the
    # lattice and lose fewer atoms to lifetime alone. That is real and wanted,
    # but it does mean raw atom number is not comparable between scan points
    # until it is divided by what a stationary lattice would have kept over
    # the same duration. Nothing here does that division for you.
    'move_accel_mps2': {'Move Acceleration (m/s^2)': '5.0'},
    # Ceiling on the transport speed. Constant-velocity motion is an inertial
    # frame and should not heat, so this is not the parameter that limits
    # survival -- it exists to bound the detuning (v_max = 75 mm/s is 189 kHz)
    # and to give the profile somewhere to flatten out on long moves. Set it
    # high and every point becomes a triangle.
    # Each driver sweeps the full beat, but with the range centred (below) no
    # AOM goes more than half of it from base -- the same peak excursion the
    # segment script's dual mode had.
    'move_vmax_mmps': {'Move Max Velocity (mm/s)': '25.0'},
    # How long before the move each gate edge is sent. Every edge is shifted by
    # the same amount, so this moves the whole transport earlier in time and
    # changes nothing about its shape. The segment script needed ~3 ms because
    # Mseg answers a Seg pulse "approx. 3 ms" late; the manual gives no latency
    # for the Sync input in sweep mode, so this starts at 0. Measure it on a
    # scope (Sync in vs. RF out through a mixer) and put it here.
    'sweep_lead_us': {'Sweep Gate Lead (us)': '0'},
    # Dwell at REST at the far point, between the outward and return legs:
    # the return gate rises T_r + this after the outward gate falls, so the
    # atoms sit still, displaced by the full transport distance, for this long.
    #
    # 0 makes the reversal one continuous ramp through zero velocity at the
    # same acceleration: smoother, but the cloud gets no time at the far point.
    #
    # It costs time twice over -- once in the move, once against the lattice
    # lifetime -- so total move = 4*ramp + 2*plateau + turn_hold.
    'turn_hold_ms': {'Hold at Far Point Between Legs (ms) [0 = continuous]': '0'},
    # +1: COM4 (up) is gated first, so the outward beat is f_up > f_down.
    # -1: COM5 (down) is gated first, which reverses the motion. The return is
    # always the other driver, so this does not change where the atoms end up.
    'move_direction': {'Move Direction (+1 or -1)': '-1'},

    # ===================== SWEEP RANGE =====================
    # 1 = Fstart = base - Df/2, Fstop = base + Df/2. Peak excursion Df/2 per
    #     AOM, but at REST both beams sit Df/2 below base for the whole cycle.
    # 0 = Fstart = base, Fstop = base + Df. At rest both beams are on base; the
    #     moving AOM goes the full Df off it.
    # Either way both drivers get the SAME range, so at rest the beat is zero
    # and the lattice is stationary.
    'sweep_centred': {'Sweep Range Centred on Base? (1=yes, 0=start at base)': '1'},

    # ===================== MOVE DISTANCE (single point) =====================
    # ONE transport distance, out and back. The three-point sweep this script
    # was cut down from is the right tool for asking "how far can they go";
    # this one is for the other half of that work -- park on one distance and
    # turn acceleration, holds or bias while watching a single panel.
    #
    # The timings are still DERIVED from it, at the acceleration and velocity
    # cap above, so this distance comes out as either a triangle or a
    # trapezoid; the panel header prints which, and the gate length.
    # 0 makes it a stationary reference: the holds and the imaging run
    # unchanged, no gate goes high, and the atoms simply sit in the lattice.
    'dist_mm': {'Move Distance (mm, one way)': '0.5'},

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
    'dio6': 0,      # COM4 sweep gate LOW -- driver at Fstart, not moving
    'dio7': 0,      # COM5 sweep gate LOW -- driver at Fstart, not moving
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
    dist_mm = float(dataset.get_input_parameter("dist_mm"))
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
        title=f"Lattice move out and back (DIM sweep) -- d={dist_mm:.3f} mm one way",
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
    wrote anything to it -- sweep mode included, since Gpar reports Rswpm --
    and left OUT of segment mode. See dim_state.py.
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

    # ONE transport distance. The list and n_scan stay because every per-point
    # structure below -- schedule, build_sub_sequence, the
    # accumulators -- indexes by scan point; with a single entry the sweep
    # machinery collapses to one point without any of it needing to know, and
    # the sequence built here is identical to the sweep script's.
    dist_values_mm = [float(dataset.get_input_parameter("dist_mm"))]
    n_scan = 1
    pre_hold_ms = int(dataset.get_input_parameter("pre_hold_ms"))
    post_hold_ms = int(dataset.get_input_parameter("post_hold_ms"))
    move_accel_mps2 = float(dataset.get_input_parameter("move_accel_mps2"))
    move_vmax_mmps = float(dataset.get_input_parameter("move_vmax_mmps"))
    sweep_lead_us = max(0.0, float(dataset.get_input_parameter("sweep_lead_us")))
    turn_hold_ms = max(0.0, float(dataset.get_input_parameter("turn_hold_ms")))
    dim_base_freq_mhz = float(dataset.get_input_parameter("dim_base_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))
    move_direction = -1 if int(float(dataset.get_input_parameter("move_direction"))) < 0 else 1
    sweep_centred = bool(int(float(dataset.get_input_parameter("sweep_centred"))))
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

    # NI sample grid. Defined here rather than with the rest of the timing
    # because the gate length is rounded to it, and the distance follows the
    # gate: d = v * T_gate.
    ni_rate = 20000
    SPM = ni_rate // 1000
    SPU = ni_rate / 1e6
    def ms2s(t): return int(round(t * SPM))
    def us2s(t): return int(round(t * SPU))

    # DIM-3000 sweep step limits. Manual: SWPt 4 - 262000 ns, rounded DOWN to
    # a whole 4 ns. The 1 us floor is this script's choice, not the driver's:
    # it still gives >= 100 steps on any ramp here, and keeps well clear of
    # the 4 ns end of the range, which the manual quotes but nobody here has
    # tested. It is relaxed to 4 ns only if no step >= 1 us can make the rate.
    SWEEP_STEP_NS_MIN = 1000
    SWEEP_STEP_NS_MAX = 262000
    SWEEP_MIN_STEPS = 100
    SWEEP_RATE_TOL = 1e-4     # 0.01% on the acceleration

    def choose_sweep(df_hz, ramp_s):
        """SWPf/SWPt pair whose ramp rate is closest to df_hz / ramp_s.

        The DIM steps SWPf whole Hz every SWPt ns. The RATE is what sets the
        acceleration, so the rate is what is matched; the beat is then rounded
        to a whole number of steps so the sweep lands exactly on Fstop rather
        than part-way through a step. The SMALLEST SWPf that gets the rate
        within SWEEP_RATE_TOL wins, i.e. the finest staircase that is good
        enough -- a slightly better rate match is not worth steps 50x coarser.
        Only if none gets that close is the best match taken outright.

        :param df_hz: (int) requested beat detuning [Hz]
        :param ramp_s: (float) requested ramp time [s]
        :return: (step_hz, step_ns, n_steps), or None for a zero move
        """
        if df_hz <= 0 or ramp_s <= 0:
            return None
        rate = df_hz / (ramp_s * 1e9)                 # Hz per ns
        f_max = max(1, df_hz // SWEEP_MIN_STEPS)
        best = None
        for t_min in (SWEEP_STEP_NS_MIN, 4):
            for f in range(1, f_max + 1):
                t = 4 * int(round(f / rate / 4.0))
                if t < t_min or t > SWEEP_STEP_NS_MAX:
                    continue
                err = abs(f / t - rate) / rate
                if best is None or err < best[0] - 1e-12:
                    best = (err, f, t)
                if err <= SWEEP_RATE_TOL:
                    break
            if best is not None:
                break
        if best is None:
            # Slower than 1 Hz per 262 us, or faster than f_max per 4 ns:
            # take the nearest end and let the logged acceleration say so.
            if rate < 1.0 / SWEEP_STEP_NS_MAX:
                best = (None, 1, SWEEP_STEP_NS_MAX)
            else:
                best = (None, f_max, 4)
        _, f, t = best
        n = max(1, int(round(df_hz / f)))
        return f, t, n

    def schedule_for_distance(d_mm):
        """Timings and sweep settings for one distance.

        Acceleration and peak velocity are given, the durations fall out of
        the distance:

            d <= v_max^2/a : triangle  v = sqrt(a*d), T_r = sqrt(d/a), T_p = 0
            otherwise      : trapezoid v = v_max,     T_r = v/a, T_p = d/v-T_r

        Then the ramp is QUANTISED to what the DIM can actually do
        (choose_sweep), and everything downstream uses the ACHIEVED beat and
        ramp time, not the requested ones.

        The gate length T_gate is what sets the distance: a gate of T_gate
        gives a trapezoid of area v * T_gate, whatever T_r is, provided
        T_gate >= T_r. So T_gate = d/v on the 50 us NI grid, but never shorter
        than the ramp -- a gate that fell mid-ramp would turn the sweep round
        before it reached v, and the move would be a smaller triangle.

        :param d_mm: (float) one-way transport distance [mm]
        :return: (dict) see the keys below
        """
        zero = {'d_mm': float(d_mm), 'd_actual_mm': 0.0, 'v_mmps': 0.0,
                'ramp_ms': 0.0, 'plat_ms': 0.0, 'gate_ms': 0.0,
                'total_ms': 0.0, 'accel_mps2': 0.0, 'df_hz': 0,
                'step_hz': 0, 'step_ns': 0, 'n_steps': 0,
                'fstart_hz': base_freq_hz, 'fstop_hz': base_freq_hz,
                'v_req_mmps': 0.0, 'accel_req_mps2': 0.0}
        d = float(d_mm) * 1e-3                       # m
        if d <= 0 or move_accel_mps2 <= 0:
            return zero
        v_cap = max(1e-9, move_vmax_mmps * 1e-3)     # m/s
        if d <= v_cap * v_cap / move_accel_mps2:
            v = np.sqrt(move_accel_mps2 * d)         # triangle
        else:
            v = v_cap                                # trapezoid
        ramp_req_s = v / move_accel_mps2
        df_req = int(round(v * 1e3 / v_per_Hz_mmps))
        sw = choose_sweep(df_req, ramp_req_s)
        if sw is None:
            return zero
        step_hz, step_ns, n_steps = sw
        # What the DIM will really do.
        df_hz = n_steps * step_hz
        ramp_s_ = n_steps * step_ns * 1e-9
        v_mmps = df_hz * v_per_Hz_mmps
        ramp_ms = ramp_s_ * 1e3
        # Gate on the NI grid. ceil for the ramp floor, so a triangle's gate
        # never falls before the sweep has reached Fstop.
        gate_samples = max(int(np.ceil(ramp_ms * SPM - 1e-9)),
                           ms2s(d_mm / v_mmps * 1e3))
        gate_ms = gate_samples / SPM
        # Centred: Fstart = base - Df/2. floor on the half, so the two halves
        # add back to exactly Df and Fstop = base + (Df - Df//2).
        fstart = base_freq_hz - (df_hz // 2 if sweep_centred else 0)
        return {'d_mm': float(d_mm),
                'd_actual_mm': v_mmps * gate_ms * 1e-3,
                'v_mmps': v_mmps,
                'ramp_ms': ramp_ms,
                'plat_ms': gate_ms - ramp_ms,
                'gate_ms': gate_ms,
                'total_ms': 2 * gate_ms + 2 * ramp_ms + turn_hold_ms,
                'accel_mps2': (v_mmps * 1e-3) / ramp_s_,
                'df_hz': df_hz,
                'step_hz': step_hz, 'step_ns': step_ns, 'n_steps': n_steps,
                'fstart_hz': fstart, 'fstop_hz': fstart + df_hz,
                'v_req_mmps': v * 1e3, 'accel_req_mps2': move_accel_mps2}

    schedule = [schedule_for_distance(d) for d in dist_values_mm]
    v_peak_mmps_per_point = [sc['v_mmps'] for sc in schedule]
    sc0 = schedule[0]
    do_sweep = sc0['df_hz'] > 0

    logger.info(
        f"Move physics (OUT AND BACK, DIM SWEEP MODE): a={move_accel_mps2:.2f} "
        f"m/s^2 requested, v_max={move_vmax_mmps:.1f} mm/s, "
        f"turn_hold={turn_hold_ms:g}ms. Triangle below "
        f"d={((move_vmax_mmps*1e-3)**2/move_accel_mps2*1e3 if move_accel_mps2 > 0 else 0):.2f}mm, "
        f"trapezoid above.")
    for sc in schedule:
        if sc['df_hz'] <= 0:
            logger.info(f"  d={sc['d_mm']:.3f}mm -> stationary reference, "
                        f"no gate will go high.")
            continue
        logger.info(
            f"  d={sc['d_mm']:.3f}mm -> v={sc['v_mmps']:.3f}mm/s "
            f"(asked {sc['v_req_mmps']:.3f}), a={sc['accel_mps2']:.4f}m/s^2 "
            f"(asked {sc['accel_req_mps2']:.4f}), T_r={sc['ramp_ms']:.4f}ms, "
            f"gate={sc['gate_ms']:.2f}ms (T_p={sc['plat_ms']:.3f}ms), "
            f"{'TRIANGLE' if sc['plat_ms'] < 1.0 / SPM else 'trapezoid'}, "
            f"move={sc['total_ms']:.2f}ms, distance actually "
            f"{sc['d_actual_mm']:.4f}mm. Beat Df={sc['df_hz']/1e3:.3f}kHz = "
            f"{sc['n_steps']} steps of {sc['step_hz']}Hz/{sc['step_ns']}ns; "
            f"sweep {(sc['fstart_hz']-base_freq_hz)/1e3:+.3f} -> "
            f"{(sc['fstop_hz']-base_freq_hz)/1e3:+.3f}kHz from base on BOTH "
            f"drivers.")
        if abs(sc['accel_mps2'] / sc['accel_req_mps2'] - 1.0) > 0.01:
            logger.warn(
                f"  The DIM's sweep steps cannot make a={sc['accel_req_mps2']} "
                f"m/s^2 closer than {sc['accel_mps2']:.4f} m/s^2 at this "
                f"velocity. The move runs at the achieved value.")
    if do_sweep and not (10_000_000 <= sc0['fstart_hz'] and sc0['fstop_hz'] <= 400_000_000):
        raise ValueError(
            f"Sweep {sc0['fstart_hz']} -> {sc0['fstop_hz']} Hz is outside the "
            f"DIM-3000's 10-400 MHz range.")

    # =========================================================================
    # CONFIGURE DIM-3000 DRIVERS (called once per experiment start)
    # =========================================================================
    DIM_COM_UP = 'COM4'     # lattice up   -- gated by GATE_UP_DO
    DIM_COM_DOWN = 'COM5'   # lattice down -- gated by GATE_DN_DO
    # Sweep gates. Each goes to the Sync I/O SMA of ONE driver -- see the
    # WIRING note in the header.
    GATE_UP_DO = "dio6"
    GATE_DN_DO = "dio7"
    amplitude_x10 = int(round(dim_amplitude_dbm * 10))
    # Manual: about ten commands a second is the ceiling. Configuration only
    # happens once, at startup, so there is no reason to push it.
    DIM_CMD_GAP_S = 0.12

    def _dim_open(port):
        try:
            return serial.Serial(port=port, baudrate=19200,
                                 bytesize=8, parity='N', stopbits=1, timeout=1)
        except Exception:
            return None

    def _dim_send(ser, cmd):
        ser.write((cmd + '\n').encode())
        time.sleep(DIM_CMD_GAP_S)

    def _dim_query(ser, cmd):
        ser.flushInput()
        _dim_send(ser, cmd)
        time.sleep(0.1)
        return ser.readline().decode(errors='replace').strip()

    def _dim_configure_sweep(ser, port):
        """Put one driver into external-gate triangle sweep. Both get this.

        Order matters:
          Mseg:0  segment play off, so the rear Seg input -- which may still
                  be teed to dio6 from the segment scripts -- does nothing.
          FM_off  manual, Appendix 1: SWP and FM cannot run together.
          SWPm:0  stop any sweep left running BEFORE its endpoints change;
                  otherwise it ramps between half-old and half-new settings
                  for the second or so the writes take.
          FRQ/AMP/OUT_on  the carrier, which is also where the driver goes
                  back to when sweep mode is switched off again.
          SWPs/SWPp/SWPf/SWPt, then SWPm:2 last.
        With the gate low the output then sits at Fstart.
        """
        _dim_send(ser, 'Mseg:0')
        _dim_send(ser, 'FM_off')
        _dim_send(ser, 'SWPm:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_send(ser, f'AMP:{amplitude_x10}')
        _dim_send(ser, 'OUT_on')
        if not do_sweep:
            logger.info(
                f"DIM-3000 {port}: {base_freq_hz/1e6:.6f} MHz, "
                f"+{amplitude_x10/10:.1f} dBm, ON, sweep OFF -- distance is 0, "
                f"stationary reference.")
            return
        _dim_send(ser, f"SWPs:{sc0['fstart_hz']}")
        _dim_send(ser, f"SWPp:{sc0['fstop_hz']}")
        _dim_send(ser, f"SWPf:{sc0['step_hz']}")
        _dim_send(ser, f"SWPt:{sc0['step_ns']}")
        _dim_send(ser, 'SWPm:2')
        logger.info(
            f"DIM-3000 {port}: SWPm:2 (triangle, external gate), "
            f"{sc0['fstart_hz']/1e6:.6f} -> {sc0['fstop_hz']/1e6:.6f} MHz in "
            f"{sc0['step_hz']} Hz / {sc0['step_ns']} ns steps "
            f"({sc0['ramp_ms']:.4f} ms per ramp), +{amplitude_x10/10:.1f} dBm, "
            f"ON, segment mode OFF.")

    def _dim_verify(ser, port):
        """Read the settings back with Gpar rather than trusting the writes.

        Every sweep parameter is reported (Rswpm/Rswps/Rswpp/Rswpf/Rswpt), so
        unlike the segment table this can be checked completely. A mismatch
        in any of them is a move that is not the one on the panel.
        """
        raw = _dim_query(ser, 'Gpar')
        got = {}
        for item in raw.split('|'):
            if ':' in item:
                k, v = item.split(':', 1)
                try:
                    got[k.strip()] = int(v.strip())
                except ValueError:
                    pass
        want = {'Rout': 1, 'Rampl': amplitude_x10, 'Rfmon': 0,
                'Rswpm': 2 if do_sweep else 0}
        if do_sweep:
            want.update({'Rswps': sc0['fstart_hz'], 'Rswpp': sc0['fstop_hz'],
                         'Rswpf': sc0['step_hz'], 'Rswpt': sc0['step_ns']})
        wrong = [f"{k}={got.get(k, '<missing>')} (want {v})"
                 for k, v in want.items() if got.get(k) != v]
        if not raw:
            logger.error(f"DIM-3000 {port}: no reply to Gpar -- its sweep "
                         f"settings could NOT be verified.")
            return False
        if wrong:
            logger.error(
                f"DIM-3000 {port}: settings did not take: {'; '.join(wrong)}. "
                f"The move will not be the one on the panel.")
            return False
        logger.info(f"DIM-3000 {port}: sweep settings verified by Gpar readback.")
        return True

    # Open both ports ONCE and keep them open for the whole run.
    ser_up = _dim_open(DIM_COM_UP)
    ser_dn = _dim_open(DIM_COM_DOWN)
    # Record the drivers BEFORE anything is written to them, so the
    # wrapper around this function can put them back exactly as found --
    # sweep mode included -- when the run stops, however it stops.
    kwargs['dim_snapshots'].append(dim_state.snapshot(ser_up, DIM_COM_UP, logger))
    kwargs['dim_snapshots'].append(dim_state.snapshot(ser_dn, DIM_COM_DOWN, logger))
    # ---- From here to the finally below, BOTH PORTS ARE OPEN. ----
    # The finally takes both drivers out of sweep mode and back to base
    # before letting go, whatever ended the run.
    try:
        if ser_up is not None and ser_dn is not None:
            id1 = _dim_query(ser_up, '*IDN?')
            id2 = _dim_query(ser_dn, '*IDN?')
            logger.info(f"DIM-3000 up ({DIM_COM_UP}): {id1}")
            logger.info(f"DIM-3000 down ({DIM_COM_DOWN}): {id2}")
            for _ser, _port in ((ser_up, DIM_COM_UP), (ser_dn, DIM_COM_DOWN)):
                _dim_configure_sweep(_ser, _port)
            for _ser, _port in ((ser_up, DIM_COM_UP), (ser_dn, DIM_COM_DOWN)):
                _dim_verify(_ser, _port)
            if do_sweep:
                logger.warn(
                    f"SWEEP MODE WIRING: {GATE_UP_DO} must reach the Sync I/O "
                    f"SMA of {DIM_COM_UP} ONLY and {GATE_DN_DO} the Sync I/O "
                    f"of {DIM_COM_DOWN} ONLY -- not the rear Seg inputs, and "
                    f"not teed. If one gate reaches both drivers they ramp "
                    f"together, the beat stays zero and the lattice never "
                    f"moves. Nothing over serial can detect that -- check the "
                    f"cables.")
        else:
            logger.warn(
                "DIM-3000: ports unavailable, running without serial control. "
                "The drivers are NOT in sweep mode, so the gates do nothing "
                "and the lattice will not move.")
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
        # ni_rate, SPM, SPU, ms2s and us2s are defined with the move physics
        # above, because the gate length is built on the same grid.
        delay_ns = int(round(1e9 / ni_rate)) - 500
        ttl_up = max(1, SPM // 20)

        mot_s = ms2s(mot_loading_ms)
        cmot_ramp_s = ms2s(CMOT_RAMP_MS)
        cmot_hold_s = ms2s(cmot_hold_ms)
        dark_s = ms2s(dark_ms) if dark_enabled else 0
        pgc_cool_s = ms2s(pgc_cool_ms)
        pgc_rampdown_s = ms2s(pgc_rampdown_ms)
        pgc_s = pgc_cool_s + pgc_rampdown_s  # total PGC duration
        pre_hold_s = ms2s(pre_hold_ms)
        post_hold_s = ms2s(post_hold_ms)
        gate_lead_s = us2s(sweep_lead_us)
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
        first_name = DIM_COM_UP if move_direction > 0 else DIM_COM_DOWN
        second_name = DIM_COM_DOWN if move_direction > 0 else DIM_COM_UP
        logger.info(f"Time at start {time.perf_counter_ns()}")
        logger.info(
            f"Timeline: MOT 0-{mot_loading_ms}ms, "
            f"CMOT {mot_loading_ms}-{cmot_end/SPM:.0f}ms, "
            f"Dark MOT {dark_str} ({dark_start/SPM:.0f}-{dark_end/SPM:.0f}ms), "
            f"PGC_cool ({pgc_start/SPM:.0f}-{pgc_cool_end/SPM:.0f}ms), "
            f"PGC_rampdown+lattice load ({pgc_cool_end/SPM:.0f}-{pgc_end/SPM:.0f}ms), "
            f"pre_hold={pre_hold_ms}ms, "
            f"move OUT AND BACK (point 0) {sc0['total_ms']:.2f}ms: "
            f"{first_name} gate {sc0['gate_ms']:.2f}ms, "
            f"{sc0['ramp_ms'] + turn_hold_ms:.2f}ms gap, "
            f"{second_name} gate {sc0['gate_ms']:.2f}ms, "
            f"{sc0['ramp_ms']:.2f}ms final ramp, "
            f"post_hold={post_hold_ms}ms, TOF={tof_us}us"
        )
        # Every gate edge fires sweep_lead_us early. A lead longer than the
        # pre-move hold would push the first one back into the PGC ramp-down,
        # starting the atoms moving while the cooling light is still coming
        # down.
        if do_sweep and sweep_lead_us >= pre_hold_ms * 1e3:
            logger.warn(
                f"Sweep Gate Lead ({sweep_lead_us:g}us) is not shorter than the "
                f"pre-move hold ({pre_hold_ms}ms), so the first gate rises "
                f"during PGC ramp-down instead of in the hold. Raise Pre-move "
                f"Hold.")

        # =========================================================================
        # WAVEFORM CONSTRUCTION
        # =========================================================================
        def build_sub_sequence(idx):
            """One cycle's waveforms."""
            sc = schedule[idx]
            do_move = sc['df_hz'] > 0
            # Sequence after PGC end:
            #   pre_hold -> [out-and-back move if moving] -> post_hold -> release
            # The move is two gate pulses, one per driver:
            #   first gate  [g1_on, g1_off)   outward trapezoid
            #   second gate [g2_on, g2_off)   return trapezoid
            # g2_on = g1_off + T_r + turn_hold: the first driver has just
            # finished ramping back to Fstart when the second sets off, so
            # with no hold the beat passes straight through zero at constant
            # acceleration. Both gates are the SAME number of samples, which
            # is what makes the return distance equal the outward one.
            # For a 0mm reference no gate goes high and the atoms just sit in
            # the stationary lattice. That cycle is correspondingly SHORTER
            # than a moving one; it shares the holds and the imaging.
            move_start = pgc_end + pre_hold_s
            gates = []
            if do_move:
                gate_s = ms2s(sc['gate_ms'])        # exact: gate_ms is on the grid
                ramp_s = ms2s(sc['ramp_ms'])
                g1_on = move_start
                g1_off = g1_on + gate_s
                g2_on = g1_off + ramp_s + ms2s(turn_hold_ms)
                g2_off = g2_on + gate_s
                gates = [(g1_on, g1_off), (g2_on, g2_off)]
                # The move is over once the second driver is back at Fstart.
                move_end = g2_off + int(np.ceil(sc['ramp_ms'] * SPM - 1e-9))
            else:
                move_end = move_start
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

            # ---- DIM-3000 sweep gates (GATE_UP_DO, GATE_DN_DO) ----
            # Levels, not triggers: while a gate is HIGH its driver ramps to
            # and sits at Fstop, and when it drops the driver ramps back to
            # Fstart. Nothing is counted, so a cycle with no move (0 mm)
            # simply leaves both low and cannot desynchronise
            # anything for the cycles after it.
            gate_up = [0] * N
            gate_dn = [0] * N
            if gates:
                first, second = ((gate_up, gate_dn) if move_direction > 0
                                 else (gate_dn, gate_up))
                for arr, (on, off) in zip((first, second), gates):
                    # The lead shifts BOTH edges, so the gate length -- and
                    # with it the distance -- is unchanged.
                    for s in range(max(0, on - gate_lead_s),
                                   min(N, off - gate_lead_s)):
                        arr[s] = 1
                # Overlap would mean both drivers ramping at once, which
                # cancels in the beat: the lattice would stop part-way.
                if any(u and d for u, d in zip(gate_up, gate_dn)):
                    logger.error(
                        f"Point {idx}: the two sweep gates overlap. Both "
                        f"drivers would be ramping together and the lattice "
                        f"would not follow the requested profile.")

            return cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, biasx, biasy, biasz, ludo, lddo, luao, ldao, gate_up, gate_dn, N

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
        gate_up_p = []
        gate_dn_p = []
        # Warm-up first, at the first point's settings, then the real sweep. They
        # are part of the SAME regenerating buffer, so they run at the head of
        # every pass and the MOT is always in its cyclic steady state by the time
        # point 1 is recorded. Their frames are discarded in the acquisition loop.
        # Scan-point INDICES, not values: every per-point quantity (timings,
        # detuning, sweep steps) lives in schedule, so passing the
        # index is what keeps a sub-sequence from ever being built against another
        # point's numbers.
        # One pass through the buffer is [warm-up] [move].
        jobs = [0] * warmup_cycles + list(range(n_scan))
        for idx in jobs:
            (cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco,
             bx_w, by_w, bz_w, ludo, lddo, luao, ldao, gu_w, gd_w, N) = build_sub_sequence(idx)
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
            gate_up_p += gu_w
            gate_dn_p += gd_w

        total = len(cam_p)
        logger.info(
            f"Lattice MOVE-AND-BACK, single point, DIM SWEEP MODE, "
            f"{first_name} out / {second_name} back: "
            f"out distance={dist_values_mm[0]}mm "
            f"(net displacement 0, imaged at the starting position), "
            f"a={sc0['accel_mps2']:.3f}m/s^2 (achieved), v_max={move_vmax_mmps:.0f}mm/s, "
            f"move={[round(sc['total_ms'], 1) for sc in schedule]}ms, "
            f"tof={tof_us}us, {warmup_cycles} warm-up cycle(s) per pass, at "
            f"{dist_values_mm[0]}mm, total={total} samples")

        # Everything the header says about the MOVE, worked out once, here, where
        # the schedule, the sweep settings and the sub-sequence length are all in scope.
        # The view never recomputes any of it: a second copy of this arithmetic is
        # a second place for it to disagree with what the hardware is actually
        # doing, and the panel is exactly where that would be least likely to be
        # noticed. N is the sub-sequence length left by the concatenation loop
        # above, so "full cycle" is the real one, holds and imaging included.
        move_summary = {
            # The distance the gate actually produces, which is the request
            # rounded to one 50 us sample of travel.
            'd_mm': sc0['d_actual_mm'] if do_sweep else sc0['d_mm'],
            'v_mmps': sc0['v_mmps'],
            'accel_mps2': sc0['accel_mps2'],
            'ramp_ms': sc0['ramp_ms'],
            'plat_ms': sc0['plat_ms'],
            'total_ms': sc0['total_ms'],
            'turn_hold_ms': turn_hold_ms,
            'df_khz': sc0['df_hz'] / 1e3,
            'gate_ms': sc0['gate_ms'],
            'fstart_off_khz': (sc0['fstart_hz'] - base_freq_hz) / 1e3,
            'fstop_off_khz': (sc0['fstop_hz'] - base_freq_hz) / 1e3,
            'step_hz': sc0['step_hz'],
            'step_ns': sc0['step_ns'],
            'shape': ('stationary' if not do_sweep else
                      'triangle' if sc0['plat_ms'] < 1.0 / SPM else 'trapezoid'),
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
            NI_card_2.set_do_voltage(do_channel=GATE_UP_DO, value=gate_up_p, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel=GATE_DN_DO, value=gate_dn_p, sample_rate=ni_rate)
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
                # No DIM reprogramming per pass: the sweep settings written at
                # startup ARE the move, and there is no chain position that a
                # stopped pass could have left half-way through.
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
                        # No DIM reprogramming between cycles: every cycle is
                        # the same move, set up once at startup.
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
        # Take both drivers OUT of sweep mode before letting go of the ports.
        # A driver left in SWPm:2 follows its Sync input for as long as it is
        # powered: any stray level on that cable detunes one beam and the
        # standing wave walks, dragging whatever atoms are in it. SWPm:0
        # returns it to its carrier (FRQ), which is base, so the lattice is on
        # and stationary -- the state park_outputs() assumes on the NI side.
        # dim_state.restore() then puts back whatever sweep mode the driver
        # had before the run, which is normally 0 as well.
        for _ser, _name in ((ser_up, DIM_COM_UP), (ser_dn, DIM_COM_DOWN)):
            if _ser is None:
                continue
            try:
                _dim_send(_ser, 'SWPm:0')
                _dim_send(_ser, f'FRQ:{base_freq_hz}')
                logger.info(f"DIM {_name}: sweep OFF, parked at base "
                            f"{base_freq_hz/1e6:.3f} MHz (lattice on, stationary)")
            except Exception as e:
                logger.warn(f"Could not take {_name} out of sweep mode: {e!r}")
            try:
                _ser.close()
            except Exception:
                pass
