# Lattice TRANSPORT (move) sweep.
# Full sequence per scan point:
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> lattice hold (20ms)
#   -> ACCELERATE (DIM seg 1: ramps 10ms, base_freq -> plateau_freq)
#   -> CONSTANT VELOCITY plateau (variable duration -> sets displacement)
#   -> DECELERATE (DIM seg 3: ramps 10ms, plateau_freq -> base_freq)
#   -> post-move hold (20ms)
#   -> release -> short TOF -> image
#
# Lattice TTL & AM held constant for entire cycle (AOM thermal stability).
# The MOVE is done by detuning ONE lattice beam (up, via DIM-3000 COM4
# segment mode) while the other (down, COM5) stays at base frequency.
# A detuning Df between the two beams makes the standing wave move at
# v = (lambda/2) * Df. With lambda=795nm: v_plateau = 3.975 mm/s @ 10 kHz.
#
# Sweep is over displacement (1, 2, 3, 4 mm). At fixed plateau velocity
# and fixed ramp time, displacement = v_plateau * (t_plateau + t_ramp).
#
# Sequence per scan point:
#   1. MOT loading (full power, coils on)
#   2a. CMOT ramp (GUI: mot_switch_time)
#   2b. CMOT hold (GUI)
#      -- coils TTL off 2ms before CMOT ends --
#   3. [Dark MOT] (field-free, far detuned, low repump)
#   4. [PGC] (field-free, very far detuned, low power, sub-Doppler cooling)
#   5. Release + TOF (everything off)
#   6. Frame 1 imaging
#   7. Atoms clear
#   8. Frame 2 background
#
# PGC (polarization gradient cooling / optical molasses):
#   - No B-field (coils already off)
#   - Cooling beam: very far red-detuned (~-100 MHz), low power (~10-20%)
#   - Repump: very low (~3-5%) or off
#   - Typical duration: 5-20 ms
#   - Cools atoms well below Doppler limit via Sisyphus effect

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

M_RB87 = 1.4431e-25   # kg
K_B = 1.3806e-23       # J/K


def _sigma_from_moment(profile):
    """Fit a Gaussian + constant offset to a 1D profile.
    Uses moment-based initial guess, then scipy least-squares refinement.
    Returns (mu, sigma, bg_subtracted_profile) or (nan, nan, None).
    The returned sigma and mu come from the fit, not from moments."""
    n = profile.size
    if n < 10:
        return float('nan'), float('nan'), None
    y = profile.astype(np.float64, copy=True)

    # Baseline: use lower percentile
    baseline = np.percentile(y, 10)
    y_bg = y - baseline
    np.clip(y_bg, 0, None, out=y_bg)
    w_sum = y_bg.sum()
    if w_sum <= 0:
        return float('nan'), float('nan'), None

    x = np.arange(n, dtype=np.float64)

    # Moment-based initial guess
    mu0 = (y_bg * x).sum() / w_sum
    var0 = (y_bg * (x - mu0) ** 2).sum() / w_sum
    if var0 <= 0 or not np.isfinite(var0):
        return float('nan'), float('nan'), None
    sigma0 = np.sqrt(var0)
    amp0 = y_bg.max()

    # Least-squares Gaussian + offset fit: y = offset + amp * exp(-(x-mu)^2/(2*sigma^2))
    try:
        from scipy.optimize import curve_fit

        def gauss_with_offset(x, offset, amplitude, mu, sigma):
            return offset + amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))

        p0 = [baseline, amp0, mu0, sigma0]
        bounds_lo = [-np.inf, 0, 0, 1]
        bounds_hi = [np.inf, np.inf, n, n]
        popt, _ = curve_fit(gauss_with_offset, x, profile,
                            p0=p0, bounds=(bounds_lo, bounds_hi),
                            maxfev=2000)
        fit_offset, fit_amp, fit_mu, fit_sigma = popt

        if fit_sigma < 1 or fit_sigma > n / 2 or not np.isfinite(fit_sigma):
            # Fit returned nonsense, fall back to moments
            return float(mu0), float(sigma0), y_bg

        # Return the fit-subtracted background profile for display
        y_display = profile - fit_offset
        np.clip(y_display, 0, None, out=y_display)
        return float(fit_mu), float(fit_sigma), y_display

    except Exception:
        # scipy not available or fit failed -- fall back to moment estimate
        return float(mu0), float(sigma0), y_bg


def _gaussian_curve(n, mu, sigma, amplitude):
    """Generate (x, y) Gaussian curve. Used for overlay display."""
    x = np.arange(n, dtype=np.float64)
    y = amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return x, y


def _linear_fit(x, y):
    n = len(x)
    if n < 2:
        return float('nan'), float('nan')
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float('nan'), float('nan')
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    n = len(x)
    det = n * sxx - sx * sx
    if abs(det) < 1e-30:
        return float('nan'), float('nan')
    a = (sxx * sy - sx * sxy) / det
    b = (n * sxy - sx * sy) / det
    return float(a), float(b)


# Fallbacks only -- the live range comes from the Display Min/Max/Autoscale
# GUI fields below, in absolute camera counts, as in temp_PGC_aziza.
#
# The old hardcoded (0, 20) is why the panels came out black and yellow with
# none of inferno's midtones: this script runs the camera at 50 dB, the
# calibration is 1.84 DN/photon at 0 dB, and FLIR gain is 10^(G/20), so one
# photon is 1.84 * 316 = 582 counts. A ceiling of 20 counts put every empty
# pixel at the bottom of the map and every pixel that caught even one photon
# past the top -- two colours, no scale.
DIFF_LEVELS = (0, 2000)
AVG_DIFF_LEVELS = (0, 2000)

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


def results_table_html(fx, fy, peak, met, reliable):
    """Two-column-pair HTML table of the cloud results.

    Laid out wide rather than tall so it fits a short strip under the image and
    never has to overlap it. Monospace keeps the numbers in line.
    """
    if met is None:
        body = "<tr><td colspan=4>no cloud region -- fit unusable</td></tr>"
    else:
        rows = [
            ("1/e&sup2; radius x",
             f"{fx['w']:.1f} px = {met['wx_um']:.0f} &micro;m",
             "brightness (2w)", f"{met['counts']:.3e} counts"),
            ("1/e&sup2; radius y",
             f"{fy['w']:.1f} px = {met['wy_um']:.0f} &micro;m",
             "atom number", f"{met['n_atoms']:.3e}"),
            ("peak brightness", f"{peak:.1f} cnt/px",
             "peak density", f"{met['density']:.3e} cm&#8315;&sup3;"),
            ("total brightness", f"{met['total_fit']:.3e} counts",
             "2w / total", f"{met['frac_2w'] * 100:.1f}% (Gauss 99.97%)"),
            # Moved down from the old title line above the image.
            ("centre", f"({fx['center']:.0f}, {fy['center']:.0f}) px",
             "", ""),
        ]
        body = "".join(
            f"<tr><td>{a}</td><td><b>{b}</b></td>"
            f"<td>&nbsp;&nbsp;&nbsp;{c}</td><td><b>{d}</b></td></tr>"
            for a, b, c, d in rows)
    warn = "" if reliable else (
        "<tr><td colspan=4>"
        "<span style='color:#ff8a80'>fit unreliable -- values above are "
        "meaningless</span></td></tr>")
    return (f"<div style='font-family:monospace'>"
            f"<table cellspacing=2 cellpadding=1>{body}{warn}</table></div>")


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


# Percentile the autoscaled top follows, for BOTH grid rows -- identical to
# temp_PGC_aziza so the two scripts render the same cloud the same way.
AUTOSCALE_PCT = 99.95
AUTOSCALE_STRIDE = 4


class ScanGridWindow:
    MARGIN_PX = 55
    POLL_MS = 500

    def __init__(self, scan_labels, diff_levels, avg_levels,
                 title="Scan grid", tile_px=320, autoscale=False,
                 exposure_us=200.0):
        self.scan_labels = list(scan_labels)
        self.n_scan = len(self.scan_labels)
        self.diff_levels = diff_levels
        self.avg_levels = avg_levels
        # Manual: every panel uses avg_levels exactly, so brightness can be
        # compared across the scan. Autoscale: each panel's top follows its
        # own 99.95th percentile, which finds a faint cloud but destroys
        # any comparison between panels.
        self.autoscale = autoscale
        # Needed for the atom-number conversion. The dtype is the RAW
        # camera container, not the averaged difference: DN/photon is a
        # property of the ADC word, and the running average is float64,
        # which would silently select the 12-bit calibration and put N
        # out by 16x. set_average() fills it in from the real frames.
        self.exposure_us = exposure_us
        self.frame_dtype = np.uint16
        # Measured camera gain, filled in by set_gain_db() once the run
        # has read it back. Defaults to the requested value only so the
        # first poll before that has something sane.
        self.gain_db = CAMERA_GAIN_DB
        self.atom_numbers = [float('nan')] * self.n_scan
        # Every quantity the headers show, kept per panel so the summary
        # row can report its mean and spread across the scan.
        self.panel_stats = [None] * self.n_scan
        self._title = title
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        # Compact sizing. The tile is CAPPED, not just fitted: on a wide screen
        # "fill the available width" made a 3-point sweep 4900 px across, which
        # is no easier to read than a small one -- the cloud is a few hundred
        # pixels wide either way. Cap the tile, then shrink below it only when
        # the screen cannot fit that many panels.
        TILE_MAX, TILE_MIN = 300, 150
        HEADER_PX, SUMMARY_PX = 96, 64
        try:
            from pyqtgraph.Qt import QtWidgets as _QtW
            geo = _QtW.QApplication.instance().primaryScreen().availableGeometry()
            avail_w, avail_h = int(geo.width() * 0.92), int(geo.height() * 0.88)
        except Exception:
            avail_w, avail_h = 1400, 850      # headless / old Qt: sane default
        col_w = max(TILE_MIN, min(TILE_MAX,
                                  (avail_w - self.MARGIN_PX) // max(1, self.n_scan)))
        row_h = max(TILE_MIN, min(TILE_MAX,
                                  (avail_h - HEADER_PX - SUMMARY_PX - self.MARGIN_PX) // 2))
        self.win.resize(
            min(avail_w, col_w * self.n_scan + self.MARGIN_PX),
            min(avail_h, row_h * 2 + HEADER_PX + SUMMARY_PX + self.MARGIN_PX))
        # Squeeze the gaps between panels and their profile strips. pyqtgraph
        # defaults leave ~10 px around every cell, which across a 4-panel grid
        # is more whitespace than image.
        try:
            self.win.ci.layout.setSpacing(1)
            self.win.ci.setContentsMargins(2, 2, 2, 2)
        except Exception:
            pass
        self.current_imgs = []
        self.avg_imgs = []
        self.x_data_curves = []
        self.x_fit_curves = []
        self.y_data_curves = []
        self.y_fit_curves = []
        self.header_labels = []
        self._latest_avgs = [None] * self.n_scan
        # Per-panel autoscale tops, collected each pass and then collapsed to
        # ONE range shared by every panel. Comparability is the whole point of
        # a scan grid: if each panel scales to itself, a point with a tenth of
        # the atoms renders just as bright as the best one.
        self._avg_tops = [float('nan')] * self.n_scan
        self._cur_tops = [float('nan')] * self.n_scan
        self._rendered_versions = [0] * self.n_scan
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            lbl = self.win.addLabel(
                f"<b>{self.scan_labels[col]}</b><br>"
                f"<span style='color:#FFFF00'>sx=-- sy=--</span>",
                row=0, col=base_col, size="10pt")
            self.header_labels.append(lbl)
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            vb = self.win.addViewBox(row=1, col=base_col, lockAspect=True)
            vb.invertY(True)
            img = pg.ImageItem(axisOrder='row-major')
            img.setLevels(self.diff_levels)
            cmap = pg.colormap.get('inferno')
            img.setLookupTable(cmap.getLookupTable())
            vb.addItem(img)
            self.current_imgs.append(img)
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            y_margin_col = 2 * col
            y_plot = self.win.addPlot(row=2, col=y_margin_col)
            y_plot.setMaximumWidth(self.MARGIN_PX)
            y_plot.hideAxis('bottom')
            y_plot.hideAxis('left')
            y_plot.setMouseEnabled(x=False, y=False)
            y_plot.setMenuEnabled(False)
            y_plot.invertX(True)
            y_plot.invertY(True)
            self.y_data_curves.append(y_plot.plot([], [], pen=pg.mkPen('w', width=1)))
            self.y_fit_curves.append(y_plot.plot([], [], pen=pg.mkPen('y', width=2)))
            vb_avg = self.win.addViewBox(row=2, col=base_col, lockAspect=True)
            vb_avg.invertY(True)
            img = pg.ImageItem(axisOrder='row-major')
            img.setLevels(self.avg_levels)
            cmap = pg.colormap.get('inferno')
            img.setLookupTable(cmap.getLookupTable())
            vb_avg.addItem(img)
            self.avg_imgs.append(img)
            y_plot.setYLink(vb_avg)
            x_plot = self.win.addPlot(row=3, col=base_col)
            x_plot.setMaximumHeight(self.MARGIN_PX)
            x_plot.hideAxis('bottom')
            x_plot.hideAxis('left')
            x_plot.setMouseEnabled(x=False, y=False)
            x_plot.setMenuEnabled(False)
            self.x_data_curves.append(x_plot.plot([], [], pen=pg.mkPen('w', width=1)))
            self.x_fit_curves.append(x_plot.plot([], [], pen=pg.mkPen('y', width=2)))
            x_plot.setXLink(vb_avg)
        # Full-width summary row under everything: mean +/- sigma of every
        # quantity in the headers, over the panels that produced a
        # trustworthy fit.
        self.summary_label = self.win.addLabel(
            'collecting...', row=4, col=0, colspan=2 * self.n_scan,
            size='10pt', justify='left')
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    def _poll_and_update(self):
        for idx in range(self.n_scan):
            entry = self._latest_avgs[idx]
            if entry is None:
                continue
            avg_copy, version = entry
            if version <= self._rendered_versions[idx]:
                continue
            self._rendered_versions[idx] = version
            try:
                img = avg_copy.astype(np.float64)
                x_proj = img.sum(axis=0)
                y_proj = img.sum(axis=1)
                fx = fit_gaussian_1d(x_proj)
                fy = fit_gaussian_1d(y_proj)
                ax = np.arange(x_proj.size, dtype=np.float64)
                ay = np.arange(y_proj.size, dtype=np.float64)
                # Both traces are drawn with the FITTED offset removed, so
                # data and curve sit on the same baseline instead of the fit
                # floating above a pedestal the data still carries.
                if fx:
                    self.x_data_curves[idx].setData(
                        ax, np.clip(x_proj - fx['offset'], 0, None))
                    self.x_fit_curves[idx].setData(
                        ax, np.clip(fx['curve'] - fx['offset'], 0, None))
                if fy:
                    self.y_data_curves[idx].setData(
                        np.clip(y_proj - fy['offset'], 0, None), ay)
                    self.y_fit_curves[idx].setData(
                        np.clip(fy['curve'] - fy['offset'], 0, None), ay)
                # sigma, NOT w. fit_gaussian_1d returns both; w = 2*sigma is
                # the 1/e^2 radius, and feeding it to a sigma^2-vs-t^2
                # temperature fit would report 4x the real temperature.
                sx = f"{fx['sigma']:.1f}" if fx else "--"
                sy = f"{fy['sigma']:.1f}" if fy else "--"
                trust = bool(fx and fy and fx['reliable'] and fy['reliable'])
                # Absolute brightness, not the raw whole-frame sum. img.sum()
                # counts the pedestal across five million pixels and is
                # dominated by it; integrated_brightness() sums inside the 2w
                # ellipse with the background taken from OUTSIDE a 3w margin,
                # so what is reported is the cloud in real camera counts.
                met = None
                if fx and fy:
                    met = cloud_metrics(img, fx, fy, self.frame_dtype,
                                        self.exposure_us,
                                        gain_db=self.gain_db)
                if met is not None:
                    area = met['counts']
                    self.atom_numbers[idx] = (met['n_atoms_total'] if trust
                                              else float('nan'))
                    self.panel_stats[idx] = ({
                        'cx': fx['center'], 'cy': fy['center'],
                        'sx': fx['sigma'], 'sy': fy['sigma'],
                        'bright': area, 'N': met['n_atoms'],
                        'Ntot': met['n_atoms_total'],
                        'frac': met['capture_fraction'],
                        'n0': met['density'],
                    } if trust else None)
                else:
                    area = float(img.sum())
                    self.atom_numbers[idx] = float('nan')
                    self.panel_stats[idx] = None
                # Absolute-brightness display, jingfastMOT style. Display
                # Min pins the bottom for good; only the top moves, and only
                # when autoscale is on. 99.95th percentile rather than max,
                # so one hot pixel cannot set the scale for the whole panel.
                # Row 2 uses the identical rule as row 1, via one helper, so
                # the averaged panel cannot end up on a different scale
                # convention from the live panel above it.
                # Panel's own top, stashed rather than applied. One shared
                # range is chosen across every panel at the end of this pass
                # (see _apply_shared_levels) so the columns can be COMPARED --
                # per-panel autoscaling makes a faint point and a bright one
                # look identical, which is exactly what a sweep must not do.
                lo, hi = self.levels_for(img, self.avg_levels)
                self._avg_tops[idx] = hi
                self.header_labels[idx].setText(
                    f"<b>{self.scan_labels[idx]}</b><br>"
                    f"<span style='color:#FFFF00'>sx={sx} sy={sy} px</span><br>"
                    + (f"<span style='color:#B0BEC5'>centre="
                       f"({fx['center']:.0f},{fy['center']:.0f}) px</span><br>"
                       if (fx and fy) else
                       "<span style='color:#B0BEC5'>centre=--</span><br>")
                    + f"<span style='color:#00FF88'>bright={area:.3e} cnt</span><br>"
                    + (f"<span style='color:#FFB74D'>N(seen)={met['n_atoms']:.3e}</span>"
                       f"<br><span style='color:#FF7043'>N(total)="
                       f"{met['n_atoms_total']:.3e}  "
                       f"({met['capture_fraction'] * 100:.1f}% seen)</span><br>"
                       f"<span style='color:#FFB74D'>n0={met['density']:.2e}"
                       f"/cm3</span><br>"
                       if met is not None else
                       "<span style='color:#FFB74D'>N=--</span><br>")
                    + f"<span style='color:#88AAFF'>levels=({lo:.0f},{hi:.0f}) "
                    f"{'auto' if self.autoscale else 'manual'}</span>"
                    + ("" if trust else
                       "<br><span style='color:#FF8A80'>UNRELIABLE FIT</span>"))
            except Exception:
                pass
        self._apply_shared_levels()
        self._update_summary()

    def _apply_shared_levels(self):
        """Put every panel in a row on one common colour range.

        The top is the LARGEST of the per-panel tops, so the brightest point
        fills the map and every other panel reads as a fraction of it. Taking
        the mean or the median instead would clip the best panel, which is the
        one you are usually trying to look at.

        With autoscale off, levels_for() has already returned the typed
        Display Min/Max for every panel, so this collapses to that pair and
        the behaviour is unchanged.
        """
        for tops, imgs, base in ((self._cur_tops, self.current_imgs, self.diff_levels),
                                 (self._avg_tops, self.avg_imgs, self.avg_levels)):
            finite = [t for t in tops if np.isfinite(t)]
            if not finite:
                continue
            lo, hi = base[0], max(finite)
            if hi <= lo:
                hi = lo + 1
            for im in imgs:
                try:
                    im.setLevels((lo, hi))
                except Exception:
                    pass

    def _update_summary(self):
        """Mean and sigma of every displayed quantity, across the scan."""
        rows = [s for s in self.panel_stats if s is not None]
        n = len(rows)
        if n == 0:
            return
        # Population spread of the points themselves (ddof=1 for n>1), NOT
        # an error on the mean: the interesting question is how much the
        # quantity moves across the scan, not how well the mean is known.

        def ms(key):
            v = np.array([r[key] for r in rows], dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return float('nan'), float('nan')
            return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1
                                     else 0.0)
        parts = []
        for key, label, fmt in (
                ('cx', 'centre x', '%.0f'), ('cy', 'centre y', '%.0f'),
                ('sx', 'sx', '%.1f'), ('sy', 'sy', '%.1f'),
                ('bright', 'bright', '%.3e'), ('N', 'N(seen)', '%.3e'),
                ('Ntot', 'N(total)', '%.3e'), ('n0', 'n0', '%.2e')):
            m, s = ms(key)
            parts.append(('%s = ' + fmt + ' +/- ' + fmt) % (label, m, s))
        fm, fs = ms('frac')
        parts.append('seen = %.1f%% +/- %.1f%%' % (fm * 100, fs * 100))
        try:
            self.summary_label.setText(
                "<b>mean +/- sigma over %d/%d trusted points</b> &nbsp; %s"
                % (n, self.n_scan, ' &nbsp;|&nbsp; '.join(parts)), size='10pt')
            self.win.setWindowTitle(
                '%s   |   mean N(total) = %.3e over %d/%d points'
                % (self._title, ms('Ntot')[0], n, self.n_scan))
        except Exception:
            pass

    def levels_for(self, image, base_levels):
        """Colour range for one panel: (lo, hi).

        Display Min always pins the bottom, so the background sits at the same
        colour in every panel and only the top moves. With autoscale off the
        pair is used exactly as typed, which is what makes panels comparable
        between scan points and between runs.
        """
        lo, hi = base_levels
        if self.autoscale:
            sample = np.asarray(image)[::AUTOSCALE_STRIDE, ::AUTOSCALE_STRIDE]
            hi = float(np.percentile(sample, AUTOSCALE_PCT))
            if hi <= lo + 1:
                hi = lo + 10          # empty frame: keep a usable range
        elif hi <= lo:
            hi = lo + 1               # a zero-width range renders as a flat block
        return lo, hi

    def set_current(self, scan_idx, diff_image):
        """Row 1: the latest single difference frame for this scan point.

        Levelled per frame from its own brightness, so a faint cloud is
        visible immediately instead of sitting in the bottom few percent of
        the colour map. Computed here, on the acquisition thread, rather than
        in the poll timer -- it keeps the sort off the GUI thread, and this
        row has a new image to level on every pair anyway.
        """
        lo, hi = self.levels_for(diff_image, self.diff_levels)
        self._cur_tops[scan_idx] = hi
        try:
            # Levels here are provisional; _apply_shared_levels() re-levels
            # every panel to the common range on the next timer tick.
            self.current_imgs[scan_idx].setImage(
                diff_image, levels=(lo, hi), autoLevels=False)
        except Exception:
            pass

    def set_gain_db(self, gain_db):
        """Gain the camera actually applied, for the DN/photon rescale."""
        self.gain_db = float(gain_db)

    def set_frame_dtype(self, dtype):
        """Raw camera container dtype, for the DN/photon calibration."""
        self.frame_dtype = np.dtype(dtype)

    def set_average(self, scan_idx, avg_image):
        try:
            self.avg_imgs[scan_idx].setImage(
                avg_image, levels=self.avg_levels, autoLevels=False)
        except Exception:
            pass
        self._latest_avgs[scan_idx] = (np.asarray(avg_image).copy(), self._rendered_versions[scan_idx] + 1)


class TempFitWindow:
    POLL_MS = 1000

    def __init__(self, pixel_size_um):
        self.pixel_size_m = pixel_size_um * 1e-6
        self.win = pg.GraphicsLayoutWidget(show=True, title="Temperature from TOF")
        self.win.resize(800, 500)
        self.plot_x = self.win.addPlot(row=0, col=0, title="<b>sigma_x^2 vs t^2</b>")
        self.plot_x.setLabel('bottom', 't^2', units='s^2')
        self.plot_x.setLabel('left', 'sigma_x^2', units='m^2')
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)
        self.data_x = self.plot_x.plot([], [], pen=None, symbol='o', symbolSize=10, symbolBrush='#2196F3')
        self.fit_x = self.plot_x.plot([], [], pen=pg.mkPen('#FF9800', width=2))
        self.label_x = pg.TextItem('', color='#FF9800', anchor=(0, 1))
        self.plot_x.addItem(self.label_x)
        self.plot_y = self.win.addPlot(row=0, col=1, title="<b>sigma_y^2 vs t^2</b>")
        self.plot_y.setLabel('bottom', 't^2', units='s^2')
        self.plot_y.setLabel('left', 'sigma_y^2', units='m^2')
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)
        self.data_y = self.plot_y.plot([], [], pen=None, symbol='o', symbolSize=10, symbolBrush='#4CAF50')
        self.fit_y = self.plot_y.plot([], [], pen=pg.mkPen('#F44336', width=2))
        self.label_y = pg.TextItem('', color='#F44336', anchor=(0, 1))
        self.plot_y.addItem(self.label_y)
        self._latest_data = None
        self._rendered_version = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    def update_data(self, tof_us_list, sigma_x_list, sigma_y_list):
        self._latest_data = (list(tof_us_list), list(sigma_x_list), list(sigma_y_list), self._rendered_version + 1)

    def _poll_and_update(self):
        entry = self._latest_data
        if entry is None:
            return
        tof_us, sx_px, sy_px, version = entry
        if version <= self._rendered_version:
            return
        self._rendered_version = version
        try:
            t_s = np.array(tof_us, dtype=np.float64) * 1e-6
            t2 = t_s ** 2
            sx_m = np.array(sx_px, dtype=np.float64) * self.pixel_size_m
            sy_m = np.array(sy_px, dtype=np.float64) * self.pixel_size_m
            sx2 = sx_m ** 2
            sy2 = sy_m ** 2
            mask_x = np.isfinite(sx2) & (sx2 > 0)
            mask_y = np.isfinite(sy2) & (sy2 > 0)
            if mask_x.sum() >= 2:
                self.data_x.setData(t2[mask_x], sx2[mask_x])
                a_x, b_x = _linear_fit(t2[mask_x], sx2[mask_x])
                if np.isfinite(b_x) and b_x > 0:
                    T_x = M_RB87 * b_x / K_B
                    t2_fit = np.linspace(0, t2.max() * 1.1, 50)
                    self.fit_x.setData(t2_fit, a_x + b_x * t2_fit)
                    self.label_x.setText(f"T_x = {T_x*1e6:.1f} uK\nsigma_0x = {np.sqrt(max(a_x,0))/self.pixel_size_m:.0f} px")
                    self.label_x.setPos(t2.max() * 0.05, sx2[mask_x].max() * 0.95)
            if mask_y.sum() >= 2:
                self.data_y.setData(t2[mask_y], sy2[mask_y])
                a_y, b_y = _linear_fit(t2[mask_y], sy2[mask_y])
                if np.isfinite(b_y) and b_y > 0:
                    T_y = M_RB87 * b_y / K_B
                    t2_fit = np.linspace(0, t2.max() * 1.1, 50)
                    self.fit_y.setData(t2_fit, a_y + b_y * t2_fit)
                    self.label_y.setText(f"T_y = {T_y*1e6:.1f} uK\nsigma_0y = {np.sqrt(max(a_y,0))/self.pixel_size_m:.0f} px")
                    self.label_y.setPos(t2.max() * 0.05, sy2[mask_y].max() * 0.95)
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
    'lat_am_v': {'Lattice AM (V)': '1.0'},

    # ===================== FIXED LATTICE HOLDS (before + after move) =====================
    'pre_hold_ms': {'Pre-move Hold (ms)': '5'},
    'post_hold_ms': {'Post-move Hold (ms)': '5'},

    # ===================== MOVE PARAMETERS (fixed across scan) =====================
    # Trapezoidal velocity profile: accel (move_ramp_ms) -> plateau (move_plateau_ms)
    # -> decel (move_ramp_ms). Distance = v_peak * (move_ramp_ms + move_plateau_ms).
    # For each scan point, v_peak is computed from the requested distance.
    # The DIM-3000 is RE-PROGRAMMED via serial between scan points.
    'move_ramp_ms': {'Move Accel/Decel Ramp (ms)': '10'},
    'move_plateau_ms': {'Move Plateau (ms)': '10'},
    'seg_lead_ms': {'DIM3000 Trigger Lead (ms)': '3'},
    # +1 or -1: which sign of detuning -> reverses lattice motion direction.
    'move_direction': {'Move Direction (+1 or -1)': '-1'},

    # ===================== MOVE DISTANCE SWEEP (3 points) =====================
    # FOV diagnostic: very small moves. If atoms appear here but not at 1+mm,
    # the move mechanism works and the issue was atoms leaving the imaging
    # field of view (camera FOV).
    # Point 1: 0.05 mm (50 um) - barely moves, should look like 0mm
    # Point 2: 0.1 mm (100 um) - small move
    # Point 3: 0.2 mm (200 um) - moderate move
    'dist_1_mm': {'Move Distance 1 (mm)': '0.1'},
    'dist_2_mm': {'Move Distance 2 (mm)': '0.5'},
    'dist_3_mm': {'Move Distance 3 (mm)': '1.5'},

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
    'pixel_size_um': {'Pixel Size (um/px)': '3.45'},

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
    dist_values_mm = [
        float(dataset.get_input_parameter("dist_1_mm")),
        float(dataset.get_input_parameter("dist_2_mm")),
        float(dataset.get_input_parameter("dist_3_mm")),
    ]
    scan_labels = [f"d={d:.2f}mm" for d in dist_values_mm]
    # Absolute camera counts, clamped to what a pixel can physically reach.
    disp_min = float(dataset.get_input_parameter("display_min"))
    disp_max = float(dataset.get_input_parameter("display_max"))
    disp_min = max(disp_min, -float(SENSOR_MAX_MONO16))
    disp_max = min(disp_max, float(SENSOR_MAX_MONO16))
    autoscale = bool(int(float(dataset.get_input_parameter("display_autoscale"))))
    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        autoscale=autoscale,
        exposure_us=float(dataset.get_input_parameter("camera_exposure_us")),
        diff_levels=(disp_min, disp_max), avg_levels=(disp_min, disp_max),
        title="Lattice MOVE sweep (transport distance)",
    )
    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))
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

    dist_values_mm = [
        float(dataset.get_input_parameter("dist_1_mm")),
        float(dataset.get_input_parameter("dist_2_mm")),
        float(dataset.get_input_parameter("dist_3_mm")),
    ]
    n_scan = len(dist_values_mm)
    pre_hold_ms = int(dataset.get_input_parameter("pre_hold_ms"))
    post_hold_ms = int(dataset.get_input_parameter("post_hold_ms"))
    move_ramp_ms = int(dataset.get_input_parameter("move_ramp_ms"))
    move_plateau_ms = int(dataset.get_input_parameter("move_plateau_ms"))
    seg_lead_ms = int(dataset.get_input_parameter("seg_lead_ms"))
    dim_base_freq_mhz = float(dataset.get_input_parameter("dim_base_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))
    move_direction = int(dataset.get_input_parameter("move_direction"))
    tof_us = int(dataset.get_input_parameter("tof_us"))
    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))
    warmup_cycles = max(0, int(dataset.get_input_parameter("warmup_cycles")))

    # ============ Move physics ============
    # v = lambda/2 * df. For 795nm: v[mm/s] = 0.0003975 * df[Hz]
    LATTICE_WAVELENGTH = 795e-9
    v_per_Hz_mmps = LATTICE_WAVELENGTH / 2 * 1e3  # mm/s per Hz of detuning
    base_freq_hz = int(round(dim_base_freq_mhz * 1e6))

    # Trapezoid: distance = v_peak * (move_ramp_ms + move_plateau_ms)
    # Per scan point, choose v_peak to match the requested distance.
    move_total_ms = move_ramp_ms + move_plateau_ms  # ramp + plateau (one side)
    v_peak_mmps_per_point = []
    move_freq_hz_per_point = []
    for d in dist_values_mm:
        v_peak_mmps = d / (move_total_ms * 1e-3)            # mm/s
        df_hz = int(round(v_peak_mmps / v_per_Hz_mmps))     # Hz of detuning
        move_freq = base_freq_hz + move_direction * df_hz
        v_peak_mmps_per_point.append(v_peak_mmps)
        move_freq_hz_per_point.append(move_freq)

    logger.info(
        f"Move physics: ramp={move_ramp_ms}ms, plateau={move_plateau_ms}ms, "
        f"v_peak per point: " +
        ", ".join(f"{d}mm->{v:.1f}mm/s ({(f-base_freq_hz)/1e3:+.1f}kHz)"
                  for d, v, f in zip(dist_values_mm, v_peak_mmps_per_point, move_freq_hz_per_point)))

    # =========================================================================
    # CONFIGURE DIM-3000 DRIVERS (called once per experiment start)
    # =========================================================================
    DIM_COM_SWEEP = 'COM4'   # lattice up -- has frequency sweep
    DIM_COM_FIXED = 'COM5'   # lattice down -- fixed frequency
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

    def _dim_program_segments(ser, move_freq):
        """Program Wseg 1-4 for a move from base_freq_hz to `move_freq` and back.

        FOUR segments: the three real legs all carry type flag 1, and a
        fourth, base->base, no-op segment carries the flag 8 end-of-chain
        marker.

            Wseg:1  flag 1   base -> move    accelerate   (real)
            Wseg:2  flag 1   move -> move    plateau      (real)
            Wseg:3  flag 1   move -> base    DECELERATE   (real)
            Wseg:4  flag 8   base -> base    sentinel     (nothing to play)

        This used to put flag 8 on Wseg:3, the deceleration. If the driver
        treats the marked segment as the wrap point rather than as a segment
        to play, the decel never runs: the chain returns to segment 1 straight
        off the plateau and the carrier is left at move_freq with the lattice
        still walking. Putting the marker on a segment that does nothing makes
        the profile come out right under either reading of the flag.
        Same ramp time across calls; only the endpoint changes per scan point.
        Re-asserts Mseg:1 at the end so the DIM stays in segment-trigger mode."""
        freq_range = abs(move_freq - base_freq_hz)
        max_step_us = 256.0
        min_steps = max(1, int(np.ceil(move_ramp_ms * 1000.0 / max_step_us)))
        num_steps = max(min_steps, 10)
        step_freq = max(1, freq_range // num_steps) if freq_range > 0 else 1
        step_time_us = (move_ramp_ms * 1000.0) / num_steps
        step_time_arb = max(1, min(65535, int(round(step_time_us * 1000.0 / 3.90625))))
        _dim_send(ser, f'Wseg:1;1;{base_freq_hz};{move_freq};{step_freq};{step_time_arb};0')
        _dim_send(ser, f'Wseg:2;1;{move_freq};{move_freq};0;1;0')
        _dim_send(ser, f'Wseg:3;1;{move_freq};{base_freq_hz};{step_freq};{step_time_arb};0')
        # The sentinel. base->base with a zero step, so it holds whatever
        # frequency the deceleration arrived at whether the driver plays it or
        # wraps straight past it.
        _dim_send(ser, f'Wseg:4;8;{base_freq_hz};{base_freq_hz};0;1;0')
        _dim_send(ser, 'Mseg:1')
        return num_steps, step_freq, step_time_arb

    # Open both ports ONCE and keep them open across the sweep so reprogramming
    # between scan points only costs ~10-20ms of serial writes (not open/close).
    ser_sweep = _dim_open(DIM_COM_SWEEP)
    ser_fixed = _dim_open(DIM_COM_FIXED)
    # Record the drivers BEFORE anything is written to them, so the
    # wrapper around this function can put them back exactly as found
    # when the run stops, however it stops (dim_state.py).
    kwargs['dim_snapshots'].append(dim_state.snapshot(ser_sweep, DIM_COM_SWEEP, logger))
    kwargs['dim_snapshots'].append(dim_state.snapshot(ser_fixed, DIM_COM_FIXED, logger))
    if ser_sweep is not None and ser_fixed is not None:
        id1 = _dim_query(ser_sweep, '*IDN?')
        id2 = _dim_query(ser_fixed, '*IDN?')
        logger.info(f"DIM-3000 sweep ({DIM_COM_SWEEP}): {id1}")
        logger.info(f"DIM-3000 fixed ({DIM_COM_FIXED}): {id2}")
        # Fixed driver: base freq, full amplitude, always on, no segments
        _dim_send(ser_fixed, f'FRQ:{base_freq_hz}')
        _dim_send(ser_fixed, f'AMP:{amplitude_x10}')
        _dim_send(ser_fixed, 'OUT_on')
        logger.info(f"DIM-3000 fixed: {base_freq_hz/1e6:.3f} MHz, +{amplitude_x10/10:.1f} dBm, ON")
        # Sweep driver: start at base, full amp, always on, segment mode enabled
        _dim_send(ser_sweep, f'FRQ:{base_freq_hz}')
        _dim_send(ser_sweep, f'AMP:{amplitude_x10}')
        _dim_send(ser_sweep, 'OUT_on')
        # Program initial segments using first scan point's frequency
        ns, sf, st = _dim_program_segments(ser_sweep, move_freq_hz_per_point[0])
        _dim_send(ser_sweep, 'Mseg:1')
        logger.info(
            f"DIM-3000 sweep: initialized with point 0 freq, "
            f"{ns} steps, step_freq={sf}, step_time_arb={st} (~{move_ramp_ms}ms ramp)")
    else:
        logger.info("DIM-3000: ports unavailable, will run without serial control")
        if ser_sweep is not None:
            ser_sweep.close()
            ser_sweep = None
        if ser_fixed is not None:
            ser_fixed.close()
            ser_fixed = None

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
    def ms2s(t): return t * SPM
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
    move_ramp_s = ms2s(move_ramp_ms)
    move_plateau_s = ms2s(move_plateau_ms)
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
    logger.info(f"Time at start {time.perf_counter_ns()}")
    logger.info(
        f"Timeline: MOT 0-{mot_loading_ms}ms, "
        f"CMOT {mot_loading_ms}-{cmot_end/SPM:.0f}ms, "
        f"Dark MOT {dark_str} ({dark_start/SPM:.0f}-{dark_end/SPM:.0f}ms), "
        f"PGC_cool ({pgc_start/SPM:.0f}-{pgc_cool_end/SPM:.0f}ms), "
        f"PGC_rampdown+lattice load ({pgc_cool_end/SPM:.0f}-{pgc_end/SPM:.0f}ms), "
        f"pre_hold={pre_hold_ms}ms, accel={move_ramp_ms}ms, "
        f"plateau={move_plateau_ms}ms (fixed), decel={move_ramp_ms}ms, "
        f"post_hold={post_hold_ms}ms, TOF={tof_us}us"
    )

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence(dist_mm):
        do_move = (dist_mm > 0)
        # Sequence after PGC end:
        #   pre_hold -> [accel -> plateau -> decel if moving] -> post_hold -> release
        # For 0mm reference (dist_mm==0), accel/plateau/decel collapse to zero
        # AND no DIM triggers fire => atoms just sit in stationary lattice.
        ramp_s = move_ramp_s if do_move else 0
        plat_s = move_plateau_s if do_move else 0
        pre_hold_end = pgc_end + pre_hold_s
        accel_start = pre_hold_end
        accel_end = accel_start + ramp_s        # = seg2 start
        plateau_end = accel_end + plat_s          # = seg3 start
        decel_end = plateau_end + ramp_s
        post_hold_end = decel_end + post_hold_s
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

        # DIM-3000 segment trigger (dio6): FOUR pulses, each ~seg_lead_ms
        # BEFORE the corresponding segment start (DIM internal latency).
        # Trigger 1: start of accel ramp (seg1)
        # Trigger 2: start of plateau   (seg2)
        # Trigger 3: start of decel ramp(seg3)
        # Trigger 4: end of decel       (seg4, the flag-8 sentinel)
        #
        # The fourth is what makes the sentinel scheme work, and it is also
        # why the pulse count has to match the chain length exactly: a segment
        # with no trigger stalls the chain until the NEXT cycle's first
        # trigger, which then plays every later cycle one segment out of
        # phase.  Four segments, four triggers.
        #
        # Trigger 4 lands seg_lead_ms before decel_end, i.e. INSIDE the decel
        # ramp -- the same relationship triggers 2 and 3 already have to the
        # legs they follow. It changes no anchor and adds no samples:
        # decel_end and everything after it are where they were.
        # For 0mm reference, no triggers fire and DIM stays at base_freq.
        seg_do = [0] * N
        if do_move:
            for trig_t in (accel_start, accel_end, plateau_end, decel_end):
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
    warm_values = [dist_values_mm[0]] * warmup_cycles
    for dist_mm in warm_values + list(dist_values_mm):
        (cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco,
         bx_w, by_w, bz_w, ludo, lddo, luao, ldao, seg_w, N) = build_sub_sequence(dist_mm)
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
    logger.info(f"Lattice MOVE sweep: distances={dist_values_mm}mm, plateau={move_plateau_ms}ms (fixed), tof={tof_us}us, {warmup_cycles} warm-up cycle(s) at {dist_values_mm[0]}mm, total={total} samples")

    diff_sums = [None] * n_scan
    diff_counts = [0] * n_scan

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
        dataset.scan_grid.set_gain_db(applied_gain)
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

        scan_grid = dataset.scan_grid
        sigma_x_vals = [float('nan')] * n_scan
        sigma_y_vals = [float('nan')] * n_scan

        try:
            # Initial reprogram for scan point 0 (in case it differs from startup)
            if ser_sweep is not None:
                _dim_program_segments(ser_sweep, move_freq_hz_per_point[0])
            stopping = False
            for run_idx in range(M):
                if not thread.running:
                    stopping = True
                    break
                logger.info(f"Run {run_idx+1} of {M}")
                # The warm-up cycles sit at the head of the buffer and fire
                # their camera triggers like any other cycle, so their frames
                # MUST be consumed here. Skip this and every pair afterwards
                # is offset by one -- each panel would then be showing another
                # point's cloud, with no error to say so.
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
                    # Reprogram DIM for the NEXT scan point so segments are
                    # ready before that sub-sequence reaches the move phase.
                    next_idx = (scan_idx + 1) % n_scan
                    if ser_sweep is not None:
                        try:
                            _dim_program_segments(ser_sweep, move_freq_hz_per_point[next_idx])
                        except Exception as e:
                            logger.info(f"  DIM reprogram failed for next point: {e!r}")
                    diff = frame1.astype(np.int32) - frame2.astype(np.int32)
                    # Identical accumulation to temp_PGC_aziza: the running
                    # sum is float64 (an int32 accumulator wraps silently past
                    # 2.15e9 and the mean comes back NEGATIVE), the average is
                    # sum/count so it is a true running mean rather than an
                    # exponential one, and both rows are handed the same data.
                    scan_grid.set_frame_dtype(frame1.dtype)
                    scan_grid.set_current(scan_idx, diff)
                    if diff_sums[scan_idx] is None:
                        diff_sums[scan_idx] = diff.astype(np.float64)
                    else:
                        diff_sums[scan_idx] += diff
                    diff_counts[scan_idx] += 1
                    avg = diff_sums[scan_idx] / diff_counts[scan_idx]
                    scan_grid.set_average(scan_idx, avg)
                    img = avg.astype(np.float64)
                    # fit_gaussian_1d, not the old moment estimator: it carries
                    # an explicit offset term (a pedestal otherwise inflates the
                    # width) and reports whether the fit can be trusted at all.
                    fx = fit_gaussian_1d(img.sum(axis=0))
                    fy = fit_gaussian_1d(img.sum(axis=1))
                    good = bool(fx and fy and fx['reliable'] and fy['reliable'])
                    sx = fx['sigma'] if fx else float('nan')
                    sy = fy['sigma'] if fy else float('nan')
                    sigma_x_vals[scan_idx] = sx
                    sigma_y_vals[scan_idx] = sy
                    met = (cloud_metrics(img, fx, fy, frame1.dtype,
                                         camera_exposure_us,
                                         gain_db=applied_gain)
                           if (fx and fy) else None)
                    logger.info(
                        f"  d={d_v:.2f}mm (v_peak={v_peak_mmps_per_point[scan_idx]:.1f}mm/s): "
                        f"n={diff_counts[scan_idx]}, sx={sx:.1f}px, sy={sy:.1f}px"
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
    # After the while loop ends (thread stopped): close DIM serial ports
    # (release REMOTE CONTROL display, free the COM port for next run)
    # Park the DIM before letting go of the port. The sweep driver is left in
    # segment mode sitting at whatever frequency the last move ended on; if
    # that is not the base frequency the two beams stay detuned and the
    # standing wave KEEPS WALKING after the run has stopped, dragging any
    # atoms still in the trap. Put it back on base so the lattice is on but
    # stationary, which is the state park_outputs() assumes on the NI side.
    if ser_sweep is not None:
        try:
            _dim_send(ser_sweep, f'FRQ:{base_freq_hz}')
            logger.info(f"DIM sweep parked at base {base_freq_hz/1e6:.3f} MHz "
                        f"(lattice on, stationary)")
        except Exception as e:
            logger.warn(f"Could not park the DIM at base frequency: {e!r}")
        try:
            ser_sweep.close()
        except Exception:
            pass
    if ser_fixed is not None:
        try:
            ser_fixed.close()
        except Exception:
            pass
