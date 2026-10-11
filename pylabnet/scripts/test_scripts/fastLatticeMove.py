# Lattice TRANSPORT: ONE WAY -- SINGLE POINT, SINGLE PANEL, COM4 ONLY.
#
# latticemove&back_single.py with the return trip removed and the move given
# by its DURATIONS instead of by acceleration and velocity cap. One direction,
# one distance, one panel, one AOM swept. Everything that is not the move --
# the MOT/CMOT/Dark MOT/PGC sequence, the lattice loading, the NI waveform
# construction, the camera, the fits and the atom number -- is carried over
# unchanged, so a cloud here and a cloud there are measured the same way.
#
# THE MOVE IS SPECIFIED BY ITS THREE DURATIONS. Distance d, accel time T_a,
# plateau time T_p and decel time T_d are typed in; the velocity and the two
# accelerations FOLLOW from them:
#
#     d = v * (T_a/2 + T_p + T_d/2)       area under the velocity trapezoid
#     v = d / (T_a/2 + T_p + T_d/2)
#     a_accel = v / T_a,   a_decel = v / T_d
#
# That is the other way round from the out-and-back scripts, which hold the
# acceleration fixed and derive the timings. Holding the timings fixed instead
# is what you want when the question is about TIME -- how short can the ramp
# be, how long a cruise does the cloud survive -- and it means the header's
# acceleration and velocity are outputs to read, not inputs to trust. Change
# the distance at fixed timings and the acceleration changes with it.
#
# T_p = 0 gives a triangle (accel straight into decel). T_a != T_d gives an
# asymmetric profile; each ramp gets its own DIM step rate. A zero accel or
# decel time is allowed but is a velocity STEP -- the lattice jumps to full
# speed in one DIM update -- and the log warns about it.
#
# Full sequence:
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> pre-move hold
#   -> ACCELERATE          (seg 1: base_freq -> move_freq,  T_a)
#   -> CONSTANT VELOCITY   (seg 2: hold move_freq,          T_p)
#   -> DECELERATE to rest  (seg 3: move_freq -> base_freq,  T_d)
#   -> post-move hold      (lattice stationary, atoms DISPLACED by d)
#   -> release -> TOF -> image
#
# THREE SEGMENTS, not seven. A zero-length leg is dropped from both the chain
# and the trigger train, so a triangle runs 2 segments. Segments 4..7 are
# blanked at startup, because the other scripts in this family leave chains up
# to 7 long behind them and flag 8 hides a stale segment without erasing it.
#
# THE IMAGE IS TAKEN AT THE FAR POINT. There is no return trip, so the cloud
# is imaged displaced by d along the lattice axis. Compare its centre against
# a d = 0 run (stationary reference: no triggers fire, same holds, same
# imaging) to read the displacement off the camera. The next cycle reloads
# the lattice from a fresh MOT at the original position, so nothing carries
# over between cycles -- the standing wave has no memory of where it was.
#
# SINGLE AOM. Only the up beam (COM4) is swept; the down beam (COM5) sits at
# base and is made DEAF to dio6, which is teed to both Seg inputs. A detuning
# Df between the beams moves the standing wave at v = (lambda/2) * Df, so with
# lambda = 795 nm: v = 3.975 mm/s per 10 kHz. COM4 carries the full Df.
#
# SINGLE SHOT, NOT AN AVERAGE. The panel draws the latest frame pair only and
# the numbers beside it are fitted from that same pair, so it answers
# immediately when something is turned. Read a faint cloud as one shot of many.
#
# Lattice TTL & AM held constant for the entire cycle (AOM thermal stability).

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
    nan = float('nan')
    move_rows = [
        ("transport (one way)", f"{d:.3f} mm",
         "expected shift on camera",
         f"{m.get('d_px', nan):.1f} px (if in image plane)"),
        ("accel time T_a", f"{m.get('accel_ms', nan):.2f} ms",
         "acceleration (derived)", f"{m.get('a_acc_mps2', nan):.2f} m/s&sup2;"),
        ("constant velocity T_p", f"{m.get('plat_ms', nan):.2f} ms",
         "peak velocity (derived)", f"{m.get('v_mmps', nan):.2f} mm/s"),
        ("decel time T_d", f"{m.get('decel_ms', nan):.2f} ms",
         "deceleration (derived)", f"{m.get('a_dec_mps2', nan):.2f} m/s&sup2;"),
        ("TOTAL MOVE TIME", f"{m.get('total_ms', nan):.2f} ms",
         "COM4 detuning",
         f"{m.get('df_khz_signed', nan):+.2f} kHz ({m.get('shape', '--')}, "
         f"{m.get('n_legs', 0)} DIM segments)"),
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
    'lat_am_v': {'Lattice AM (V)': '1.0'},

    # ===================== MOVE (one way, COM4 only) =====================
    # The move is given by its DISTANCE and its THREE DURATIONS. Velocity and
    # acceleration are derived from them and printed in the panel header:
    #
    #     v       = d / (T_a/2 + T_p + T_d/2)
    #     a_accel = v / T_a        a_decel = v / T_d
    #
    # The defaults (0.5 mm, 5 / 15 / 5 ms) reproduce the out-and-back script's
    # defaults on its outward leg: v = 25 mm/s, a = 5 m/s^2.
    #
    # 0 distance is a stationary reference: no DIM triggers fire, the holds and
    # imaging run unchanged, and the cloud is imaged where it was loaded -- the
    # position to measure every moved cloud's displacement against.
    'dist_mm': {'Move Distance (mm, one way)': '0.5'},
    # Whole-sample resolution is 50 us (NI clock 20 kHz), so these accept
    # fractional ms. A leg of 0 ms is dropped from the chain: T_p = 0 makes a
    # triangle; T_a or T_d = 0 makes a velocity STEP, which the log warns about.
    'accel_ms': {'Accel Time T_a (ms)': '5.0'},
    'plat_ms': {'Plateau (constant velocity) Time T_p (ms)': '15.0'},
    'decel_ms': {'Decel Time T_d (ms)': '5.0'},
    'seg_lead_ms': {'DIM3000 Trigger Lead (ms)': '3'},
    # +1 or -1: the sign of COM4's detuning, i.e. which way along the lattice
    # axis the atoms go. There is no return leg, so this DOES decide where the
    # cloud ends up.
    'move_direction': {'Move Direction (+1 or -1)': '-1'},

    # ===================== FIXED LATTICE HOLDS (before + after move) =====================
    'pre_hold_ms': {'Pre-move Hold (ms)': '5'},
    # After the decel the lattice is stationary with the atoms at the far
    # point. This is how long they sit there before release.
    'post_hold_ms': {'Post-move Hold (ms)': '5'},

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
        title=f"Lattice move ONE WAY (COM4) -- d={dist_mm:.3f} mm",
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

    # ONE transport distance. The list and n_scan stay because every per-point
    # structure below -- schedule, legs_per_point, build_sub_sequence, the
    # accumulators -- indexes by scan point; with a single entry the sweep
    # machinery collapses to one point without any of it needing to know, and
    # the sequence built here is identical to the sweep script's.
    dist_values_mm = [float(dataset.get_input_parameter("dist_mm"))]
    n_scan = 1
    pre_hold_ms = int(dataset.get_input_parameter("pre_hold_ms"))
    post_hold_ms = int(dataset.get_input_parameter("post_hold_ms"))
    accel_ms = float(dataset.get_input_parameter("accel_ms"))
    plat_ms_in = float(dataset.get_input_parameter("plat_ms"))
    decel_ms = float(dataset.get_input_parameter("decel_ms"))
    seg_lead_ms = int(dataset.get_input_parameter("seg_lead_ms"))
    # Refuse the inputs that have no meaning rather than guess at one. A
    # negative duration, or a distance with no time to cover it in, would
    # otherwise surface as a divide-by-zero or a nonsense detuning deep inside
    # the DIM programming, after the ports are already open.
    if min(accel_ms, plat_ms_in, decel_ms) < 0:
        raise ValueError(
            f"Move durations must be >= 0 (got T_a={accel_ms}, "
            f"T_p={plat_ms_in}, T_d={decel_ms} ms).")
    if float(dataset.get_input_parameter("dist_mm")) < 0:
        raise ValueError(
            "Move Distance must be >= 0; the direction is set by Move "
            "Direction (+1 or -1), not by the sign of the distance.")
    dim_base_freq_mhz = float(dataset.get_input_parameter("dim_base_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))
    move_direction = int(dataset.get_input_parameter("move_direction"))
    if move_direction not in (1, -1):
        raise ValueError(f"Move Direction must be +1 or -1, got {move_direction}.")
    tof_us = int(dataset.get_input_parameter("tof_us"))
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

    def schedule_for_distance(d_mm):
        """Velocity, accelerations and detuning for one move of given durations.

        The durations are the inputs here; everything kinematic follows from
        the area under the velocity trapezoid:

            d = v * (T_a/2 + T_p + T_d/2)
            v = d / (T_a/2 + T_p + T_d/2),  a_acc = v/T_a,  a_dec = v/T_d

        Durations stay FLOATS in ms and are snapped to the 50 us NI grid only
        where the waveform is built, as in the out-and-back script.

        :param d_mm: (float) one-way transport distance [mm]
        :return: (dict) d, v, the three durations, both accelerations, the
            COM4 plateau frequency and the total move time.
        """
        d = float(d_mm) * 1e-3                       # m
        eff_s = (0.5 * accel_ms + plat_ms_in + 0.5 * decel_ms) * 1e-3
        stationary = {'d_mm': float(d_mm), 'v_mmps': 0.0,
                      'accel_ms': 0.0, 'plat_ms': 0.0, 'decel_ms': 0.0,
                      'a_acc_mps2': 0.0, 'a_dec_mps2': 0.0,
                      'freq_hz': base_freq_hz, 'df_hz': 0, 'total_ms': 0.0}
        if d <= 0:
            return stationary
        if eff_s <= 0:
            raise ValueError(
                f"Distance {d_mm} mm needs a non-zero move time, but "
                f"T_a/2 + T_p + T_d/2 = 0. Set at least one duration.")
        v = d / eff_s                                # m/s
        v_mmps = v * 1e3
        # Detuning rounded to whole Hz -- the DIM takes integer frequencies.
        # The distance actually travelled is then v_real * eff, which is
        # reported below so the rounding is visible rather than assumed away.
        df_hz = int(round(v_mmps / v_per_Hz_mmps))
        return {'d_mm': float(d_mm), 'v_mmps': v_mmps,
                'accel_ms': accel_ms, 'plat_ms': plat_ms_in,
                'decel_ms': decel_ms,
                'a_acc_mps2': v / (accel_ms * 1e-3) if accel_ms > 0 else float('inf'),
                'a_dec_mps2': v / (decel_ms * 1e-3) if decel_ms > 0 else float('inf'),
                'freq_hz': base_freq_hz + move_direction * df_hz,
                'df_hz': df_hz,
                'd_real_mm': df_hz * v_per_Hz_mmps * eff_s,
                'total_ms': accel_ms + plat_ms_in + decel_ms}

    schedule = [schedule_for_distance(d) for d in dist_values_mm]
    v_peak_mmps_per_point = [sc['v_mmps'] for sc in schedule]

    sc0 = schedule[0]
    if sc0['d_mm'] > 0:
        logger.info(
            f"Move physics (ONE WAY, COM4 only): d={sc0['d_mm']:.3f}mm in "
            f"T_a={sc0['accel_ms']:.2f} + T_p={sc0['plat_ms']:.2f} + "
            f"T_d={sc0['decel_ms']:.2f} = {sc0['total_ms']:.2f}ms -> "
            f"v={sc0['v_mmps']:.2f}mm/s, a_accel={sc0['a_acc_mps2']:.2f}m/s^2, "
            f"a_decel={sc0['a_dec_mps2']:.2f}m/s^2, "
            f"{'TRIANGLE' if sc0['plat_ms'] <= 0 else 'trapezoid'}, "
            f"COM4 detuning {(sc0['freq_hz'] - base_freq_hz)/1e3:+.3f}kHz "
            f"(distance after 1 Hz rounding: {sc0['d_real_mm']:.4f}mm)")
        # A zero ramp is legal but it is a velocity step: the lattice goes from
        # rest to full speed in one DIM update. Atoms cannot follow that, so
        # say so plainly instead of letting a dark image be the first clue.
        for _name, _t in (('Accel', sc0['accel_ms']), ('Decel', sc0['decel_ms'])):
            if _t <= 0:
                logger.warn(
                    f"{_name} time is 0: the lattice {'jumps to' if _name == 'Accel' else 'stops from'} "
                    f"{sc0['v_mmps']:.1f} mm/s in a single step. That is an "
                    f"unbounded acceleration and will shake most atoms out.")
    else:
        logger.info("Move distance 0: STATIONARY reference -- no DIM triggers, "
                    "cloud imaged where it was loaded.")

    # =========================================================================
    # CONFIGURE DIM-3000 DRIVERS (called once per experiment start)
    # =========================================================================
    DIM_COM_SWEEP = 'COM4'   # lattice up -- has frequency sweep
    DIM_COM_FIXED = 'COM5'   # lattice down -- fixed frequency, but its Seg
    # input is teed to dio6, so it has to be told to
    # ignore triggers rather than merely left alone

    # This script needs at most three legs, but the out-and-back scripts in
    # this family write up to seven and leave them behind. The DIM-3000
    # allows twenty (manual, Wseg: "N -- min 1, max 20"), so 7 is a fact about
    # these scripts, not a hardware limit. Used to scrub segments a PREVIOUS,
    # longer run left behind, which flag 8 hides but does not erase.
    DIM_MAX_SEGMENTS = 7
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
        well. If that driver is still in segment-play mode from an earlier run
        -- which is exactly the state the two-AOM script leaves it in -- it
        plays its stale chain alongside ours, detunes itself off base, and the
        pair is then a WALKING standing wave rather than a trap. There is no
        symptom except a lattice that will not hold, and nothing in this run's
        log would mention the driver it never asked about.

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
        # Verified, not just sent: a base frequency the driver quietly ignored
        # would leave both beams wherever the last run left them.
        dim_state.set_frequency(ser, base_freq_hz, port, logger)
        logger.info(
            f"DIM-3000 {port}: {base_freq_hz/1e6:.6f} MHz, "
            f"+{amplitude_x10/10:.1f} dBm, ON, segments neutralised and Seg "
            f"DISABLED (Mseg:0) -- this run does not sweep it, and dio6 "
            f"reaches it through the tee.")

    def _dim_scrub_tail(ser, port, n_active):
        """Blank segments above the chain this run uses.

        flag 8 makes the chain wrap before reaching them, so they should never
        play -- but "should never" is doing a lot of work there, and a stale
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

    def move_legs(sc):
        """The one-way move: one leg per DIM segment, in play order.

        ONE definition, two consumers -- _dim_program_segments() turns it into
        the segment chain and build_sub_sequence() turns it into the dio6
        trigger train. They MUST agree on how many legs there are: a segment
        with no trigger stalls the chain until the NEXT cycle's first trigger,
        playing every later cycle one segment out of phase, and a trigger with
        no segment runs the chain off its end. Deriving both from this list is
        what makes disagreement impossible rather than merely unlikely.

        The last leg ends AT BASE. That is what leaves the lattice stationary
        for the post-move hold and the imaging; a chain that ended anywhere else
        would leave the two beams detuned and the standing wave still walking.

        :param sc: (dict) schedule entry for this point
        :return: (list of dict) name, dur_ms, f0, f1. f0 == f1 marks a hold
            leg, whose duration is set by the trigger train, not the segment.
        """
        move_freq = sc['freq_hz']
        legs = [
            {'name': 'accel', 'dur_ms': sc['accel_ms'],
             'f0': base_freq_hz, 'f1': move_freq},
            {'name': 'plateau', 'dur_ms': sc['plat_ms'],
             'f0': move_freq, 'f1': move_freq},
            {'name': 'decel to rest', 'dur_ms': sc['decel_ms'],
             'f0': move_freq, 'f1': base_freq_hz},
        ]
        # Zero-length legs must not reach the chain: their trigger would land
        # on the SAME sample as the next leg's, the two pulses would merge, and
        # the DIM would be a trigger short for the rest of the run. Dropping a
        # hold is exact (f0 == f1, so the chain stays continuous). Dropping a
        # RAMP is a frequency step -- which is precisely what a 0 ms ramp asks
        # for, and the schedule already warned about it.
        return [leg for leg in legs if leg['dur_ms'] > 0]

    legs_per_point = [move_legs(sc) for sc in schedule]
    n_legs_max = max(len(lg) for lg in legs_per_point)

    def _ramp_params(dur_ms, span_hz):
        """DIM step size and step time for one ramp leg.

        Per LEG, not per point: with T_a != T_d the two ramps cover the same
        frequency span in different times, so they need different step rates.
        (The out-and-back scripts could share one rate because all their ramps
        were the same length.) Same rules as those scripts otherwise: at least
        10 steps, no step longer than 256 us, step time in units of
        3.90625 NANOseconds (65535 of them is the 256 us ceiling).

        :return: (step_freq_hz, step_time_arb, num_steps, played_ms) where
            played_ms is num_steps * step time as the DIM will actually run it.
        """
        max_step_us = 256.0
        min_steps = max(1, int(np.ceil(dur_ms * 1000.0 / max_step_us)))
        num_steps = max(min_steps, 10)
        step_freq = max(1, span_hz // num_steps) if span_hz > 0 else 1
        step_time_us = (dur_ms * 1000.0) / num_steps if dur_ms > 0 else 1.0
        step_time_arb = max(1, min(65535, int(round(step_time_us * 1000.0 / 3.90625))))
        played_ms = num_steps * step_time_arb * 3.90625e-6   # ns -> ms
        return step_freq, step_time_arb, num_steps, played_ms

    def _dim_program_segments(ser, idx):
        """Write the one-way chain for scan point `idx` and enter segment mode.

        One Wseg per leg of move_legs(), in play order. The LAST leg carries
        type flag 8 -- end of chain, next trigger goes back to segment 1 -- and
        every earlier leg carries 1. Derived from the list rather than typed,
        because flag 8 on the wrong leg is the one mistake a readback still
        looks healthy for.

        Each ramp leg gets its own step rate from _ramp_params(); hold legs get
        step 0 and sit until the next trigger.

        :return: (list of str) one human-readable line per leg, for the log
        """
        legs = legs_per_point[idx]
        lines = []
        for i, leg in enumerate(legs, start=1):
            flag = 8 if i == len(legs) else 1
            if leg['f0'] == leg['f1']:
                _dim_send(ser, f"Wseg:{i};{flag};{leg['f0']};{leg['f1']};0;1;0")
                lines.append(f"seg {i} {leg['name']}: hold {leg['f0']} Hz, "
                             f"{leg['dur_ms']:.2f} ms (trigger-timed)")
            else:
                span = abs(leg['f1'] - leg['f0'])
                sf, st, ns, played = _ramp_params(leg['dur_ms'], span)
                _dim_send(ser, f"Wseg:{i};{flag};{leg['f0']};{leg['f1']};"
                               f"{sf};{st};0")
                lines.append(
                    f"seg {i} {leg['name']}: {leg['f0']} -> {leg['f1']} Hz, "
                    f"{ns} steps x {sf} Hz, step_time_arb={st} -> DIM plays "
                    f"{played:.3f} ms (requested {leg['dur_ms']:.3f} ms)")
        _dim_send(ser, 'Mseg:1')
        return lines

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
        # Fixed driver: base freq, full amplitude, always on -- and DEAF.
        # It used to get FRQ/AMP/OUT_on and nothing else, on the reasoning
        # that it had no Seg cable so its segment state could not matter.
        # The tee to dio6 ended that; see _dim_make_deaf.
        _dim_make_deaf(ser_fixed, DIM_COM_FIXED)
        if n_legs_max == 0:
            # Stationary reference. There is no chain to play, and a driver
            # left in segment mode with NO chain written would play whatever
            # a previous run left in segment 1 on the first dio6 edge -- and
            # dio6 still reaches it. So COM4 is parked and made deaf exactly
            # like COM5: both beams at base, standing wave stationary.
            _dim_make_deaf(ser_sweep, DIM_COM_SWEEP)
        else:
            # Sweep driver: start at base, full amp, always on.
            dim_state.set_frequency(ser_sweep, base_freq_hz, DIM_COM_SWEEP, logger)
            _dim_send(ser_sweep, f'AMP:{amplitude_x10}')
            _dim_send(ser_sweep, 'OUT_on')
            seg_lines = _dim_program_segments(ser_sweep, 0)
            _dim_scrub_tail(ser_sweep, DIM_COM_SWEEP, n_legs_max)
            _dim_send(ser_sweep, 'Mseg:1')
            logger.info(
                f"DIM-3000 sweep ({DIM_COM_SWEEP}): {n_legs_max}-segment "
                f"one-way chain written, segment mode ON:")
            for _ln in seg_lines:
                logger.info(f"  {_ln}")
            # Read the chain back before trusting it. An empty Rseg reply is
            # what a segment the driver did not accept looks like; the move
            # would then stop part-way with the beams still detuned and the
            # lattice walking, which no other line in this log would reveal.
            missing = []
            _names0 = [lg['name'] for lg in legs_per_point[0]]
            for idx in range(1, n_legs_max + 1):
                reply = _dim_query(ser_sweep, f'Rseg:{idx}')
                label = _names0[idx - 1] if idx <= len(_names0) else 'padding'
                logger.info(f"  Rseg:{idx} ({label}) -> {reply or '<no reply>'}")
                if not reply:
                    missing.append(idx)
            if missing:
                logger.error(
                    f"DIM-3000 did not read back segment(s) {missing} of "
                    f"{n_legs_max}. The move plays only as far as the last "
                    f"segment that took, and if that is not the decel the "
                    f"lattice is LEFT MOVING after the move. Do not trust "
                    f"this run's images.")
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
        f"move ONE WAY {schedule[0]['total_ms']:.2f}ms in "
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
                f"the lattice would be left moving. Lengthen the accel/decel "
                f"time, or raise the NI sample rate.")
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
                f"segment. Shorten Trigger Lead, or lengthen that leg.")

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence(idx):
        sc = schedule[idx]
        do_move = (sc['d_mm'] > 0)
        # Sequence after PGC end:
        #   pre_hold -> [one-way move if moving] -> post_hold -> release
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
        # BEFORE that leg's waveform start (DIM internal latency). Three
        # pulses for a trapezoid, two for a triangle -- the count comes from
        # the leg list, so it follows whichever durations are zero without
        # anything here needing to know about it.
        # For 0mm reference, no triggers fire and DIM stays at base_freq.
        seg_do = [0] * N
        if do_move:
            for trig_t in leg_starts:
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
    logger.info(
        f"Lattice MOVE ONE WAY (COM4), single point: d={dist_values_mm[0]}mm, "
        f"T_a/T_p/T_d = {accel_ms:.2f}/{plat_ms_in:.2f}/{decel_ms:.2f}ms, "
        f"v={schedule[0]['v_mmps']:.2f}mm/s, imaged at the FAR point, "
        f"tof={tof_us}us, {warmup_cycles} warm-up cycle(s), total={total} samples")

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
        # Expected shift of the cloud centre on the sensor. Exact only if the
        # lattice axis lies in the image plane; any tilt toward the camera
        # shortens it by cos(angle), which is itself worth knowing.
        'd_px': sc0['d_mm'] * 1e3 / PIXEL_UM,
        'v_mmps': sc0['v_mmps'],
        'accel_ms': sc0['accel_ms'],
        'plat_ms': sc0['plat_ms'],
        'decel_ms': sc0['decel_ms'],
        'a_acc_mps2': sc0['a_acc_mps2'],
        'a_dec_mps2': sc0['a_dec_mps2'],
        'total_ms': sc0['total_ms'],
        'df_khz_signed': (sc0['freq_hz'] - base_freq_hz) / 1e3,
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
            # Re-write the chain at every re-arm, so each pass starts from
            # segment 1 whatever state the last pass left it in. NOT for a
            # stationary run: that has no chain, and this call ends in Mseg:1,
            # which would switch the deafened COM4 back into segment mode.
            if ser_sweep is not None and n_legs_max > 0:
                _dim_program_segments(ser_sweep, 0)
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
            dim_state.set_frequency(ser_sweep, base_freq_hz, DIM_COM_SWEEP, logger)
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
