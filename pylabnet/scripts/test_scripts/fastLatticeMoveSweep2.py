# Lattice TRANSPORT: ONE WAY, DRIVEN BY THE DIM-3000 FREQUENCY SWEEP (MODE 2)
#
# fastLatticeMove.py with the segment chain replaced by the driver's built-in
# frequency sweep, "SWPm:2 -- triangle with external trigger". Same inputs,
# same sequence, same camera, fits and atom number; only the way COM4's
# frequency is moved differs.
#
# COMMANDS: this driver's firmware only acts on the S-prefix form of the
# sweep commands (Sswpm:2, Sswps:..., Sswpp:..., Sswpf:..., Sswpt:...), not
# the manual's SWPm:/SWPs:/... -- found on the bench.
#
# A second window, "RF signals", draws the sub-sequence as it is sent: each
# AOM's frequency, the lattice velocity those frequencies imply, and the dio6
# gate taken from the actual NI buffer.
#
# HOW MODE 2 BEHAVES (manual v9, p.8-9, "Sync In/Out and output frequency
# relationship"). The rear-panel "Sync" SMA is a GATE, not an edge trigger:
#
#     Sync LOW            output sits at Fstart (SWPs)
#     Sync goes HIGH      ramps Fstart -> Fstop (SWPp) at SWPf Hz per SWPt ns,
#                         then HOLDS at Fstop for as long as Sync stays HIGH
#     Sync goes LOW       ramps Fstop -> Fstart at the same rate, then sits
#
# So ONE dio6 pulse is the whole move: its rising edge starts the accel, the
# time it stays high beyond the ramp is the plateau, and its falling edge
# starts the decel. With Fstart = base, the lattice starts and ends at rest.
#
#     ramp time   T_r = (|Fstop - Fstart| / SWPf) * SWPt      (manual, p.8)
#     gate width  T_g = T_r + T_p            (dio6 HIGH time)
#     sweep time  T   = T_g + T_r            (Sync rising edge -> back at base)
#
# The area under a symmetric trapezoid is v*(T_r + T_p) = v*T_g, so the
# distance is set by the detuning and the GATE WIDTH alone:
#
#     d = v * T_g,    v = (lambda/2) * |Fstop - Fstart|
#
# which is why the stop frequency below is computed from the gate width as the
# NI card will actually produce it (50 us grid), not from the typed durations.
#
# WHAT MODE 2 CANNOT DO. There is one step rate, used both ways, so the accel
# and decel are always the SAME length. T_a and T_d are still read, but the
# sweep uses their MEAN for both ramps. That keeps d, the peak velocity and the
# total move time exactly what fastLatticeMove would give for the same inputs;
# only the two accelerations change (they become equal). The log warns when
# T_a != T_d.
#
# Also unlike the segment chain there is no ~3 ms firmware delay documented
# for Sync -- the manual gives none at all. "Trigger Lead" is kept as an input
# but defaults to 0 here; measure the real latency (scope on dio6 against the
# RF, or a beat note) and put it in. A lead delays nothing inside the move:
# it shifts both gate edges together, so the move itself is unchanged.
#
# Full sequence:
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> pre-move hold
#   -> dio6 HIGH:  accel (base -> Fstop, T_r) then plateau at Fstop (T_p)
#   -> dio6 LOW:   decel (Fstop -> base, T_r)
#   -> post-move hold      (lattice stationary, atoms DISPLACED by d)
#   -> release -> TOF -> image
#
# WIRING. dio6 must reach COM4's rear "Sync" SMA, NOT its "Seg" SMA -- "Seg"
# does nothing outside segment-play mode. COM5 is left in normal operation
# (sweep off, segment mode off), so whatever dio6 is teed into there is
# ignored.
#
# Single AOM, single shot, lattice TTL & AM constant for the whole cycle: all
# exactly as in fastLatticeMove.py.

from pyqtgraph.Qt import QtCore, QtWidgets
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
    # Every duration here is what the DRIVER will play (SWPf/SWPt as sent,
    # gate on the NI grid), not what was typed; the typed values are in the log.
    move_rows = [
        ("transport (one way)",
         f"{d:.3f} mm (realised {m.get('d_real_mm', nan):.4f})",
         "expected shift on camera",
         f"{m.get('d_px', nan):.1f} px (if in image plane)"),
        ("accel = decel ramp T_r", f"{m.get('ramp_ms', nan):.3f} ms",
         "acceleration (= decel)", f"{m.get('a_mps2', nan):.2f} m/s&sup2;"),
        ("plateau at Fstop T_p", f"{m.get('plat_real_ms', nan):.3f} ms",
         "peak velocity", f"{m.get('v_mmps', nan):.2f} mm/s"),
        ("SWEEP START  SWPs", f"{m.get('f_start_hz', nan) / 1e6:.6f} MHz",
         "SWEEP STOP  SWPp", f"{m.get('f_stop_hz', nan) / 1e6:.6f} MHz"),
        ("SWEEP DURATION", f"{m.get('total_ms', nan):.3f} ms",
         "COM4 detuning at Fstop",
         f"{m.get('df_khz_signed', nan):+.3f} kHz ({m.get('shape', '--')})"),
        ("Sync gate (dio6 HIGH)", f"{m.get('gate_ms', nan):.3f} ms",
         "SWPf x SWPt",
         f"{m.get('fstep_hz', 0)} Hz x {m.get('tstep_ns', 0)} ns "
         f"({m.get('n_steps', 0)} steps/ramp)"),
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
    HEADER_PX = 300          # two rows taller than fastLatticeMove: sweep settings

    def __init__(self, title, levels):
        self._title = title
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(880, 1070)
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


def rf_signal_trace(base_hz, sc, move_start_s, gate, spm, v_per_hz_mmps):
    """One sub-sequence of the RF drive, as plottable piecewise-linear lines.

    :param base_hz: (int) COM5's frequency and COM4's rest frequency
    :param sc: (dict) schedule entry -- f_stop_hz, ramp_ms, gate_ms, d_mm
    :param move_start_s: (int) sample at which COM4 starts to ramp (the gate
        rises Gate Lead earlier; see gate)
    :param gate: (list of 0/1) the dio6 samples EXACTLY as sent to the card
    :param spm: (int) NI samples per ms
    :param v_per_hz_mmps: (float) lattice velocity per Hz of detuning
    :return: (dict) t_ms / f4_hz, t5_ms / f5_hz, v_mps (on t_ms),
        gate_t_ms / gate
    """
    end_ms = len(gate) / spm
    t0 = move_start_s / spm
    if sc['d_mm'] > 0:
        stop = float(sc['f_stop_hz'])
        ramp, g_ms = sc['ramp_ms'], sc['gate_ms']
        t = [0.0, t0, t0 + ramp, t0 + g_ms, t0 + g_ms + ramp, end_ms]
        f4 = [base_hz, base_hz, stop, stop, base_hz, base_hz]
    else:
        t, f4 = [0.0, end_ms], [base_hz, base_hz]
    f4 = np.array(f4, dtype=float)
    # v = (lambda/2) * (f_COM4 - f_COM5); COM5 is fixed at base. Signed, so
    # the plot shows which way the lattice goes, in m/s for pyqtgraph's SI
    # prefix.
    v_mps = v_per_hz_mmps * (f4 - base_hz) * 1e-3
    # The gate from the real buffer: its edges, drawn as a step.
    g = np.asarray(gate, dtype=np.int8)
    edges = np.flatnonzero(np.diff(g)) + 1
    gt, gv = [0.0], [float(g[0]) if g.size else 0.0]
    for e in edges:
        gt += [e / spm, e / spm]
        gv += [gv[-1], float(g[e])]
    gt.append(end_ms)
    gv.append(gv[-1])
    return {'t_ms': np.array(t), 'f4_hz': f4, 'v_mps': v_mps,
            't5_ms': np.array([0.0, end_ms]),
            'f5_hz': np.array([base_hz, base_hz], dtype=float),
            'gate_t_ms': np.array(gt), 'gate': np.array(gv)}


class RFSignalView:
    """Second window: the RF drive over one sub-sequence.

    Three plots on one time axis -- each AOM's frequency, the lattice velocity
    they imply, and the dio6 sweep gate as it is in the NI buffer -- with the
    sequence's phase boundaries marked. It opens zoomed on the move; drag or
    right-click -> "View All" to see the whole cycle.

    Same threading rule as MoveLiveView: callers only store data (set_trace),
    a GUI-thread QTimer draws it.
    """

    POLL_MS = 200

    def __init__(self, title="RF signals (COM4 swept, COM5 fixed)"):
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle(title)
        self.win.setStyleSheet("background-color: black;")
        box = QtWidgets.QVBoxLayout(self.win)
        box.setContentsMargins(6, 6, 6, 6)
        self.label = QtWidgets.QLabel('RF signals appear when the run starts.')
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setStyleSheet("color: #DDDDDD; font-size: 10pt;")
        box.addWidget(self.label)
        canvas = pg.GraphicsLayoutWidget()
        box.addWidget(canvas, stretch=1)
        self.win.resize(900, 820)
        self.win.show()

        self.f_plot = canvas.addPlot(row=0, col=0)
        self.f_plot.setLabel('left', 'RF frequency', units='Hz')
        self.f_plot.showGrid(x=True, y=True, alpha=0.3)
        self.f_plot.addLegend(offset=(10, 10))
        self.f4_curve = self.f_plot.plot(
            [], [], pen=pg.mkPen('#4DD0E1', width=2), name='COM4 (swept)')
        self.f5_curve = self.f_plot.plot(
            [], [], pen=pg.mkPen('#FFB74D', width=2, style=QtCore.Qt.DashLine),
            name='COM5 (fixed)')

        self.v_plot = canvas.addPlot(row=1, col=0)
        self.v_plot.setLabel('left', 'lattice velocity', units='m/s')
        self.v_plot.showGrid(x=True, y=True, alpha=0.3)
        self.v_curve = self.v_plot.plot([], [], pen=pg.mkPen('#81C784', width=2))

        self.g_plot = canvas.addPlot(row=2, col=0)
        self.g_plot.setMaximumHeight(130)
        self.g_plot.setLabel('left', 'dio6 gate')
        self.g_plot.setLabel('bottom', 'time in sub-sequence', units='s')
        self.g_plot.setYRange(-0.1, 1.1)
        self.g_plot.showGrid(x=True, y=False, alpha=0.3)
        self.g_curve = self.g_plot.plot([], [], pen=pg.mkPen('w', width=2))

        self.v_plot.setXLink(self.g_plot)
        self.f_plot.setXLink(self.g_plot)
        self._markers = []

        self._pending = None
        self._drawn = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(self.POLL_MS)

    def set_trace(self, trace, markers, html, zoom_ms):
        """Called from the acquisition thread; stores data, touches no Qt.

        :param markers: (list of (t_ms, text)) phase boundaries to mark
        :param zoom_ms: (lo, hi) initial x range in ms
        """
        self._pending = (trace, markers, html, zoom_ms, self._drawn + 1)

    def _poll(self):
        entry = self._pending
        if entry is None or entry[-1] <= self._drawn:
            return
        trace, markers, html, zoom_ms, version = entry
        self._drawn = version
        try:
            self.label.setText(html)
            s = 1e-3                                      # ms -> s
            self.f4_curve.setData(trace['t_ms'] * s, trace['f4_hz'])
            self.f5_curve.setData(trace['t5_ms'] * s, trace['f5_hz'])
            self.v_curve.setData(trace['t_ms'] * s, trace['v_mps'])
            self.g_curve.setData(trace['gate_t_ms'] * s, trace['gate'])
            # Flat lines autoscale to a zero-height range; give them some.
            lo = min(trace['f4_hz'].min(), trace['f5_hz'].min())
            hi = max(trace['f4_hz'].max(), trace['f5_hz'].max())
            pad = max((hi - lo) * 0.1, 1e3)
            self.f_plot.setYRange(lo - pad, hi + pad)
            vlo, vhi = trace['v_mps'].min(), trace['v_mps'].max()
            vpad = max((vhi - vlo) * 0.1, 1e-4)
            self.v_plot.setYRange(vlo - vpad, vhi + vpad)
            for plot, line in self._markers:
                plot.removeItem(line)
            self._markers = []
            for t_ms, text in markers:
                for plot in (self.f_plot, self.v_plot, self.g_plot):
                    line = pg.InfiniteLine(
                        pos=t_ms * s, angle=90,
                        pen=pg.mkPen('#9E9E9E', width=1, style=QtCore.Qt.DotLine),
                        label=text if plot is self.f_plot else None,
                        labelOpts={'position': 0.95, 'color': '#BDBDBD'})
                    plot.addItem(line)
                    self._markers.append((plot, line))
            self.g_plot.setXRange(zoom_ms[0] * s, zoom_ms[1] * s, padding=0)
        except Exception:
            pass          # a display problem must never stop the run


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
    'pgc_rampdown_ms': {'PGC Rampdown Duration (ms)': '0'},

    # ===================== LATTICE AM =====================
    # DIM-3000 set to AMP=320 (+32dBm) via dim_rf_on.py; AM=1.0V => full max.
    # AOM thermal stability: lattice TTL stays ON & AM stays at lat_am_v
    # for the ENTIRE cycle.
    'lat_am_v': {'Lattice AM (V)': '0.6'},

    # ===================== MOVE (one way, COM4 frequency sweep, mode 2) =====================
    # Same inputs as fastLatticeMove. Velocity is derived exactly as there:
    #
    #     v = d / (T_a/2 + T_p + T_d/2)
    #
    # but the sweep has ONE step rate for both ramps, so accel and decel both
    # run for (T_a + T_d)/2. With T_a == T_d (the defaults) nothing changes.
    # The start/stop frequencies and the sweep duration the driver will
    # actually play are printed at the top of the panel.
    #
    # 0 distance is a stationary reference: the gate never opens, sweep mode is
    # left OFF, and the cloud is imaged where it was loaded.
    'dist_mm': {'Move Distance (mm, one way)': '0.5'},
    # T_p = 0 is a triangle (the gate closes the moment Fstop is reached).
    # T_a = T_d = 0 is a velocity STEP, which the log warns about.
    'accel_ms': {'Accel Time T_a (ms)': '5.0'},
    'plat_ms': {'Plateau (constant velocity) Time T_p (ms)': '15.0'},
    'decel_ms': {'Decel Time T_d (ms)': '5.0'},
    # How far AHEAD of the move the gate is raised, to absorb the driver's
    # Sync-input latency. The manual documents none for the sweep (the ~3 ms
    # in fastLatticeMove is the SEGMENT engine's), so 0 until measured.
    # Fractional ms allowed; snapped to the 50 us NI grid.
    'seg_lead_ms': {'Sweep Gate Lead (ms) [Sync latency]': '0'},
    # +1 or -1: the sign of COM4's detuning, i.e. which way along the lattice
    # axis the atoms go. There is no return leg, so this DOES decide where the
    # cloud ends up.
    'move_direction': {'Move Direction (+1 or -1)': '1'},

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
    'dio6': 0,      # DIM-3000 sweep gate LOW -> COM4 back at Fstart (= base)
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
        title=f"Lattice move ONE WAY (COM4 sweep mode 2) -- d={dist_mm:.3f} mm",
        levels=(disp_min, disp_max))
    dataset.img_win = dataset.live.win
    dataset.img_item = dataset.live.img_item
    # Second window: the RF drive, filled in by experiment() once the
    # sub-sequence is built (the timeline is only known there).
    dataset.rf_view = RFSignalView()
    dataset.graph.hide()


def experiment(**kwargs):
    """Run the sequence, then put the RF drivers back the way they were found.

    DataTaker calls this; the sequence itself is _run_experiment(). This
    wrapper exists for its `finally:` alone -- whatever ends the run (Stop, an
    exception, a device dropping out), every DIM-3000 driver the run opened is
    restored to the settings dim_state.snapshot() recorded before the run
    wrote anything to it, and left OUT of segment mode. See dim_state.py.

    Sweep mode is switched off here FIRST, with Sswpm:0. A driver left in
    mode 2 walks the lattice to Fstop on any stray HIGH at its Sync input.

    COMMAND FORM: this firmware ignores the manual's SWPm:/SWPs:/SWPp:/SWPf:/
    SWPt: commands and only acts on the Gpar-style S-prefix writes (Sswpm:,
    Sswps:, ...), found on the bench. Every sweep command in this file uses
    that form.
    """
    snaps = []
    try:
        _run_experiment(dim_snapshots=snaps, **kwargs)
    finally:
        for snap in snaps:
            if not snap:
                continue
            try:
                if snap['_ser'].is_open:
                    snap['_ser'].write(b'Sswpm:0\n')
                    time.sleep(0.12)
            except Exception:
                pass
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

    # ONE transport distance, kept as a one-entry list so the per-point
    # structures below index the same way fastLatticeMove's do.
    dist_values_mm = [float(dataset.get_input_parameter("dist_mm"))]
    n_scan = 1
    pre_hold_ms = int(dataset.get_input_parameter("pre_hold_ms"))
    post_hold_ms = int(dataset.get_input_parameter("post_hold_ms"))
    accel_ms = float(dataset.get_input_parameter("accel_ms"))
    plat_ms_in = float(dataset.get_input_parameter("plat_ms"))
    decel_ms = float(dataset.get_input_parameter("decel_ms"))
    # Float, not int as in fastLatticeMove: the Sync latency may well be
    # sub-millisecond, and int("0.2") would stop the run before it started.
    seg_lead_ms = float(dataset.get_input_parameter("seg_lead_ms"))
    if min(accel_ms, plat_ms_in, decel_ms) < 0:
        raise ValueError(
            f"Move durations must be >= 0 (got T_a={accel_ms}, "
            f"T_p={plat_ms_in}, T_d={decel_ms} ms).")
    if seg_lead_ms < 0:
        raise ValueError(f"Sweep Gate Lead must be >= 0, got {seg_lead_ms} ms.")
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

    # NI sample clock. Needed this early because the gate width -- and through
    # it the stop frequency -- is quantised to it.
    ni_rate = 20000
    SPM = ni_rate // 1000
    SAMPLE_MS = 1000.0 / ni_rate

    def ceil_samples(t_ms):
        """ms -> whole NI samples, rounding UP (1e-9 absorbs float fuzz)."""
        return int(np.ceil(t_ms * SPM - 1e-9))

    # DIM-3000 sweep limits (manual p.4 and p.13). SWPt is in ns, 4..262000,
    # and the firmware rounds it DOWN to a multiple of 4 -- so only multiples
    # of 4 are ever sent, and what is sent is what plays.
    #
    # SWEEP_STEP_NS_MIN is ours, not the manual's: the driver accepts 4 ns,
    # but 1 us steps already make a 5 ms ramp 5000 steps long, far smoother
    # than the lattice can resolve, and nothing is gained by asking the DDS
    # for more.
    SWEEP_STEP_NS_QUANTUM = 4
    SWEEP_STEP_NS_MAX = 262000
    SWEEP_STEP_NS_MIN = 1000
    SWEEP_FSTEP_SEARCH = 20000   # Fstep candidates tried, from the finest up

    def sweep_steps(span_hz, ramp_ms):
        """SWPf and SWPt for a ramp covering span_hz in (at most) ramp_ms.

        The ramp is n steps of SWPf Hz, each SWPt ns long, and the driver
        plays exactly that, so n * SWPf is the span it actually covers.
        Candidates are tried from the FINEST step upward, and the first one
        within 2e-4 of the span (the span IS the velocity, and so the
        distance) and 2e-3 of the ramp time is taken -- so the ramp is as
        smooth as the step-time floor allows. If none gets that close, the
        best-scoring one is used. SWPt is always rounded DOWN, so the ramp is
        never longer than asked for and therefore always fits inside the gate.

        :return: (dict) fstep_hz, tstep_ns, n_steps, span_hz (as played),
            ramp_ms (as played), clipped (True if the ramp had to be made
            SHORTER than asked because even 1 Hz steps at the longest step
            time cannot fill it)
        """
        q = SWEEP_STEP_NS_QUANTUM
        target_ns = ramp_ms * 1e6
        if target_ns < SWEEP_STEP_NS_MIN:
            # A ramp shorter than one step -- including 0, a velocity step.
            # One step straight to Fstop at the shortest time the driver takes.
            t_ns = max(q, int(target_ns // q) * q)
            return {'fstep_hz': span_hz, 'tstep_ns': t_ns, 'n_steps': 1,
                    'span_hz': span_hz, 'ramp_ms': t_ns * 1e-6,
                    'clipped': False}
        n_min = max(1, int(np.ceil(target_ns / SWEEP_STEP_NS_MAX)))
        n_max = int(target_ns // SWEEP_STEP_NS_MIN)
        if span_hz < n_min:
            # Too slow a ramp for the span: 1 Hz steps at 262 us each run out
            # of frequency before they run out of time. Take the slowest ramp
            # there is and say so.
            t_ns = (SWEEP_STEP_NS_MAX // q) * q
            return {'fstep_hz': 1, 'tstep_ns': t_ns, 'n_steps': span_hz,
                    'span_hz': span_hz, 'ramp_ms': span_hz * t_ns * 1e-6,
                    'clipped': True}
        best = None
        f_lo = max(1, span_hz // max(n_max, 1))
        for f in range(f_lo, f_lo + SWEEP_FSTEP_SEARCH):
            n = int(round(span_hz / f))
            if n < n_min:
                break          # steps only get coarser from here
            if n > n_max:
                continue
            t_ns = int((target_ns / n) // q) * q
            if t_ns < q or t_ns > SWEEP_STEP_NS_MAX:
                continue
            span_err = abs(n * f - span_hz) / span_hz
            ramp_err = abs(n * t_ns - target_ns) / target_ns
            score = 10.0 * span_err + ramp_err
            if best is None or score < best[0]:
                best = (score, f, t_ns, n)
            if span_err <= 2e-4 and ramp_err <= 2e-3:
                best = (score, f, t_ns, n)
                break
        if best is None:
            raise ValueError(
                f"No DIM sweep step settings cover {span_hz} Hz in "
                f"{ramp_ms:.3f} ms. Change the ramp time or the distance.")
        _, f, t_ns, n = best
        return {'fstep_hz': f, 'tstep_ns': t_ns, 'n_steps': n,
                'span_hz': n * f, 'ramp_ms': n * t_ns * 1e-6,
                'clipped': False}

    def schedule_for_distance(d_mm):
        """Everything the sweep needs for one one-way move of distance d_mm.

        Velocity comes from the typed durations exactly as in fastLatticeMove,
        v = d / (T_a/2 + T_p + T_d/2). Mode 2 then forces both ramps to one
        length, T_r = (T_a + T_d)/2, which keeps T_r + T_p -- and so v and
        the total time -- unchanged.

        The gate is T_r + T_p, rounded UP to the NI grid so the ramp always
        completes inside it (a gate that closed early would turn round below
        Fstop, like the second pulse in the manual's mode-2 diagram). Since
        d = v * gate for this profile, the detuning is then recomputed from
        the gate as it will actually be played, so the rounding costs a few
        tens of microseconds of plateau rather than any distance.

        :return: (dict) with f_start_hz / f_stop_hz / fstep_hz / tstep_ns
            for the driver, gate_s for the NI card, and every derived
            quantity the panel and log print.
        """
        stationary = {'d_mm': float(d_mm), 'd_real_mm': 0.0, 'v_mmps': 0.0,
                      'ramp_req_ms': 0.0, 'ramp_ms': 0.0,
                      'plat_ms': 0.0, 'plat_real_ms': 0.0,
                      'gate_s': 0, 'gate_ms': 0.0, 'total_ms': 0.0,
                      'a_mps2': 0.0, 'f_start_hz': base_freq_hz,
                      'f_stop_hz': base_freq_hz, 'df_hz': 0,
                      'fstep_hz': 0, 'tstep_ns': 0, 'n_steps': 0,
                      'clipped': False}
        if d_mm <= 0:
            return stationary
        ramp_req = 0.5 * (accel_ms + decel_ms)
        if ramp_req + plat_ms_in <= 0:
            raise ValueError(
                f"Distance {d_mm} mm needs a non-zero move time, but "
                f"T_a/2 + T_p + T_d/2 = 0. Set at least one duration.")
        gate_s = ceil_samples(ramp_req + plat_ms_in)
        gate_ms = gate_s * SAMPLE_MS
        span_target = int(round(d_mm / (gate_ms * 1e-3) / v_per_Hz_mmps))
        if span_target < 1:
            raise ValueError(
                f"{d_mm} mm in {gate_ms:.3f} ms is under 1 Hz of detuning, "
                f"below the driver's resolution.")
        st = sweep_steps(span_target, ramp_req)
        span = st['span_hz']
        v_mmps = span * v_per_Hz_mmps
        ramp_ms = st['ramp_ms']
        return {'d_mm': float(d_mm),
                'd_real_mm': v_mmps * gate_ms * 1e-3,
                'v_mmps': v_mmps,
                'ramp_req_ms': ramp_req,
                'ramp_ms': ramp_ms,
                'plat_ms': plat_ms_in,
                'plat_real_ms': gate_ms - ramp_ms,
                'gate_s': gate_s,
                'gate_ms': gate_ms,
                # Sync rising edge -> back at Fstart: up, hold, down.
                'total_ms': gate_ms + ramp_ms,
                'a_mps2': (v_mmps * 1e-3) / (ramp_ms * 1e-3),
                'f_start_hz': base_freq_hz,
                'f_stop_hz': base_freq_hz + move_direction * span,
                'df_hz': span,
                'fstep_hz': st['fstep_hz'],
                'tstep_ns': st['tstep_ns'],
                'n_steps': st['n_steps'],
                'clipped': st['clipped']}

    schedule = [schedule_for_distance(d) for d in dist_values_mm]
    v_peak_mmps_per_point = [sc['v_mmps'] for sc in schedule]

    sc0 = schedule[0]
    do_move_any = sc0['d_mm'] > 0
    if do_move_any:
        logger.info(
            f"Move physics (ONE WAY, COM4 sweep mode 2): d={sc0['d_mm']:.3f}mm, "
            f"typed T_a={accel_ms:.3f} / T_p={plat_ms_in:.3f} / "
            f"T_d={decel_ms:.3f} ms -> played: ramp {sc0['ramp_ms']:.4f} ms "
            f"each way, plateau {sc0['plat_real_ms']:.4f} ms, gate "
            f"{sc0['gate_ms']:.3f} ms, sweep {sc0['total_ms']:.4f} ms; "
            f"v={sc0['v_mmps']:.3f}mm/s, a={sc0['a_mps2']:.3f}m/s^2, "
            f"SWPs={sc0['f_start_hz']} Hz, SWPp={sc0['f_stop_hz']} Hz "
            f"({move_direction * sc0['df_hz'] / 1e3:+.3f} kHz), "
            f"SWPf={sc0['fstep_hz']} Hz, SWPt={sc0['tstep_ns']} ns, "
            f"{sc0['n_steps']} steps; realised distance {sc0['d_real_mm']:.4f}mm")
        if abs(accel_ms - decel_ms) > 1e-9:
            logger.warn(
                f"T_a ({accel_ms} ms) != T_d ({decel_ms} ms). Sweep mode 2 has "
                f"one step rate for both ramps, so BOTH run for their mean, "
                f"{sc0['ramp_req_ms']:.3f} ms. Distance, peak velocity and total "
                f"time are as typed; the accelerations are not. Use "
                f"fastLatticeMove.py for an asymmetric profile.")
        if accel_ms + decel_ms <= 0:
            logger.warn(
                f"Accel and decel times are 0: the lattice jumps to "
                f"{sc0['v_mmps']:.1f} mm/s in a single step and stops the same "
                f"way. That is an unbounded acceleration and will shake most "
                f"atoms out.")
        if sc0['clipped']:
            logger.warn(
                f"The ramp is only {sc0['ramp_ms']:.3f} ms, not the "
                f"{sc0['ramp_req_ms']:.3f} ms asked for: {sc0['df_hz']} Hz of "
                f"detuning in 1 Hz steps at the longest step time the driver "
                f"has (262 us) cannot take longer. The plateau is longer to "
                f"match, so the distance is unaffected.")
    else:
        logger.info("Move distance 0: STATIONARY reference -- gate never opens, "
                    "sweep mode OFF, cloud imaged where it was loaded.")

    # =========================================================================
    # CONFIGURE DIM-3000 DRIVERS (called once per experiment start)
    # =========================================================================
    DIM_COM_SWEEP = 'COM4'   # lattice up -- swept, its rear "Sync" on dio6
    DIM_COM_FIXED = 'COM5'   # lattice down -- fixed at base

    # Only used to neutralise segment tables other scripts leave behind; this
    # script never plays a segment.
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
        time.sleep(0.12)          # manual: ~10 commands/s at most

    def _dim_query(ser, cmd):
        ser.flushInput()
        _dim_send(ser, cmd)
        time.sleep(0.1)
        return ser.readline().decode().strip()

    def _dim_gpar(ser):
        """Gpar parsed to {field: int}; unparseable fields left out."""
        out = {}
        for item in _dim_query(ser, 'Gpar').split('|'):
            if ':' in item:
                key, val = item.split(':', 1)
                try:
                    out[key.strip()] = int(val.strip())
                except ValueError:
                    pass
        return out

    def _dim_make_deaf(ser, port):
        """Park a driver at base in plain single-frequency operation.

        Sweep OFF, so its Sync input does nothing, and segment mode OFF, so its
        Seg input does nothing -- whichever of the two dio6 happens to be teed
        into on this driver. Segments are neutralised before Mseg:0 for the
        same reason as in fastLatticeMove: a stale chain is harmless only
        while nothing can play it.
        """
        _dim_send(ser, 'Sswpm:0')
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
            f"+{amplitude_x10/10:.1f} dBm, ON, sweep OFF (Sswpm:0), segment "
            f"mode OFF (Mseg:0) -- ignores dio6 on either input.")

    def _dim_program_sweep(ser, sc):
        """Load the move into COM4's sweep engine and arm mode 2.

        Order matters. Segment mode goes off first so the two engines never
        both own the carrier. Sweep goes to 0 before its limits change, so the
        driver is never armed with a half-written set. FRQ is set to base too
        -- mode 0's "Fnormal" -- so entering and leaving mode 2 does not step
        the frequency (mode 2 idles at Fstart, which is also base).

        :return: (list of str) mismatches found on Gpar readback; empty = OK
        """
        # The bench-tested sequence (S-prefix forms; see experiment()):
        #   Mseg:0  Sswpm:0  FRQ  AMP  OUT_on  Sswps  Sswpp  Sswpf  Sswpt  Sswpm:2
        _dim_send(ser, 'Mseg:0')
        _dim_send(ser, 'Sswpm:0')
        dim_state.set_frequency(ser, base_freq_hz, DIM_COM_SWEEP, logger)
        _dim_send(ser, f'AMP:{amplitude_x10}')
        _dim_send(ser, 'OUT_on')
        _dim_send(ser, f"Sswps:{sc['f_start_hz']}")
        _dim_send(ser, f"Sswpp:{sc['f_stop_hz']}")
        _dim_send(ser, f"Sswpf:{sc['fstep_hz']}")
        _dim_send(ser, f"Sswpt:{sc['tstep_ns']}")
        _dim_send(ser, 'Sswpm:2')
        # Read it back. A dropped or refused command (the manual quotes ~10
        # commands/s) would otherwise give a move of the wrong length or
        # distance with nothing in the log to say so -- and a stop frequency
        # BELOW the start (Move Direction -1) is not something the manual
        # says the sweep engine accepts.
        time.sleep(0.1)
        got = _dim_gpar(ser)
        want = {'Rswpm': 2, 'Rswps': sc['f_start_hz'],
                'Rswpp': sc['f_stop_hz'], 'Rswpf': sc['fstep_hz'],
                'Rswpt': sc['tstep_ns']}
        return [f"{k} is {got.get(k, '<missing>')} (sent {v})"
                for k, v in want.items() if got.get(k) != v]

    # Open both ports ONCE and keep them open for the whole run.
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
        _dim_make_deaf(ser_fixed, DIM_COM_FIXED)
        if not do_move_any:
            # Stationary reference: both beams at base, standing wave still.
            _dim_make_deaf(ser_sweep, DIM_COM_SWEEP)
        else:
            # Scrub the segment table too: sweep and segment engines are
            # separate, but a stale chain is one Mseg:1 from playing.
            for i in range(1, DIM_MAX_SEGMENTS + 1):
                flag = 8 if i == DIM_MAX_SEGMENTS else 1
                _dim_send(ser_sweep,
                          f'Wseg:{i};{flag};{base_freq_hz};{base_freq_hz};0;1;0')
            bad = _dim_program_sweep(ser_sweep, sc0)
            logger.info(
                f"DIM-3000 sweep ({DIM_COM_SWEEP}): mode 2 armed -- "
                f"SWPs {sc0['f_start_hz']} Hz, SWPp {sc0['f_stop_hz']} Hz, "
                f"SWPf {sc0['fstep_hz']} Hz, SWPt {sc0['tstep_ns']} ns. "
                f"Sync HIGH ramps to Fstop and holds; Sync LOW ramps back.")
            if bad:
                logger.error(
                    f"DIM-3000 {DIM_COM_SWEEP} did not take the sweep settings: "
                    f"{'; '.join(bad)}. The move will not be the one on the "
                    f"panel. Do not trust this run's images.")
            else:
                logger.info(f"DIM-3000 {DIM_COM_SWEEP}: sweep settings verified "
                            f"by Gpar readback.")
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
    # ni_rate and SPM are set with the move physics above: the gate width,
    # and through it the stop frequency, is quantised to this clock.
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
        f"pre_hold={pre_hold_ms}ms, "
        f"move ONE WAY {schedule[0]['total_ms']:.3f}ms (gate "
        f"{schedule[0]['gate_ms']:.3f}ms + decel {schedule[0]['ramp_ms']:.3f}ms), "
        f"post_hold={post_hold_ms}ms, TOF={tof_us}us"
    )
    # The gate rises seg_lead_ms before the move is meant to start, and the
    # move starts pre_hold_ms after PGC ends. A lead longer than the pre-move
    # hold would push it back into the PGC ramp-down, where it would start the
    # atoms moving while the cooling light is still coming down.
    if seg_lead_ms >= pre_hold_ms:
        logger.warn(
            f"Gate Lead ({seg_lead_ms}ms) is not shorter than the pre-move "
            f"hold ({pre_hold_ms}ms), so the sweep gate opens during PGC "
            f"ramp-down instead of in the hold. Raise Pre-move Hold above "
            f"{seg_lead_ms}ms.")

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence(idx):
        sc = schedule[idx]
        do_move = (sc['d_mm'] > 0)
        # Sequence after PGC end:
        #   pre_hold -> [one-way move if moving] -> post_hold -> release
        # The move is ONE gate pulse on dio6: HIGH for gate_s samples (accel
        # + plateau), then the driver decelerates on its own for the ramp
        # time after the falling edge. move_end is when it is back at base.
        # For 0mm reference (dist_mm==0) the move collapses to nothing and the
        # gate never opens => atoms just sit in the stationary lattice.
        move_start = pgc_end + pre_hold_s
        if do_move:
            gate_s = sc['gate_s']
            move_end = move_start + gate_s + ceil_samples(sc['ramp_ms'])
        else:
            gate_s = 0
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

        # DIM-3000 sweep gate (dio6 -> COM4 rear "Sync"): HIGH for the whole
        # accel + plateau, raised seg_lead_ms early to absorb the Sync latency.
        # The lead moves BOTH edges, so the gate width -- which is what sets
        # the distance -- is untouched by it.
        # For 0mm reference the gate never opens and COM4 stays at base_freq.
        seg_do = [0] * N
        if do_move:
            t0 = max(0, move_start - seg_lead_s)
            for s in range(gate_s):
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
    # detuning, gate width) lives in schedule, so passing the index is what
    # keeps a sub-sequence from ever being built against another point's
    # numbers.
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
        f"Lattice MOVE ONE WAY (COM4 sweep mode 2), single point: "
        f"d={dist_values_mm[0]}mm, "
        f"T_a/T_p/T_d = {accel_ms:.2f}/{plat_ms_in:.2f}/{decel_ms:.2f}ms, "
        f"v={schedule[0]['v_mmps']:.2f}mm/s, imaged at the FAR point, "
        f"tof={tof_us}us, {warmup_cycles} warm-up cycle(s), total={total} samples")

    # Everything the header says about the MOVE, worked out once, here, where
    # the schedule and the sub-sequence length are both in scope.
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
        'd_real_mm': sc0['d_real_mm'],
        'v_mmps': sc0['v_mmps'],
        'ramp_ms': sc0['ramp_ms'],
        'plat_real_ms': sc0['plat_real_ms'],
        'a_mps2': sc0['a_mps2'],
        'gate_ms': sc0['gate_ms'],
        'total_ms': sc0['total_ms'],
        'f_start_hz': sc0['f_start_hz'],
        'f_stop_hz': sc0['f_stop_hz'],
        'fstep_hz': sc0['fstep_hz'],
        'tstep_ns': sc0['tstep_ns'],
        'n_steps': sc0['n_steps'],
        'df_khz_signed': (sc0['f_stop_hz'] - base_freq_hz) / 1e3,
        # Judged on the TYPED T_p: a triangle still plays up to one NI sample
        # (50 us) of plateau, from rounding the gate up.
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

    # RF signals window. The gate is the LAST sub-sequence's dio6 buffer
    # (seg_w, left by the concatenation loop) -- the real samples, not a
    # re-derivation. The frequency lines are anchored at the same move_start
    # build_sub_sequence() uses: the RF starts ramping there, Gate Lead after
    # the gate rises (which is what the lead is for).
    try:
        rf_move_start = pgc_end + pre_hold_s
        rf_move_end = rf_move_start + (
            sc0['gate_s'] + ceil_samples(sc0['ramp_ms']) if sc0['d_mm'] > 0 else 0)
        rf_release = rf_move_end + post_hold_s
        rf_image = rf_release + tof_s_fixed
        rf_trace = rf_signal_trace(base_freq_hz, sc0, rf_move_start, seg_w,
                                   SPM, v_per_Hz_mmps)
        rf_markers = [(pgc_end / SPM, 'PGC end'),
                      (rf_move_start / SPM, 'move start'),
                      (rf_move_end / SPM, 'move end'),
                      (rf_release / SPM, 'release'),
                      (rf_image / SPM, 'image 1')]
        rf_html = (
            f"<span style='font-family:monospace'>"
            f"COM4 SWPs <b>{sc0['f_start_hz'] / 1e6:.6f} MHz</b> &rarr; SWPp "
            f"<b>{sc0['f_stop_hz'] / 1e6:.6f} MHz</b>, "
            f"{sc0['fstep_hz']} Hz x {sc0['tstep_ns']} ns | COM5 fixed "
            f"<b>{base_freq_hz / 1e6:.6f} MHz</b><br>"
            f"sweep {sc0['total_ms']:.3f} ms (ramp {sc0['ramp_ms']:.3f} + hold "
            f"{sc0['plat_real_ms']:.3f} + ramp) | gate {sc0['gate_ms']:.3f} ms, "
            f"lead {seg_lead_ms:g} ms | peak v {sc0['v_mmps']:.2f} mm/s | "
            f"sub-sequence {N / SPM:.1f} ms</span>")
        margin = 5.0
        zoom = ((rf_move_start - seg_lead_s) / SPM - pre_hold_ms - margin,
                rf_image / SPM + margin)
        dataset.rf_view.set_trace(rf_trace, rf_markers, rf_html, zoom)
    except Exception as exc:
        logger.warn(f"Could not draw the RF signals window: {exc!r}")

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
            # Nothing to re-write per pass, unlike fastLatticeMove's segment
            # chain: mode 2 has no counter to fall out of step. It follows
            # the LEVEL on Sync, so every gate plays the same move from base
            # no matter how the previous pass ended.
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
                    # No DIM reprogramming between cycles: every cycle is the
                    # same move, so the sweep armed at startup is already
                    # right.
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
    # Park the DIM before letting go of the port. Left in mode 2, COM4 would
    # ramp to Fstop and STAY there on any HIGH at Sync -- a stop landing
    # mid-gate, or the next script's dio6 pattern -- and the standing wave
    # would keep walking. Sweep off first, then base, so the lattice is on but
    # stationary, which is the state park_outputs() assumes on the NI side.
    if ser_sweep is not None:
        try:
            _dim_send(ser_sweep, 'Sswpm:0')
            dim_state.set_frequency(ser_sweep, base_freq_hz, DIM_COM_SWEEP, logger)
            logger.info(f"DIM sweep OFF (Sswpm:0) and parked at base "
                        f"{base_freq_hz/1e6:.3f} MHz (lattice on, stationary)")
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
