# Lattice TRANSPORT ONE WAY, DIM-3000 SWEEP MODE 2, BOTH AOMs SWEEPING
# -- BOTTOM CAMERA VERSION.
#
# The transport half of lattice_to_bottom_twoaoms_spinview.py, unchanged, with
# the imaging half of lattice_to_Bottom.py bolted on in place of the lit hold.
#
#   from lattice_to_bottom_twoaoms_spinview.py (byte for byte):
#       MOT, CMOT, Dark MOT, PGC, lattice loading, the pre-move hold, the
#       dual-AOM move, the hold at the far point, the DIM-3000 programming,
#       the gate polarity and rest states, the RF signals window.
#   from lattice_to_Bottom.py:
#       the bottom camera client, the dio8 trigger, the two 200 us frames
#       200 ms apart, the Gaussian fits, the atom number and the live panel.
#
# So a number from this script is comparable with a number from
# lattice_to_Bottom.py, and a cloud seen here is the same cloud the SpinView
# script shows -- the only thing that changed is how it is recorded.
#
# WHAT REPLACED THE LIT HOLD. The SpinView script lights the atoms for
# bottom_hold_ms and lets you watch. This one releases at the end of the hold
# at the far point, waits TOF, then fires one 200 us flash for frame 1 and a
# second, identical flash 200 ms later for frame 2. The difference is the
# cloud; nothing is accumulated between shots.
#
# THE PHYSICS. Only the beat between the two beams moves the standing wave:
#
#     v = (lambda/2) * (f_up - f_down)
#
# so sweeping one beam by df, or both by df/2 in opposite directions, gives
# the SAME transport. What halves is how far either AOM is from the frequency
# it was aligned at, which is what lets the velocity cap be set twice as high.
#
# THE PROBLEM. The DIM-3000 sweep engine only ramps UP from SWPs to SWPp
# (bench-tested: a stop frequency below the start does not sweep). Sweeping
# BOTH drivers up would leave the beat unchanged and the lattice at rest.
#
# THE TRICK. Manual v9 p.9, mode 2 diagram: Sync is a level.
#     Sync HIGH   ->  ramp SWPs -> SWPp at SWPf per SWPt, then HOLD at SWPp
#     Sync LOW    ->  ramp back DOWN to SWPs at the same rate, then hold there
# The falling edge IS a downward sweep. So the "down" driver is given a sweep
# window that ENDS at base, and its gate is run INVERTED:
#
#     UP driver    SWPs = base          SWPp = base + df/2
#                  gate LOW at rest, HIGH for accel + plateau
#     DOWN driver  SWPs = base - df/2   SWPp = base
#                  gate HIGH at rest (sits at SWPp = base), LOW for the move
#
# One NI window [t0, t0 + gate) is written HIGH on the up driver's line and
# LOW on the down driver's line. At t0 the up driver ramps up and the down
# driver ramps down -- both at the same SWPf/SWPt, so the beat opens at twice
# the per-driver rate. At t0 + gate both ramp back to base and the lattice
# comes to rest, displaced by d. Both then sit at base -- one with Sync LOW at
# its SWPs, the other with Sync HIGH at its SWPp -- which is single-frequency
# operation in everything but name. That is the hold at the far point.
#
# WHY THE HOLD IS NOT "OUTSIDE SWEEP MODE". Mode changes go over serial at
# about 10 commands a second with no hardware timing, so Sswpm:0 cannot be
# placed at a defined point in a 20 kHz NI sequence, and it is not needed: at
# rest both drivers are at base and neither Sync line moves. Sweep mode is
# switched off, over serial, only when the run stops.
#
# WIRING (both drivers' rear "Sync" SMA, each on its own NI line):
#   dio6 -> COM4 Sync        dio7 -> COM5 Sync
# NOT a tee: the two gates are complements of each other during the move.
# As a safeguard the FSK frequency and amplitude of BOTH drivers are set to
# base / AMP for the run (restored on stop), so a HIGH at Sync in sweep mode
# 0 -- which jumps this firmware to its FSK frequency -- changes nothing.
#
# REST STATE OF THE DOWN DRIVER'S LINE IS HIGH. It is written HIGH at every
# sample of the sub-sequence except the move window, so there is no edge at
# the buffer boundary, and park_outputs() holds it HIGH between passes so the
# lattice stays stationary while the buffer is re-armed. At the very start of
# the run it is raised as soon as the driver is armed, so the down driver
# ramps SWPs -> base once, before any atoms exist. Both lines go LOW only
# after sweep mode has been switched off at the end of the run.
#
# COMMANDS: this firmware only acts on the S-prefix form of the sweep commands
# (Sswpm:2, Sswps:..., Sswpp:..., Sswpf:..., Sswpt:..., and Sffreq/Sfampl for
# FSK), not the manual's SWPm:/SWPs:/... -- found on the bench.
#
#     ramp time   T_r = (df/2 / SWPf) * SWPt          (per driver)
#     gate width  T_g = T_r + T_p
#     distance    d   = v * T_g,   v = (lambda/2) * df
#
# One step rate is used both ways, so accel and decel are always the same
# length: T_a and T_d are read but both ramps run for their MEAN.
#
# Full sequence:
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> pre-move hold
#   -> ACCELERATE   (both ramps,   T_r)
#   -> DRIFT        (both hold,    T_p)
#   -> DECELERATE   (both ramp back, T_r)
#   -> HOLD AT FAR POINT (both at base, sweep mode still armed, Sync idle)
#   -> release -> TOF -> image at the FAR point
#
# 'Both AOMs Sweep' = 0 falls back to the single-driver move of the
# bothBeams script: the up driver carries the whole df, the other driver is
# parked at base with sweep and segment mode OFF and its line held LOW.
#
# DIRECTION. The atoms move toward the beam with the LOWER frequency.
#   +1  COM4 up, COM5 down   ->  the "+" direction of the bothBeams script
#   -1  COM5 up, COM4 down   ->  the "-" direction
# Which of those is physically "down" is fixed by which beam comes from
# where; check it once on the camera and leave the default set to it.
#
# A second window, "RF signals", draws the sub-sequence as it is sent: both
# AOM frequencies, the lattice velocity they imply, and the dio6 / dio7 gates
# taken from the actual NI buffers.

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
CAMERA_GAIN_DB = 40.0

# ---- Imaging geometry ----
# Sensor pitch divided by the magnification gives the size of one pixel
# projected back into the cloud.
#
# !! MAGNIFICATION, WINDOW_DIAMETER_MM and APERTURE_DISTANCE_MM are
# !! PLACEHOLDERS for the bottom camera, carried over (as in
# !! fastMOT_twoCameras.py). Measure them for the bottom camera, or its atom
# !! number and widths in um are wrong.
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
        ("MOVE MODE", f"{m.get('mode_desc', '--')}",
         "legs (driver, gate, dir)", f"{m.get('legs_desc', '--')}"),
        ("distance per leg",
         f"{d:.3f} mm (realised {m.get('d_real_mm', nan):.4f})",
         "net shift on camera",
         f"{m.get('net_mm', nan):+.4f} mm = {m.get('d_px', nan):+.1f} px"),
        ("accel = decel ramp T_r", f"{m.get('ramp_ms', nan):.3f} ms",
         "acceleration (= decel)", f"{m.get('a_mps2', nan):.2f} m/s&sup2;"),
        ("plateau at Fstop T_p", f"{m.get('plat_real_ms', nan):.3f} ms",
         "peak velocity", f"{m.get('v_mmps', nan):.2f} mm/s"),
        ("SWEEP START  SWPs", f"{m.get('f_start_hz', nan) / 1e6:.6f} MHz",
         "SWEEP STOP  SWPp", f"{m.get('f_stop_hz', nan) / 1e6:.6f} MHz"),
        ("SWEEP DURATION (per leg)", f"{m.get('total_ms', nan):.3f} ms",
         "detuning at Fstop",
         f"+{m.get('df_khz', nan):.3f} kHz ({m.get('shape', '--')})"),
        ("Sync gate HIGH (per leg)", f"{m.get('gate_ms', nan):.3f} ms",
         "SWPf x SWPt",
         f"{m.get('fstep_hz', 0)} Hz x {m.get('tstep_ns', 0)} ns "
         f"({m.get('n_steps', 0)} steps/ramp)"),
        ("TOTAL TRANSPORT", f"{m.get('transport_ms', nan):.3f} ms",
         "far-point hold", f"{m.get('far_hold_ms', 0):.2f} ms"
                           + ("" if m.get('n_legs', 0) > 1 else " (one way)")),
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
    HEADER_PX = 340          # mode, legs and transport rows on top of Sweep2's

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


def rf_signal_trace(port_freqs, start_s, ramp_ms, gate_ms, gates, gate_line,
                    spm, v_per_hz_mmps):
    """One sub-sequence of the RF drive, as plottable piecewise-linear lines.

    :param port_freqs: (dict) port -> (rest_hz, moved_hz). A driver that does
        not sweep has moved_hz == rest_hz and draws a flat line.
    :param start_s: (int) sample at which BOTH drivers start ramping
    :param ramp_ms, gate_ms: (float) per-driver ramp and the NI gate width
    :param gates: (dict) NI line -> list of 0/1 samples, EXACTLY as sent
    :param gate_line: (dict) port -> NI line, e.g. {'COM4': 'dio6', ...}
    :param spm: (int) NI samples per ms
    :param v_per_hz_mmps: (float) lattice velocity per Hz of beat
    :return: (dict) t_ms, f4_hz, f5_hz, v_mps (all on t_ms), and
        gate_<port> -> (t_ms, 0/1) step points for each line
    """
    any_line = next(iter(gates.values()))
    end_ms = len(any_line) / spm

    def corners(port):
        rest, moved = (float(v) for v in port_freqs[port])
        pts = [(0.0, rest)]
        if start_s is not None and moved != rest:
            t0 = start_s / spm
            pts += [(t0, rest), (t0 + ramp_ms, moved), (t0 + gate_ms, moved),
                    (t0 + gate_ms + ramp_ms, rest)]
        pts.append((end_ms, rest))
        return np.array(pts, dtype=float)

    c4, c5 = corners('COM4'), corners('COM5')
    t = np.unique(np.concatenate([c4[:, 0], c5[:, 0]]))
    f4 = np.interp(t, c4[:, 0], c4[:, 1])
    f5 = np.interp(t, c5[:, 0], c5[:, 1])
    # v = (lambda/2) * (f_COM4 - f_COM5), in m/s so pyqtgraph picks the prefix.
    v_mps = v_per_hz_mmps * (f4 - f5) * 1e-3
    out = {'t_ms': t, 'f4_hz': f4, 'f5_hz': f5, 'v_mps': v_mps}
    for port, line in gate_line.items():
        g = np.asarray(gates[line], dtype=np.int8)
        edges = np.flatnonzero(np.diff(g)) + 1
        gt, gv = [0.0], [float(g[0]) if g.size else 0.0]
        for e in edges:
            gt += [e / spm, e / spm]
            gv += [gv[-1], float(g[e])]
        gt.append(end_ms)
        gv.append(gv[-1])
        out[f'gate_{port}'] = (np.array(gt), np.array(gv))
    return out


class RFSignalView:
    """Second window: the RF drive over one sub-sequence.

    Both AOM frequencies, the lattice velocity they imply, and the two sweep
    gates (dio6 -> COM4, dio7 -> COM5) as they are in the NI buffers. Same
    threading rule as MoveLiveView.
    """

    POLL_MS = 200
    COLOURS = {'COM4': '#4DD0E1', 'COM5': '#FFB74D'}

    def __init__(self, title="RF signals (COM4 + COM5)"):
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
        self.win.resize(900, 860)
        self.win.show()

        self.f_plot = canvas.addPlot(row=0, col=0)
        self.f_plot.setLabel('left', 'RF frequency', units='Hz')
        self.f_plot.showGrid(x=True, y=True, alpha=0.3)
        self.f_plot.addLegend(offset=(10, 10))
        self.f4_curve = self.f_plot.plot(
            [], [], pen=pg.mkPen(self.COLOURS['COM4'], width=2), name='COM4')
        self.f5_curve = self.f_plot.plot(
            [], [], pen=pg.mkPen(self.COLOURS['COM5'], width=2,
                                 style=QtCore.Qt.DashLine), name='COM5')

        self.v_plot = canvas.addPlot(row=1, col=0)
        self.v_plot.setLabel('left', 'lattice velocity', units='m/s')
        self.v_plot.showGrid(x=True, y=True, alpha=0.3)
        self.v_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('#616161', width=1)))
        self.v_curve = self.v_plot.plot([], [], pen=pg.mkPen('#81C784', width=2))

        self.g_plot = canvas.addPlot(row=2, col=0)
        self.g_plot.setMaximumHeight(150)
        self.g_plot.setLabel('left', 'Sync gates')
        self.g_plot.setLabel('bottom', 'time in sub-sequence', units='s')
        self.g_plot.setYRange(-0.1, 1.2)
        self.g_plot.showGrid(x=True, y=False, alpha=0.3)
        self.g_plot.addLegend(offset=(10, 5))
        self.g_curves = {
            'COM4': self.g_plot.plot([], [], pen=pg.mkPen(self.COLOURS['COM4'], width=2),
                                     name='dio6 -> COM4'),
            'COM5': self.g_plot.plot([], [], pen=pg.mkPen(self.COLOURS['COM5'], width=2),
                                     name='dio7 -> COM5'),
        }
        self.G_OFFSET = {'COM4': 0.0, 'COM5': -0.04}

        self.v_plot.setXLink(self.g_plot)
        self.f_plot.setXLink(self.g_plot)
        self._markers = []

        self._pending = None
        self._drawn = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(self.POLL_MS)

    def set_trace(self, trace, markers, html, zoom_ms):
        """Called from the acquisition thread; stores data, touches no Qt."""
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
            self.f5_curve.setData(trace['t_ms'] * s, trace['f5_hz'])
            self.v_curve.setData(trace['t_ms'] * s, trace['v_mps'])
            for port, curve in self.g_curves.items():
                gt, gv = trace[f'gate_{port}']
                curve.setData(gt * s, gv + self.G_OFFSET[port])
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
    'mot_loading_time': {'MOT Loading Time (ms)': '2000'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '6'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0'},
    'Repump_VCO': {'Repump VCO (V)': '0'},

    # ===================== PHASE 2: CMOT =====================
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
    'pgc_rampdown_ms': {'PGC Rampdown Duration (ms)': '0'},

    # ===================== LATTICE AM =====================
    # AOM thermal stability: lattice TTL stays ON & AM stays at lat_am_v
    # for the ENTIRE cycle.
    'lat_am_v': {'Lattice AM (V)': '0.6'},

    # ===================== MOVE DIRECTION AND SWEEP MODE =====================
    # +1: COM4 sweeps UP, COM5 sweeps DOWN -- the "+" direction of the
    #     bothBeams script (COM4 above COM5).
    # -1: COM5 sweeps UP, COM4 sweeps DOWN -- the "-" direction.
    # The atoms move toward the beam with the LOWER frequency. Which sign is
    # physically "down" depends on which beam comes from where: check it once
    # on the camera and set the default to it.
    'move_direction': {'Move Direction (+1: COM4 up/COM5 down, -1: COM5 up/COM4 down)': '-1'},
    # 1 = both drivers sweep, each carrying HALF the beat: the down driver
    #     uses the falling-edge ramp of sweep mode 2 (see header). Its Sync
    #     line idles HIGH.
    # 0 = single driver: the up driver carries the whole beat, the other is
    #     parked at base with sweep and segment mode OFF, its line LOW.
    'dual_sweep_enabled': {'Both AOMs Sweep? (1 = yes, 0 = up driver only)': '1'},

    # ===================== MOVE (one leg, one way) =====================
    #     v = d / (T_a/2 + T_p + T_d/2)
    # The sweep has ONE step rate for both ramps, so accel and decel both run
    # for (T_a + T_d)/2. With T_a == T_d (the defaults) nothing changes.
    # 0 distance is a stationary reference: no gate opens, sweep mode stays
    # OFF on both drivers, and the cloud is imaged where it was loaded.
    'dist_mm': {'Move Distance (mm, one way)': '256.0'},
    'accel_ms': {'Accel Time T_a (ms)': '1.0'},
    'plat_ms': {'Drift (constant velocity) Time T_p (ms)': '80.0'},
    'decel_ms': {'Decel Time T_d (ms)': '1.0'},
    # How far AHEAD of the move both gates switch, to absorb the driver's
    # Sync-input latency. The manual documents none for the sweep, so 0
    # until measured. Fractional ms allowed; snapped to the 50 us NI grid.
    'seg_lead_ms': {'Sweep Gate Lead (ms) [Sync latency]': '0'},

    # ===================== HOLDS (before + after move) =====================
    'pre_hold_ms': {'Pre-move Hold (ms)': '5'},
    # After the decel both drivers sit at base with their Sync lines idle:
    # the lattice is stationary at the far point for this long, then release.
    'post_hold_ms': {'Hold at Far Point After Move (ms)': '5'},

    # ===================== DIM-3000 AOM DRIVER =====================
    'dim_base_freq_mhz': {'DIM3000 Base Freq (MHz)': '99.800'},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)': '32.0'},

    # ===================== LIT HOLD (watch this in SpinView) =====================
    # After the move the lattice is stationary at the far point and the
    # cooling and repump light come ON and STAY on, so the cloud fluoresces
    # continuously and SpinView shows it live. No camera trigger is produced
    # by this script at all.
    #
    # ===================== FIXED TOF after release =====================
    # Release is the end of the hold at the far point. 0 images the cloud
    # still in the lattice, which is what lattice_to_Bottom.py does.
    'tof_us': {'TOF (us)': '0'},

    # ===================== IMAGING =====================
    # Exactly the inputs lattice_to_Bottom.py uses, with the same defaults.
    # Camera Exposure drives BOTH the camera's ExposureTime and the length of
    # the cooling/repump pulse, the latter quantised UP to the 50 us NI grid
    # -- keep it a multiple of 50 us and the two stay locked.
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before Dark MOT)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '200'},

    # ===================== DISPLAY =====================
    'display_autoscale': {'Autoscale Display? (1=yes, 0=manual)': '1'},
    'display_min': {'Display Min / Background (counts)': '0'},
    'display_max': {'Display Max (counts) [used when autoscale=0]': '2000'},

    # Warm-up cycles at the head of the buffer: their camera triggers fire
    # like any other cycle and their frames are consumed and thrown away.
    'warmup_cycles': {'Warm-up Cycles (0 = none)': '3'},

    # ===================== BIAS COILS =====================
    'bias_x_v': {'Bias X (V)': '0.0'},
    'bias_y_v': {'Bias Y (V)': '0.0'},
    'bias_z_v': {'Bias Z (V)': '0.0'},

    # ===================== RUN =====================
    # Cycles the OPX clocks per NI arm. The cards are re-armed between passes,
    # which leaves a gap of a second or so with the lattice at base.

    # ===================== GENERAL =====================
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},
}


# ---------------------------------------------------------------------------
# Safe state applied when the run stops.
# ---------------------------------------------------------------------------
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

# THE LATTICE IS DELIBERATELY LEFT ON so the AOM crystals stay in one thermal
# state. dio6/dio7 are the two sweep gates; their idle value is LOW, but the
# DOWN driver's line is overridden to HIGH for as long as that driver is in
# sweep mode 2 (its rest state is Sync HIGH at SWPp = base). See park_outputs.
SAFE_DO_LATTICE_ON = {
    'dio8': 0,      # bottom camera trigger idle
    'dio1': 0,      # MOT AOM enable -- light off
    'dio2': 1,      # coils TTL: the sequence's "coils off" state
    'dio3': 0,      # repump AOM enable -- light off
    'dio4': 0,      # lattice up   -- LOW = ON, left running on purpose
    'dio5': 0,      # lattice down -- LOW = ON, left running on purpose
    'dio6': 0,      # COM4 sweep gate
    'dio7': 0,      # COM5 sweep gate
}


def park_do(NI_do, safe_do, logger):
    """Drive every DO line to the given value. Best effort, never raises."""
    try:
        NI_do.write_static(do=safe_do)
        return
    except Exception as exc:
        logger.warn(f"write_static unavailable ({exc}); parking DO channel by channel")
    for channel, value in safe_do.items():
        try:
            NI_do.set_do_voltage(do_channel=channel, value=bool(value))
        except Exception as e:
            logger.warn(f"Could not park {channel} at {value}: {e}")


# Bottom-chamber camera: datataker client key and the NI card 2 line wired to
# its Line0. Same as camera B in fastMOT_twoCameras.py.
CAMERA_CLIENT = 'fluorescence_imaging_camera_bfs_u3_51s5m_bottom_chamber'
CAMERA_TRIGGER_DO = 'dio8'
PIXEL_FORMAT = 'Mono16'


def get_camera_client(kwargs):
    client = kwargs.get(CAMERA_CLIENT)
    if client is None:
        raise RuntimeError(
            f"Bottom camera client '{CAMERA_CLIENT}' is not connected to this "
            f"datataker. Add its server to the datataker config.")
    return client


def park_outputs(NI_do, NI_ao, logger, lattice_am=0.0, lattice_on=True,
                 do_overrides=None):
    """Drive every line to its safe value, lattice left ON and stationary.

    :param lattice_am: AM voltage the run used, held on ao6/ao7
    :param lattice_on: False parks the lattice off (ao6/ao7 = 0, TTL high)
    :param do_overrides: (dict) DO lines whose parked value differs from
        SAFE_DO_LATTICE_ON -- the down driver's gate, held HIGH while that
        driver is in sweep mode 2 so it sits at base rather than at SWPs.
    """
    safe_ao = dict(SAFE_AO_BASE)
    safe_do = dict(SAFE_DO_LATTICE_ON)
    if do_overrides:
        safe_do.update(do_overrides)
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
    except Exception as exc:
        logger.warn(f"write_static unavailable ({exc}); parking AO channel by channel")
        for channel, value in safe_ao.items():
            try:
                NI_ao.set_ao_voltage(ao_channel=channel, voltages=value)
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value} V: {e}")
    park_do(NI_do, safe_do, logger)
    logger.info(
        "Outputs parked: VCOs 0 V, coils 0 V, bias 0 V, AOM drive 0.8 V, "
        "cooling/repump light off, lattice %s, gates dio6=%d dio7=%d."
        % ((f"ON at {float(lattice_am):.2f} V (stationary)" if lattice_on
            else "OFF"), safe_do['dio6'], safe_do['dio7'])
    )


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")
    dataset.camera_client = get_camera_client(kwargs)
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
    _dirn = int(float(dataset.get_input_parameter('move_direction')))
    dataset.live = MoveLiveView(
        title=f"Lattice to bottom, TWO AOMs, BOTTOM camera "
              f"(dir {'+1' if _dirn > 0 else '-1'}) -- d={dist_mm:.3f} mm",
        levels=(disp_min, disp_max))
    dataset.img_win = dataset.live.win
    dataset.img_item = dataset.live.img_item
    # Second window: the RF drive, filled in by experiment() once the
    # sub-sequence is built (the timeline is only known there).
    dataset.rf_view = RFSignalView()
    dataset.graph.hide()


# FSK frequency / amplitude of each driver as the run found them, keyed by
# port; filled by _run_experiment, restored by experiment()'s finally.
FSK_FOUND = {}


def experiment(**kwargs):
    """Run the sequence, then put the RF drivers back the way they were found.

    Order on the way out matters:
      1. both gate lines LOW  -- a driver in mode 2 with Sync HIGH is sitting
         at SWPp; dropping the line ramps it back to SWPs. And a driver put
         into mode 0 while its Sync is HIGH jumps to its FSK frequency, which
         is about to be restored to whatever it was before the run.
      2. Sswpm:0 on both, FSK put back, then dim_state restores the rest.
    """
    snaps = []
    FSK_FOUND.clear()
    try:
        _run_experiment(dim_snapshots=snaps, **kwargs)
    finally:
        try:
            park_do(kwargs['nidaqmx_ni_daq_2'], dict(SAFE_DO_LATTICE_ON),
                    kwargs['dataset'].log)
        except Exception:
            pass
        time.sleep(0.3)      # longest ramp back to SWPs is a few ms; be generous
        for snap in snaps:
            if not snap:
                continue
            ser = snap['_ser']
            try:
                if not ser.is_open:
                    ser.open()
                ser.write(b'Sswpm:0\n')
                time.sleep(0.12)
                found = FSK_FOUND.get(snap['_port'], {})
                if 'Rffreq' in found:
                    ser.write(f"Sffreq:{found['Rffreq']}\n".encode())
                    time.sleep(0.12)
                if 'Rfampl' in found:
                    ser.write(f"Sfampl:{found['Rfampl']}\n".encode())
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
    OPX_client = kwargs["OPX_OPX"]
    camera_client = dataset.camera_client       # bottom chamber, dio8 -> Line0

    # =========================================================================
    # READ GUI PARAMETERS
    # =========================================================================
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    CMOT_RAMP_MS = int(dataset.get_input_parameter("mot_switch_time"))
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_v = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

    dark_enabled = int(dataset.get_input_parameter("dark_mot_enabled")) != 0
    dark_ms = int(dataset.get_input_parameter("dark_mot_duration"))
    dark_power = float(dataset.get_input_parameter("dark_mot_power"))
    dark_detuning = float(dataset.get_input_parameter("dark_mot_detuning"))
    dark_rep_power = float(dataset.get_input_parameter("dark_mot_repump_power"))
    dark_rep_vco = float(dataset.get_input_parameter("dark_mot_repump_vco"))

    pgc_cool_ms = int(dataset.get_input_parameter("pgc_cool_ms"))
    pgc_rampdown_ms = int(dataset.get_input_parameter("pgc_rampdown_ms"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))
    lat_am_v = float(dataset.get_input_parameter("lat_am_v"))

    dist_mm_in = float(dataset.get_input_parameter("dist_mm"))
    pre_hold_ms = int(dataset.get_input_parameter("pre_hold_ms"))
    post_hold_ms = int(dataset.get_input_parameter("post_hold_ms"))
    accel_ms = float(dataset.get_input_parameter("accel_ms"))
    plat_ms_in = float(dataset.get_input_parameter("plat_ms"))
    decel_ms = float(dataset.get_input_parameter("decel_ms"))
    seg_lead_ms = float(dataset.get_input_parameter("seg_lead_ms"))
    if min(accel_ms, plat_ms_in, decel_ms) < 0:
        raise ValueError(
            f"Move durations must be >= 0 (got T_a={accel_ms}, "
            f"T_p={plat_ms_in}, T_d={decel_ms} ms).")
    if seg_lead_ms < 0:
        raise ValueError(f"Sweep Gate Lead must be >= 0, got {seg_lead_ms} ms.")
    if dist_mm_in < 0:
        raise ValueError(
            "Move Distance must be >= 0; the direction is set by Move "
            "Direction, not by the sign of the distance.")
    dim_base_freq_mhz = float(dataset.get_input_parameter("dim_base_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))
    move_direction = int(float(dataset.get_input_parameter("move_direction")))
    if move_direction not in (1, -1):
        raise ValueError(f"Move Direction must be +1 or -1, got {move_direction}.")
    dual_sweep = int(float(dataset.get_input_parameter("dual_sweep_enabled"))) != 0

    tof_us = int(float(dataset.get_input_parameter("tof_us")))
    warmup_cycles = max(0, int(float(dataset.get_input_parameter("warmup_cycles"))))
    # Display range, read here rather than in the view: the worker picks the
    # colour range and does the fitting, the view only draws what it is handed.
    disp_lo = max(float(dataset.get_input_parameter("display_min")),
                  -float(SENSOR_MAX_MONO16))
    disp_hi = min(float(dataset.get_input_parameter("display_max")),
                  float(SENSOR_MAX_MONO16))
    if disp_hi <= disp_lo:
        disp_hi = disp_lo + 1.0
    disp_auto = bool(int(float(dataset.get_input_parameter("display_autoscale"))))

    # Which driver goes UP and which goes DOWN. v = (lambda/2)*(f4 - f5):
    # +1 puts COM4 above COM5.
    UP_PORT = 'COM4' if move_direction > 0 else 'COM5'
    DN_PORT = 'COM5' if move_direction > 0 else 'COM4'
    DIM_PORTS = ('COM4', 'COM5')
    GATE_LINE = {'COM4': 'dio6', 'COM5': 'dio7'}
    # WHICH WAY THE CLOUD ACTUALLY GOES, in two words, for the panel header.
    # Confirmed on the bottom camera: sweeping COM5 up (move_direction = -1)
    # carries the cloud DOWN to the bottom chamber.
    #
    # THE ONE PLACE TO CHANGE IF THE LAB SAYS OTHERWISE. A moving lattice
    # travels along the propagation direction of the HIGHER-frequency beam, so
    # which word is right depends on which beam points where. Nothing else
    # reads these two strings; the driver-level detail is in legs_desc.
    MOVE_WORD = {1: 'one way up', -1: 'one way down'}
    move_word = MOVE_WORD[move_direction]
    mode_desc = (f"one way, {'+' if move_direction > 0 else '-'}: "
                 f"{UP_PORT} up" + (f", {DN_PORT} down" if dual_sweep else ""))
    sweep_desc = ("both AOMs (down driver on the falling-edge ramp)"
                  if dual_sweep else f"{UP_PORT} only, {DN_PORT} parked at base")

    # ============ Move physics ============
    LATTICE_WAVELENGTH = 795e-9
    v_per_Hz_mmps = LATTICE_WAVELENGTH / 2 * 1e3  # mm/s per Hz of BEAT
    base_freq_hz = int(round(dim_base_freq_mhz * 1e6))

    ni_rate = 20000
    SPM = ni_rate // 1000
    SAMPLE_MS = 1000.0 / ni_rate

    def ceil_samples(t_ms):
        """ms -> whole NI samples, rounding UP (1e-9 absorbs float fuzz)."""
        return int(np.ceil(t_ms * SPM - 1e-9))

    # DIM-3000 sweep limits (manual p.4 and p.13). SWPt is in ns, 4..262000,
    # rounded DOWN by the firmware to a multiple of 4.
    SWEEP_STEP_NS_QUANTUM = 4
    SWEEP_STEP_NS_MAX = 262000
    SWEEP_STEP_NS_MIN = 1000
    SWEEP_FSTEP_SEARCH = 20000

    def sweep_steps(span_hz, ramp_ms):
        """SWPf and SWPt for a ramp covering span_hz in (at most) ramp_ms.

        Candidates are tried from the FINEST step upward, and the first one
        within 2e-4 of the span and 2e-3 of the ramp time is taken. SWPt is
        always rounded DOWN, so the ramp is never longer than asked for and
        therefore always fits inside the gate.

        :return: (dict) fstep_hz, tstep_ns, n_steps, span_hz (as played),
            ramp_ms (as played), clipped
        """
        q = SWEEP_STEP_NS_QUANTUM
        target_ns = ramp_ms * 1e6
        if target_ns < SWEEP_STEP_NS_MIN:
            t_ns = max(q, int(target_ns // q) * q)
            return {'fstep_hz': span_hz, 'tstep_ns': t_ns, 'n_steps': 1,
                    'span_hz': span_hz, 'ramp_ms': t_ns * 1e-6,
                    'clipped': False}
        n_min = max(1, int(np.ceil(target_ns / SWEEP_STEP_NS_MAX)))
        n_max = int(target_ns // SWEEP_STEP_NS_MIN)
        if span_hz < n_min:
            t_ns = (SWEEP_STEP_NS_MAX // q) * q
            return {'fstep_hz': 1, 'tstep_ns': t_ns, 'n_steps': span_hz,
                    'span_hz': span_hz, 'ramp_ms': span_hz * t_ns * 1e-6,
                    'clipped': True}
        best = None
        f_lo = max(1, span_hz // max(n_max, 1))
        for f in range(f_lo, f_lo + SWEEP_FSTEP_SEARCH):
            n = int(round(span_hz / f))
            if n < n_min:
                break
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
        """Everything both sweeps need for one one-way move of d_mm.

        v = d / (T_a/2 + T_p + T_d/2) as before. The gate is T_r + T_p rounded
        UP to the NI grid; d = v * gate, so the BEAT is recomputed from the
        gate as played. With both drivers sweeping each carries half the beat
        with IDENTICAL SWPf/SWPt, so the two ramps are the same length by
        construction and the beat ramps at twice the per-driver rate.

        :return: (dict) per-driver windows f_up_start/stop, f_dn_start/stop,
            the shared fstep/tstep, gate_s for the NI card, and every derived
            quantity the panel and log print.
        """
        stationary = {'d_mm': float(d_mm), 'd_real_mm': 0.0, 'v_mmps': 0.0,
                      'ramp_req_ms': 0.0, 'ramp_ms': 0.0,
                      'plat_ms': 0.0, 'plat_real_ms': 0.0,
                      'gate_s': 0, 'gate_ms': 0.0, 'total_ms': 0.0,
                      'a_mps2': 0.0, 'df_hz': 0, 'df_drv_hz': 0,
                      'f_up_start': base_freq_hz, 'f_up_stop': base_freq_hz,
                      'f_dn_start': base_freq_hz, 'f_dn_stop': base_freq_hz,
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
        beat_target = int(round(d_mm / (gate_ms * 1e-3) / v_per_Hz_mmps))
        if beat_target < (2 if dual_sweep else 1):
            raise ValueError(
                f"{d_mm} mm in {gate_ms:.3f} ms is under the driver's 1 Hz "
                f"resolution (per driver).")
        if dual_sweep:
            # Half each, EXACTLY the same on both drivers: the sweep engine
            # is programmed once and copied, so any 1 Hz remainder is
            # dropped rather than given to one driver only.
            st = sweep_steps(beat_target // 2, ramp_req)
            df_drv = st['span_hz']
            beat = 2 * df_drv
        else:
            st = sweep_steps(beat_target, ramp_req)
            df_drv = st['span_hz']
            beat = df_drv
        v_mmps = beat * v_per_Hz_mmps
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
                'total_ms': gate_ms + ramp_ms,      # edge -> back at rest
                'a_mps2': (v_mmps * 1e-3) / (ramp_ms * 1e-3),
                'df_hz': beat,
                'df_drv_hz': df_drv,
                # UP driver: rest at SWPs = base, moved at SWPp = base + df_drv.
                'f_up_start': base_freq_hz,
                'f_up_stop': base_freq_hz + df_drv,
                # DOWN driver: rest at SWPp = base (Sync HIGH), moved at
                # SWPs = base - df_drv (Sync LOW). Equal to base when single.
                'f_dn_start': base_freq_hz - (df_drv if dual_sweep else 0),
                'f_dn_stop': base_freq_hz,
                'fstep_hz': st['fstep_hz'],
                'tstep_ns': st['tstep_ns'],
                'n_steps': st['n_steps'],
                'clipped': st['clipped']}

    sc0 = schedule_for_distance(dist_mm_in)
    do_move = sc0['d_mm'] > 0
    for f in (sc0['f_up_start'], sc0['f_up_stop'], sc0['f_dn_start'], sc0['f_dn_stop']):
        if not (10_000_000 <= f <= 400_000_000):
            raise ValueError(f"Sweep frequency {f} Hz is outside the DIM-3000's "
                             f"10-400 MHz range.")

    if do_move:
        logger.info(
            f"Move ({mode_desc}), {sweep_desc}: d={sc0['d_mm']:.3f}mm, "
            f"typed T_a={accel_ms:.3f} / T_p={plat_ms_in:.3f} / "
            f"T_d={decel_ms:.3f} ms -> played: ramp {sc0['ramp_ms']:.4f} ms "
            f"each way, drift {sc0['plat_real_ms']:.4f} ms, gate "
            f"{sc0['gate_ms']:.3f} ms, edge-to-rest {sc0['total_ms']:.4f} ms; "
            f"v={sc0['v_mmps']:.3f}mm/s, a={sc0['a_mps2']:.3f}m/s^2, "
            f"BEAT {sc0['df_hz']/1e3:.3f} kHz = {sc0['df_drv_hz']/1e3:.3f} kHz "
            f"per driver; {UP_PORT}: SWPs {sc0['f_up_start']} -> SWPp "
            f"{sc0['f_up_stop']} Hz"
            + (f"; {DN_PORT}: SWPs {sc0['f_dn_start']} -> SWPp "
               f"{sc0['f_dn_stop']} Hz (idles HIGH at SWPp = base)"
               if dual_sweep else f"; {DN_PORT} parked at base")
            + f"; SWPf={sc0['fstep_hz']} Hz, SWPt={sc0['tstep_ns']} ns, "
            f"{sc0['n_steps']} steps; realised distance {sc0['d_real_mm']:.4f}mm")
        if abs(accel_ms - decel_ms) > 1e-9:
            logger.warn(
                f"T_a ({accel_ms} ms) != T_d ({decel_ms} ms). Sweep mode 2 has "
                f"one step rate for both ramps, so BOTH run for their mean, "
                f"{sc0['ramp_req_ms']:.3f} ms. Distance, peak velocity and "
                f"move time are as typed; the accelerations are not.")
        if accel_ms + decel_ms <= 0:
            logger.warn(
                f"Accel and decel times are 0: the lattice jumps to "
                f"{sc0['v_mmps']:.1f} mm/s in a single step and stops the same "
                f"way. That is an unbounded acceleration and will shake most "
                f"atoms out.")
        if sc0['clipped']:
            logger.warn(
                f"The ramp is only {sc0['ramp_ms']:.3f} ms, not the "
                f"{sc0['ramp_req_ms']:.3f} ms asked for: {sc0['df_drv_hz']} Hz "
                f"per driver in 1 Hz steps at the longest step time (262 us) "
                f"cannot take longer. The drift is longer to match, so the "
                f"distance is unaffected.")
    else:
        logger.info("Move distance 0: STATIONARY reference -- no gate opens, "
                    "sweep mode OFF on both drivers, cloud imaged where it "
                    "was loaded.")

    # =========================================================================
    # CONFIGURE DIM-3000 DRIVERS (called once per experiment start)
    # =========================================================================
    swept_ports = set()
    if do_move:
        swept_ports.add(UP_PORT)
        if dual_sweep:
            swept_ports.add(DN_PORT)
    # The down driver's rest state is Sync HIGH, for as long as it is armed.
    gate_rest = {GATE_LINE[DN_PORT]: 1} if (do_move and dual_sweep) else {}

    DIM_MAX_SEGMENTS = 7          # only to neutralise other scripts' tables
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

    def _dim_neutralise_segments(ser):
        for i in range(1, DIM_MAX_SEGMENTS + 1):
            flag = 8 if i == DIM_MAX_SEGMENTS else 1
            _dim_send(ser, f'Wseg:{i};{flag};{base_freq_hz};{base_freq_hz};0;1;0')

    def _dim_fsk_safe(ser, port):
        """FSK frequency = base and FSK amplitude = AMP, for the run.

        In sweep mode 0 a HIGH at Sync jumps this firmware to its FSK
        frequency and amplitude (seen on the bench). With FSK = base such a
        jump goes nowhere. The values found are put back by experiment().
        """
        found = _dim_gpar(ser)
        FSK_FOUND[port] = {k: found[k] for k in ('Rffreq', 'Rfampl') if k in found}
        _dim_send(ser, f'Sffreq:{base_freq_hz}')
        _dim_send(ser, f'Sfampl:{amplitude_x10}')

    def _dim_make_deaf(ser, port):
        """Park a driver at base in plain single-frequency operation."""
        _dim_send(ser, 'Sswpm:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_send(ser, f'AMP:{amplitude_x10}')
        _dim_send(ser, 'OUT_on')
        _dim_neutralise_segments(ser)
        _dim_send(ser, 'Mseg:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_fsk_safe(ser, port)
        logger.info(
            f"DIM-3000 {port}: {base_freq_hz/1e6:.6f} MHz, "
            f"+{amplitude_x10/10:.1f} dBm, ON, sweep OFF (Sswpm:0), segment "
            f"mode OFF (Mseg:0), FSK = base -- not swept this run.")

    def _dim_program_sweep(ser, port, f_start, f_stop):
        """Load one sweep window into this driver and arm mode 2.

        Bench-tested sequence (S-prefix forms):
            Mseg:0  Sswpm:0  FRQ  AMP  OUT_on  Sswps  Sswpp  Sswpf  Sswpt  Sswpm:2
        SWPf/SWPt are the same on both drivers, so the two ramps are the
        same length whichever way each one goes.

        :return: (list of str) mismatches found on Gpar readback; empty = OK
        """
        _dim_neutralise_segments(ser)
        _dim_fsk_safe(ser, port)
        _dim_send(ser, 'Mseg:0')
        _dim_send(ser, 'Sswpm:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_send(ser, f'AMP:{amplitude_x10}')
        _dim_send(ser, 'OUT_on')
        _dim_send(ser, f"Sswps:{f_start}")
        _dim_send(ser, f"Sswpp:{f_stop}")
        _dim_send(ser, f"Sswpf:{sc0['fstep_hz']}")
        _dim_send(ser, f"Sswpt:{sc0['tstep_ns']}")
        _dim_send(ser, 'Sswpm:2')
        time.sleep(0.1)
        got = _dim_gpar(ser)
        logger.info(
            f"DIM-3000 {port} after arming: FRQ={got.get('Rfreq')} "
            f"SWPm={got.get('Rswpm')} SWPs={got.get('Rswps')} "
            f"SWPp={got.get('Rswpp')} SWPf={got.get('Rswpf')} "
            f"SWPt={got.get('Rswpt')} FFRQ={got.get('Rffreq')} "
            f"FAMP={got.get('Rfampl')}")
        want = {'Rswpm': 2, 'Rswps': f_start, 'Rswpp': f_stop,
                'Rswpf': sc0['fstep_hz'], 'Rswpt': sc0['tstep_ns'],
                'Rffreq': base_freq_hz, 'Rfampl': amplitude_x10}
        return [f"{k} is {got.get(k, '<missing>')} (sent {v})"
                for k, v in want.items() if got.get(k) != v]

    # Both gate lines LOW before any driver is touched: a driver that is
    # still in mode 0 from a previous run must not see a HIGH (FSK jump), and
    # the down driver must enter mode 2 sitting at SWPs before its line is
    # raised, so that its one ramp up to base happens now, with no atoms.
    park_do(NI_card_2, dict(SAFE_DO_LATTICE_ON), logger)

    sers = {port: _dim_open(port) for port in DIM_PORTS}
    for port in DIM_PORTS:
        kwargs['dim_snapshots'].append(dim_state.snapshot(sers[port], port, logger))
    if all(sers[port] is not None for port in DIM_PORTS):
        for port in DIM_PORTS:
            logger.info(f"DIM-3000 {port}: {_dim_query(sers[port], '*IDN?')}")
        for port in DIM_PORTS:
            ser = sers[port]
            if port not in swept_ports:
                _dim_make_deaf(ser, port)
                continue
            if port == UP_PORT:
                f_start, f_stop = sc0['f_up_start'], sc0['f_up_stop']
                rest = f"LOW at SWPs = base; HIGH ramps to SWPp"
            else:
                f_start, f_stop = sc0['f_dn_start'], sc0['f_dn_stop']
                rest = f"HIGH at SWPp = base; LOW ramps down to SWPs"
            bad = _dim_program_sweep(ser, port, f_start, f_stop)
            logger.info(
                f"DIM-3000 {port}: mode 2 armed on {GATE_LINE[port]} -- SWPs "
                f"{f_start} Hz, SWPp {f_stop} Hz, SWPf {sc0['fstep_hz']} Hz, "
                f"SWPt {sc0['tstep_ns']} ns. Sync {rest}.")
            if bad:
                logger.error(
                    f"DIM-3000 {port} did not take the sweep settings: "
                    f"{'; '.join(bad)}. The move will not be the one on the "
                    f"panel. Do not trust this run's images.")
            else:
                logger.info(f"DIM-3000 {port}: sweep settings verified by "
                            f"Gpar readback.")
        if gate_rest:
            # Raise the down driver's line NOW: it ramps SWPs -> base once,
            # over ramp_ms, long before the first atoms are loaded, and then
            # sits at base until the first move window pulls the line LOW.
            park_do(NI_card_2, {**SAFE_DO_LATTICE_ON, **gate_rest}, logger)
            time.sleep(0.05 + sc0['ramp_ms'] * 1e-3)
            logger.info(
                f"{GATE_LINE[DN_PORT]} raised: {DN_PORT} has ramped up to base "
                f"and idles there with Sync HIGH. It goes LOW only for the move "
                f"window, and is held HIGH by park_outputs() between passes.")
    else:
        logger.info("DIM-3000: ports unavailable, will run without serial control")
        for port in DIM_PORTS:
            if sers[port] is not None:
                sers[port].close()
                sers[port] = None

    # Imaging (the lit hold)
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(float(dataset.get_input_parameter("camera_exposure_us")))
    atoms_clear_ms = int(float(dataset.get_input_parameter("atoms_clear_time")))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    bias_x_v = float(dataset.get_input_parameter("bias_x_v"))
    bias_y_v = float(dataset.get_input_parameter("bias_y_v"))
    bias_z_v = float(dataset.get_input_parameter("bias_z_v"))

    # =========================================================================
    # CHANNELS
    # =========================================================================
    camera_trigger_do = CAMERA_TRIGGER_DO      # 'dio8' -> bottom camera Line0
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
    SPU = ni_rate / 1e6
    delay_ns = int(round(1e9 / ni_rate)) - 500
    ttl_up = max(1, SPM // 20)
    def ms2s(t): return int(round(t * SPM))
    def us2s(t): return int(round(t * SPU))

    mot_s = ms2s(mot_loading_ms)
    cmot_ramp_s = ms2s(CMOT_RAMP_MS)
    cmot_hold_s = ms2s(cmot_hold_ms)
    dark_s = ms2s(dark_ms) if dark_enabled else 0
    pgc_cool_s = ms2s(pgc_cool_ms)
    pgc_rampdown_s = ms2s(pgc_rampdown_ms)
    pgc_s = pgc_cool_s + pgc_rampdown_s
    pre_hold_s = ms2s(pre_hold_ms)
    post_hold_s = ms2s(post_hold_ms)
    seg_lead_s = ms2s(seg_lead_ms)
    tof_s_fixed = us2s(tof_us)
    clear_s = ms2s(atoms_clear_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)

    # =========================================================================
    # TIMELINE ANCHORS
    # =========================================================================
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    dark_start = cmot_end
    dark_end = dark_start + dark_s
    pgc_start = dark_end
    pgc_cool_end = pgc_start + pgc_cool_s
    pgc_end = pgc_cool_end + pgc_rampdown_s

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
        f"pre_hold={pre_hold_ms}ms, move {mode_desc}: {sc0['total_ms']:.3f}ms "
        f"(gate {sc0['gate_ms']:.3f}ms + decel {sc0['ramp_ms']:.3f}ms), "
        f"hold at far point={post_hold_ms}ms (DARK), release, tof={tof_us}us, "
        f"then frame 1 and frame 2 {atoms_clear_ms}ms apart"
    )
    if seg_lead_ms >= pre_hold_ms:
        logger.warn(
            f"Gate Lead ({seg_lead_ms}ms) is not shorter than the pre-move "
            f"hold ({pre_hold_ms}ms), so the gates switch during PGC "
            f"ramp-down instead of in the hold. Raise Pre-move Hold above "
            f"{seg_lead_ms}ms.")

    def leg_layout():
        """The one move window in samples -- ONE definition, used by both the
        waveform and the RF signals window.

        :return: (start, gate_s, move_end). start is the sample at which both
            drivers begin ramping (the gates switch seg_lead_s earlier);
            move_end is when both are back at base. start is None for a
            0 mm reference.
        """
        t = pgc_end + pre_hold_s
        if not do_move:
            return None, 0, t
        return t, sc0['gate_s'], t + sc0['gate_s'] + ceil_samples(sc0['ramp_ms'])

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence():
        start, gate_s, move_end = leg_layout()
        post_hold_end = move_end + post_hold_s
        # Release is the end of the hold at the far point; then TOF, then the
        # two frames. Identical to lattice_to_Bottom.py from here on.
        release = post_hold_end
        f1 = release + tof_s_fixed
        f2 = f1 + clear_s
        crt = f2 + aom_pulse_s + coils_restore_s
        N = max(crt + ttl_up + 1, f2 + ttl_up + 1)

        # dio8 -> the bottom camera's Line0. One sample (50 us) per frame: the
        # trigger is configured FrameStart/RisingEdge, so only the edge counts
        # and the camera's own ExposureTime sets the integration.
        cam = [0] * N
        for s in range(ttl_up):
            cam[f1 + s] = 1
            cam[f2 + s] = 1

        ado = [0] * N
        aao = [mot_aom_power] * N
        rdo = [0] * N
        rao = [repump_aom_power] * N

        for s in range(cmot_start):
            ado[s] = 1
            aao[s] = mot_aom_power
            rdo[s] = 1
            rao[s] = repump_aom_power
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            ado[t] = 1
            aao[t] = mot_aom_power + frac * (cmot_mot_power - mot_aom_power)
            rdo[t] = 1
            rao[t] = repump_aom_power + frac * (cmot_repump_power - repump_aom_power)
        for s in range(cmot_hold_s):
            t = cmot_hold_start + s
            ado[t] = 1
            aao[t] = cmot_mot_power
            rdo[t] = 1
            rao[t] = cmot_repump_power
        if dark_enabled:
            for s in range(dark_s):
                t = dark_start + s
                ado[t] = 1
                aao[t] = dark_power
                rdo[t] = 1
                rao[t] = dark_rep_power
        for s in range(pgc_cool_s):
            t = pgc_start + s
            ado[t] = 1
            aao[t] = pgc_mot_power
            rdo[t] = 1
            rao[t] = pgc_rep_power
        for s in range(pgc_rampdown_s):
            t = pgc_cool_end + s
            frac = s / max(pgc_rampdown_s - 1, 1)
            ado[t] = 1
            aao[t] = pgc_mot_power * (1.0 - frac)
            rdo[t] = 1
            rao[t] = pgc_rep_power * (1.0 - frac)
        # Lattice hold, the whole move and the hold after it: TTLs off (no
        # cooling light), analog stays at full for AOM thermal stability.
        for s in range(pgc_end, release):
            ado[s] = 0
            rdo[s] = 0
        # TOF: same (digital off, analog stays at full).
        for s in range(release, min(f1, N)):
            ado[s] = 0
            rdo[s] = 0

        # Imaging pulses: full MOT power, one per frame, identical by
        # construction so the difference is the cloud and nothing else.
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

        cdo = [0] * N
        for s in range(coils_off, min(crt, N)):
            cdo[s] = 1

        cao = [mot_coils_v] * N
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            cao[t] = mot_coils_v + frac * (cmot_coils_v - mot_coils_v)
        for s in range(cmot_hold_start, N):
            cao[s] = cmot_coils_v

        mvco = [vco_loading] * N
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            mvco[t] = vco_loading + frac * (cmot_detuning - vco_loading)
        for s in range(cmot_hold_start, cmot_end):
            mvco[s] = cmot_detuning
        if dark_enabled:
            for s in range(dark_s):
                mvco[dark_start + s] = dark_detuning
        for s in range(pgc_s):
            mvco[pgc_start + s] = pgc_detuning
        for s in range(pgc_end, N):
            mvco[s] = pgc_detuning
        # Imaging VCO override: on for the whole lit window, switched
        # vco_lead_us early so the light is already at the imaging detuning.
        # Imaging VCO override: on from vco_lead_us before frame 1 right
        # through to the end of frame 2, so both flashes are at the same
        # detuning by construction.
        vlo = max(0, f1 - vco_lead_s)
        vhi = min(N, f2 + aom_pulse_s)
        for s in range(vlo, vhi):
            mvco[s] = vco_imaging

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

        biasx = [bias_x_v] * N
        biasy = [bias_y_v] * N
        biasz = [bias_z_v] * N

        ludo = [0] * N
        lddo = [0] * N
        luao = [lat_am_v] * N
        ldao = [lat_am_v] * N

        # ---- DIM-3000 sweep gates ----
        # UP driver's line:   LOW everywhere, HIGH for [t0, t0 + gate).
        # DOWN driver's line: HIGH everywhere, LOW for the SAME window (dual)
        #                     or LOW everywhere (single: driver is deaf).
        # The lead moves both edges of both lines equally, so the gate width,
        # which sets the distance, is untouched by it. The down line is HIGH
        # at sample 0 and at sample N-1, so the buffer boundary has no edge.
        gates = {GATE_LINE[port]: [0] * N for port in DIM_PORTS}
        if do_move and dual_sweep:
            gates[GATE_LINE[DN_PORT]] = [1] * N
        if start is not None:
            t0 = max(0, start - seg_lead_s)
            up_line = gates[GATE_LINE[UP_PORT]]
            dn_line = gates[GATE_LINE[DN_PORT]]
            for s in range(gate_s):
                if t0 + s < N:
                    up_line[t0 + s] = 1
                    if dual_sweep:
                        dn_line[t0 + s] = 0

        return (cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, biasx, biasy,
                biasz, ludo, lddo, luao, ldao, gates, N)

    # Every cycle of this script is identical -- one distance, one move -- so
    # the buffer is just (warm-ups + 1) copies of the same sub-sequence. The
    # warm-ups matter because they fire their camera triggers like any other
    # cycle: their frames are consumed and discarded in the acquisition loop,
    # which is what keeps the kept pair aligned with the cycle that produced
    # it. They also let the MOT come back to its cyclic steady state after the
    # re-arm before the frame that is kept.
    (cam_1, ado_1, aao_1, rdo_1, rao_1, cdo_1, cao_1, mvco_1, rvco_1,
     biasx_1, biasy_1, biasz_1, ludo_1, lddo_1, luao_1, ldao_1,
     gates_1, N) = build_sub_sequence()
    _reps = warmup_cycles + 1
    cam_p = cam_1 * _reps
    ado_p, aao_p = ado_1 * _reps, aao_1 * _reps
    rdo_p, rao_p = rdo_1 * _reps, rao_1 * _reps
    cdo_p, cao_p = cdo_1 * _reps, cao_1 * _reps
    mvco_p, rvco_p = mvco_1 * _reps, rvco_1 * _reps
    biasx_p, biasy_p, biasz_p = biasx_1 * _reps, biasy_1 * _reps, biasz_1 * _reps
    ludo_p, lddo_p = ludo_1 * _reps, lddo_1 * _reps
    luao_p, ldao_p = luao_1 * _reps, ldao_1 * _reps
    gates_p = {line: samples * _reps for line, samples in gates_1.items()}
    # The RF window draws ONE cycle, not the whole buffer.
    gates_w = gates_1

    total = N * _reps
    start0, gate_s0, move_end0 = leg_layout()
    net_sign = move_direction if do_move else 0
    release0 = move_end0 + post_hold_s
    f1_0 = release0 + tof_s_fixed
    f2_0 = f1_0 + clear_s
    logger.info(
        f"Lattice MOVE ONE WAY ({mode_desc}), {sweep_desc}, BOTTOM camera: "
        f"d={dist_mm_in}mm, T_a/T_p/T_d = {accel_ms:.2f}/{plat_ms_in:.2f}/"
        f"{decel_ms:.2f}ms, v={sc0['v_mmps']:.2f}mm/s, imaged at the FAR point; "
        f"release {release0 / SPM:.1f}ms, frame 1 {f1_0 / SPM:.1f}ms, frame 2 "
        f"{f2_0 / SPM:.1f}ms ({atoms_clear_ms}ms apart), tof={tof_us}us, "
        f"exposure={camera_exposure_us}us (AOM pulse {aom_pulse_s / SPM * 1000:.0f}us), "
        f"VCO {vco_imaging:.2f}V, {warmup_cycles} warm-up cycle(s), "
        f"cycle {N / SPM:.1f}ms, {total} samples")

    move_start0 = pgc_end + pre_hold_s

    # Everything the panel header says about the MOVE, worked out once, here,
    # where the schedule and the sub-sequence length are both in scope. The
    # view never recomputes any of it: a second copy of this arithmetic is a
    # second place for it to disagree with the hardware.
    #
    # Two fields mean something different here than in lattice_to_Bottom.py,
    # because this script sweeps BOTH drivers. SWPs/SWPp are the UP driver's
    # window -- the DOWN driver mirrors it below base -- and the detuning row
    # carries the BEAT, which is what actually moves the lattice and is twice
    # the per-driver excursion when both are sweeping. "legs" names both
    # drivers with their gate lines and which way each one goes.
    move_summary = {
        # Two words only. The full description -- which driver sweeps which
        # way, on which gate line, dual or single -- is one column to the
        # right in legs_desc, and spelling it out here as well pushed that
        # column off the panel.
        'mode_desc': move_word,
        'legs_desc': (f"{UP_PORT} up ({GATE_LINE[UP_PORT]}), "
                      f"{DN_PORT} down ({GATE_LINE[DN_PORT]}, idles HIGH)"
                      if dual_sweep else
                      f"{UP_PORT} up ({GATE_LINE[UP_PORT]}), {DN_PORT} parked"),
        'd_mm': sc0['d_mm'],
        'd_real_mm': sc0['d_real_mm'],
        'net_mm': net_sign * sc0['d_real_mm'],
        'd_px': net_sign * sc0['d_real_mm'] * 1e3 / PIXEL_UM,
        'v_mmps': sc0['v_mmps'],
        'a_mps2': sc0['a_mps2'],
        'ramp_ms': sc0['ramp_ms'],
        'plat_real_ms': sc0['plat_real_ms'],
        'gate_ms': sc0['gate_ms'],
        'total_ms': sc0['total_ms'],
        'transport_ms': (move_end0 - move_start0) / SPM,
        # One leg, one way. far-point hold has no meaning here: the cloud is
        # imaged AT the far point rather than brought back.
        'n_legs': 1,
        'far_hold_ms': 0.0,
        # The UP driver's window. The DOWN driver runs base - df_drv -> base.
        'f_start_hz': sc0['f_up_start'],
        'f_stop_hz': sc0['f_up_stop'],
        'fstep_hz': sc0['fstep_hz'],
        'tstep_ns': sc0['tstep_ns'],
        'n_steps': sc0['n_steps'],
        # The BEAT, not one driver's excursion.
        'df_khz': sc0['df_hz'] / 1e3,
        'shape': ('stationary' if sc0['d_mm'] <= 0 else
                  'triangle' if plat_ms_in <= 0 else 'trapezoid'),
        'pre_hold_ms': pre_hold_ms,
        'post_hold_ms': post_hold_ms,
        'tof_us': tof_us,
        'cycle_ms': N / SPM,
    }

    # RF signals window. The gates are the real dio6 / dio7 buffers, so the
    # picture cannot drift from what is played.
    try:
        port_freqs = {
            UP_PORT: (sc0['f_up_start'], sc0['f_up_stop']),
            DN_PORT: (sc0['f_dn_stop'], sc0['f_dn_start']),   # rest, moved
        }
        rf_trace = rf_signal_trace(port_freqs, start0, sc0['ramp_ms'],
                                   sc0['gate_ms'], gates_w, GATE_LINE, SPM,
                                   v_per_Hz_mmps)
        rf_markers = [(pgc_end / SPM, 'PGC end')]
        if start0 is not None:
            rf_markers.append((start0 / SPM, 'move'))
        rf_markers += [(move_end0 / SPM, 'at rest (far point)'),
                       (release0 / SPM, 'release'),
                       (f1_0 / SPM, 'frame 1'),
                       (f2_0 / SPM, 'frame 2')]
        rf_html = (
            f"<span style='font-family:monospace'>"
            f"<b>{mode_desc}</b> | {sweep_desc}<br>"
            f"{UP_PORT}: {sc0['f_up_start']/1e6:.6f} -> {sc0['f_up_stop']/1e6:.6f} MHz | "
            + (f"{DN_PORT}: {sc0['f_dn_stop']/1e6:.6f} &larr; {sc0['f_dn_start']/1e6:.6f} MHz"
               if dual_sweep else f"{DN_PORT}: parked at base") +
            f" | {sc0['fstep_hz']} Hz x {sc0['tstep_ns']} ns<br>"
            f"ramp {sc0['ramp_ms']:.3f} + drift {sc0['plat_real_ms']:.3f} + ramp "
            f"{sc0['ramp_ms']:.3f} ms | gate {sc0['gate_ms']:.3f} ms | beat "
            f"{sc0['df_hz']/1e3:.3f} kHz | peak v {sc0['v_mmps']:.2f} mm/s | "
            f"a {sc0['a_mps2']:.2f} m/s^2<br>"
            f"<b>frame 1 at {f1_0 / SPM:.1f} ms, frame 2 at {f2_0 / SPM:.1f} ms</b> "
            f"&mdash; {camera_exposure_us} us each, {atoms_clear_ms} ms apart, after a "
            f"{post_hold_ms:g} ms dark hold at the far point and {tof_us} us TOF "
            f"(VCO {vco_imaging:.2f} V) | cycle {N / SPM:.1f} ms | dio8 &rarr; "
            f"bottom camera</span>")
        margin = 5.0
        zoom = ((move_start0 - seg_lead_s) / SPM - pre_hold_ms - margin,
                (f1_0 + clear_s // 8) / SPM + margin)
        dataset.rf_view.set_trace(rf_trace, rf_markers, rf_html, zoom)
    except Exception as exc:
        logger.warn(f"Could not draw the RF signals window: {exc!r}")

    # =========================================================================
    # MAIN LOOP: arm, clock M buffers, keep one frame pair per buffer, re-arm
    # =========================================================================
    # Paced by the CAMERA, not by a timer: every pass blocks on frames, so the
    # panel always shows the shot that just happened. reps_per_arm is not used
    # here -- the buffer already carries its warm-ups, and M below is how many
    # times it plays per arm, exactly as in lattice_to_Bottom.py.
    cycles_done = 0

    while thread.running:
        # Re-asserted every pass: another script, or SpinView, may have left
        # the camera free-running or on a different pixel format.
        camera_client.set_hardware_trigger(
            line="Line0", activation="RisingEdge",
            selector="FrameStart", overlap="ReadOut",
            acquisition_mode="Continuous")
        if not camera_client.try_set_enum("PixelFormat", PIXEL_FORMAT):
            logger.warn(f"Bottom camera: could not set PixelFormat to "
                        f"{PIXEL_FORMAT}; it keeps whatever it had.")
        camera_client.set_exposure(camera_exposure_us)
        # Request the gain, then READ IT BACK: try_set_float is best effort and
        # a value past the node maximum is swallowed, leaving the camera on its
        # previous gain. N scales as 10^(-G/20), so assuming the requested
        # value when the camera refused it is a large photometric error.
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
        camera_client.start_acquisition()

        try:
            NI_card_1.finalize_clock()
        except Exception:
            pass

        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(do_channel=camera_trigger_do, value=cam_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=MOT_AOM_do, value=ado_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=mot_coils_do, value=cdo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=Repump_AOM_do, value=rdo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio4", value=ludo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio5", value=lddo_p, sample_rate=ni_rate)
        # Sweep gates: dio6 -> COM4 Sync, dio7 -> COM5 Sync. No camera line.
        for line, samples in gates_p.items():
            NI_card_2.set_do_voltage(do_channel=line, value=samples, sample_rate=ni_rate)
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

        h1 = NI_card_2.arm(regeneration=True)
        h2 = NI_card_3.arm(regeneration=True)
        NI_card_1.arm_clock(length=total, sample_rate=ni_rate)

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
            b, shape, dtype = camera_client.get_frame_bytes(timeout_ms)
            return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        # Stop is only a flag on the worker thread, so it takes effect where
        # the script LOOKS at it. A single get_frame() blocking for 100 s made
        # the red Stop button appear dead for up to that long. Poll in short
        # slices and check the flag between them.
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
            stopping = False
            for run_idx in range(M):
                if not thread.running:
                    stopping = True
                    break
                logger.info(f"Run {run_idx+1} of {M}")
                # The warm-up cycles at the head of the buffer fire their
                # camera triggers like any other cycle, so their frames MUST
                # be consumed here. Skip this and every pair afterwards is
                # offset by one, and the panel shows a warm-up cycle cloud --
                # taken while the MOT was still coming back from the re-arm --
                # as though it were data, with no error to say so.
                for _ in range(2 * warmup_cycles):
                    if grab(100000) is None:
                        stopping = True
                        break
                if stopping:
                    break
                is_first = (run_idx == 0)
                frame1 = grab(10000 if is_first else 100000)
                if frame1 is None:
                    stopping = True
                    break
                frame2 = grab(100000)
                if frame2 is None:
                    stopping = True
                    break
                # No DIM reprogramming between cycles: every cycle is the same
                # move, so the sweep armed at startup is already right.
                diff = frame1.astype(np.int32) - frame2.astype(np.int32)
                # THIS PAIR AND NOTHING ELSE. Nothing is accumulated between
                # cycles, so the panel always shows the shot that just
                # happened and answers immediately when a parameter is turned.
                img = diff.astype(np.float64)
                # Fits run on the SUMMED profiles, not on single rows:
                # collapsing the rows lifts the cloud linearly while the noise
                # grows only as sqrt(rows), which is what makes a fit possible
                # on a cloud too faint to see.
                x_proj = img.sum(axis=0)
                y_proj = img.sum(axis=1)
                fx = fit_gaussian_1d(x_proj)
                fy = fit_gaussian_1d(y_proj)
                good = bool(fx and fy and fx['reliable'] and fy['reliable'])
                sx = fx['sigma'] if fx else float('nan')
                sy = fy['sigma'] if fy else float('nan')
                # frame1.dtype, NOT img.dtype: DN/photon is a property of the
                # raw ADC word, and img is float64, which would select the
                # 12-bit calibration and put N out by 16x. applied_gain is
                # what the camera reported back, not what was requested.
                met = (cloud_metrics(img, fx, fy, frame1.dtype,
                                     camera_exposure_us,
                                     gain_db=applied_gain)
                       if (fx and fy) else None)
                # Display Min always pins the bottom, so the background sits
                # at the same colour in every shot and only the top moves.
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
                    f"  d={sc0['d_mm']:.2f}mm (v={sc0['v_mmps']:.1f}mm/s): "
                    f"sx={sx:.1f}px, sy={sy:.1f}px"
                    + (f", N(seen)={met['n_atoms']:.3e}, "
                       f"N(total)={met['n_atoms_total']:.3e} "
                       f"({met['capture_fraction'] * 100:.1f}% seen), "
                       f"n0={met['density']:.2e}/cm3"
                       if met is not None else ", no cloud region")
                    + ("" if good else "  [UNRELIABLE FIT]"))
                cycles_done += 1
        finally:
            logger.info("Stopping acquisition")
            try:
                camera_client.stop_acquisition()
            except Exception:
                pass

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
        # Sequence tasks are closed now, so static writes can land. The gate
        # idle states matter here: the DOWN driver rests with its Sync HIGH.
        park_outputs(NI_card_2, NI_card_3, logger, lattice_am=lat_am_v,
                     do_overrides=gate_rest)
        logger.info(f"Arm done: {cycles_done} cycles so far.")
        time.sleep(wait_time)

    # After the while loop ends (thread stopped). Order:
    #   1. both gate lines LOW  (down driver ramps SWPp -> SWPs, a few ms)
    #   2. Sswpm:0 + FRQ base on both, so nothing at Sync moves them again
    #   3. close the ports; experiment() then restores the snapshots.
    # 1 before 2 because a driver put into mode 0 with Sync HIGH jumps to its
    # FSK frequency, and FSK is about to be restored to whatever it was.
    park_do(NI_card_2, dict(SAFE_DO_LATTICE_ON), logger)
    time.sleep(0.05 + sc0['ramp_ms'] * 1e-3)
    for port in DIM_PORTS:
        ser = sers[port]
        if ser is None:
            continue
        try:
            _dim_send(ser, 'Sswpm:0')
            _dim_send(ser, f'FRQ:{base_freq_hz}')
            logger.info(f"DIM {port} sweep OFF (Sswpm:0) and parked at base "
                        f"{base_freq_hz/1e6:.3f} MHz (lattice on, stationary)")
        except Exception as e:
            logger.warn(f"Could not park DIM {port} at base frequency: {e!r}")
        try:
            ser.close()
        except Exception:
            pass
