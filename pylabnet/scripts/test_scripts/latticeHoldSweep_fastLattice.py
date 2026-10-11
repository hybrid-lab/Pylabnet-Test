# ---------------------------------------------------------------------------
# Duplicate of latticeHoldSweep.py with every INIT_DICT default that fastLattice.py also
# has set to fastLattice.py's value, so this sweep runs on top of the same
# MOT/CMOT/dark-MOT/PGC/lattice/imaging conditions. The swept axis and the
# analysis code are unchanged from latticeHoldSweep.py.
# CMOT_RAMP_MS is likewise 15 ms here (fastLattice 'MOT Switch Time'),
# not the 10 ms of latticeHoldSweep.py.
#
# The atom number, peak density and their calibration come from
# temp_PGC_aziza.py (which in turn took them from motAtomNumber.py): the
# per-panel Gaussian fit is the same least-squares fit with an explicit
# offset, the brightness is integrated inside the fitted 2w ellipse against a
# background taken from outside it, and the same DN/photon, collection-solid-
# angle and scattering-rate constants turn that into N. So the number printed
# over each hold time here is directly comparable to the one that script
# prints over each TOF.
#
# On top of that, N vs hold time is fitted to a single exponential in its own
# window -- that fit IS the lattice lifetime, and it replaces the log-slope of
# the raw whole-frame area that was only written to the log before.
#
# Both DIM-3000 drivers are forced onto one base frequency at the start of
# every run (reset_dim_to_base). Without that the script inherited whatever
# the last one left programmed, and a move script that was stopped mid-move
# leaves the up beam parked kHz away from the down beam -- a standing wave
# that walks at (lambda/2)*Df for the whole hold, displacing the cloud
# linearly in hold time. That is a moving lattice masquerading as physics.
#
# The image display and its scaling come from jingfastPGC.py: Display Min /
# Display Max / Autoscale on the GUI instead of the hardcoded (0, 20) levels,
# the range clamped to what a pixel can physically reach (sensor_full_scale),
# a colour bar in real camera counts beside each row of tiles, and the
# per-frame diagnostics that say WHY a frame looks black.
# ---------------------------------------------------------------------------
# Lattice HOLD-TIME sweep.
# Full sequence:
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> lattice HOLD (swept, 6 points) -> release -> short TOF -> image
#
# Lattice runs continuously at full AM/TTL for AOM thermal stability.
# Atoms transfer from molasses to lattice during PGC rampdown.
# Then we hold them in the lattice for a variable time, release, and image.
# Loss vs hold-time => lattice lifetime measurement.
#
# Sequence per scan point:
#   1. MOT loading (full power, coils on)
#   2a. CMOT ramp (15ms hardcoded, = fastLattice MOT Switch Time)
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

import numpy as np
import dim_state
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
import time
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

try:
    from scipy.optimize import curve_fit
except ImportError:      # fits fall back to their moment / log-linear seeds
    curve_fit = None

try:
    import serial
except ImportError:      # no pyserial -> DIM reset is skipped, loudly
    serial = None

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

# A 1/e^2 radius is 2 sigma, and integrating a 1/e^2-parameterised Gaussian
# gives w*sqrt(pi/2) -- used by cloud_metrics to turn the two 1D fits back
# into a 2D peak.
ROOT_HALF_PI = np.sqrt(np.pi / 2.0)


# ---------------------------------------------------------------------------
# Atom number and density  -- ported unchanged from temp_PGC_aziza.py
# ---------------------------------------------------------------------------
# Same apparatus, same camera and the same 50 dB of gain, so the calibration
# constants carry over as they are. The exposure is the one difference: it is
# read from the GUI here, not fixed in the module.
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

# The imaging pulse is the same full-power MOT light in both scripts, so the
# scattering rate above describes it here too. Note that this is the IMAGING
# light only: the lattice is 795 nm, far off the D2 cycling transition, and
# contributes nothing to the fluorescence being counted.

# FLIR gain is 10^(G/20), so 50 dB is a factor of 316. Getting it wrong is a
# two-and-a-half order of magnitude error in N, which is why the run reads the
# value back off the camera instead of trusting the request.
CAMERA_GAIN_DB = 50.0

# ---- Imaging geometry ----
# Sensor pitch divided by the magnification gives the size of one pixel
# projected back into the cloud.
PIXEL_PITCH_UM = 3.45
MAGNIFICATION = 0.59
PIXEL_UM = PIXEL_PITCH_UM / MAGNIFICATION      # ~5.847 um per pixel at the atoms

# The line-of-sight axis is not imaged, so its extent has to be assumed. Taking
# it equal to the axis along the 2448-pixel direction (x) is the usual choice
# for a cloud that is round in the plane perpendicular to the camera.
DEPTH_FOLLOWS_X = True

# Fraction of the 1/e^2 radius the brightness is integrated over. At 2w an
# elliptical aperture already encloses 1 - exp(-8) = 99.97% of a Gaussian, so
# this is the total in all but name, while still being a measurement of real
# pixels rather than an extrapolation of the fit.
INTEGRATION_RADII = 2.0

# ---- Fixed-box atom counting ----
# Half-width of the atom-number box, in units of the FIRST scan point's fitted
# sigma. The box is latched once from that point and then frozen for the run,
# so every hold time is summed over exactly the same pixels.
#
# WHY A FIXED BOX. The elliptical aperture above follows the per-point fit, so
# a cloud that breathes with hold time is integrated over a region that
# breathes with it, and capture_fraction then extrapolates whatever the fit
# believes fell off the sensor. Both are model-dependent, and both act hardest
# at the long hold times where there is least signal -- which is exactly where
# tau is decided. A fixed box is a plain sum of the same pixels every time: no
# per-point fit enters the number, so a fit that has gone wrong can no longer
# move it.
#
# The cost is deliberate. Atoms that leave the box are counted as lost. At
# +/-3 sigma the box holds erf(3/sqrt(2))^2 = 99.46% of a Gaussian, so for a
# cloud that keeps its size this is the whole cloud; for one that genuinely
# expands it is not, and tau then measures loss FROM THE BOX. That is the
# honest thing to quote when expansion cannot be separated from loss, but it
# is not the same quantity the ellipse used to report.
BOX_HALF_SIGMA = 3.0

# Reject a scan point from the lifetime fit when its fitted sigma exceeds this
# multiple of the FIRST point's sigma on EITHER axis. 2.0 means "grown by more
# than 100%". A fit that has found noise instead of atoms comes back wide, and
# at the longest hold time -- weakest fit, largest leverage on tau -- that is
# the single point most able to bend the answer. Referenced to the first point
# rather than an absolute number so it travels with the cloud size of the day.
WIDTH_REJECT_FACTOR = 2.0


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
    # annulus can fall entirely OFF it. Assuming offset = 0 there would leave
    # the pedestal in the sum and inflate N by whatever it integrates to, so
    # step the margin in instead, and fall back to the whole-frame median if
    # even the 2w boundary leaves too little. Which estimator was used is
    # returned, so a caller can say when the number is not to be trusted.
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
        # dtype=float64 is load-bearing, not decoration. An int32 accumulator
        # (np.dtype(int) is int32 on Windows) wraps silently past 2.15e9, and a
        # bright cloud filling a 2w ellipse reaches that -- the sum comes back
        # NEGATIVE, taking the atom number with it. No error, just a sign flip.
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


def reference_box(shape, fx, fy, half_sigma=BOX_HALF_SIGMA):
    """The fixed counting box, built once from one reference fit.

    Half-widths are half_sigma * sigma about the fitted centre on each axis,
    clipped to the sensor. Latched from the first scan point and reused
    unchanged for every hold time -- that reuse is the whole point, because it
    is what makes the atom numbers comparable across the sweep.

    :param shape: (tuple) frame shape the box must live inside
    :param fx, fy: (dict) fit_gaussian_1d results for the reference point
    :param half_sigma: (float) box half-width in sigma
    :return: (dict) slice bounds x0/x1/y0/y1, the centre and half-widths, and
        the reference sigmas the width test compares against; or None if the
        fit does not describe a usable box.
    """
    h, w = int(shape[0]), int(shape[1])
    sx, sy = float(fx['sigma']), float(fy['sigma'])
    if not (np.isfinite(sx) and np.isfinite(sy)) or sx <= 0 or sy <= 0:
        return None
    hx, hy = half_sigma * sx, half_sigma * sy
    x0 = int(max(0, np.floor(fx['center'] - hx)))
    x1 = int(min(w, np.ceil(fx['center'] + hx) + 1))
    y0 = int(max(0, np.floor(fy['center'] - hy)))
    y1 = int(min(h, np.ceil(fy['center'] + hy) + 1))
    # A box clipped down to nothing is not a measurement of anything.
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
            'cx': float(fx['center']), 'cy': float(fy['center']),
            'hx': float(hx), 'hy': float(hy),
            'sigma_x': sx, 'sigma_y': sy,
            'half_sigma': float(half_sigma), 'shape': (h, w)}


def box_brightness(diff, box):
    """Background-subtracted counts inside the FIXED box.

    The background is treated exactly as in integrated_brightness -- median of
    the pixels outside a margin, median so stray light and hot pixels cannot
    drag it, and the margin stepped in if it runs off the sensor -- so the two
    estimators differ only in the shape of the aperture and each can be read
    against the other.

    The box does NOT move with the cloud and is not refitted, so nothing in
    this function depends on the current point's fit succeeding.

    :param diff: (ndarray) frame1 - frame2, in camera counts
    :param box: (dict) reference_box() result
    :return: (dict) counts, offset, pixels, ...; or None if unusable here
    """
    h, w = diff.shape
    if box['shape'] != (int(h), int(w)):
        # The box belongs to a different sensor geometry than this frame.
        # Silently counting the wrong pixels is the one outcome worth ruling
        # out, so refuse instead.
        return None
    y0, y1, x0, x1 = box['y0'], box['y1'], box['x0'], box['x1']
    n_px = int((y1 - y0) * (x1 - x0))
    if n_px < 4:
        return None

    # The box never moves, so neither does the background region. Build the
    # mask once and cache it on the box: this function is now called once per
    # CYCLE as well as once per point (the per-cycle call is what supplies the
    # shot-to-shot error on N), and rebuilding a five-megapixel boolean array
    # every time would be the most expensive thing in the acquisition loop for
    # no reason at all.
    cached = box.get('_bg')
    if cached is None:
        MIN_BG_PX = 20000
        outside, n_bg, bg_source = None, 0, 'none'
        for margin, tag in ((1.5, '1.5x box margin'),
                            (1.25, '1.25x box margin'),
                            (1.0, 'box boundary')):
            mx, my = margin * box['hx'], margin * box['hy']
            gx0 = int(max(0, np.floor(box['cx'] - mx)))
            gx1 = int(min(w, np.ceil(box['cx'] + mx) + 1))
            gy0 = int(max(0, np.floor(box['cy'] - my)))
            gy1 = int(min(h, np.ceil(box['cy'] + my) + 1))
            cand = np.ones((h, w), dtype=bool)
            cand[gy0:gy1, gx0:gx1] = False
            n_bg = int(cand.sum())
            if n_bg >= MIN_BG_PX:
                outside, bg_source = cand, tag
                break
        else:
            outside = None
            n_bg = int(h * w)
            bg_source = 'whole-frame median (box fills the sensor)'
        cached = (outside, n_bg, bg_source)
        box['_bg'] = cached
    outside, n_bg, bg_source = cached
    offset = float(np.median(diff if outside is None else diff[outside]))

    return {
        # float64 accumulator for the same reason as integrated_brightness:
        # an int32 sum wraps silently past 2.15e9 and comes back NEGATIVE.
        'counts': float(diff[y0:y1, x0:x1].sum(dtype=np.float64)
                        - offset * n_px),
        'offset': offset,
        'pixels': n_px,
        'bg_pixels': n_bg,
        'bg_source': bg_source,
        'fill_fraction': n_px / float(diff.size),
    }


def capture_fraction(shape, fx, fy, inside):
    """Fraction of a perfect 2D Gaussian that actually lands in the measured region.

    The 2w ellipse is 4 sigma across, which encloses 1 - exp(-8) = 99.97% of a
    Gaussian -- IF it fits on the sensor. After the release and TOF here it
    need not, and a clipped ellipse means `inside` is the ellipse intersected
    with the frame.

    Integrating the FITTED Gaussian over exactly that mask, against its
    analytic whole-plane integral 2*pi*sx*sy, gives the fraction collected.
    Dividing the measured atom number by it extrapolates to the whole cloud --
    on the explicit assumption that the cloud really is Gaussian out in the
    wings nobody measured.

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


def cloud_metrics(diff, fx, fy, dtype, exposure_us, gain_db=CAMERA_GAIN_DB,
                  box=None):
    """Brightness, atom number and peak density for one difference frame.

    :param diff: (ndarray) frame1 - frame2, in camera counts
    :param fx, fy: (dict) fit_gaussian_1d results for the two axes
    :param dtype: raw frame dtype, which fixes the DN/photon scale
    :param exposure_us: (float) camera exposure actually used
    :param box: (dict) reference_box() result. When given, the brightness is
        summed over that FIXED region instead of the per-point ellipse, and no
        capture-fraction correction is applied -- see below. The fits are still
        used for the widths, the density and the extrapolated total, all of
        which are diagnostics here rather than inputs to N.
    :return: (dict) or None if the region could not be formed
    """
    if box is not None:
        band = box_brightness(diff, box)
        region = 'fixed box %.1f sigma' % box['half_sigma']
    else:
        band = integrated_brightness(diff, fx, fy)
        region = 'ellipse %.1fw' % INTEGRATION_RADII
    if band is None:
        return None

    # Total brightness straight from the fitted centre and radii, out to
    # infinity: integrating amplitude*exp(-2r^2/w^2) along one axis gives
    # amplitude*w*sqrt(pi/2). Each marginal was already summed over the other
    # axis, so each integral is the WHOLE cloud on its own -- two independent
    # estimates of one number, not two factors to multiply.
    #
    # This is the extrapolated counterpart to band['counts']: same quantity,
    # one from the model and one from the pixels. Their ratio should sit at
    # 1 - exp(-8) = 99.97% for a real Gaussian; well under that means the fit
    # is inventing wings the sensor never saw, and over 100% means the aperture
    # is collecting something the fit does not describe.
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
    # whole-cloud number that implies -- in ELLIPSE mode. There, for a compact
    # cloud this is ~1.000 and n_atoms_total == n_atoms; for an expanded one it
    # is the correction that matters most, and it is what let a cloud that
    # BREATHES with hold time still be compared like for like.
    #
    # In BOX mode that correction is deliberately not applied, so the two modes
    # measure genuinely different quantities: the ellipse estimates how many
    # atoms there were, the box counts how many are still in the same pixels.
    # Do not compare a tau from one against a tau from the other.
    if box is not None:
        # NO capture-fraction correction in box mode. Dividing a fixed-region
        # sum by a fraction derived from this point's Gaussian would put the
        # per-point fit straight back into the number the box exists to keep it
        # out of -- and it is the long-hold points, where the fit is worst,
        # that the correction would inflate hardest. Atoms outside the box are
        # counted as lost, which is the stated meaning of this tau.
        frac = 1.0
        n_total = n_atoms
    else:
        frac = capture_fraction(diff.shape, fx, fy, band['inside'])
        n_total = (n_atoms / frac if (np.isfinite(frac) and frac > 0)
                   else float('nan'))
    return {
        'region': region,
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
    # Standard errors on the four parameters. nan unless a least-squares fit
    # actually ran and returned a usable covariance: the moment estimate is a
    # calculation, not a fit, and has no covariance to quote. Do not silently
    # substitute zero -- "no uncertainty" and "uncertainty unknown" are
    # different statements and only one of them is true here.
    perr = [float('nan')] * 4
    if curve_fit is not None:
        try:
            p, pcov = curve_fit(_gaussian, x, y, p0=p, maxfev=5000)
            var = np.diag(np.asarray(pcov, dtype=np.float64))
            if var.shape == (4,):
                perr = [float(np.sqrt(v))
                        if (np.isfinite(v) and v >= 0) else float('nan')
                        for v in var]
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
    # whether it can be trusted. That matters more here than in a TOF sweep:
    # the whole point of a lifetime measurement is the LAST points, where the
    # cloud is nearly gone and a fit to noise would masquerade as surviving
    # atoms and bend tau upwards.
    noise = float(np.std(y - curve))
    # Significance of the whole peak, not of one sample. The amplitude is
    # constrained by every point across the width, so a per-point ratio badly
    # understates a fit built from thousands of them.
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
        # 1-sigma errors on the fitted parameters, from the covariance matrix.
        # absolute_sigma is left False, so these are scaled by the reduced
        # chi-square -- the profile carries no per-sample error bars, so the
        # residual scatter is the only noise estimate there is.
        'offset_err': perr[0],
        'amplitude_err': perr[1],
        'center_err': perr[2],
        'sigma_err': abs(perr[3]) if np.isfinite(perr[3]) else float('nan'),
        'w_err': (2.0 * abs(perr[3]) if np.isfinite(perr[3])
                  else float('nan')),
        'snr': snr,
        'reliable': reliable,
        'curve': curve,
        'x': x,
    }


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


def _linear_fit_err(x, y):
    """Ordinary least squares with standard errors on both coefficients.

    Separate from _linear_fit because that one is used where only the
    coefficients matter and changing its signature would be a trap.

    The errors come from the residual scatter -- there are no per-point error
    bars on N to propagate, so the fit's own spread about the line is the only
    evidence available about how well determined the slope is. That makes them
    meaningful only when there are more points than parameters: with n == 2 the
    line passes exactly through both and the residuals say nothing, so the
    errors come back as nan rather than as a reassuring zero.

    :return: (a, b, sigma_a, sigma_b) for y = a + b x
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    nan = float('nan')
    if n < 2:
        return nan, nan, nan, nan
    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()
    det = n * sxx - sx * sx
    if abs(det) < 1e-30:
        return nan, nan, nan, nan
    a = (sxx * sy - sx * sxy) / det
    b = (n * sxy - sx * sy) / det
    if n < 3:
        return float(a), float(b), nan, nan
    resid = y - (a + b * x)
    s2 = float((resid ** 2).sum()) / (n - 2)       # residual variance
    return (float(a), float(b),
            float(np.sqrt(s2 * sxx / det)), float(np.sqrt(s2 * n / det)))


def _pm(value, err, fmt='%.1f', pct=False):
    """'12.3 +/- 0.4 (3.3%)' -- one value with its error, for a plot label.

    A nan error prints as "--" rather than being hidden: a fit with too few
    points to support an uncertainty should say so on the picture, not look
    like a fit whose uncertainty happened not to be mentioned.
    """
    if not np.isfinite(value):
        return '--'
    base = fmt % value
    if not np.isfinite(err):
        return base + ' +/- --'
    out = base + ' +/- ' + (fmt % err)
    if pct and value != 0:
        out += ' (%.1f%%)' % (100.0 * abs(err / value))
    return out


def _exp_decay_fit(t_ms, n_vals, n_errs=None):
    """Fit N(t) = N0 exp(-t/tau) + N1 to atom number vs lattice hold time.

    Two stages, because neither alone is enough. Taking logs turns a
    pedestal-free decay into a straight line, which needs no starting guess
    and cannot diverge -- but it also weights the points wrongly: log()
    magnifies the scatter of the small, late N values, which is exactly where
    a lifetime measurement has the least signal, and a log-linear fit alone
    therefore lets the last point drag tau around. So the straight line is
    used as a SEED, and the fit is then redone in linear space, with the
    pedestal free, where the bright early points carry the weight they should.

    THE PEDESTAL. N1 is the floor the decay does not remove: a population that
    survives on a timescale far longer than tau, plus whatever the per-point
    background subtraction leaves inside the counting region. Letting it float
    is what stops a two-component decay from being reported as one long tau.
    It cuts the other way too -- a free pedestal can equally absorb real atoms
    and make tau come out SHORT -- so it is bounded at or above zero and
    reported next to tau rather than hidden. Read them together: an N1 that is
    a large fraction of N0 means this is not a clean single exponential, and
    the tau beside it should not be quoted on its own.

    Without scipy there is no three-parameter stage; the log-linear answer is
    returned with N1 = 0 and `refined` False, which the panel labels say.

    UNCERTAINTIES. Every parameter comes back with a 1-sigma standard error:
    from the covariance matrix when scipy refined the fit, and from the
    residual scatter about the straight line when it did not. There are no
    per-point error bars on N to propagate -- the atom numbers carry no
    measured uncertainty of their own -- so these are scatter-derived, which
    means they describe how well the MODEL is pinned by the points and say
    nothing about a systematic that moves every point the same way. A tau with
    a 2% error can still be 80% wrong if the counting region is wrong.

    They also need more points than parameters to exist at all: 3 points on a
    3-parameter fit is an exact solution with nothing left over, so the errors
    come back nan rather than zero.

    WEIGHTING. When n_errs is supplied -- the shot-to-shot standard error on
    each point's N, measured by the scatter of the individual cycles that were
    averaged into it -- the fit is weighted by it and the parameter errors come
    from absolute_sigma=True, so they are propagated from real measured
    uncertainties instead of inferred from how well the model happened to fit.
    That is the honest version, and it also stops a noisy late point from
    pulling tau as hard as a clean early one.

    It brings a test with it. chi2/dof near 1 means the model describes the
    data to within the measured noise; much larger means it does not, and the
    parameter errors are then too small to believe whatever they say. Both are
    returned so the panel can show them together -- a tau with a tight error
    and a chi2/dof of 30 is not a precise measurement, it is a bad model.

    :param t_ms: (sequence) hold times [ms]
    :param n_vals: (sequence) atom number at each hold time
    :param n_errs: (sequence) 1-sigma error on each N, or None for an
        unweighted fit. Points with a non-finite or non-positive error are
        kept but fall back to unweighted treatment for the whole fit, because
        mixing weighted and unweighted points has no consistent meaning.
    :return: (dict) n0, tau_ms, n1 and their *_err, plus mask, refined,
        weighted, chi2_red and dof; or None if unfittable
    """
    t = np.asarray(t_ms, dtype=np.float64)
    y = np.asarray(n_vals, dtype=np.float64)
    # Only positive N can be logged, and a non-positive N means the fit that
    # produced it was noise anyway.
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return None
    t_m, y_m = t[mask], y[mask]
    # Errors, if every surviving point has a usable one. All or nothing: a fit
    # in which some points are weighted by a measured error and others by an
    # implicit 1.0 is weighted by the accident of which points had enough
    # cycles to estimate a spread, which is not a weighting scheme.
    e_m = None
    if n_errs is not None:
        e = np.asarray(n_errs, dtype=np.float64)
        if e.shape == y.shape:
            e_sel = e[mask]
            if np.all(np.isfinite(e_sel)) and np.all(e_sel > 0):
                e_m = e_sel
    # Stage 1, pedestal-free: both the fallback answer and the N0/tau seed.
    # Weighted in log space too when errors are available: d(log N) = dN/N.
    a, b, sa, sb = _linear_fit_err(t_m, np.log(y_m))
    if not (np.isfinite(a) and np.isfinite(b)) or b >= 0:
        # b >= 0 is N flat or RISING with hold time: no decay to quote. The
        # points are still plotted; only the curve is withheld.
        nan = float('nan')
        return {'n0': nan, 'tau_ms': nan, 'n1': nan,
                'n0_err': nan, 'tau_err': nan, 'n1_err': nan,
                'mask': mask, 'refined': False, 'weighted': False,
                'chi2_red': nan, 'dof': 0}
    n0, tau, n1 = float(np.exp(a)), float(-1.0 / b), 0.0
    # Propagate the log-space errors into the parameters they stand for:
    # N0 = exp(a) so dN0 = N0 da, and tau = -1/b so dtau = db / b^2. This is
    # the answer that survives if scipy is missing, so it is worth having.
    n0_err = float(n0 * sa) if np.isfinite(sa) else float('nan')
    tau_err = float(sb / (b * b)) if np.isfinite(sb) else float('nan')
    n1_err = 0.0            # held, not fitted
    refined = False
    # Three free parameters need more than three points to mean anything. At
    # exactly 3 the fit is exact and tau is whatever the pedestal leaves over,
    # so the pedestal stage is held back until there are 4.
    if curve_fit is not None and mask.sum() >= 4:
        # Seed the floor below the smallest surviving point -- the most the
        # data supports without asserting the tail is all pedestal.
        n1_seed = max(0.0, 0.5 * float(np.min(y_m)))
        try:
            # bounds keeps N0 and N1 physical (no negative atom numbers) and
            # tau off zero. NOTE: with bounds, curve_fit routes through
            # least_squares, which takes max_nfev -- maxfev is a TypeError.
            popt, pcov = curve_fit(
                lambda x, N0, T, N1: N0 * np.exp(-x / T) + N1, t_m, y_m,
                p0=[n0, tau, n1_seed],
                sigma=e_m, absolute_sigma=(e_m is not None),
                bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]),
                max_nfev=20000)
            if np.all(np.isfinite(popt)) and popt[0] > 0 and popt[1] > 0:
                n0, tau, n1 = float(popt[0]), float(popt[1]), float(popt[2])
                # Unweighted, absolute_sigma is False and pcov comes back
                # already scaled by the reduced chi-square, which is what is
                # wanted when the points carry no error bars of their own.
                # Weighted, absolute_sigma is True and the errors are
                # propagated from the measured per-point spread instead -- the
                # two are different statements about where the error came
                # from, and `weighted` below records which one this is.
                #
                # A parameter that ran into its bound, or a fit with no degrees
                # of freedom left, comes back with an infinite or negative
                # variance. sqrt() of that is nan with a RuntimeWarning, so the
                # diagonal is checked before it is rooted: a nan error is
                # displayed as "--", which is honest, whereas a warning in the
                # log and a silent nan is not.
                refined = True
                n0_err = tau_err = n1_err = float('nan')
                try:
                    var = np.diag(np.asarray(pcov, dtype=np.float64))
                    if var.shape == (3,):
                        err = [float(np.sqrt(v))
                               if (np.isfinite(v) and v >= 0) else float('nan')
                               for v in var]
                        n0_err, tau_err, n1_err = err
                except Exception:
                    pass
        except Exception:
            pass        # keep the log-linear seed rather than losing the fit
    # Goodness of fit. Only defined against a real noise scale, so it is
    # quoted only for a weighted fit; unweighted, chi2/dof is 1 by
    # construction and would be a reassurance rather than a test.
    weighted = bool(refined and e_m is not None)
    n_par = 3 if refined else 2
    dof = int(mask.sum()) - n_par
    chi2_red = float('nan')
    if weighted and dof > 0 and np.isfinite(tau) and tau > 0:
        resid = (y_m - (n0 * np.exp(-t_m / tau) + n1)) / e_m
        chi2_red = float((resid ** 2).sum() / dof)
    return {'n0': n0, 'tau_ms': tau, 'n1': n1,
            'n0_err': n0_err, 'tau_err': tau_err, 'n1_err': n1_err,
            'mask': mask, 'refined': refined, 'weighted': weighted,
            'chi2_red': chi2_red, 'dof': dof}


# ---------------------------------------------------------------------------
# DISPLAY / SCALING  -- ported from jingfastPGC.py
# ---------------------------------------------------------------------------
# One name for both the image LUT and the colour bars, so they cannot disagree.
COLORMAP = 'inferno'

# Pixel format the camera is expected to be in. Nothing here sets it -- it is
# only named in the 8-bit error below -- but the camera remembers whatever was
# last set on it, so a run can silently come back 8-bit and throw away four
# bits of an already faint signal.
PIXEL_FORMAT = 'Mono16'

# Measured from this camera's own frames rather than assumed: every Mono16 word
# it produces satisfies v = 16*k + 12 (12-bit ADC left-shifted by four, plus a
# 12 count pedestal). So the ceiling is 16*4095 + 12 = 65532, NOT the 65535 the
# container would suggest, and observed saturation piles up at 65516.
SENSOR_MAX_MONO16 = 16 * 4095 + 12   # 65532
SENSOR_MAX_MONO12 = 4095
SENSOR_MAX_MONO8 = 255

# Percentile the autoscaled top follows. Not the max: one hot pixel would then
# set the scale for the whole frame and push the cloud back into the dark end.
AUTOSCALE_PCT = 99.95

# Autoscale is measured on every 4th row and column. A percentile over all
# 5.0 M pixels is a full sort per frame; 313 k pixels give the same number to
# well under a count. Same value as temp_PGC_aziza and latticeMoveSweep, so
# all three scripts level an identical frame identically.
AUTOSCALE_STRIDE = 4

# Fallback only. configure() and experiment() both read Display Min/Max from
# the GUI; these are what a missing or malformed field degrades to, and they
# match the INIT_DICT defaults. The old (0, 20) was narrower than a single
# photon: the calibration is 1.84 DN/photon at 0 dB and FLIR gain is 10^(G/20),
# so at the 50 dB this script sets, one photon is 1.84 * 316 = 582 counts. A 20
# count ceiling put every pixel that caught even one photon past the top.
DIFF_LEVELS = (-5, 500)
AVG_DIFF_LEVELS = (-5, 500)


def sensor_full_scale(frame):
    """Brightest value one pixel can report, deduced from the frame itself.

    An 8-bit container is unambiguous. A 16-bit one is not: it carries either
    left-shifted Mono16 or raw 12-bit, and both arrive as uint16.

    The peak value cannot tell them apart -- a dim frame comes nowhere near
    either ceiling, so a faint cloud would be misread as 12-bit and the range
    wrongly capped at 4095. The low nibble can: left-shifting by four leaves
    it constant across the whole frame (12 on this camera), whereas raw
    12-bit data spreads over all sixteen residues.

    :param frame: (ndarray) one raw camera frame
    :return: (int) maximum single-pixel brightness in counts
    """
    dt = frame.dtype
    if dt == np.uint8:
        return SENSOR_MAX_MONO8
    if dt != np.uint16:
        return int(np.iinfo(dt).max)

    counts = np.bincount(np.ravel(frame) & 0xF, minlength=16)
    if counts.max() > 0.9 * counts.sum():
        return SENSOR_MAX_MONO16
    return SENSOR_MAX_MONO12


def localized_signal(diff, blocks=32):
    """Find the strongest spatially-clustered excess in a difference frame.

    The integrated area cannot tell a cloud from a uniform baseline shift, and
    across five million pixels that distinction is everything: a pedestal
    difference of 0.04 counts per pixel -- far below one ADU, invisible in every
    percentile -- integrates to 2e5 counts and scores 300 sigma against a noise
    model that assumes pixels are independent. It looks like an enormous signal
    and is just the two frames sitting at slightly different black levels.

    Binning separates them. Atoms concentrate into a few blocks; a pedestal
    lifts every block equally, so it cancels against the median. The comparison
    is against the median absolute deviation over blocks, which a pedestal does
    not inflate either.

    :param diff: (ndarray) frame1 - frame2
    :param blocks: (int) grid is blocks x blocks
    :return: (dict) peak block level, background, excess in sigma, and where
    """
    h, w = diff.shape
    by, bx = h // blocks, w // blocks
    if by < 1 or bx < 1:
        return None
    binned = diff[:by * blocks, :bx * blocks].reshape(
        blocks, by, blocks, bx).mean(axis=(1, 3))
    med = float(np.median(binned))
    mad = float(np.median(np.abs(binned - med))) * 1.4826
    iy, ix = np.unravel_index(int(np.argmax(binned)), binned.shape)
    peak = float(binned[iy, ix])
    return {
        'peak': peak,
        'background': med,
        'excess_sigma': (peak - med) / mad if mad > 0 else float('inf'),
        'x': int(ix * bx + bx // 2),
        'y': int(iy * by + by // 2),
        'block_px': by * bx,
    }


def autoscale_top(image, lo):
    """Top of the colour range for one image, from its own brightness.

    Percentile rather than max, so one hot pixel cannot set the scale for the
    whole panel and push the cloud back into the dark end. Subsampled for the
    same reason temp_PGC_aziza subsamples: this runs on every frame pair.
    """
    sample = np.asarray(image)[::AUTOSCALE_STRIDE, ::AUTOSCALE_STRIDE]
    hi = float(np.percentile(sample, AUTOSCALE_PCT))
    if hi <= lo + 1.0:
        hi = lo + 10.0        # empty frame: keep a usable, non-degenerate range
    return hi


def read_display_levels(dataset):
    """(lo, hi, autoscale) from the GUI, clamped to what a pixel can reach.

    Kept in one place so configure() and experiment() cannot drift apart, and
    so a missing or malformed field degrades to something viewable instead of
    raising in the middle of a run. No frame exists when configure() calls
    this, so the clamp here is the widest any container allows; experiment()
    tightens it once it can inspect a real frame.
    """
    def _get(name, default):
        try:
            return float(dataset.get_input_parameter(name))
        except Exception:
            return default
    lo = _get("display_min", DIFF_LEVELS[0])
    hi = _get("display_max", DIFF_LEVELS[1])
    auto = bool(int(_get("display_autoscale", 1.0)))
    # This is a difference image, so it legitimately spans -full .. +full.
    lo = max(lo, -float(SENSOR_MAX_MONO16))
    hi = min(hi, float(SENSOR_MAX_MONO16))
    if hi <= lo:
        hi = lo + 1.0         # a zero-width range renders as a flat block
    return lo, hi, auto


class StopRequested(Exception):
    """Raised in place of a frame when the user presses Stop.

    Unwinds straight out of the run and scan loops to the teardown, so the
    stop is acted on where it is noticed instead of every level in between
    having to test thread.running and pass a flag outwards.
    """


class ScanGridWindow:
    MARGIN_PX = 55
    POLL_MS = 500

    def __init__(self, scan_labels, diff_levels, avg_levels,
                 title="Scan grid", tile_px=320,
                 colorbar_label='Brightness (counts, frame 1 - frame 2)'):
        self.scan_labels = list(scan_labels)
        self.n_scan = len(self.scan_labels)
        self.diff_levels = diff_levels
        self.avg_levels = avg_levels
        # Atom number per hold point, and everything else the headers show,
        # kept per panel so the lifetime fit and the summary row read the same
        # numbers that are on screen.
        self.atom_numbers = [float('nan')] * self.n_scan
        self.panel_stats = [None] * self.n_scan
        # Filled by set_metrics() from the acquisition thread; read by the
        # timer below. Plain floats, so nothing here touches Qt off-thread.
        self._latest_metrics = [None] * self.n_scan
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
        self._rendered_versions = [0] * self.n_scan
        # Filled by the acquisition thread, applied by the timer below. Only
        # the timer runs on the GUI thread, and the colour bars are Qt.
        self._levels_hint = None
        for col in range(self.n_scan):
            base_col = 2 * col + 1
            lbl = self.win.addLabel(
                f"<b>{self.scan_labels[col]}</b><br>"
                f"<span style='color:#FFFF00'>sx=-- sy=--</span><br>"
                f"<span style='color:#FFB74D'>N=--</span>",
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
        # Colour bars in real camera counts, one for each row of tiles. The
        # tiles carry no axis of their own, so without these nothing on screen
        # says which count value a colour means. Non-interactive: the levels
        # are driven from the acquisition side and a draggable bar would only
        # fight it. Each bar is linked to EVERY tile in its row, so all scan
        # points share one scale -- a per-tile scale would make a column with
        # half the atoms look identical to a full one, which is exactly the
        # comparison a sweep exists to make.
        self.colorbar_cur = self._add_colorbar(
            self.current_imgs, self.diff_levels, row=1, col=2 * self.n_scan,
            label=colorbar_label)
        self.colorbar_avg = self._add_colorbar(
            self.avg_imgs, self.avg_levels, row=2, col=2 * self.n_scan,
            label=colorbar_label + ', averaged')
        # Full-width summary row under everything: mean +/- sigma of every
        # quantity in the headers, over the panels that produced a trustworthy
        # fit. On a lifetime sweep N is MEANT to vary across the panels, so the
        # spread quoted here is the decay itself, not a repeatability figure --
        # the number to read is each panel's own N, and the tau in the lifetime
        # window.
        self.summary_label = self.win.addLabel(
            'collecting...', row=4, col=0, colspan=2 * self.n_scan,
            size='10pt', justify='left')
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    def _add_colorbar(self, imgs, levels, row, col, label):
        """Colour bar for one row of tiles, or None if it cannot be built.

        ColorBarItem arrived in pyqtgraph 0.12; on anything older the fallback
        is simply the picture with no scale beside it, which is what this
        window showed before.
        """
        try:
            bar = pg.ColorBarItem(
                values=(float(levels[0]), float(levels[1])),
                colorMap=pg.colormap.get(COLORMAP),
                label=label, interactive=False)
            bar.setImageItem(imgs)
            self.win.addItem(bar, row=row, col=col)
            return bar
        except Exception:
            return None

    def set_levels_hint(self, diff_levels, avg_levels):
        """Called from the acquisition thread. Stores floats, touches no Qt.

        The tiles get their levels passed straight into setImage; this is only
        how the colour bars find out, and the timer is what applies it.
        """
        self._levels_hint = (tuple(diff_levels), tuple(avg_levels))

    def _poll_and_update(self):
        hint = self._levels_hint
        if hint is not None:
            self._levels_hint = None
            self.diff_levels, self.avg_levels = hint
            for bar, lv in ((self.colorbar_cur, self.diff_levels),
                            (self.colorbar_avg, self.avg_levels)):
                if bar is None:
                    continue
                try:
                    # Drive the bar, not the tiles: ColorBarItem.setLevels
                    # propagates to every linked ImageItem, so the tick labels
                    # and the pictures cannot drift apart.
                    bar.setLevels(low=lv[0], high=lv[1])
                except Exception:
                    pass
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
                sx = (_pm(fx['sigma'], fx.get('sigma_err', float('nan')))
                      if fx else "--")
                sy = (_pm(fy['sigma'], fy.get('sigma_err', float('nan')))
                      if fy else "--")
                # The atom number is NOT recomputed here. It arrives from the
                # acquisition thread through set_metrics(): the same call
                # feeds the log line and the lifetime fit, so the header
                # cannot quote a different N from the curve, and the
                # five-megapixel masking inside cloud_metrics is done once
                # instead of once per thread. The 1D fits above are redone
                # because the profile curves need them and they cost nothing
                # -- they run on two projections, not on the image.
                entry_met = self._latest_metrics[idx]
                met = entry_met['met'] if entry_met else None
                trust = bool(entry_met['trust']) if entry_met else False
                if met is not None:
                    area = met['counts']
                    # nan on an untrustworthy panel rather than a plausible
                    # looking number: this list is what the lifetime fit reads,
                    # and one width fitted to noise at the longest hold time
                    # would bend tau more than any other single point.
                    self.atom_numbers[idx] = (met['n_atoms_total'] if trust
                                              else float('nan'))
                    self.panel_stats[idx] = ({
                        'cx': entry_met['cx'], 'cy': entry_met['cy'],
                        'sx': entry_met['sx'], 'sy': entry_met['sy'],
                        'bright': area, 'N': met['n_atoms'],
                        'Ntot': met['n_atoms_total'],
                        'frac': met['capture_fraction'],
                        'n0': met['density'],
                    } if trust else None)
                else:
                    # No cloud region, so there is no 2w brightness to quote.
                    # The whole-frame sum is a different quantity -- it counts
                    # the pedestal across every pixel -- so it is labelled as
                    # such below rather than shown as if it were the cloud.
                    area = float(img.sum())
                    self.atom_numbers[idx] = float('nan')
                    self.panel_stats[idx] = None
                bright_label = 'bright' if met is not None else 'frame sum'
                self.header_labels[idx].setText(
                    f"<b>{self.scan_labels[idx]}</b><br>"
                    f"<span style='color:#FFFF00'>sx={sx} sy={sy} px</span><br>"
                    + (f"<span style='color:#B0BEC5'>centre="
                       f"({fx['center']:.0f},{fy['center']:.0f}) px</span><br>"
                       if (fx and fy) else
                       "<span style='color:#B0BEC5'>centre=--</span><br>")
                    + f"<span style='color:#00FF88'>{bright_label}="
                      f"{area:.3e} cnt</span><br>"
                    + ((f"<span style='color:#FFB74D'>N="
                        + _pm(met['n_atoms_total'],
                              entry_met.get('n_err', float('nan'))
                              if entry_met else float('nan'),
                              '%.3e', pct=True)
                        + f"</span><br><span style='color:#FF7043'>"
                        f"{met['region']}"
                        # In box mode there is no capture-fraction correction,
                        # so quoting a "% seen" would be reporting a 100% that
                        # was assumed rather than measured. The region label
                        # says what was actually summed instead.
                        + ("" if met['region'].startswith('fixed box')
                           else "  (%.1f%% seen)"
                                % (met['capture_fraction'] * 100))
                        + "</span><br>"
                        f"<span style='color:#FFB74D'>n0={met['density']:.2e}"
                        f"/cm3</span>")
                       if met is not None else
                       "<span style='color:#FFB74D'>N=--</span>")
                    + ("" if trust else
                       "<br><span style='color:#FF8A80'>EXCLUDED FROM FIT"
                       "</span>"))
            except Exception:
                pass
        self._update_summary()

    def _update_summary(self):
        """Mean and sigma of every displayed quantity, across the scan."""
        rows = [s for s in self.panel_stats if s is not None]
        n = len(rows)
        if n == 0:
            return
        # Population spread of the points themselves (ddof=1 for n>1), NOT an
        # error on the mean: the question is how much the quantity moves across
        # the scan.

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
        except Exception:
            pass

    def set_metrics(self, scan_idx, fit_x, fit_y, met, trust, n_err=None):
        """Cloud fit and atom number for one panel, from the acquisition thread.

        Computed there rather than in the timer because the same numbers also
        go into the log line and into the lifetime fit. Recomputing them here
        would repeat a five-megapixel masking operation and, worse, let the
        header quote an N from a different average than the curve was fitted
        to. Stores plain floats and dicts; touches no Qt.

        :param scan_idx: (int) which panel
        :param fit_x, fit_y: (dict) fit_gaussian_1d results, or None
        :param met: (dict) cloud_metrics result, or None
        :param trust: (bool) whether the point may enter the lifetime fit
        :param n_err: (float) shot-to-shot 1-sigma error on N, or nan
        """
        nan = float('nan')
        self._latest_metrics[scan_idx] = {
            'met': met,
            'trust': bool(trust),
            'n_err': float(n_err) if n_err is not None else nan,
            'cx': float(fit_x['center']) if fit_x else nan,
            'cy': float(fit_y['center']) if fit_y else nan,
            'sx': float(fit_x['sigma']) if fit_x else nan,
            'sy': float(fit_y['sigma']) if fit_y else nan,
            # Gaussian-fit 1-sigma errors. In BOX mode these do NOT enter N --
            # the box is fixed, so the atom number does not depend on this
            # point's fit at all -- they are the uncertainty on the WIDTHS and
            # on anything derived from them, such as the density.
            'sx_err': float(fit_x.get('sigma_err', nan)) if fit_x else nan,
            'sy_err': float(fit_y.get('sigma_err', nan)) if fit_y else nan,
        }

    def levels_for(self, image, base_levels):
        """Colour range for one panel: (lo, hi). Same rule as temp_PGC_aziza.

        Display Min always pins the bottom, so the background is the same
        colour in every panel and only the top moves.
        """
        lo, hi = base_levels
        if self.autoscale:
            hi = autoscale_top(image, lo)
        elif hi <= lo:
            hi = lo + 1               # a zero-width range renders as a flat block
        return lo, hi

    def set_current(self, scan_idx, diff_image, levels=None):
        lv = self.diff_levels if levels is None else levels
        try:
            self.current_imgs[scan_idx].setImage(
                diff_image, levels=lv, autoLevels=False)
        except Exception:
            pass

    def set_average(self, scan_idx, avg_image, levels=None):
        lv = self.avg_levels if levels is None else levels
        try:
            self.avg_imgs[scan_idx].setImage(
                avg_image, levels=lv, autoLevels=False)
        except Exception:
            pass
        self._latest_avgs[scan_idx] = (np.asarray(avg_image).copy(), self._rendered_versions[scan_idx] + 1)


class LifetimeFitWindow:
    """Atom number vs lattice hold time, and the exponential fitted to it.

    The grid window answers "how many atoms at this hold time"; this one
    answers the question the sweep exists for, "how fast are they leaving",
    and it is a separate window because the fit is over the WHOLE scan and
    only becomes meaningful once every point has been averaged a few times.

    Two panels, same data and same fitted curve:
      - linear, which is where the fit is actually done and where a bad point
        is obvious as a point off the curve;
      - semi-log, where a single exponential is a straight line. Curvature
        there is the signal that one decay constant is not the whole story --
        a fast light-assisted-collision loss on top of a slow background-gas
        one bends it down early, and a lattice that is still being loaded, or
        untrapped atoms drifting out of the imaging region, bends it the other
        way. tau from a single exponential is only the whole answer when this
        panel is straight.
    """

    POLL_MS = 1000

    def __init__(self, hold_values_ms):
        self.hold_values_ms = list(hold_values_ms)
        self.win = pg.GraphicsLayoutWidget(
            show=True, title="Lattice lifetime (N vs hold time)")
        self.win.resize(900, 480)

        self.plot_lin = self.win.addPlot(
            row=0, col=0, title="<b>N vs hold time</b>")
        self.plot_lin.setLabel('bottom', 'lattice hold time', units='ms')
        self.plot_lin.setLabel('left', 'atom number N (total)')
        self.plot_lin.showGrid(x=True, y=True, alpha=0.3)
        self.data_lin = self.plot_lin.plot(
            [], [], pen=None, symbol='o', symbolSize=10,
            symbolBrush='#2196F3')
        self.fit_lin = self.plot_lin.plot([], [], pen=pg.mkPen('#FF9800', width=2))
        # Shot-to-shot error on each N, drawn with the points it belongs to.
        # Added before the fit curve is ever set so the item exists whether or
        # not errors turn out to be available.
        self.err_lin = pg.ErrorBarItem(
            x=np.array([]), y=np.array([]), top=np.array([]),
            bottom=np.array([]), pen=pg.mkPen('#2196F3', width=1), beam=2.0)
        self.plot_lin.addItem(self.err_lin)

        self.plot_log = self.win.addPlot(
            row=0, col=1, title="<b>N vs hold time (log)</b>")
        self.plot_log.setLabel('bottom', 'lattice hold time', units='ms')
        self.plot_log.setLabel('left', 'atom number N (total)')
        self.plot_log.showGrid(x=True, y=True, alpha=0.3)
        # pyqtgraph does the log transform itself, so everything below stays
        # in real atom numbers and the two panels are fed identical arrays.
        self.plot_log.setLogMode(x=False, y=True)
        self.data_log = self.plot_log.plot(
            [], [], pen=None, symbol='o', symbolSize=10,
            symbolBrush='#4CAF50')
        self.fit_log = self.plot_log.plot([], [], pen=pg.mkPen('#F44336', width=2))
        # ErrorBarItem is NOT a PlotDataItem and pyqtgraph does not apply the
        # panel's log transform to it, so this one is fed log10 coordinates by
        # hand in _poll_and_update. Getting that wrong draws bars of plausible
        # length in the wrong place, which is why the transform is done in one
        # spot and not inline.
        self.err_log = pg.ErrorBarItem(
            x=np.array([]), y=np.array([]), top=np.array([]),
            bottom=np.array([]), pen=pg.mkPen('#4CAF50', width=1), beam=0.02)
        self.plot_log.addItem(self.err_log)

        # The fit result goes in the plot titles rather than a TextItem: a
        # TextItem has to be positioned in data coordinates, which are
        # logarithmic in the right-hand panel and would need the placement
        # transformed by hand every update.
        self.summary_label = self.win.addLabel(
            'collecting...', row=1, col=0, colspan=2, size='11pt',
            justify='left')

        self._latest_data = None
        self._rendered_version = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    @staticmethod
    def _method_text(res):
        """How the fit was done and whether it can be believed.

        chi2/dof is the part worth reading. It only exists for a weighted fit
        -- unweighted it is 1 by construction -- and it is the only thing on
        this panel that can say the model is wrong. A tight error on tau next
        to a chi2/dof of 30 means the single exponential does not describe the
        data to within the measured noise, and the error bar on tau is then
        far too small whatever it says.
        """
        if not res.get('refined'):
            return ('log-linear seed, N1 held at 0, '
                    '1-sigma from residual scatter')
        if not res.get('weighted'):
            return ('least-squares, unweighted; 1-sigma from residual scatter '
                    '(no per-point errors yet)')
        chi2 = res.get('chi2_red', float('nan'))
        dof = res.get('dof', 0)
        if not np.isfinite(chi2):
            return 'weighted least-squares, 1-sigma from measured shot noise'
        verdict = ('model fits the noise' if chi2 < 2.0
                   else 'MODEL DOES NOT FIT -- tau error is optimistic')
        return ('weighted least-squares, 1-sigma from measured shot noise '
                '&nbsp;|&nbsp; chi2/dof = %.2f (%d dof, %s)'
                % (chi2, dof, verdict))

    def update_data(self, hold_ms_list, n_list, n_err_list=None):
        """Called from the acquisition thread. Stores plain lists, no Qt.

        :param hold_ms_list: (sequence) hold time of each scan point [ms]
        :param n_list: (sequence) atom number at that point, nan where the
            panel's Gaussian fit was not trustworthy
        :param n_err_list: (sequence) 1-sigma shot-to-shot error on each N,
            nan where too few cycles have been averaged to measure a spread
        """
        if n_err_list is None:
            n_err_list = [float('nan')] * len(n_list)
        self._latest_data = (list(hold_ms_list), list(n_list),
                             list(n_err_list), self._rendered_version + 1)

    def _poll_and_update(self):
        entry = self._latest_data
        if entry is None:
            return
        hold_ms, n_vals, n_errs, version = entry
        if version <= self._rendered_version:
            return
        self._rendered_version = version
        try:
            t = np.asarray(hold_ms, dtype=np.float64)
            n = np.asarray(n_vals, dtype=np.float64)
            ne = np.asarray(n_errs, dtype=np.float64)
            res = _exp_decay_fit(t, n, ne)
            if res is None:
                self.summary_label.setText(
                    "<span style='color:#FF8A80'>fewer than 2 usable points "
                    "-- no fit yet</span>", size='11pt')
                return
            mask = res['mask']
            self.data_lin.setData(t[mask], n[mask])
            self.data_log.setData(t[mask], n[mask])
            # ---- error bars ----
            tm, nm, em = t[mask], n[mask], ne[mask]
            have_err = np.isfinite(em) & (em > 0)
            if have_err.any():
                self.err_lin.setData(x=tm[have_err], y=nm[have_err],
                                     top=em[have_err], bottom=em[have_err])
                # Log panel: everything is drawn in log10 space, so the bar is
                # asymmetric. A lower end at or below zero has no logarithm --
                # it is clipped to one decade below the point rather than
                # dropped, so a point whose error swallows its own value is
                # still visibly uncertain instead of silently losing its bar.
                lo = nm[have_err] - em[have_err]
                y_lg = np.log10(nm[have_err])
                top_lg = np.log10(nm[have_err] + em[have_err]) - y_lg
                bot_lg = np.where(lo > 0, y_lg - np.log10(np.maximum(lo, 1e-300)),
                                  1.0)
                self.err_log.setData(x=tm[have_err], y=y_lg,
                                     top=top_lg, bottom=bot_lg)
            else:
                empty = np.array([])
                self.err_lin.setData(x=empty, y=empty, top=empty, bottom=empty)
                self.err_log.setData(x=empty, y=empty, top=empty, bottom=empty)
            tau = res['tau_ms']
            n0 = res['n0']
            if np.isfinite(tau) and tau > 0:
                # From 0 rather than from the first hold time, so N0 -- the
                # number that would have been in the lattice at zero hold -- is
                # visible as the intercept instead of being off the left edge.
                t_fit = np.linspace(0.0, float(t[mask].max()) * 1.05, 200)
                n1 = float(res.get('n1', 0.0) or 0.0)
                curve = n0 * np.exp(-t_fit / tau) + n1
                self.fit_lin.setData(t_fit, curve)
                self.fit_log.setData(t_fit, curve)
                tau_err = float(res.get('tau_err', float('nan')))
                n0_err = float(res.get('n0_err', float('nan')))
                n1_err = float(res.get('n1_err', float('nan')))
                # The error goes on the TITLE, not only in the summary line:
                # tau without its uncertainty is the number most likely to be
                # copied off a screenshot into a logbook, and a tau of 90 +/- 4
                # ms and one of 90 +/- 60 ms are not the same result.
                self.plot_lin.setTitle(
                    "<b>N vs hold time</b>&nbsp;&nbsp;tau = %s ms"
                    % _pm(tau, tau_err, '%.1f', pct=True))
                # The log panel no longer shows a straight line once N1 > 0 --
                # the curve flattens onto the pedestal -- so the slope is
                # quoted as what tau means, not as what the picture looks like.
                self.plot_log.setTitle(
                    "<b>N vs hold time (log)</b>&nbsp;&nbsp;"
                    "decay -1/(%s) ms above N1"
                    % _pm(tau, tau_err, '%.1f'))
                # N1/N0 is the number to judge the fit by: a pedestal that is a
                # large fraction of the amplitude means the data is not one
                # exponential and tau alone does not describe it.
                frac_n1 = (100.0 * n1 / n0) if n0 > 0 else float('nan')
                self.summary_label.setText(
                    "<b>N(t) = N0 exp(-t/tau) + N1</b> &nbsp;|&nbsp; "
                    "<span style='color:#FF9800'>tau = %s ms</span> "
                    "&nbsp;|&nbsp; half-life = %s ms &nbsp;|&nbsp; "
                    "N0 = %s &nbsp;|&nbsp; "
                    "<span style='color:#80CBC4'>N1 = %s (%.1f%% of N0)"
                    "</span> &nbsp;|&nbsp; %d/%d points &nbsp;|&nbsp; %s"
                    % (_pm(tau, tau_err, '%.1f', pct=True),
                       _pm(tau * np.log(2.0), tau_err * np.log(2.0), '%.1f'),
                       _pm(n0, n0_err, '%.3e', pct=True),
                       _pm(n1, n1_err, '%.3e'), frac_n1,
                       int(mask.sum()), t.size,
                       self._method_text(res)),
                    size='11pt')
            else:
                # Points but no decay: N flat or rising with hold time. Clear
                # the old curve rather than leaving a stale one on screen next
                # to fresh points it no longer describes.
                self.fit_lin.setData([], [])
                self.fit_log.setData([], [])
                self.plot_lin.setTitle("<b>N vs hold time</b>")
                self.plot_log.setTitle("<b>N vs hold time (log)</b>")
                self.summary_label.setText(
                    "<span style='color:#FF8A80'>N is not decaying with hold "
                    "time (%d points) -- no lifetime to quote</span>"
                    % int(mask.sum()), size='11pt')
        except Exception:
            pass


# =============================================================================
# ---------------------------------------------------------------------------
# GUI parameters that may predate the form in front of the user.
# ---------------------------------------------------------------------------
def gui_param(dataset, varname, default, logger):
    """A GUI input parameter, or `default` if this form has never heard of it.

    take_data.py caches INIT_DICT into self.init_dict when the experiment is
    CLICKED in the tree, and builds the entry form from that cache. Pressing
    Configure reloads the module -- so the CODE is current -- but does not
    refresh the cache, so a GUI opened before a parameter was added keeps
    handing experiment() an input dict without it. Reading such a parameter
    directly raises KeyError and kills the run before a single frame.

    A parameter added after the fact is not worth losing a run over when the
    script already knows a sensible value for it, so fall back and say so
    loudly. The remedy is one click: re-select the script in the experiment
    tree, which re-executes it and rebuilds the form with the new entries.

    Deliberately NOT used for the parameters that were always here. Those
    have no defensible default -- a hold sweep that silently invents its own
    MOT loading time is worse than one that refuses to start -- and the whole
    point of the form is that the operator sets them.

    :param default: value to use, as float, when the form lacks `varname`
    :return: (float) the parameter value
    """
    try:
        return float(dataset.get_input_parameter(varname))
    except KeyError:
        logger.warn(
            f"The GUI form has no entry for '{varname}' -- it was built from an "
            f"older copy of this script. Falling back to {default}. To set it "
            f"yourself, re-select this script in the experiment tree (that rebuilds "
            f"the form) and press Configure again.")
        return float(default)


# ---------------------------------------------------------------------------
# DIM-3000 lattice RF drivers -- put both beams back on the same frequency.
# ---------------------------------------------------------------------------
# The lattice is a standing wave made by two beams, each driven by its own
# DIM-3000. A frequency difference Df between them makes the standing wave walk
# at v = (lambda/2)*Df -- that is exactly how latticeMoveSweep.py transports
# atoms, and at 795 nm it is 0.4 um/s per Hz (10 kHz -> 3.975 mm/s).
#
# This script wants none of that: a hold sweep is supposed to leave the atoms
# in one place while they decay. But it never used to touch the drivers, so it
# inherited whatever the last script left programmed -- and the move scripts
# leave the sweep driver in segment mode, parked at the PLATEAU frequency if a
# move run was stopped between segment 2 and segment 3, since only segment 3
# walks it back to base. The lattice then crawls all through the hold and the
# cloud is displaced LINEARLY in hold time: ~3 px over a 200 ms hold for only
# 120 Hz of stale detuning, and off the sensor entirely for a stale kHz-scale
# plateau. That drift looks exactly like physics, which is why it is reset here
# rather than left to a convention about who cleans up after whom.
#
# The reset is deliberately belt-and-braces, and uses only commands this
# codebase already relies on:
#   FRQ/AMP/OUT_on on BOTH drivers  -> both beams on base, Df = 0
#   Wseg 1-3 rewritten base->base   -> the segment chain still exists with the
#                                      shape the move scripts give it, but
#                                      every leg is a no-op, so even a spurious
#                                      trigger on dio6 cannot detune anything.
#                                      (dio6 is held low for the whole sequence
#                                      anyway; this covers the case where the
#                                      driver sees an edge regardless.)
#   Mseg:0 on BOTH drivers          -> and then the Seg connector is switched
#                                      off outright. Per the DIM-3000 manual,
#                                      Mseg:0 is "single frequency operation"
#                                      with the Seg connector inactive, so no
#                                      edge of any width can play a segment.
#                                      Belt AND braces: the no-op chain covers
#                                      a driver that somehow stays in segment
#                                      mode, this covers one whose chain still
#                                      holds a move script's real frequencies.
#   FRQ re-asserted at the end      -> nothing above gets to leave the output
#                                      somewhere other than base.
#
# BOTH DRIVERS GET THE FULL TREATMENT, which is a change. It used to be the up
# beam only, on the reasoning that the down beam sat at base and had no Seg
# cable, so its segment state could not matter. That stopped being true the day
# dio6 was teed to both Seg inputs for the two-AOM transport script. A down
# driver left in Mseg:1 by a move run -- which is exactly the state that script
# leaves it in -- now plays a segment on the first edge it sees and detunes
# itself off base, and a detuned pair is a WALKING lattice, not a trap. The
# symptom is no lattice at all, with nothing in this script's log to explain
# it, because this script never asked the down beam anything.
#
# Defaults for the two GUI entries below. They are named rather than typed
# straight into INIT_DICT because experiment() also falls back to them when
# the GUI form predates those entries (see gui_param) -- one definition, so
# the fallback cannot quietly drift away from what the form offers.
DIM_BASE_FREQ_MHZ = 100.800
DIM_AMPLITUDE_DBM = 32.0

DIM_COM_SWEEP = 'COM4'   # lattice up   -- the one the move scripts detune
DIM_COM_FIXED = 'COM5'   # lattice down -- detuned too by the two-AOM script,
# and its Seg input is teed to dio6, so it needs the
# same reset as the up beam rather than a lighter one


def reset_dim_to_base(base_freq_hz, amplitude_x10, logger, snaps=None):
    """Put both lattice drivers on base_freq_hz: stationary standing wave.

    Opens the ports, resets, and closes them again. This script never triggers
    a segment, so there is nothing to hold them open for, and releasing them
    lets the move scripts -- which DO keep them open across a sweep -- run
    afterwards without a port conflict.

    Never raises: a driver that cannot be reached must not kill a run that
    would otherwise be fine. It does log at ERROR, because a run whose lattice
    frequencies are unverified is precisely the run that produces a drifting
    spot, and that has to be visible in the log beside the images.

    :param base_freq_hz: (int) frequency for BOTH drivers [Hz]
    :param amplitude_x10: (int) amplitude in tenths of a dBm, the DIM's units
    :return: (bool) True only if both drivers were confirmed reset
    """
    if serial is None:
        logger.error(
            "pyserial is not installed, so the DIM-3000 frequencies could NOT "
            "be reset. If a move script ran before this one the two lattice "
            "beams may still be detuned, in which case the lattice is walking "
            "and the spot will move with hold time. Install pyserial, or reset "
            "the drivers by hand before trusting these positions.")
        return False

    def _open(port):
        try:
            return serial.Serial(port=port, baudrate=19200, bytesize=8,
                                 parity='N', stopbits=1, timeout=1)
        except Exception as exc:
            logger.error(
                f"Could not open the DIM-3000 on {port} ({exc}). Its frequency "
                f"is UNVERIFIED -- if it is left over from a move run the "
                f"lattice is moving and every position in this sweep is "
                f"suspect. Close whatever else is holding the port and re-run.")
            return None

    def _send(ser, cmd):
        ser.write((cmd + '\n').encode())
        time.sleep(0.05)

    def _query(ser, cmd):
        ser.flushInput()
        _send(ser, cmd)
        time.sleep(0.1)
        return ser.readline().decode(errors='replace').strip()

    def _park(ser, port):
        """Base frequency, full amplitude, output on, Seg made harmless.

        Two independent guards, because the failure they prevent is invisible:
        a driver that is off base makes the pair detuned, and a detuned pair is
        a walking lattice that looks like bad physics rather than bad state.

        Neutralised chain first, THEN Mseg:0. That order matters if the driver
        is mid-chain when this runs: rewriting the legs while segment mode is
        still on means an edge arriving during the rewrite plays a no-op leg
        rather than a stale one.
        """
        _send(ser, f'FRQ:{base_freq_hz}')
        _send(ser, f'AMP:{amplitude_x10}')
        _send(ser, 'OUT_on')
        # The longest chain any script in this repo writes is the transport
        # script's seven legs, so seven is what has to be neutralised -- the
        # old three left legs 4-7 holding a move's real frequencies. The
        # manual allows 20; seven covers everything that is actually written.
        for _i in range(1, 8):
            flag = 8 if _i == 7 else 1
            _send(ser, f'Wseg:{_i};{flag};{base_freq_hz};{base_freq_hz};0;1;0')
        # Seg connector off outright. Manual, Mseg: "N=0 - off; Connector
        # inactive, single frequency operation."
        _send(ser, 'Mseg:0')
        # Verified, not just sent. This firmware silently ignores commands it
        # does not like (its sweep commands only work in the S-prefix form),
        # and a base frequency typed into the GUI and dropped would leave both
        # beams wherever the last run put them, with nothing looking wrong.
        dim_state.set_frequency(ser, base_freq_hz, port, logger)
        logger.info(
            f"DIM-3000 {port}: {base_freq_hz / 1e6:.6f} MHz, "
            f"+{amplitude_x10 / 10:.1f} dBm, ON, segments rewritten "
            f"base->base and Seg disabled (Mseg:0)")

    ser_sweep = _open(DIM_COM_SWEEP)
    ser_fixed = _open(DIM_COM_FIXED)
    ok = False
    try:
        if ser_sweep is not None and ser_fixed is not None:
            logger.info(f"DIM-3000 sweep ({DIM_COM_SWEEP}): "
                        f"{_query(ser_sweep, '*IDN?')}")
            logger.info(f"DIM-3000 fixed ({DIM_COM_FIXED}): "
                        f"{_query(ser_fixed, '*IDN?')}")
            # BOTH, identically. See the note above _park: the down beam's
            # Seg input is teed to dio6 now, so "it has no trigger cable" is
            # no longer a reason to leave its segment state alone.
            # Record the drivers BEFORE anything is written to them, so the
            # wrapper around this function can put them back exactly as found
            # when the run stops, however it stops (dim_state.py).
            if snaps is not None:
                snaps.append(dim_state.snapshot(ser_fixed, DIM_COM_FIXED, logger))
                snaps.append(dim_state.snapshot(ser_sweep, DIM_COM_SWEEP, logger))
            _park(ser_fixed, DIM_COM_FIXED)
            _park(ser_sweep, DIM_COM_SWEEP)
            ok = True
            logger.info(
                f"Both lattice beams reset to {base_freq_hz / 1e6:.6f} MHz: "
                f"Df = 0, standing wave stationary. Spot motion with hold time "
                f"from here is NOT a stale DIM detuning.")
        else:
            # One port open and the other not is the worst state to leave
            # behind: resetting a single beam can CREATE a detuning that was
            # not there before. So change nothing at all.
            logger.error(
                "Only one of the two DIM-3000 drivers could be opened, so "
                "NEITHER was changed -- resetting one beam alone would detune "
                "the pair rather than match it. Lattice frequencies are "
                "UNVERIFIED for this run.")
    except Exception as exc:
        logger.error(
            f"DIM-3000 reset failed partway through ({exc!r}). The two beams "
            f"may be detuned; treat this run's positions as unverified.")
        ok = False
    finally:
        for ser in (ser_sweep, ser_fixed):
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
    return ok


# Default number of hold times the sweep visits, used only when the GUI form
# has no "Hold Points" entry (see n_hold_points_from_gui). The grid, the scan
# labels and the lifetime fit all size themselves off the list
# hold_values_from_gui produces, so this one number sets the whole sweep --
# nothing downstream hardcodes four.
N_HOLD_POINTS = 8

# Two points is the fewest an exponential can be fitted through -- with N0 and
# tau free there is then no freedom left over, so the fit is exact and says
# nothing about whether the decay IS exponential. Below two there is no sweep
# at all. It is a floor, not a recommendation: three or more points are what
# make tau mean anything.
N_HOLD_POINTS_MIN = 2


INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '1500'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '6'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0'},
    'Repump_VCO': {'Repump VCO (V)': '0'},

    # ===================== PHASE 2: CMOT =====================
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
    # 0 = no ramp: the cooling and repump light switch off in one step at the
    # end of PGC cool, and the lattice hold starts right there.
    'pgc_rampdown_ms': {'PGC Rampdown Duration (ms)': '0'},

    # ===================== LATTICE AM =====================
    # AM=1.0V => the full DIM output, whatever AMP is set to. AMP itself is
    # no longer inherited from dim_rf_on.py -- reset_dim_to_base writes it (and
    # the frequency) on both drivers at the start of every run.
    # AOM thermal stability: lattice TTL stays ON & AM stays at lat_am_v
    # for the ENTIRE cycle.
    'lat_am_v': {'Lattice AM (V)': '0.6'},

    # ===================== DIM-3000 AOM DRIVERS =====================
    # Both lattice beams are forced onto this ONE frequency at the start of
    # every run, so the standing wave cannot be left walking by whatever ran
    # before (see reset_dim_to_base). Same defaults as latticeMoveSweep.py's
    # base, which is the frequency its moves start and end on -- keep the two
    # in step, or a move script and a hold sweep will disagree about where
    # "stationary" is.
    'dim_base_freq_mhz': {'DIM3000 Base Freq (MHz)':
                          '%.3f' % DIM_BASE_FREQ_MHZ},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)':
                          '%.1f' % DIM_AMPLITUDE_DBM},

    # ===================== PHASE 5: LATTICE HOLD SWEEP =====================
    # Hold time = atoms sit in lattice with all PGC light off.
    # Lifetime usually 30-40 ms (D1 photon scattering at 795 nm).
    # Lattice hold is swept over Hold Points values spaced linearly from
    # min to max and rounded to whole ms. 30..200 over 4 points gives
    # 30/87/143/200, a span wide enough to see a real decay: a lifetime fit
    # needs the longest hold to be comparable to tau, or every point sits on
    # the flat part of the exponential and tau is unconstrained.
    'hold_min_ms': {'Lattice Hold Min (ms)': '30'},
    'hold_max_ms': {'Lattice Hold Max (ms)': '200'},

    # How many hold times the lifetime sweep visits -- and therefore how many
    # panel columns appear, since the grid gets one column per hold time.
    # More points constrain tau better and show whether the decay is actually
    # a single exponential, but each one is another sub-sequence in every
    # buffer cycle: the time to a given number of averages per point, and the
    # screen width, both grow in proportion. Four fits comfortably; much past
    # six or seven the tiles shrink to their 150 px floor.
    'n_hold_points': {'Hold Points (lifetime sweep)': '%d' % N_HOLD_POINTS},

    # ===================== FIXED TOF after release =====================
    'tof_us': {'TOF (us)': '0'},

    # ===================== IMAGING =====================
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '200'},

    # ===================== ATOM NUMBER =====================
    # Half-width of the FIXED counting box, in units of the first scan point's
    # fitted sigma. The box is latched once from that point and every hold time
    # is then summed over the same pixels -- see BOX_HALF_SIGMA at the top for
    # what that buys and what it costs.
    'box_half_sigma': {'Counting Box Half-Width (x sigma of 1st point)':
                       '%.1f' % BOX_HALF_SIGMA},
    # Drop a point from the lifetime fit when its sigma exceeds this multiple
    # of the first point's, on either axis. 2.0 = "grown by more than 100%";
    # 1.0 = "grown at all", which is the strict reading and will also reject
    # ordinary shot-to-shot scatter, so start at 2.0 and tighten if the tail
    # still looks contaminated.
    'width_reject_factor': {'Reject Point if sigma > this x 1st point':
                            '%.1f' % WIDTH_REJECT_FACTOR},

    # ===================== CALIBRATION =====================
    # OBJECT-SPACE pixel size, i.e. what one pixel covers AT THE ATOMS: the
    # 3.45 um sensor pitch divided by the 0.59x magnification. Shown for
    # reference alongside the widths; the atom-number and density conversions
    # use the PIXEL_UM constant at the top, computed from those same two
    # numbers, so there is one definition and not two that can drift apart.
    'pixel_size_um': {'Pixel Size at the Atoms (um/px)': '%.4f' % PIXEL_UM},

    # ===================== BIAS COILS =====================
    # Held CONSTANT at these values for the ENTIRE sub-sequence
    # (no ramping, on through MOT loading too). Matches sweep scripts.
    'bias_x_v': {'Bias X (V)': '0.0'},
    'bias_y_v': {'Bias Y (V)': '0.0'},
    'bias_z_v': {'Bias Z (V)': '0.0'},

    # ===================== DISPLAY =====================
    # Autoscale by default (1): the top of the colour range follows the
    # signal in each frame (99.95th percentile, so one hot pixel cannot set
    # it), while Display Min stays pinned as the background. This is the
    # setting that makes a faint cloud visible at all -- inferno needs about
    # 20% of its range before it shows anything, so a fixed Display Max more
    # than ~5x the real peak renders the whole cloud as black even while the
    # profile fits find it easily (they sum 2048 rows, a ~45x gain in SNR,
    # so they see what the eye cannot).
    #
    # Set to 0 once a range is known, to compare brightness between runs:
    # then the colour range is exactly Display Min .. Display Max, clamped to
    # what a pixel can physically reach. The log's p99.95 line tells you what
    # to type.
    'display_autoscale': {'Autoscale Display? (1=yes, 0=manual)': '1'},
    'display_min': {'Display Min / Background (counts)': '0'},
    'display_max': {'Display Max (counts)': '500'},

    # ===================== WARM-UP =====================
    # Cycles run before any data is kept, at the FIRST hold time's settings.
    # Between one arming and the next the coils are off and the light is down
    # for seconds while the waveform is finalized, uploaded and re-armed, so
    # the MOT is completely gone. The first sub-sequence after that loads from
    # ZERO atoms, while every later one only tops up a MOT that never fully
    # decayed -- which is why the first point or two come out different. On a
    # hold sweep that transient lands on the shortest hold, i.e. the point the
    # lifetime fit leans on hardest, so it biases tau and not just one panel.
    # These cycles absorb it. Set to 0 to keep every shot; each one costs a
    # full sub-sequence of run time.
    'warmup_cycles': {'Warm-up Cycles (0 = none)': '3'},

    # ===================== SCAN ORDER =====================
    # 1  = shortest hold first (what this sweep has always done)
    # -1 = longest hold first  -- the reversal test
    # 0  = flip every buffer cycle, to average the order out for good
    # See scan_play_order for why the order is worth controlling at all.
    'scan_order': {'Scan Order (1=up, -1=down, 0=alternate)': '1'},

    # ===================== GENERAL =====================
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},
}


# Scan-order modes for the reversal test (see scan_play_order).
ORDER_UP = 1
ORDER_DOWN = -1
ORDER_ALTERNATE = 0


def scan_play_order(n_points, mode, cycle):
    """The order to PLAY the hold points in, as canonical indices.

    The sweep always ran shortest hold first, which quietly ties hold time to
    wall-clock time: with the default four points the first is imaged about
    0.9 s into a pass and the last about 4.5 s in, always in that order. Every
    shot at a given point therefore lands at the same phase of the pass, so
    anything drifting on a few-second scale -- imaging beam pointing, fibre
    coupling, an AOM still thermalising -- adds a displacement that grows
    monotonically with hold time and does NOT average away with more runs. It
    is indistinguishable from real hold-time physics as long as the order
    never changes.

    Reversing the order breaks the tie. Real physics is a function of the hold
    itself, so it comes out the same either way; an artefact of acquisition
    order flips sign against hold time. Run the sweep once each way and
    compare the two sets of panels.

    ORDER_ALTERNATE flips on every buffer cycle instead. That is the remedy
    rather than the test: each point then samples both ends of the pass and
    the drift averages out of the running mean. Use it once the test has said
    there is something worth averaging out.

    :param n_points: (int) number of hold points
    :param mode: ORDER_UP, ORDER_DOWN or ORDER_ALTERNATE
    :param cycle: (int) 0-based buffer cycle, only read for ORDER_ALTERNATE
    :return: (list of int) indices into hold_values_ms, in playback order
    """
    up = list(range(n_points))
    if mode == ORDER_DOWN:
        return up[::-1]
    if mode == ORDER_ALTERNATE:
        return up[::-1] if (cycle % 2) else up
    return up


def n_hold_points_from_gui(dataset, logger=None):
    """How many hold times to sweep, from the GUI.

    This one entry sets both halves of what the operator sees: the number of
    points the lattice-lifetime fit is made from, AND the number of panel
    columns on screen, because every downstream size (scan labels, grid
    columns, running-average buffers, the per-point atom numbers, the fit's x
    axis) is taken from len(hold_values_from_gui(...)).

    Read defensively rather than with get_input_parameter, for the same reason
    gui_param exists: a GUI form built before this entry existed hands
    experiment() an input dict without it, and a bare read would KeyError and
    kill the run before a single frame. configure() has no logger to hand, so
    the logger is optional and the fallback is silent without one -- the
    warning still reaches the log from experiment(), which does pass it.

    :return: (int) number of hold points, at least N_HOLD_POINTS_MIN
    """
    try:
        n_points = int(round(float(dataset.get_input_parameter("n_hold_points"))))
    except (KeyError, TypeError, ValueError):
        if logger is not None:
            logger.warn(
                f"The GUI form has no usable entry for 'n_hold_points' -- it "
                f"was built from an older copy of this script. Sweeping the "
                f"default {N_HOLD_POINTS} hold times. To set it yourself, "
                f"re-select this script in the experiment tree (that rebuilds "
                f"the form) and press Configure again.")
        return N_HOLD_POINTS
    if n_points < N_HOLD_POINTS_MIN:
        if logger is not None:
            logger.warn(
                f"Hold Points = {n_points} is below {N_HOLD_POINTS_MIN}; a "
                f"lifetime cannot be fitted through fewer than two hold "
                f"times, so running {N_HOLD_POINTS_MIN} points.")
        return N_HOLD_POINTS_MIN
    return n_points


def hold_values_from_gui(dataset, logger=None):
    """The lattice hold times to sweep, in whole ms.

    Hold Points values spaced linearly from Lattice Hold Min to Lattice Hold
    Max and rounded to the nearest millisecond. Rounding is not cosmetic: the
    waveform is built in whole-ms blocks (ms2s multiplies by samples per ms),
    so a fractional hold could not be played anyway, and returning the rounded
    values keeps the panel labels and the lifetime fit's x axis showing what
    was actually run rather than what was asked for.

    Negative holds are clamped to 0 -- there is no such thing as a negative
    time in the trap, and a negative block length would silently produce an
    empty range and shift every later anchor in the sequence.

    :return: (list of int) hold times in ms, one per GUI Hold Point
    """
    n_points = n_hold_points_from_gui(dataset, logger)
    lo = float(dataset.get_input_parameter("hold_min_ms"))
    hi = float(dataset.get_input_parameter("hold_max_ms"))
    values = [max(0, int(round(v)))
              for v in np.linspace(lo, hi, n_points)]
    return values


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
    hold_values_ms = hold_values_from_gui(dataset)
    scan_labels = [f"hold={t}ms" for t in hold_values_ms]
    # Display range from the GUI rather than the old hardcoded (0, 20).
    disp_min, disp_max, _ = read_display_levels(dataset)
    dataset.diff_levels = (disp_min, disp_max)
    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        diff_levels=(disp_min, disp_max), avg_levels=(disp_min, disp_max),
        title="Lattice HOLD sweep (PGC_cool+rampdown then hold in lattice)",
    )
    # Second window: N vs hold time and the exponential fitted through it.
    dataset.lifetime_fit = LifetimeFitWindow(hold_values_ms)
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
    CMOT_RAMP_MS = 15
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
    # gui_param, not get_input_parameter: these two are newer than some of
    # the saved GUI forms out there, and a form that predates them must not
    # take the run down with a KeyError.
    dim_base_freq_mhz = gui_param(dataset, "dim_base_freq_mhz",
                                  DIM_BASE_FREQ_MHZ, logger)
    dim_amplitude_dbm = gui_param(dataset, "dim_amplitude_dbm",
                                  DIM_AMPLITUDE_DBM, logger)
    base_freq_hz = int(round(dim_base_freq_mhz * 1e6))

    hold_values_ms = hold_values_from_gui(dataset, logger)
    n_scan = len(hold_values_ms)
    # gui_param, not get_input_parameter: a form built before this entry
    # existed would otherwise KeyError and kill the run (see gui_param).
    warmup_cycles = max(0, int(gui_param(dataset, "warmup_cycles", 3, logger)))
    box_half_sigma = float(gui_param(dataset, "box_half_sigma",
                                     BOX_HALF_SIGMA, logger))
    width_reject_factor = float(gui_param(dataset, "width_reject_factor",
                                          WIDTH_REJECT_FACTOR, logger))
    if box_half_sigma <= 0:
        raise ValueError(
            "Counting Box Half-Width must be positive; %g would make the box "
            "empty and every atom number zero." % box_half_sigma)
    if width_reject_factor < 1.0:
        logger.warn(
            f"Width reject factor {width_reject_factor:.2f} is below 1.0, so "
            f"the reference point is wider than its own limit and EVERY point "
            f"will be rejected, including the one the box came from.")
    scan_order_mode = int(gui_param(dataset, "scan_order", ORDER_UP, logger))
    if scan_order_mode not in (ORDER_UP, ORDER_DOWN, ORDER_ALTERNATE):
        logger.warn(
            f"Scan Order {scan_order_mode} is not 1, -1 or 0; running the "
            f"sweep shortest-hold-first.")
        scan_order_mode = ORDER_UP
    tof_us = int(dataset.get_input_parameter("tof_us"))
    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))

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
    # DIM-3000: both beams onto base BEFORE anything else happens
    # =========================================================================
    # First thing in the run, ahead of building any waveform: if this fails we
    # want it at the top of the log, not buried after the timeline. A hold
    # sweep never changes these frequencies again, so one reset per Start is
    # all it takes -- the ports are closed again immediately.
    reset_dim_to_base(base_freq_hz, int(round(dim_amplitude_dbm * 10)), logger,
                      snaps=kwargs['dim_snapshots'])

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
        f"then sweep lattice HOLD = {hold_values_ms}ms, "
        f"TOF (after release) = {tof_us}us"
    )

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence(hold_ms):
        hold_s = ms2s(hold_ms)
        release = pgc_end + hold_s  # release happens AFTER lattice hold
        f1 = release + tof_s_fixed
        f2 = f1 + clear_s
        crt = f2 + aom_pulse_s + coils_restore_s  # coils stay OFF through frame 2
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

        return cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, biasx, biasy, biasz, ludo, lddo, luao, ldao, N

    # Each hold point is built ONCE. Which order they are concatenated in is
    # then just a permutation, chosen per buffer cycle -- so ORDER_ALTERNATE
    # can flip direction without rebuilding a single waveform.
    subseq = [build_sub_sequence(h) for h in hold_values_ms]
    n_wf = len(subseq[0]) - 1        # last tuple element is N, not a waveform

    def concat(order):
        """Splice the sub-sequences into one buffer, in playback order.

        Warm-up cycles sit at the head and take the hold of whichever point
        plays FIRST, so they stay a faithful rehearsal of the cycle that
        follows them in either direction -- a 200 ms warm-up in front of a
        200 ms first point, not a 30 ms one left over from the other order.

        :param order: (list of int) indices into hold_values_ms, playback order
        :return: (list of list, int) the 16 channel buffers and their length
        """
        plan = [order[0]] * warmup_cycles + list(order)
        buffers = [[] for _ in range(n_wf)]
        for idx in plan:
            sub_wf = subseq[idx]
            for ch in range(n_wf):
                buffers[ch] += sub_wf[ch]
        return buffers, len(buffers[0])

    def order_label(order):
        """Playback order written out as the hold times themselves."""
        return " -> ".join(str(hold_values_ms[i]) for i in order) + " ms"

    ORDER_NAMES = {ORDER_UP: "ascending (shortest hold first)",
                   ORDER_DOWN: "DESCENDING -- reversal test",
                   ORDER_ALTERNATE: "alternating every buffer cycle"}
    logger.info(
        f"Lattice hold sweep: {n_scan} hold points, "
        f"hold_ms={hold_values_ms}, tof={tof_us}us, "
        f"{warmup_cycles} warm-up cycle(s), "
        f"scan order = {ORDER_NAMES[scan_order_mode]}")
    # Rounding to whole ms can collapse two requested points onto the same
    # value when min and max are close together -- 10..12 over four points
    # gives 10, 11, 11, 12. The scan still runs, but two columns are then
    # duplicates and the lifetime fit gets a repeated x, so say so.
    if len(set(hold_values_ms)) < len(hold_values_ms):
        logger.warn(
            f"Hold values {hold_values_ms} contain duplicates after rounding "
            f"to whole ms; widen the Min/Max span, or lower Hold Points, "
            f"to get {n_scan} distinct points."
        )

    diff_sums = [None] * n_scan
    diff_counts = [0] * n_scan
    # These live out here with the running sums they are computed from, NOT
    # inside the `while thread.running` loop below. That loop re-enters every
    # time the NI/OPX buffer is re-armed (every M runs), and initialising these
    # inside it wiped every point out of the lifetime plot on each re-arm --
    # the fit fell back to "fewer than 2 usable points" while the averaged
    # images it is derived from carried on improving, because diff_sums was
    # correctly scoped out here and these were not.
    sigma_x_vals = [float('nan')] * n_scan
    sigma_y_vals = [float('nan')] * n_scan
    # Whole-cloud atom number at each hold time, nan until that point has a
    # trustworthy fit. This is what the lifetime window plots and fits.
    n_atom_vals = [float('nan')] * n_scan
    # The FIXED counting box, latched once from scan point 0 (the shortest
    # hold, so the brightest and best-determined cloud) and then frozen for the
    # whole run. Scoped out here with the running sums for the same reason
    # sigma_x_vals is: re-arming the buffer must not silently re-latch the box
    # on a later, dimmer cloud and change what N means half way through a
    # lifetime curve.
    ref_box = None
    ref_sigma = None        # (sigma_x, sigma_y) of the reference point, px
    # 1-sigma error on each point's N, from the SPREAD OF THE INDIVIDUAL
    # CYCLES that were averaged into it. This is the error that matters: the
    # average of n shots is only as good as the shot-to-shot scatter divided
    # by sqrt(n), and nothing else in this script knew that number.
    n_err_vals = [float('nan')] * n_scan
    # Running sum and sum-of-squares of the PER-CYCLE box counts, so the
    # standard error is available without keeping every cycle's frame. Scoped
    # out here with diff_sums for the same reason: a buffer re-arm must not
    # reset the statistics half way through a lifetime curve.
    box_n = [0] * n_scan
    box_sum = [0.0] * n_scan
    box_sumsq = [0.0] * n_scan

    # ---- display / scaling state (jingfastPGC) ----
    disp_min, disp_max, autoscale = read_display_levels(dataset)
    # Brightest raw pixel seen this run, and the resulting physical ceiling.
    # Both settle on the first frame and are refined if anything brighter turns
    # up; `limit` is what the requested range is clamped against.
    run_peak = None
    limit = None
    # Latest p99.95 of each tile, so one range can cover a whole row.
    cur_tops = [None] * n_scan
    avg_tops = [None] * n_scan
    # Each diagnostic fires once per run; the conditions persist for as long as
    # whatever caused them, and one warning per scan point would bury the log.
    warned = set()

    # =========================================================================
    # MAIN ACQUISITION LOOP
    # =========================================================================
    cycle_idx = 0
    while thread.running:
        # Order first, then splice the buffer to match. The acquisition loop
        # below walks this same list, so the frames coming back and the panels
        # they land in cannot disagree about which hold time is on screen.
        play_order = scan_play_order(n_scan, scan_order_mode, cycle_idx)
        (cam_p, ado_p, aao_p, rdo_p, rao_p, cdo_p, cao_p, mvco_p, rvco_p,
         biasx_p, biasy_p, biasz_p, ludo_p, lddo_p, luao_p,
         ldao_p), total = concat(play_order)
        logger.info(
            f"Buffer cycle {cycle_idx + 1}: playing {order_label(play_order)}"
            f", {warmup_cycles} warm-up at "
            f"{hold_values_ms[play_order[0]]} ms, total={total} samples")
        camera_client.set_hardware_trigger(
            line="Line0", activation="RisingEdge",
            selector="FrameStart", overlap="ReadOut",
            acquisition_mode="Continuous")
        camera_client.set_exposure(camera_exposure_us)
        # Gain: request it, then READ IT BACK and use whatever the camera
        # actually applied. try_set_float is best-effort -- a value past the
        # node maximum raises inside SetValue, the driver swallows it, and the
        # camera keeps what it had. Assuming the requested value instead makes
        # every atom number wrong by 10^(dG/20), and since the lifetime fit
        # only cares about RATIOS of N it would not even show up there: tau
        # would look perfectly healthy while every N beside it was out by
        # orders of magnitude.
        gain_set = camera_client.try_set_float("Gain", CAMERA_GAIN_DB)
        gain_db = None
        try:
            gain_db = camera_client.get_float("Gain")
        except Exception as exc:
            logger.warn(
                f"Camera server has no get_float ({exc}); cannot confirm "
                f"the gain. Restart it to enable the read-back.")
        if gain_db is None:
            gain_db = CAMERA_GAIN_DB if gain_set else 0.0
            logger.warn(
                f"Gain could not be read back; assuming {gain_db:.2f} dB "
                f"from the set() {'succeeding' if gain_set else 'FAILING'}.")
        if abs(gain_db - CAMERA_GAIN_DB) > 0.01:
            logger.error(
                f"Camera gain is {gain_db:.2f} dB, NOT the "
                f"{CAMERA_GAIN_DB:.2f} dB requested. Atom numbers use the "
                f"ACTUAL value so they are correct, but set CAMERA_GAIN_DB "
                f"to something the camera accepts.")
        else:
            logger.info(f"Camera gain confirmed at {gain_db:.2f} dB")
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
        NI_card_2.set_do_voltage(do_channel="dio6", value=[0] * total, sample_rate=ni_rate)
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

        def get_frame(timeout_ms=1000, poll_ms=500):
            """One frame, waited for in short slices so Stop is seen promptly.

            The camera is triggered by the sequence, so a grab legitimately
            blocks for a whole scan point -- about a second here -- and the
            timeout has to allow for a missed trigger on top of that. Asking
            for the entire wait in one call means Stop cannot be noticed until
            the frame lands, which is what made stopping slow: thread.running
            was tested once per pass through the buffer, so a stop pressed at
            the wrong moment sat through the remaining runs and scan points.

            Spinnaker's GetNextImage does not consume anything when it times
            out, so a queue of short waits collects exactly the frame one long
            wait would have -- it just gets a chance to look at thread.running
            between them.
            """
            deadline = time.perf_counter() + timeout_ms / 1000.0
            while True:
                if not thread.running:
                    raise StopRequested()
                remaining_ms = (deadline - time.perf_counter()) * 1000.0
                if remaining_ms <= 0:
                    raise TimeoutError(
                        f"No camera frame within {timeout_ms} ms")
                started = time.perf_counter()
                try:
                    b, shape, dtype = dataset.camera_client.get_frame_bytes(
                        int(min(poll_ms, remaining_ms)))
                except Exception:
                    # A timeout has already burned its slice. A real fault --
                    # incomplete frame, RPC hiccup -- comes back at once, so
                    # pause before retrying rather than spinning on it.
                    if time.perf_counter() - started < 0.05:
                        time.sleep(0.05)
                    continue
                return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        scan_grid = dataset.scan_grid
        lifetime_fit = dataset.lifetime_fit
        try:
            for run_idx in range(M):
                if not thread.running:
                    raise StopRequested()
                logger.info(f"Run {run_idx+1} of {M}")
                # The warm-up cycles sit at the head of the buffer and fire
                # their camera triggers like any other cycle, so their frames
                # MUST be consumed here. Skip this and every pair afterwards
                # is offset by one -- each panel would then be showing another
                # hold time's cloud, and the lifetime fit would read a tau off
                # a shuffled x axis, with no error to say so.
                for _ in range(2 * warmup_cycles):
                    get_frame(timeout_ms=100000)
                # pos is the position in the PASS, scan_idx the hold point.
                # Everything downstream -- diff_sums, the panels, the lifetime
                # fit -- stays indexed by scan_idx, so a reversed run still
                # files each cloud under its own hold time and the two
                # directions compare panel for panel.
                for pos, scan_idx in enumerate(play_order):
                    if not thread.running:
                        raise StopRequested()
                    hold_v = hold_values_ms[scan_idx]
                    is_first = (run_idx == 0 and pos == 0)
                    frame1 = get_frame(timeout_ms=10000 if is_first else 100000)
                    frame2 = get_frame(timeout_ms=100000)
                    diff = frame1.astype(np.int32) - frame2.astype(np.int32)
                    if diff_sums[scan_idx] is None:
                        diff_sums[scan_idx] = diff.astype(np.float64)
                    else:
                        diff_sums[scan_idx] += diff
                    diff_counts[scan_idx] += 1
                    avg = diff_sums[scan_idx] / diff_counts[scan_idx]

                    # ------------------ display / scaling ------------------
                    # Full scale is a property of the ADC, not of the
                    # difference, so it is read off the RAW frames.
                    frame_peak = int(max(frame1.max(), frame2.max()))
                    if run_peak is None or frame_peak > run_peak:
                        run_peak = frame_peak
                    if limit is None:
                        limit = sensor_full_scale(frame1)
                        lo_req, hi_req = disp_min, disp_max
                        # The difference of two frames spans -limit .. +limit.
                        disp_min = max(disp_min, -float(limit))
                        disp_max = min(disp_max, float(limit))
                        logger.info(
                            f"Camera full scale: {limit} counts "
                            f"({frame1.dtype}). Display range limited to "
                            f"[{-limit}, {limit}].")
                        if frame1.dtype == np.uint8:
                            logger.error(
                                f"Camera is sending 8-bit frames despite the "
                                f"request for {PIXEL_FORMAT}. Every count is "
                                f"16x coarser than Mono16, so a cloud that "
                                f"should read tens of counts arrives as 1-3 "
                                f"and is quantised into a handful of grey "
                                f"levels. Nothing on the display side can undo "
                                f"that -- fix the PixelFormat (check SpinView "
                                f"is not holding the camera, and that "
                                f"Mono16/Mono12p is available on this model).")
                        if (lo_req, hi_req) != (disp_min, disp_max):
                            logger.warn(
                                f"Requested display range ({lo_req:.0f}, "
                                f"{hi_req:.0f}) exceeds what a pixel can "
                                f"reach; using ({disp_min:.0f}, "
                                f"{disp_max:.0f}).")
                    if autoscale:
                        # Top follows the signal, bottom stays pinned at
                        # Display Min so the background does not breathe. The
                        # top is the largest over the scan points seen so far,
                        # not this tile's own, so the columns stay comparable.
                        cur_tops[scan_idx] = autoscale_top(diff, disp_min)
                        avg_tops[scan_idx] = autoscale_top(avg, disp_min)
                        cur_levels = (disp_min,
                                      max(t for t in cur_tops if t is not None))
                        avg_levels = (disp_min,
                                      max(t for t in avg_tops if t is not None))
                    else:
                        cur_levels = (disp_min, disp_max)
                        avg_levels = (disp_min, disp_max)
                    scan_grid.set_current(scan_idx, diff, levels=cur_levels)
                    scan_grid.set_average(scan_idx, avg, levels=avg_levels)
                    scan_grid.set_levels_hint(cur_levels, avg_levels)
                    # ---------------- atom number at this hold ----------------
                    # Fitted on the running AVERAGE, not the single shot: at
                    # the long hold times there are few enough atoms left that
                    # one frame will not support a fit, and averaging is the
                    # only thing that keeps the tail of the decay -- the part
                    # tau is most sensitive to -- from being thrown away as
                    # unreliable.
                    img = avg.astype(np.float64)
                    fx = fit_gaussian_1d(img.sum(axis=0))
                    fy = fit_gaussian_1d(img.sum(axis=1))
                    good = bool(fx and fy and fx['reliable'] and fy['reliable'])
                    # sigma, NOT w: w = 2*sigma is the 1/e^2 radius.
                    sx = fx['sigma'] if fx else float('nan')
                    sy = fy['sigma'] if fy else float('nan')
                    sigma_x_vals[scan_idx] = sx
                    sigma_y_vals[scan_idx] = sy

                    # ---- the fixed counting box, latched once ----
                    # Scan point 0 is the shortest hold: the brightest cloud
                    # and the best-determined fit in the sweep, which is the
                    # one worth building a permanent region from. Deliberately
                    # keyed to scan_idx and not to acquisition order, so a
                    # reversed or shuffled play order still measures the box
                    # off the same physical point. Under a reversed order that
                    # point arrives LAST in the first pass, so the panels ahead
                    # of it publish no N until pass 2 -- they fill in then,
                    # because every point is re-evaluated on its running
                    # average on every pass.
                    if ref_box is None and scan_idx == 0 and good:
                        ref_box = reference_box(img.shape, fx, fy,
                                                box_half_sigma)
                        if ref_box is not None:
                            ref_sigma = (ref_box['sigma_x'], ref_box['sigma_y'])
                            n_box_px = ((ref_box['x1'] - ref_box['x0'])
                                        * (ref_box['y1'] - ref_box['y0']))
                            logger.info(
                                f"Counting box LATCHED from hold="
                                f"{hold_values_ms[0]}ms at "
                                f"{box_half_sigma:.1f} sigma "
                                f"(sx={ref_sigma[0]:.1f}, "
                                f"sy={ref_sigma[1]:.1f} px): "
                                f"x {ref_box['x0']}-{ref_box['x1']}, "
                                f"y {ref_box['y0']}-{ref_box['y1']} px, "
                                f"{n_box_px} px. Every hold time is summed "
                                f"over these same pixels from here on; points "
                                f"wider than {width_reject_factor:.2f}x this "
                                f"are dropped from the lifetime fit.")

                    # ---- width test against the reference point ----
                    # A fit that has found noise rather than atoms comes back
                    # wide. `reliable` already catches the worst of it, but a
                    # broad bump can pass that test and still be nothing, and
                    # at the longest hold time one such point bends tau more
                    # than any other. Referred to the FIRST point's sigma, so
                    # the criterion travels with the cloud size of the day
                    # instead of being a pixel count that goes stale.
                    width_ok = True
                    width_msg = ""
                    if ref_sigma is not None:
                        lim_x = width_reject_factor * ref_sigma[0]
                        lim_y = width_reject_factor * ref_sigma[1]
                        width_ok = bool(np.isfinite(sx) and np.isfinite(sy)
                                        and sx <= lim_x and sy <= lim_y)
                        if not width_ok:
                            width_msg = (f"sigma ({sx:.1f}, {sy:.1f}) px over "
                                         f"{width_reject_factor:.2f}x the "
                                         f"first point "
                                         f"({lim_x:.1f}, {lim_y:.1f} px)")
                    # frame1.dtype, NOT img.dtype: DN/photon is a property
                    # of the raw ADC word, and img is the float64 running
                    # average, which would silently select the 12-bit
                    # calibration and put N out by 16x. gain_db is the value
                    # read back off the camera, not the one requested.
                    #
                    # No box yet means no atom number at all. Falling back to
                    # the per-point ellipse here would mix two definitions of N
                    # inside one lifetime curve, which is worse than a gap --
                    # the gap is visible, the mix is not.
                    met = (cloud_metrics(img, fx, fy, frame1.dtype,
                                         camera_exposure_us, gain_db=gain_db,
                                         box=ref_box)
                           if (fx and fy and ref_box is not None) else None)
                    usable = bool(met is not None and good and width_ok)
                    n_atom_vals[scan_idx] = (met['n_atoms_total'] if usable
                                             else float('nan'))

                    # ---- shot-to-shot error on this point's N ----
                    # THIS cycle's box counts, measured on the single-shot
                    # difference rather than the running average, accumulated
                    # as sum and sum-of-squares. The spread of those per-cycle
                    # numbers is what the error bar on N is: sigma/sqrt(n).
                    #
                    # Measured, not modelled. Photon shot noise would give a
                    # floor, but the real scatter here is dominated by how
                    # many atoms the MOT loaded that cycle, and no noise model
                    # knows that. The cycles themselves do.
                    #
                    # The mean of the per-cycle counts and the counts of the
                    # mean image differ slightly, because each cycle gets its
                    # own MEDIAN background and a median is not linear. N
                    # itself keeps coming from the averaged image, which has
                    # the better-determined background; only the SPREAD comes
                    # from here.
                    if ref_box is not None:
                        cyc = box_brightness(diff.astype(np.float64), ref_box)
                        if cyc is not None:
                            c = float(cyc['counts'])
                            box_n[scan_idx] += 1
                            box_sum[scan_idx] += c
                            box_sumsq[scan_idx] += c * c
                    n_err = float('nan')
                    if usable and box_n[scan_idx] >= 2:
                        nb = box_n[scan_idx]
                        mean_c = box_sum[scan_idx] / nb
                        # Sample variance via E[x^2]-E[x]^2 with Bessel's
                        # correction. Clipped at 0: with counts of order 1e9
                        # the two terms are close and rounding can take the
                        # difference slightly negative, and sqrt of that is a
                        # nan that would silently disable the weighting.
                        var = max(0.0, (box_sumsq[scan_idx] / nb - mean_c ** 2)
                                  * nb / (nb - 1.0))
                        sem_counts = float(np.sqrt(var / nb))
                        # counts -> atoms is a single linear factor, so the
                        # fractional error carries straight across.
                        if met['counts'] > 0 and np.isfinite(sem_counts):
                            n_err = (met['n_atoms_total']
                                     * sem_counts / abs(met['counts']))
                    n_err_vals[scan_idx] = n_err if usable else float('nan')
                    # One evaluation, three consumers: this panel's header,
                    # the log line below, and the lifetime fit.
                    scan_grid.set_metrics(scan_idx, fx, fy, met, usable,
                                          n_err=n_err_vals[scan_idx])
                    lifetime_fit.update_data(hold_values_ms, n_atom_vals,
                                             n_err_vals)
                    if met is not None:
                        _n_err = n_err_vals[scan_idx]
                        _err_txt = (f" +/- {_n_err:.2e} "
                                    f"({100 * _n_err / met['n_atoms_total']:.1f}%"
                                    f", {box_n[scan_idx]} cycles)"
                                    if np.isfinite(_n_err)
                                    and met['n_atoms_total'] > 0
                                    else " +/- -- (need 2 cycles)")
                        _what = (f"bright={met['counts']:.3e}cnt "
                                 f"[{met['region']}, bg {met['bg_source']}], "
                                 f"N={met['n_atoms']:.3e}" + _err_txt +
                                 f", n0={met['density']:.2e}/cm3")
                    elif ref_box is None:
                        _what = (f"no counting box yet -- waiting on hold="
                                 f"{hold_values_ms[0]}ms to fit")
                    else:
                        _what = "no cloud region"
                    if usable:
                        _why = ""
                    elif ref_box is None:
                        _why = "  [EXCLUDED: no counting box yet]"
                    elif not good:
                        _why = "  [EXCLUDED: unreliable fit]"
                    else:
                        _why = f"  [EXCLUDED: {width_msg}]"
                    logger.info(
                        f"  hold={hold_v}ms: n={diff_counts[scan_idx]}, "
                        f"sx={sx:.1f}px, sy={sy:.1f}px, " + _what + _why)

                    # Diagnostics, from jingfastPGC. "All dark" has several
                    # distinct causes and the numbers below separate them: no
                    # light on the sensor at all, light but no atoms, a signal
                    # pointing the wrong way, or a display range that simply
                    # does not cover the data.
                    d_lo = float(np.percentile(diff, 0.05))
                    d_hi = float(np.percentile(diff, 99.95))
                    # float64 accumulator: five million int32 pixels overflow
                    # an int32 sum at a mean of only ~430 counts.
                    area = float(diff.sum(dtype=np.float64))
                    # What the integrated area would be from noise alone.
                    # Per-pixel scatter grows as sqrt(N) when summed, so
                    # anything much larger than this is real signal -- and its
                    # SIGN says which frame the atoms were in.
                    noise_area = float(diff.std()) * np.sqrt(diff.size)
                    # Area alone is not evidence of atoms -- see
                    # localized_signal.
                    loc = localized_signal(diff)
                    lo, hi = cur_levels
                    logger.info(
                        f"    raw f1={float(frame1.mean()):.0f} "
                        f"f2={float(frame2.mean()):.0f} "
                        f"peak={run_peak}/{limit} | "
                        f"diff min={float(diff.min()):.0f} p0.05={d_lo:.1f} "
                        f"med={float(np.median(diff)):.1f} "
                        f"p99.95={d_hi:.1f} max={float(diff.max()):.0f} | "
                        f"area={area:.3e} "
                        f"({area / max(noise_area, 1):+.0f}sig, "
                        f"mean {area / diff.size:+.3f}/px) | "
                        + (f"cloud {loc['peak'] - loc['background']:+.3f}/px "
                           f"({loc['excess_sigma']:+.1f}sig) at "
                           f"({loc['x']},{loc['y']}) | " if loc else "")
                        + f"levels=({lo:.0f},{hi:.0f}) "
                        + ('auto' if autoscale else 'manual'))

                    # An area that is huge but spatially flat is a black-level
                    # shift between the two frames, not atoms. Say so
                    # explicitly -- the raw sigma on the area is the single
                    # most misleading number here.
                    if loc is not None and abs(area) > 5 * noise_area \
                            and loc['excess_sigma'] < 5 and 'flat' not in warned:
                        warned.add('flat')
                        logger.warn(
                            f"Integrated area is "
                            f"{area / max(noise_area, 1):+.0f} sigma but the "
                            f"excess is UNIFORM: the brightest "
                            f"{loc['block_px']}-pixel block is only "
                            f"{loc['excess_sigma']:+.1f} sigma above the rest "
                            f"of the frame. That is a "
                            f"{area / diff.size:+.3f} count/px baseline "
                            f"difference between frame 1 and frame 2, not a "
                            f"cloud. Ignore the area sigma; it counts a "
                            f"sub-ADU pedestal across {diff.size / 1e6:.1f}M "
                            f"pixels as signal.")

                    # Is the sensor seeing light at all? Compare against the
                    # pedestal, not against zero: this camera's floor sits
                    # around 1180 counts.
                    if run_peak is not None and limit \
                            and run_peak < 0.02 * limit:
                        if 'dark' not in warned:
                            warned.add('dark')
                            logger.warn(
                                f"Sensor is nearly dark: brightest raw pixel "
                                f"{run_peak} of {limit}. No light is reaching "
                                f"the camera -- check the imaging pulse, "
                                f"exposure and trigger, not the display.")
                    # A cloud in frame 2 rather than frame 1 integrates
                    # NEGATIVE. The percentile tails cannot see this -- read
                    # noise makes them nearly symmetric either way -- but the
                    # area cannot miss it.
                    elif area < -5.0 * noise_area:
                        if 'negative' not in warned:
                            warned.add('negative')
                            logger.warn(
                                f"Signal is NEGATIVE: area={area:.3e} "
                                f"({area / max(noise_area, 1):.0f} sigma below "
                                f"zero). Frame 2 is brighter than frame 1, so "
                                f"the frame pair is swapped and the atoms are "
                                f"being subtracted the wrong way round. "
                                f"Nothing shows because the cloud sits below "
                                f"Display Min.")
                    # No signal either way -- the display is not the problem.
                    elif abs(area) < 3.0 * noise_area:
                        if 'nosignal' not in warned:
                            warned.add('nosignal')
                            logger.warn(
                                f"No signal: area={area:.3e} is within noise "
                                f"({noise_area:.1e}). The two frames are "
                                f"identical to within read noise, so there "
                                f"are no atoms in frame 1.")
                    # A range that cannot show the data is the commonest
                    # 'all dark'.
                    elif not autoscale and d_hi > 0 and hi > 10.0 * d_hi:
                        if 'range_high' not in warned:
                            warned.add('range_high')
                            logger.warn(
                                f"Display Max {hi:.0f} is {hi / d_hi:.0f}x the "
                                f"actual signal (p99.95={d_hi:.1f}); "
                                f"everything renders near the bottom of the "
                                f"colour range. Try Display Max ~{d_hi:.0f}.")
                    elif not autoscale and d_hi > hi:
                        if 'range_low' not in warned:
                            warned.add('range_low')
                            logger.warn(
                                f"Signal (p99.95={d_hi:.1f}) exceeds Display "
                                f"Max {hi:.0f}; the cloud is clipped. Try "
                                f"Display Max ~{d_hi * 1.2:.0f}.")

                # Lifetime fit on the ATOM NUMBER, which is what decays.
                # The old version fitted the raw whole-frame sum, and that sum
                # is dominated by the pedestal difference between the two
                # frames across five million pixels -- a quantity with no
                # reason to follow the atoms at all. n_atom_vals comes from
                # the fitted 2w ellipse with the background taken from outside
                # it, so what is fitted here is the cloud.
                res = _exp_decay_fit(hold_values_ms, n_atom_vals)
                if res is not None and np.isfinite(res['tau_ms']):
                    logger.info(
                        f"  >> Lattice lifetime tau = {res['tau_ms']:.1f} ms "
                        f"(N0 = {res['n0']:.3e}, "
                        f"{int(res['mask'].sum())}/{n_scan} points)")

                time.sleep(wait_time)
                logger.info(f"Updated plots for run {run_idx+1} of {M}")
        except StopRequested:
            logger.info(
                "Stop requested -- ending this cycle here instead of "
                "finishing the remaining runs and scan points.")
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
        cycle_idx += 1
        # Sequence tasks are closed now, so static writes can land.
        # The lattice is deliberately left ON and stationary; see
        # park_outputs() for why parking it off would cost coupling.
        park_outputs(NI_card_2, NI_card_3, logger, lattice_am=lat_am_v)
