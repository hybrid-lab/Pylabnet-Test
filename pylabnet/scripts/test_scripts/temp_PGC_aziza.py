# TOF sweep with temperature measurement.
# Full sequence: MOT -> CMOT -> [Dark MOT] -> [PGC] -> release -> TOF -> image
# Dark MOT and PGC independently toggleable.
# 6 TOF points, sigma^2 vs t^2 linear fit for temperature.
#
# Sequence per scan point:
#   1. MOT loading (full power, coils on)
#   2a. CMOT ramp (10ms hardcoded)
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
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
import time
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

try:
    from scipy.optimize import curve_fit
except ImportError:      # fall back to moments; see fit_gaussian_1d
    curve_fit = None

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

# Local gravity at Berkeley. International Gravity Formula 1980,
#   g = 9.780327 (1 + 0.0053024 sin^2(lat) - 0.0000058 sin^2(2 lat))
# at lat 37.8716 N gives 9.79981 m/s^2, less the free-air correction
# 3.086e-6 * height for ~100 m of elevation -> 9.7996 m/s^2. That is 0.02%
# from the 9.80665 standard value, which matters here: g sets the pixel
# scale, and the scale enters the temperature SQUARED.
G_BERKELEY = 9.7996          # m/s^2

M_RB87 = 1.4431e-25   # kg
K_B = 1.3806e-23       # J/K


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


# Fallbacks only -- the live range comes from the Display Min/Max GUI
# fields below, in absolute camera counts, exactly as in jingfastMOT.
DIFF_LEVELS = (0, 20)
AVG_DIFF_LEVELS = (0, 20)

# Brightest a single pixel can report on this camera. Mono16 is 12-bit data
# left-shifted by four plus a 12 count pedestal, so the ceiling is
# 16*4095 + 12, NOT the 65535 the container suggests. A typed display range
# is clamped to +/- this, so the scale can never open on an impossible value.
SENSOR_MAX_MONO16 = 16 * 4095 + 12   # 65532

# Percentile the autoscaled top follows, for BOTH grid rows. Not the max: a
# single hot pixel would otherwise set the scale and push the cloud back into
# the dark end of the colour map.
AUTOSCALE_PCT = 99.95

# Autoscale is measured on every 4th row and column. The top row is re-levelled
# on every frame pair, and a percentile over all 5.0 M pixels is a full sort
# each time; 313 k pixels give the same answer to well under a count.
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
        # Free-fall panel. The cloud centre falls y = y0 + v0*t + g t^2/2.
        # Everything on the left of that is in metres and everything
        # measured is in pixels, so fitting the parabola in PIXELS gives
        # the quadratic coefficient a2 = g / (2 * pixel_size), and g is
        # known -- which turns the fit into a direct measurement of the
        # pixel scale, with no assumed magnification anywhere in it.
        self.plot_c = self.win.addPlot(
            row=0, col=2, title="<b>cloud centre vs t (free fall)</b>")
        self.plot_c.setLabel('bottom', 't', units='s')
        self.plot_c.setLabel('left', 'centre (px)')
        # Both axes are parabolas: gravity has a component along each,
        # and on this setup most of it is along x.
        self.plot_c.showGrid(x=True, y=True, alpha=0.3)
        self.data_cy = self.plot_c.plot(
            [], [], pen=None, symbol='o', symbolSize=9, symbolBrush='#4CAF50')
        self.fit_cy = self.plot_c.plot([], [], pen=pg.mkPen('#F44336', width=2))
        self.data_cx = self.plot_c.plot(
            [], [], pen=None, symbol='s', symbolSize=8, symbolBrush='#2196F3')
        self.fit_cx = self.plot_c.plot(
            [], [], pen=pg.mkPen('#90CAF9', width=2, style=QtCore.Qt.DashLine))
        self.label_c = pg.TextItem('', color='#F44336', anchor=(0, 1))
        self.plot_c.addItem(self.label_c)
        # Pixel scale measured from gravity; None until enough points land.
        # experiment() reads this for its own log so the two agree.
        self.measured_pixel_size_m = None
        self._latest_data = None
        self._rendered_version = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll_and_update)
        self._timer.start(self.POLL_MS)

    def update_data(self, tof_us_list, sigma_x_list, sigma_y_list,
                    cx_list=None, cy_list=None):
        n = len(tof_us_list)
        nan = [float('nan')] * n
        self._latest_data = (list(tof_us_list), list(sigma_x_list),
                             list(sigma_y_list),
                             list(cx_list) if cx_list is not None else nan,
                             list(cy_list) if cy_list is not None else nan,
                             self._rendered_version + 1)

    def _poll_and_update(self):
        entry = self._latest_data
        if entry is None:
            return
        tof_us, sx_px, sy_px, cx_px, cy_px, version = entry
        if version <= self._rendered_version:
            return
        self._rendered_version = version
        try:
            t_s = np.array(tof_us, dtype=np.float64) * 1e-6
            t2 = t_s ** 2

            # ---- free fall in 2D: measure the pixel scale from gravity ---
            # Gravity is NOT along a sensor axis on this setup -- it is mostly
            # along x. Fitting y alone would measure only g_y = g cos(theta)
            # and report a pixel scale too large by 1/cos(theta), which for
            # "basically all x" would be wildly wrong.
            #
            # So fit BOTH centres as parabolas and take the MAGNITUDE of the
            # acceleration vector. |a| = g / pixel_size holds whatever the
            # orientation, and the direction is a free by-product that says
            # how the camera is rotated relative to gravity.
            cy = np.array(cy_px, dtype=np.float64)
            cx = np.array(cx_px, dtype=np.float64)
            mc = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(t_s)
            px_m, mag = None, float('nan')
            v0x = v0y = v0 = theta = float('nan')
            if mc.sum() >= 3:
                # c(t) = q t^2 + l t + c0, so the acceleration is 2q px/s^2.
                qx, lx, x0f = np.polyfit(t_s[mc], cx[mc], 2)
                qy, ly, y0f = np.polyfit(t_s[mc], cy[mc], 2)
                ax_px, ay_px = 2.0 * qx, 2.0 * qy
                a_mag = float(np.hypot(ax_px, ay_px))
                if np.isfinite(a_mag) and a_mag > 0:
                    px_m = G_BERKELEY / a_mag
                    mag = (PIXEL_PITCH_UM * 1e-6) / px_m
                    # Direction of gravity on the sensor. 0 deg = +x (columns),
                    # 90 deg = +y (rows). Tells you the camera rotation.
                    theta = float(np.degrees(np.arctan2(ay_px, ax_px)))
                    v0x, v0y = lx * px_m, ly * px_m
                    v0 = float(np.hypot(v0x, v0y))
                    self.measured_pixel_size_m = px_m
                tf = np.linspace(0, t_s[mc].max() * 1.05, 60)
                self.data_cx.setData(t_s[mc], cx[mc])
                self.fit_cx.setData(tf, qx * tf ** 2 + lx * tf + x0f)
                self.data_cy.setData(t_s[mc], cy[mc])
                self.fit_cy.setData(tf, qy * tf ** 2 + ly * tf + y0f)
            if px_m is not None:
                self.label_c.setText(
                    'measured %.4f um/px\nmagnification %.3f\n'
                    'g at %+.1f deg from +x\n'
                    '|v0| = %.1f mm/s  (%+.1f, %+.1f)\n(assumed %.4f um/px)'
                    % (px_m * 1e6, mag, theta, v0 * 1e3, v0x * 1e3, v0y * 1e3,
                       self.pixel_size_m * 1e6))
                self.label_c.setPos(
                    0.0, float(np.nanmax(np.concatenate([cx[mc], cy[mc]]))))

            # The temperature uses the MEASURED scale when gravity has
            # given us one -- that is the point of the free-fall fit. The
            # configured value is only the fallback.
            scale = px_m if px_m is not None else self.pixel_size_m
            sx_m = np.array(sx_px, dtype=np.float64) * scale
            sy_m = np.array(sy_px, dtype=np.float64) * scale
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
                    self.label_x.setText(
                        f"T_x = {T_x*1e6:.1f} uK\n"
                        f"sigma_0x = {np.sqrt(max(a_x,0))/scale:.0f} px\n"
                        f"scale {scale*1e6:.4f} um/px "
                        f"({'gravity' if px_m is not None else 'assumed'})")
                    self.label_x.setPos(t2.max() * 0.05, sx2[mask_x].max() * 0.95)
            if mask_y.sum() >= 2:
                self.data_y.setData(t2[mask_y], sy2[mask_y])
                a_y, b_y = _linear_fit(t2[mask_y], sy2[mask_y])
                if np.isfinite(b_y) and b_y > 0:
                    T_y = M_RB87 * b_y / K_B
                    t2_fit = np.linspace(0, t2.max() * 1.1, 50)
                    self.fit_y.setData(t2_fit, a_y + b_y * t2_fit)
                    self.label_y.setText(
                        f"T_y = {T_y*1e6:.1f} uK\n"
                        f"sigma_0y = {np.sqrt(max(a_y,0))/scale:.0f} px\n"
                        f"scale {scale*1e6:.4f} um/px "
                        f"({'gravity' if px_m is not None else 'assumed'})")
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
    # MOT -> CMOT ramp duration. Was hardcoded at 10 ms (CMOT_RAMP_MS),
    # which made it impossible to match jingfastPGC without editing the file.
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

    # ===================== PHASE 4: PGC (optical molasses) =====================
    # Sub-Doppler cooling: no B-field, very far red-detuned, low power.
    # Sisyphus effect cools atoms well below Doppler limit.
    'pgc_enabled': {'PGC Enabled (1=yes, 0=no)': '1'},
    'pgc_duration': {'PGC Duration (ms)': '5'},
    'pgc_mot_power': {'PGC Cooling Power (V)': '0.2'},
    'pgc_detuning': {'PGC Cooling VCO (V) [-122MHz]': '-2.3'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.03'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '-0.05'},

    # ===================== PHASE 5: TOF (swept) =====================
    'tof_1_us': {'TOF 1 (us)': '9000'},
    'tof_2_us': {'TOF 2 (us)': '12000'},
    'tof_3_us': {'TOF 3 (us)': '15000'},
    'tof_4_us': {'TOF 4 (us)': '18000'},
    'tof_5_us': {'TOF 5 (us)': '21000'},
    'tof_6_us': {'TOF 6 (us)': '24000'},

    # ===================== IMAGING =====================
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'coils_rampback_ms': {'Coils Rampback Duration (ms)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '100'},

    # ---------- Imaging repump prepulse (same as jingfastPGC) ----------
    # Repump fires alone for this long before the MOT light joins it, so the
    # atoms are pumped up out of F=1 and are all cycling by the time the
    # camera opens. Both frames get the identical sequence -- the subtraction
    # only works if frame 2 saw exactly the same light as frame 1. The repump
    # then STAYS on at these settings through the MOT pulse; turning it off
    # there would let atoms fall dark part-way through the exposure. Set the
    # time to 0 to go back to repump and MOT light starting together.
    'img_repump_time_us': {'Imaging Repump Lead Time (us)': '200'},
    'img_repump_power': {'Imaging Repump Power (V)': '0.8'},
    'img_repump_vco': {'Imaging Repump VCO (V)': '0.06'},

    # ===================== CALIBRATION =====================
    # OBJECT-SPACE pixel size, i.e. what one pixel covers AT THE ATOMS:
    # sensor pitch divided by the magnification. Using the bare 3.45 um
    # sensor pitch here (as this field used to) scales every sigma by the
    # magnification and the temperature by its SQUARE -- at 0.59 that
    # UNDERSTATES T by a factor of ~2.9. Seeded from PIXEL_UM so it cannot
    # disagree with the density calculation, which uses the same scale.
    'pixel_size_um': {'Pixel Size at the Atoms (um/px)': '%.4f' % PIXEL_UM},

    # ===================== BIAS COILS =====================
    # 0V during MOT+CMOT; linearly ramped to these values starting at
    # coils_off (just before dark MOT), held through both imaging frames,
    # then ramped back to 0V after frame 2.
    'pgc_bias_x_v': {'PGC Bias X (V)': '0.0'},
    'pgc_bias_y_v': {'PGC Bias Y (V)': '0.0'},
    'pgc_bias_z_v': {'PGC Bias Z (V)': '0.0'},
    'bias_ramp_ms': {'Bias Ramp Duration (ms)': '5'},

    # ===================== GENERAL =====================
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},

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
    'display_max': {'Display Max (counts)': '2000'},

    # ===================== WARM-UP =====================
    # Cycles run before any data is kept, at the FIRST scan point's
    # settings. Between one arming and the next the coils are off and the
    # light is down for seconds while the waveform is finalized, uploaded
    # and re-armed, so the MOT is completely gone. The first sub-sequence
    # after that loads from ZERO atoms, while every later one only tops up
    # a MOT that never fully decayed -- which is why the first point or two
    # come out different. These cycles absorb that transient. Set to 0 to
    # keep every shot; each one costs a full sub-sequence of run time.
    'warmup_cycles': {'Warm-up Cycles (0 = none)': '3'},
}


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")
    dataset.camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    tof_values = [
        int(dataset.get_input_parameter("tof_1_us")),
        int(dataset.get_input_parameter("tof_2_us")),
        int(dataset.get_input_parameter("tof_3_us")),
        int(dataset.get_input_parameter("tof_4_us")),
        int(dataset.get_input_parameter("tof_5_us")),
        int(dataset.get_input_parameter("tof_6_us")),
    ]
    scan_labels = [f"TOF={t}us" for t in tof_values]
    # Absolute camera counts, clamped to what a pixel can physically reach.
    disp_min = float(dataset.get_input_parameter("display_min"))
    disp_max = float(dataset.get_input_parameter("display_max"))
    disp_min = max(disp_min, -float(SENSOR_MAX_MONO16))
    disp_max = min(disp_max, float(SENSOR_MAX_MONO16))
    autoscale = bool(int(dataset.get_input_parameter("display_autoscale")))
    dataset.scan_grid = ScanGridWindow(
        scan_labels=scan_labels,
        autoscale=autoscale,
        exposure_us=float(dataset.get_input_parameter("camera_exposure_us")),
        diff_levels=(disp_min, disp_max), avg_levels=(disp_min, disp_max),
        title="TOF sweep + Temp (MOT->CMOT->[DarkMOT]->[PGC]->image)",
    )
    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))
    dataset.temp_fit = TempFitWindow(pixel_size_um)
    dataset.graph.hide()


def experiment(**kwargs):
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
    mot_switch_ms = int(dataset.get_input_parameter("mot_switch_time"))
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

    # Phase 4: PGC (toggleable)
    pgc_enabled = int(dataset.get_input_parameter("pgc_enabled")) != 0
    pgc_ms = int(dataset.get_input_parameter("pgc_duration"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))

    # TOF sweep
    tof_values_us = [
        int(dataset.get_input_parameter("tof_1_us")),
        int(dataset.get_input_parameter("tof_2_us")),
        int(dataset.get_input_parameter("tof_3_us")),
        int(dataset.get_input_parameter("tof_4_us")),
        int(dataset.get_input_parameter("tof_5_us")),
        int(dataset.get_input_parameter("tof_6_us")),
    ]
    n_scan = len(tof_values_us)
    pixel_size_um = float(dataset.get_input_parameter("pixel_size_um"))

    # Imaging
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))
    coils_rampback_ms = int(dataset.get_input_parameter("coils_rampback_ms"))
    img_repump_us = int(dataset.get_input_parameter("img_repump_time_us"))
    img_rep_power = float(dataset.get_input_parameter("img_repump_power"))
    img_rep_vco = float(dataset.get_input_parameter("img_repump_vco"))
    wait_time = float(dataset.get_input_parameter("wait_time"))
    warmup_cycles = max(0, int(dataset.get_input_parameter("warmup_cycles")))

    pgc_bias_x_v = float(dataset.get_input_parameter("pgc_bias_x_v"))
    pgc_bias_y_v = float(dataset.get_input_parameter("pgc_bias_y_v"))
    pgc_bias_z_v = float(dataset.get_input_parameter("pgc_bias_z_v"))
    bias_ramp_ms = int(dataset.get_input_parameter("bias_ramp_ms"))

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
    cmot_ramp_s = ms2s(mot_switch_ms)
    cmot_hold_s = ms2s(cmot_hold_ms)
    dark_s = ms2s(dark_ms) if dark_enabled else 0
    pgc_s = ms2s(pgc_ms) if pgc_enabled else 0
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    bias_ramp_s = ms2s(bias_ramp_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))
    coils_rampback_s = ms2s(coils_rampback_ms)
    # Repump-only prepulse, then the MOT pulse. One sample is 50 us at
    # 20 kS/s, so anything under 25 us rounds away -- 200 us is 4 samples.
    img_rep_s = us2s(img_repump_us)
    img_win_s = img_rep_s + aom_pulse_s     # whole light window per frame

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
    pgc_end = pgc_start + pgc_s

    release = pgc_end

    # Coils OFF: before Dark MOT if it's enabled, else before PGC, else before release
    # Dark MOT and PGC are both field-free phases
    if dark_enabled:
        coils_off = cmot_end - coils_lead_s
    elif pgc_enabled:
        coils_off = pgc_start - coils_lead_s
    else:
        coils_off = release - coils_lead_s

    dark_str = "ON" if dark_enabled else "OFF"
    pgc_str = "ON" if pgc_enabled else "OFF"
    logger.info(f"Time at start {time.perf_counter_ns()}")
    logger.info(
        f"Timeline: MOT 0-{mot_loading_ms}ms, "
        f"CMOT {mot_loading_ms}-{cmot_end/SPM:.0f}ms, "
        f"Dark MOT {dark_str} ({dark_start/SPM:.0f}-{dark_end/SPM:.0f}ms), "
        f"PGC {pgc_str} ({pgc_start/SPM:.0f}-{pgc_end/SPM:.0f}ms), "
        f"coils OFF at {coils_off/SPM:.0f}ms, "
        f"release at {release/SPM:.0f}ms, "
        f"TOF sweep={tof_values_us}us"
    )

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence(tof_us):
        tof_s = us2s(tof_us)
        # f1/f2 are where the LIGHT starts (repump first). The camera fires
        # img_rep_s later, when the MOT light joins -- the prepulse produces
        # almost no fluorescence, so an exposure covering only it would come
        # back empty. Note the repump lead adds to the expansion time, so
        # the effective TOF for the temperature fit is tof + img_rep_s.
        f1 = release + tof_s
        f2 = f1 + clear_s
        img_light_1 = f1 + img_rep_s        # MOT light + camera, frame 1
        img_light_2 = f2 + img_rep_s        # MOT light + camera, frame 2
        crt = f2 + img_win_s + coils_restore_s  # coils stay OFF through frame 2
        N = crt + coils_rampback_s + 2

        cam = [0] * N
        for s in range(ttl_up):
            cam[img_light_1 + s] = 1
            cam[img_light_2 + s] = 1

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

        # Phase 4: PGC (if enabled) -- very far detuned, low power, no B-field
        if pgc_enabled:
            for s in range(pgc_s):
                t = pgc_start + s
                ado[t] = 1
                aao[t] = pgc_mot_power
                rdo[t] = 1
                rao[t] = pgc_rep_power

        # TOF: digital off, analog stays at full power (thermal stability)
        for s in range(release, min(f1, N)):
            ado[s] = 0
            rdo[s] = 0

        # Imaging: repump alone for img_rep_s, then the MOT light joins it.
        # Repump holds its imaging power for the whole window rather than
        # handing back to repump_aom_power, so nothing changes about the
        # light mid-exposure.
        for s in range(img_win_s):
            for base in (f1, f2):
                t = base + s
                if t >= N:
                    continue
                rdo[t] = 1
                rao[t] = img_rep_power
                if s >= img_rep_s:      # prepulse over -- cooling light on
                    ado[t] = 1
                    aao[t] = mot_aom_power

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
        # Rampback: analog 0 -> mot_coils_v over coils_rampback_s, so the
        # quadrupole comes back softly instead of stepping to full current.
        for s in range(coils_rampback_s):
            t = crt + s
            if t >= N:
                break
            frac = s / max(coils_rampback_s - 1, 1)
            cao[t] = frac * mot_coils_v
        for s in range(crt + coils_rampback_s, N):
            cao[s] = mot_coils_v

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
        # PGC
        if pgc_enabled:
            for s in range(pgc_s):
                mvco[pgc_start + s] = pgc_detuning
        # After last phase: fill with last detuning
        if pgc_enabled:
            last_det = pgc_detuning
            last_end = pgc_end
        elif dark_enabled:
            last_det = dark_detuning
            last_end = dark_end
        else:
            last_det = cmot_detuning
            last_end = cmot_end
        for s in range(last_end, N):
            mvco[s] = last_det
        # Imaging VCO override. The lead is measured from the MOT AOM
        # opening, which is now img_rep_s after the repump's, so the cooling
        # VCO still gets its full settling time.
        vlo = max(0, img_light_1 - vco_lead_s)
        vhi = min(N, f2 + img_win_s)
        for s in range(vlo, vhi):
            mvco[s] = vco_imaging

        # Repump VCO
        rvco = [repump_vco] * N
        # CMOT ramp
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            rvco[t] = repump_vco + frac * (cmot_repump_vco - repump_vco)
        # CMOT hold
        for s in range(cmot_hold_start, cmot_end):
            rvco[s] = cmot_repump_vco
        # Dark MOT
        if dark_enabled:
            for s in range(dark_s):
                rvco[dark_start + s] = dark_rep_vco
        # PGC
        if pgc_enabled:
            for s in range(pgc_s):
                rvco[pgc_start + s] = pgc_rep_vco
        # After last phase
        if pgc_enabled:
            last_rvco = pgc_rep_vco
        elif dark_enabled:
            last_rvco = dark_rep_vco
        else:
            last_rvco = cmot_repump_vco
        for s in range(last_end, N):
            rvco[s] = last_rvco
        # Imaging repump VCO. Gets the same lead as the cooling VCO,
        # referenced to the repump AOM opening at f1 -- it has to be settled
        # before the prepulse, which is the whole point of firing early.
        rvlo = max(0, f1 - vco_lead_s)
        rvhi = min(N, f2 + img_win_s)
        for s in range(rvlo, rvhi):
            rvco[s] = img_rep_vco

        # Bias coils: 0V during MOT+CMOT, ramp UP to PGC values at coils_off,
        # hold through both imaging frames, then ramp DOWN to 0V after F2.
        bias_down_start = f2 + img_win_s
        biasx = [0.0] * N
        biasy = [0.0] * N
        biasz = [0.0] * N
        # Ramp UP at coils_off
        for s in range(bias_ramp_s):
            t = coils_off + s
            if t >= N:
                break
            frac = s / max(bias_ramp_s - 1, 1)
            biasx[t] = frac * pgc_bias_x_v
            biasy[t] = frac * pgc_bias_y_v
            biasz[t] = frac * pgc_bias_z_v
        # Hold at PGC values from end of up-ramp through F2
        for s in range(coils_off + bias_ramp_s, min(bias_down_start, N)):
            biasx[s] = pgc_bias_x_v
            biasy[s] = pgc_bias_y_v
            biasz[s] = pgc_bias_z_v
        # Ramp DOWN to 0V after F2
        for s in range(bias_ramp_s):
            t = bias_down_start + s
            if t >= N:
                break
            frac = 1.0 - s / max(bias_ramp_s - 1, 1)
            biasx[t] = frac * pgc_bias_x_v
            biasy[t] = frac * pgc_bias_y_v
            biasz[t] = frac * pgc_bias_z_v

        return cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, biasx, biasy, biasz, N

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
    # Warm-up first, at the first point's settings, then the real sweep.
    # They are part of the SAME regenerating buffer, so they run at the head
    # of every pass and the MOT is always in its cyclic steady state by the
    # time point 1 is recorded. Their frames are discarded in the
    # acquisition loop below.
    warm_values = [tof_values_us[0]] * warmup_cycles
    for tof_us in warm_values + tof_values_us:
        cam, ado, aao, rdo, rao, cdo, cao, mvco, rvco, bx_w, by_w, bz_w, N = build_sub_sequence(tof_us)
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

    total = len(cam_p)
    logger.info(
        f"TOF sweep+temp: tof={tof_values_us}us, "
        f"{warmup_cycles} warm-up cycle(s) at {tof_values_us[0]}us, "
        f"total={total} samples ({total / SPM / 1000:.2f} s per pass)")

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
        # Gain: request it, then READ IT BACK and use whatever the camera
        # actually applied. try_set_float is best-effort -- a value past
        # the node maximum raises inside SetValue, the driver swallows it,
        # and the camera keeps what it had. Assuming the requested value
        # instead makes every atom number wrong by 10^(dG/20), and is
        # exactly why N here disagreed with jingfastPGC, which has always
        # used the measured value.
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
        # dataset.scan_grid, not the local `scan_grid` -- that name is not
        # bound until further down the function.
        dataset.scan_grid.set_gain_db(gain_db)
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
        NI_card_2.set_do_voltage(do_channel="dio4", value=[1] * total, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio5", value=[1] * total, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio6", value=[0] * total, sample_rate=ni_rate)
        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel=MOT_AOM_ao, voltages=aao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=mot_coils_ao, voltages=cao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=MOT_freq_VCO_ao, voltages=mvco_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_freq_VCO_ao, voltages=rvco_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=Repump_AOM_ao, voltages=rao_p, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=[0.0] * total, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=[0.0] * total, sample_rate=ni_rate)
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

        scan_grid = dataset.scan_grid
        temp_fit = dataset.temp_fit
        sigma_x_vals = [float('nan')] * n_scan
        sigma_y_vals = [float('nan')] * n_scan
        centre_x_vals = [float('nan')] * n_scan
        centre_y_vals = [float('nan')] * n_scan

        # Stop responsiveness. thread.running is what the GUI Stop button
        # clears, and it used to be tested ONLY by the outer while loop.
        # Everything below -- M passes x n_scan points x 2 blocking reads --
        # ran to completion regardless, so Stop took at minimum a full set
        # of passes, and if frames had stopped arriving it took up to
        # 2*n_scan*M reads x 100 s of timeout before thread.running was
        # looked at again. It is now tested at every level.
        #
        # The timeout is derived from the sequence instead of a flat 100 s:
        # the longest legitimate gap between frames is one sub-sequence, so
        # anything past a few of those is a dead trigger, not a slow one.
        # N is one sub-sequence. total/n_scan would be wrong now that the
        # buffer also carries the warm-up cycles.
        sub_ms = N / SPM
        frame_timeout_ms = int(3 * sub_ms + 2000)
        first_timeout_ms = max(15000, frame_timeout_ms)
        logger.info(
            f"Sub-sequence {sub_ms:.0f} ms; frame timeout "
            f"{frame_timeout_ms} ms (first {first_timeout_ms} ms)")

        def grab(timeout_ms):
            """One frame, or None if it timed out or Stop was pressed."""
            if not thread.running:
                return None
            try:
                return get_frame(timeout_ms=timeout_ms)
            except Exception as exc:
                if thread.running:
                    logger.warn(
                        f"Frame timed out after {timeout_ms} ms ({exc}); "
                        f"abandoning this pass and re-arming.")
                return None

        stopping = False
        try:
            for run_idx in range(M):
                if not thread.running:
                    stopping = True
                    break
                logger.info(f"Run {run_idx+1} of {M}")
                # The warm-up cycles sit at the head of the buffer and fire
                # their camera triggers like any other cycle, so their
                # frames must be consumed -- otherwise every pair after them
                # would be offset by one.
                for _ in range(2 * warmup_cycles):
                    if grab(frame_timeout_ms) is None:
                        stopping = True
                        break
                if stopping:
                    break
                for scan_idx, tof_v in enumerate(tof_values_us):
                    if not thread.running:
                        stopping = True
                        break
                    is_first = (run_idx == 0 and scan_idx == 0)
                    frame1 = grab(first_timeout_ms if is_first
                                  else frame_timeout_ms)
                    if frame1 is None:
                        stopping = True
                        break
                    frame2 = grab(frame_timeout_ms)
                    if frame2 is None:
                        stopping = True
                        break
                    diff = frame1.astype(np.int32) - frame2.astype(np.int32)
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
                    fx = fit_gaussian_1d(img.sum(axis=0))
                    fy = fit_gaussian_1d(img.sum(axis=1))
                    good = bool(fx and fy and fx['reliable'] and fy['reliable'])
                    # sigma, NOT w -- see the note in _poll_and_update.
                    sx = fx['sigma'] if fx else float('nan')
                    sy = fy['sigma'] if fy else float('nan')
                    # Only trustworthy widths reach the temperature fit. A
                    # width fitted to noise is not a wide cloud, and letting
                    # one into the sigma^2-vs-t^2 line corrupts the slope --
                    # i.e. the temperature -- for the whole scan. NaN keeps
                    # the point out; _linear_fit already masks non-finite.
                    sigma_x_vals[scan_idx] = sx if good else float('nan')
                    sigma_y_vals[scan_idx] = sy if good else float('nan')
                    # Centres feed the free-fall fit. Kept on the same
                    # reliability condition: a centre from a fit that found
                    # noise is not a cloud position.
                    centre_x_vals[scan_idx] = (fx['center'] if good
                                               else float('nan'))
                    centre_y_vals[scan_idx] = (fy['center'] if good
                                               else float('nan'))
                    logger.info(
                        f"  TOF={tof_v}us: n={diff_counts[scan_idx]}, "
                        f"sx={sx:.1f}px, sy={sy:.1f}px, "
                        f"snr=({fx['snr'] if fx else 0:.0f},"
                        f"{fy['snr'] if fy else 0:.0f}), "
                        f"area={float(avg.sum()):.0f}"
                        + ("" if good else "   [UNRELIABLE - excluded from T fit]"))

                # Update temperature fit after each full sweep
                temp_fit.update_data(tof_values_us, sigma_x_vals, sigma_y_vals,
                                     centre_x_vals, centre_y_vals)
                # Prefer the scale gravity measured, so this log and the
                # plot window cannot report two different temperatures.
                pxm = temp_fit.measured_pixel_size_m or (pixel_size_um * 1e-6)
                t_s = np.array(tof_values_us, dtype=np.float64) * 1e-6
                sx_m = np.array(sigma_x_vals) * pxm
                sy_m = np.array(sigma_y_vals) * pxm
                mask_x = np.isfinite(sx_m) & (sx_m > 0)
                mask_y = np.isfinite(sy_m) & (sy_m > 0)
                if mask_x.sum() >= 2:
                    _, bx = _linear_fit(t_s[mask_x]**2, sx_m[mask_x]**2)
                    if np.isfinite(bx) and bx > 0:
                        logger.info(f"  >> T_x = {M_RB87*bx/K_B*1e6:.1f} uK")
                if mask_y.sum() >= 2:
                    _, by = _linear_fit(t_s[mask_y]**2, sy_m[mask_y]**2)
                    if np.isfinite(by) and by > 0:
                        logger.info(f"  >> T_y = {M_RB87*by/K_B*1e6:.1f} uK")

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
