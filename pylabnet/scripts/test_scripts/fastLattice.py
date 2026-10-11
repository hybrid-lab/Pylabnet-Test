# Fast single image: MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown -> release -> TOF -> image
#
# LATTICE TOGGLE ('Lattice On? (1=yes, 0=no)' on the GUI):
#   ON  -- dio4/dio5 held LOW and ao6/ao7 held at Lattice Ramp End for the
#          ENTIRE cycle, so the AOM crystal sees constant RF drive (stable
#          thermal state -> stable fiber coupling). The Lattice Hold phase
#          between PGC and release is honoured.
#   OFF -- dio4/dio5 held HIGH and ao6/ao7 at 0 V for the entire cycle, exactly
#          as the no-lattice PGC script does. Lattice Hold is FORCED to 0 ms:
#          with no trap it would just be extra dark expansion before release,
#          silently lengthening the effective TOF. Set PGC Rampdown to 0 as
#          well to reproduce the PGC script's timeline sample for sample.
#
# Two-stage PGC:
#   PGC_cool (5ms, default): full PGC light -- sub-Doppler cooling
#   PGC_rampdown (10ms, default): PGC cooling + repump amplitude ramp to 0.
#     With the lattice on this is the adiabatic handoff from molasses into the
#     lattice trap; with it off it is simply a soft extinction of the light.
#
# Both up (COM4/dio4/ao6) and down (COM5/dio5/ao7) lattices follow identically.
# NB: DIM-3000 max amplitude assumed AMP=320 (+32 dBm) via serial setup;
#     ao6/ao7 at +1.0V => max scaling, at -1.0V => parked off.

import time
import dim_state

import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

try:
    import serial
except ImportError:      # no pyserial -> the DIM reset is skipped, loudly
    serial = None

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
# left-shifted by four plus a 12 count pedestal, so the ceiling is 16*4095 + 12,
# NOT the 65535 the container suggests. Used only to warn when the ADC itself
# has clipped -- at that point the display range is irrelevant and the fix is
# less gain or a shorter exposure.
SENSOR_MAX_MONO16 = 16 * 4095 + 12   # 65532

# Where saturated pixels ACTUALLY pile up on this sensor, which is one ADU
# below the arithmetic ceiling. Testing against 65532 never fires and a
# clipped frame then reports as healthy -- the exact failure that made a
# saturated image look like a display problem.
SATURATION_COUNTS = 65516

INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '800'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '6'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0'},
    'Repump_VCO': {'Repump VCO (V)': '0'},

    # MOT -> CMOT ramp duration. Was hardcoded at 10 ms (CMOT_RAMP_MS), which
    # made it impossible to match temp_PGC_aziza without editing the file.
    'mot_switch_time': {'MOT Switch Time (ms) [MOT->CMOT ramp]': '15'},

    # ===================== PHASE 2: CMOT =====================
    'cmot_hold_time': {'CMOT Hold Time (ms)': '5'},
    'cmot_mot_power': {'CMOT MOT AOM Power (V)': '0.6'},
    'cmot_repump_power': {'CMOT Repump AOM Power (V)': '0.2'},
    'cmot_detuning_voltage': {'CMOT MOT VCO (V) [-40MHz]': '-0.6'},
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '8'},
    'cmot_repump_vco': {'CMOT Repump VCO (V) [-3MHz]': '-0.2'},

    # ===================== PHASE 3: DARK MOT =====================
    'dark_mot_duration': {'Dark MOT Duration (ms)': '4'},
    'dark_mot_power': {'Dark MOT Cooling Power (V)': '0.2'},
    'dark_mot_detuning': {'Dark MOT Cooling VCO (V) [-55MHz]': '-0.6'},
    'dark_mot_repump_power': {'Dark MOT Repump Power (V)': '0.2'},
    'dark_mot_repump_vco': {'Dark MOT Repump VCO (V)': '0.0'},

    # ===================== PHASE 4a: PGC COOL =====================
    # Field-free sub-Doppler cooling at full PGC light.
    'pgc_cool_ms': {'PGC Cool Duration (ms)': '5'},
    'pgc_mot_power': {'PGC Cooling Power (V)': '0.2'},
    'pgc_detuning': {'PGC Cooling VCO (V) [-122MHz]': '-2.3'},
    'pgc_repump_power': {'PGC Repump Power (V)': '0.03'},
    'pgc_repump_vco': {'PGC Repump VCO (V)': '-0.05'},

    # ===================== PHASE 4b: PGC RAMP-DOWN =====================
    # PGC light ramps to 0. With the lattice on this is the adiabatic transfer
    # into the trap; with it off it is just a soft extinction. Set to 0 to
    # match the no-lattice PGC script exactly.
    'pgc_rampdown_ms': {'PGC Rampdown Duration (ms)': '5'},

    # ===================== PHASE 5: LATTICE HOLD =====================
    # Atoms held in the lattice with all cooling light off, before release.
    # Mimics what the lattice_holdSweep / lattice_biasSweep scripts do.
    # IGNORED (forced to 0) when the lattice is switched off below.
    'lattice_hold_ms': {'Lattice Hold (ms)': '5'},

    # ===================== RELEASE + IMAGING =====================
    'tof_us': {'TOF (us)': '0'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'coils_rampback_ms': {'Coils Rampback Duration (ms)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '100'},

    # ===================== BIAS COILS =====================
    # Held CONSTANT at these values for the ENTIRE cycle (no ramping).
    # Matches the sweep scripts (lattice_biasXSweep etc.).
    'pgc_bias_x_v': {'PGC Bias X (V)': '0.0'},
    'pgc_bias_y_v': {'PGC Bias Y (V)': '0.0'},
    'pgc_bias_z_v': {'PGC Bias Z (V)': '0.0'},

    # ===================== LATTICE =====================
    # Master on/off switch. Anything that is not 0 counts as ON.
    #
    # 0 => dio4/dio5 HIGH and ao6/ao7 at 0.0 V for the whole cycle -- byte for
    #      byte what the no-lattice PGC script writes to those channels -- and
    #      Lattice Hold forced to 0 ms. Lattice Ramp Start/End are then unused.
    # 1 => lattice on, constant, as described at the top of the file.
    'lattice_enabled': {'Lattice On? (1=yes, 0=no)': '1'},

    # ===================== DISPLAY =====================
    # What the colour scale spans, in raw camera counts on the DIFFERENCE
    # image. This used to be hardcoded (0, 20), inherited from the bias sweep
    # scripts, and at 50 dB of gain that is narrower than a single photon:
    # the calibration is 1.84 DN/photon at 0 dB, and FLIR gain is 10^(G/20),
    # so one photon is 1.84 * 316 = 582 counts. A 20 count ceiling therefore
    # put every empty pixel at the bottom of the map and every pixel that
    # caught even one photon past the top, which is why the picture came out
    # black and yellow with none of inferno's midtones.
    #
    # Autoscale (default) sizes the top to the frame itself, so the picture
    # is correct before anyone knows what the right number is. Turn it off
    # once Display Max is set from the stats line in the log.
    # Camera gain in dB. FLIR gain is 10^(G/20), and the sensor holds 4095
    # ADU, so the full well in PHOTONS is 4095*16 / (1.84 * 10^(G/20)):
    #
    #     50 dB ->    113 photons      20 dB ->   3562 photons
    #     30 dB ->   1126 photons      10 dB ->  11262 photons
    #      0 dB -> 35615 photons
    #
    # 50 dB was clipping this cloud hard -- 0.43% of pixels pinned at the
    # ceiling and everything from the 99.9th percentile up flat -- which is
    # what produced a two-tone image no display range could rescue. Start
    # low, then raise it only until the stats line stops reporting saturation.
    'camera_gain_db': {'Camera Gain (dB)': '10'},

    'display_autoscale': {'Autoscale Display? (1=yes, 0=manual)': '1'},
    'display_min': {'Display Min / Background (counts)': '0'},
    'display_max': {'Display Max (counts) [used when autoscale=0]': '20000'},

    # Lattice AM scaling on ao6/ao7. With DIM-3000 set to AMP=320 (+32 dBm)
    # via dim_rf_on.py serial setup, AM=+1.0V => full +32 dBm output.
    # AM=-1.0V parks below threshold (effectively off).
    # lattice_ramp_start_v is kept for backward compatibility only: the AM is
    # held constant at lattice_ramp_end_v, it is never ramped.
    'lattice_ramp_start_v': {'Lattice Ramp Start (V) [UNUSED]': '-1.0'},
    'lattice_ramp_end_v': {'Lattice Ramp End (V) [held constant]': '0.6'},

    # ===================== DIM-3000 =====================
    # Where BOTH beams are parked before the sequence runs. This script used to
    # inherit whatever the last script left on the drivers; it now writes them
    # itself (see reset_dim_to_base) because inheriting is how a stale detuning
    # from a transport run ends up looking like a lattice that will not load.
    'dim_base_freq_mhz': {'DIM3000 Base Freq (MHz)': '100.800'},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)': '32.0'},
}


DIM_BASE_FREQ_MHZ = 100.800
DIM_AMPLITUDE_DBM = 32.0

DIM_COM_UP = 'COM4'     # lattice up
DIM_COM_DOWN = 'COM5'   # lattice down


def gui_float(dataset, name, default, logger):
    """A GUI value, or the default when the form predates the entry.

    A form built before these two entries existed raises KeyError on
    get_input_parameter, which would kill an otherwise fine run at startup.
    """
    try:
        return float(dataset.get_input_parameter(name))
    except Exception:
        logger.info(f"No GUI entry '{name}'; using {default}")
        return default


def reset_dim_to_base(base_freq_hz, amplitude_x10, logger, snaps=None):
    """Put BOTH lattice drivers on base and make their Seg inputs inert.

    WHY THIS SCRIPT NOW TOUCHES THE SERIAL PORTS AT ALL. It never did: it wrote
    an AM voltage and a TTL level and trusted whatever motdriverinit.py, or the
    last script to run, had left on the drivers. That trust is what breaks when
    a transport run leaves a driver in segment-play mode.

    Per the DIM-3000 manual, a driver in Mseg:N>0 plays the next segment on
    every RISING EDGE at its Seg connector, and the minimum pulse it will act
    on is 2 us. dio6 is held low for this whole sequence, but a DO line goes
    high-impedance when its task closes, and a floating line with a pull-up is
    a rising edge as far as the driver is concerned. The driver then jumps to a
    segment frequency, the two beams are detuned, and a detuned pair is a
    WALKING standing wave -- no trap, no lattice, and nothing in the images to
    say why. Now that dio6 is teed to BOTH Seg inputs, that applies to both
    drivers rather than only the up beam.

    So: base frequency on both, the segment chain rewritten to no-ops, and
    Mseg:0 to switch the Seg connector off outright.

    Never raises -- a driver that cannot be reached must not kill a run that
    would otherwise be fine. It logs at ERROR instead, because a run whose
    lattice frequencies are unverified is exactly the run that produces an
    inexplicably missing lattice, and that has to be visible in the log.

    :param base_freq_hz: (int) frequency for BOTH drivers [Hz]
    :param amplitude_x10: (int) amplitude in tenths of a dBm, the DIM's units
    :return: (bool) True only if both drivers were confirmed reset
    """
    if serial is None:
        logger.error(
            "pyserial is not installed, so the DIM-3000 drivers could NOT be "
            "reset. If a transport run left either one in segment mode, it "
            "will detune itself on the first edge dio6 gives it and there "
            "will be no lattice. Install pyserial, or run motdriverinit.py "
            "by hand before this script.")
        return False

    def _open(port):
        try:
            return serial.Serial(port=port, baudrate=19200, bytesize=8,
                                 parity='N', stopbits=1, timeout=1)
        except Exception as exc:
            logger.error(f"Could not open {port}: {exc!r}")
            return None

    def _send(ser, cmd):
        ser.write((cmd + '\n').encode())
        time.sleep(0.05)      # manual: about 10 commands/sec is the ceiling

    def _query(ser, cmd):
        ser.flushInput()
        _send(ser, cmd)
        time.sleep(0.1)
        return ser.readline().decode(errors='replace').strip()

    def _park(ser, port):
        _send(ser, f'FRQ:{base_freq_hz}')
        _send(ser, f'AMP:{amplitude_x10}')
        _send(ser, 'OUT_on')
        # Seven legs, because seven is the longest chain any script here
        # writes; the leftovers of a shorter chain are covered too. Every leg
        # starts and ends on base with a zero step, so playing one moves
        # nothing even if segment mode somehow survives.
        for i in range(1, 8):
            flag = 8 if i == 7 else 1
            _send(ser, f'Wseg:{i};{flag};{base_freq_hz};{base_freq_hz};0;1;0')
        # Then switch the connector off. Manual, Mseg: "N=0 - off; Connector
        # inactive, single frequency operation."
        _send(ser, 'Mseg:0')
        _send(ser, f'FRQ:{base_freq_hz}')
        logger.info(
            f"DIM-3000 {port}: {base_freq_hz / 1e6:.6f} MHz, "
            f"+{amplitude_x10 / 10:.1f} dBm, ON, segments neutralised and "
            f"Seg disabled (Mseg:0)")

    ser_up = _open(DIM_COM_UP)
    ser_dn = _open(DIM_COM_DOWN)
    ok = False
    try:
        if ser_up is not None and ser_dn is not None:
            logger.info(f"DIM-3000 up ({DIM_COM_UP}): "
                        f"{_query(ser_up, '*IDN?')}")
            logger.info(f"DIM-3000 down ({DIM_COM_DOWN}): "
                        f"{_query(ser_dn, '*IDN?')}")
            # Record the drivers BEFORE anything is written to them, so the
            # wrapper around this function can put them back exactly as found
            # when the run stops, however it stops (dim_state.py).
            if snaps is not None:
                snaps.append(dim_state.snapshot(ser_up, DIM_COM_UP, logger))
                snaps.append(dim_state.snapshot(ser_dn, DIM_COM_DOWN, logger))
            _park(ser_up, DIM_COM_UP)
            _park(ser_dn, DIM_COM_DOWN)
            ok = True
            logger.info(
                f"Both lattice beams on {base_freq_hz / 1e6:.6f} MHz: Df = 0, "
                f"standing wave stationary, neither driver able to act on a "
                f"dio6 edge.")
        else:
            # Resetting one beam alone can CREATE a detuning that was not there
            # before, so a half-open pair is left completely alone.
            logger.error(
                "Only one of the two DIM-3000 drivers could be opened, so "
                "NEITHER was changed -- resetting one beam alone would detune "
                "the pair rather than match it. Lattice frequencies are "
                "UNVERIFIED for this run. The usual cause is another program "
                "holding the port: close YAT or any other serial terminal.")
    except Exception as exc:
        logger.error(
            f"DIM-3000 reset failed partway through ({exc!r}). The two beams "
            f"may be detuned; treat this run's lattice as unverified.")
        ok = False
    finally:
        # Released immediately: this script never triggers a segment, so there
        # is nothing to hold them open for, and the transport scripts -- which
        # DO keep them open across a run -- must be able to take them next.
        for ser in (ser_up, ser_dn):
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
    return ok


# Percentile used when autoscaling the top of the colour range. Not the max:
# one hot pixel would then set the scale for the whole frame and push the
# cloud back down into the dark end. 99.95% of a 2448x2048 frame still leaves
# ~2500 pixels above the ceiling, which is inside the core of any real cloud.
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
# Fallback only. Unlike jingfastPGC, this script takes gain from the GUI and
# reads back what the camera actually applied, then passes that to
# cloud_metrics(). Every N scales as 10^(-G/20), so using this constant when
# the camera is at the 10 dB default would overstate N by a factor of 100.
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


AUTOSCALE_PCT = 99.95

# Autoscale is computed on every 4th row and column. A percentile over all
# 5.0 M pixels is a full sort on each frame pair; 313 k pixels give the same
# number to well under a count and keep the display off the acquisition's back.
AUTOSCALE_STRIDE = 4


def read_display_levels(dataset):
    """(lo, hi, autoscale) from the GUI, with the old hardcoded pair as fallback.

    Kept in one place so configure() and experiment() cannot drift apart, and
    so a missing or malformed field degrades to something viewable instead of
    raising in the middle of a run.
    """
    def _get(name, default):
        try:
            return float(dataset.get_input_parameter(name))
        except Exception:
            return default
    lo = _get("display_min", 0.0)
    hi = _get("display_max", 20000.0)
    auto = bool(int(_get("display_autoscale", 1.0)))
    if hi <= lo:
        hi = lo + 1.0          # a zero-width range renders as a flat block
    return lo, hi, auto


def autoscale_top(diff, lo):
    """Top of the colour range for this frame, from its own brightness."""
    sample = diff[::AUTOSCALE_STRIDE, ::AUTOSCALE_STRIDE]
    hi = float(np.percentile(sample, AUTOSCALE_PCT))
    if hi <= lo + 1.0:
        hi = lo + 10.0        # empty frame: keep a usable, non-degenerate range
    return hi


def release_ni(NI_clock, NI_do, NI_ao, h_do, h_ao, logger=None):
    """Give every armed NI task back, best effort, never raising.

    DAQmx owns a physical line for as long as the task holding it is open. If
    a run dies between arm() and the acquisition loop -- an OPX error, a lost
    connection, a stop -- and nothing finalises, those lines stay claimed
    inside the still-running server process. The NEXT run then fails at arm()
    with -200597 "requires lines that are currently in use by another output
    operation", and points at a line the new run has not touched yet.

    Each card is released in its own try so one failure cannot strand the rest.
    """
    for card, handle, name in ((NI_do, h_do, 'DO'), (NI_ao, h_ao, 'AO')):
        if handle is None:
            continue
        try:
            card.finalize(handle, timeout=120.0, force_finish=True)
        except Exception as exc:
            if logger is not None:
                logger.warn(f"Could not finalize the {name} task: {exc}")
    try:
        NI_clock.finalize_clock()
    except Exception as exc:
        if logger is not None:
            logger.warn(f"Could not finalize the clock task: {exc}")


class LiveView:
    """Image + results panel, updated from the GUI thread only.

    pyqtgraph is not thread safe. Calling setImage/setText straight from the
    acquisition thread appears to work and then takes the WHOLE PROCESS down
    with an access violation -- no Python traceback, every RPyC client
    disconnecting at once. Worse, the NI tasks are armed inside the server
    processes, which survive; nothing runs the finally block, so those servers
    keep holding dio0..dio6 and every later run dies at arm() with -200597.

    So the worker only calls submit(), which copies plain data and touches no
    Qt at all, and a QTimer owned by the GUI thread does the drawing. Same
    split jingfastPGC and the ScanGridWindow scripts already use.
    """

    POLL_MS = 200

    def __init__(self, title, levels):
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(700, 760)
        vb = self.win.addViewBox(row=0, col=0, lockAspect=True)
        vb.invertY(True)
        self.img_item = pg.ImageItem(axisOrder='row-major')
        self.img_item.setLookupTable(pg.colormap.get('inferno').getLookupTable())
        self.img_item.setLevels(levels)
        vb.addItem(self.img_item)
        # Results row UNDER the image, never over it.
        self.results_label = self.win.addLabel(
            'collecting...', row=1, col=0, justify='left', size='11pt')
        self.win.ci.layout.setRowFixedHeight(1, 150)

        self._pending = None
        self._rendered = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(self.POLL_MS)

    def submit(self, image, levels, table_html):
        """Called from the acquisition thread. Copies data, touches no Qt.

        The copy is deliberate: the worker reuses and mutates its arrays, and
        handing the widget a live reference means Qt can read it mid-write.
        """
        self._pending = (np.asarray(image).copy(), tuple(levels), table_html,
                         self._rendered + 1)

    def _poll(self):
        entry = self._pending
        if entry is None:
            return
        image, levels, table_html, version = entry
        if version <= self._rendered:
            return
        self._rendered = version
        try:
            self.img_item.setImage(image, levels=levels, autoLevels=False)
            if table_html is not None:
                self.results_label.setText(table_html, size='11pt')
        except Exception:
            pass          # a closed window must not kill the run


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

    # Title says which mode the window is showing, so two windows open side by
    # side cannot be confused for one another.
    try:
        lattice_on = bool(int(float(dataset.get_input_parameter("lattice_enabled"))))
    except Exception:
        lattice_on = True
    title = ("PGC + Lattice (atoms - background)" if lattice_on
             else "PGC, lattice OFF (atoms - background)")

    # Every Qt object lives in LiveView, which redraws on its own timer in the
    # GUI thread. experiment() only calls submit().
    dataset.live = LiveView(title, read_display_levels(dataset)[:2])
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

    # Before anything else touches the hardware: both beams onto base, both Seg
    # connectors switched off. This script cannot see a stale detuning -- it
    # would show up only as a lattice that will not load -- so it is asserted
    # rather than assumed. See reset_dim_to_base for what goes wrong without it.
    reset_dim_to_base(
        int(round(gui_float(dataset, "dim_base_freq_mhz",
                            DIM_BASE_FREQ_MHZ, logger) * 1e6)),
        int(round(gui_float(dataset, "dim_amplitude_dbm",
                            DIM_AMPLITUDE_DBM, logger) * 10)),
        logger, snaps=kwargs['dim_snapshots'])

    # =========================================================================
    # READ GUI PARAMETERS
    # =========================================================================
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    mot_switch_ms = int(dataset.get_input_parameter("mot_switch_time"))
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_v = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

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

    # Lattice master switch. float() first so "1.0" from the GUI is accepted;
    # anything non-zero is ON.
    lattice_on = bool(int(float(dataset.get_input_parameter("lattice_enabled"))))

    lattice_hold_ms = int(dataset.get_input_parameter("lattice_hold_ms"))
    # A "lattice hold" with no lattice is not a hold, it is free fall. Leaving
    # it in would add that many ms of dark expansion between PGC and release,
    # so the cloud in the image would be from tof + hold while the GUI still
    # reads TOF -- an invisible error that only shows up as a mysteriously
    # large cloud. Drop it and say so.
    if not lattice_on and lattice_hold_ms != 0:
        logger.warn(
            f"Lattice is OFF, so Lattice Hold ({lattice_hold_ms} ms) is being "
            f"ignored -- with no trap it would just be {lattice_hold_ms} ms of "
            f"extra free expansion before release, on top of the TOF. Release "
            f"now happens immediately at the end of PGC."
        )
        lattice_hold_ms = 0

    tof_us = int(dataset.get_input_parameter("tof_us"))
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    coils_rampback_ms = int(dataset.get_input_parameter("coils_rampback_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))

    pgc_bias_x_v = float(dataset.get_input_parameter("pgc_bias_x_v"))
    pgc_bias_y_v = float(dataset.get_input_parameter("pgc_bias_y_v"))
    pgc_bias_z_v = float(dataset.get_input_parameter("pgc_bias_z_v"))

    lat_start_v = float(dataset.get_input_parameter("lattice_ramp_start_v"))
    lat_end_v = float(dataset.get_input_parameter("lattice_ramp_end_v"))

    disp_lo, disp_hi, disp_auto = read_display_levels(dataset)
    gain_db = float(dataset.get_input_parameter("camera_gain_db"))

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
    dark_s = ms2s(dark_ms)
    pgc_cool_s = ms2s(pgc_cool_ms)
    pgc_rampdown_s = ms2s(pgc_rampdown_ms)
    pgc_s = pgc_cool_s + pgc_rampdown_s
    lattice_hold_s = ms2s(lattice_hold_ms)
    tof_samp = us2s(tof_us)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    coils_rampback_s = ms2s(coils_rampback_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    # =========================================================================
    # TIMELINE: MOT -> CMOT -> Dark MOT -> PGC -> [hold] -> release -> TOF -> image
    # =========================================================================
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    dark_start = cmot_end
    dark_end = dark_start + dark_s
    pgc_start = dark_end
    pgc_cool_end = pgc_start + pgc_cool_s
    pgc_end = pgc_cool_end + pgc_rampdown_s
    # lattice_hold_s is 0 when the lattice is off, so release == pgc_end there,
    # which is what the no-lattice PGC script does.
    release = pgc_end + lattice_hold_s
    coils_off = cmot_end - coils_lead_s
    f1 = release + tof_samp
    f2 = f1 + clear_s
    coils_restore_t = f2 + aom_pulse_s + coils_restore_s  # coils stay OFF through frame 2
    N = coils_restore_t + coils_rampback_s + 2

    logger.info(
        f"Fast PGC{'+Lattice' if lattice_on else ' (LATTICE OFF)'}: "
        f"mot={mot_loading_ms}ms, cmot={mot_switch_ms}+{cmot_hold_ms}ms, "
        f"dark={dark_ms}ms, pgc_cool={pgc_cool_ms}ms, "
        f"pgc_rampdown={pgc_rampdown_ms}ms, "
        f"lattice_hold={lattice_hold_ms}ms, tof={tof_us}us, "
        + (f"(lattice CONSTANT at {lat_end_v:.2f}V whole cycle), " if lattice_on
           else "(dio4/dio5 HIGH, ao6/ao7 at 0V whole cycle), ")
        + f"release={release/SPM:.0f}ms, N={N}"
    )
    if not lattice_on and pgc_rampdown_ms != 0:
        logger.info(
            f"Lattice OFF with a {pgc_rampdown_ms} ms PGC rampdown: the cooling "
            f"light is being extinguished over a ramp rather than cut, so the "
            f"sequence is {pgc_rampdown_ms} ms longer than the plain PGC "
            f"script. Set PGC Rampdown to 0 to match it exactly."
        )

    # =========================================================================
    # BUILD WAVEFORMS
    # =========================================================================
    cam = [0] * N
    for s in range(ttl_up):
        cam[f1 + s] = 1
        cam[f2 + s] = 1

    ado = [0] * N
    aao = [mot_aom_power] * N
    rdo = [0] * N
    rao = [repump_aom_power] * N

    # MOT
    for s in range(cmot_start):
        ado[s] = 1
        aao[s] = mot_aom_power
        rdo[s] = 1
        rao[s] = repump_aom_power
    # CMOT ramp
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        ado[t] = 1
        aao[t] = mot_aom_power + frac * (cmot_mot_power - mot_aom_power)
        rdo[t] = 1
        rao[t] = repump_aom_power + frac * (cmot_repump_power - repump_aom_power)
    # CMOT hold
    for s in range(cmot_hold_s):
        t = cmot_hold_start + s
        ado[t] = 1
        aao[t] = cmot_mot_power
        rdo[t] = 1
        rao[t] = cmot_repump_power
    # Dark MOT
    for s in range(dark_s):
        t = dark_start + s
        ado[t] = 1
        aao[t] = dark_power
        rdo[t] = 1
        rao[t] = dark_rep_power
    # PGC cool (full PGC light)
    for s in range(pgc_cool_s):
        t = pgc_start + s
        ado[t] = 1
        aao[t] = pgc_mot_power
        rdo[t] = 1
        rao[t] = pgc_rep_power
    # PGC rampdown: cooling AND repump ramp linearly to 0 over rampdown_s.
    # TTLs stay HIGH (light on) until amplitude hits 0; cleaner to keep TTL high
    # throughout and just let amplitude fall, so atoms see a smooth handoff.
    for s in range(pgc_rampdown_s):
        t = pgc_cool_end + s
        frac = s / max(pgc_rampdown_s - 1, 1)
        ado[t] = 1
        aao[t] = pgc_mot_power * (1.0 - frac)
        rdo[t] = 1
        rao[t] = pgc_rep_power * (1.0 - frac)
    # PGC end -> release: lattice hold if enabled (zero-length otherwise).
    # Then release -> F1: TOF. Digital off for both; analog stays at full power
    # (thermal stability).
    for s in range(pgc_end, min(f1, N)):
        ado[s] = 0
        rdo[s] = 0
    # Imaging pulses
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

    # Coils: TTL off from coils_off through frame 2 + restore delay,
    # then back ON with analog ramped 0 -> MOT value (soft turn-on).
    cdo = [0] * N
    for s in range(coils_off, min(coils_restore_t, N)):
        cdo[s] = 1
    cao = [mot_coils_v] * N
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        cao[t] = mot_coils_v + frac * (cmot_coils_v - mot_coils_v)
    for s in range(cmot_hold_start, N):
        cao[s] = cmot_coils_v
    # Rampback: analog 0 -> mot_coils_v over coils_rampback_s
    for s in range(coils_rampback_s):
        t = coils_restore_t + s
        if t >= N:
            break
        frac = s / max(coils_rampback_s - 1, 1)
        cao[t] = frac * mot_coils_v
    for s in range(coils_restore_t + coils_rampback_s, N):
        cao[s] = mot_coils_v

    # MOT VCO
    mvco = [vco_loading] * N
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        mvco[t] = vco_loading + frac * (cmot_detuning - vco_loading)
    for s in range(cmot_hold_start, cmot_end):
        mvco[s] = cmot_detuning
    for s in range(dark_s):
        mvco[dark_start + s] = dark_detuning
    # PGC cool + rampdown: hold cooling beam at pgc_detuning throughout
    for s in range(pgc_s):
        mvco[pgc_start + s] = pgc_detuning
    for s in range(pgc_end, N):
        mvco[s] = pgc_detuning
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
    for s in range(dark_s):
        rvco[dark_start + s] = dark_rep_vco
    for s in range(pgc_s):
        rvco[pgc_start + s] = pgc_rep_vco
    for s in range(pgc_end, N):
        rvco[s] = pgc_rep_vco

    # =========================================================================
    # LATTICE CHANNELS -- the one thing the toggle actually changes
    # =========================================================================
    # ON: both AM and TTL held constant for the entire cycle so the AOM crystal
    #     sees identical RF drive at all times -> stable thermal state ->
    #     stable fiber coupling. Lattice light is always present at the
    #     chamber, but during MOT loading there are no cold atoms to care.
    #     TTL LOW = beam ON on these lines.
    # OFF: TTL HIGH and AM at 0.0 V, which is exactly what the no-lattice PGC
    #     script writes to dio4/dio5/ao6/ao7, and also what its SAFE_DO/SAFE_AO
    #     park values are -- so a run that ends mid-buffer leaves the lattice
    #     in the same state either way.
    # Up beam: dio4/ao6  |  Down beam: dio5/ao7
    if lattice_on:
        ludo = [0] * N
        lddo = [0] * N
        luao = [lat_end_v] * N
        ldao = [lat_end_v] * N
    else:
        ludo = [1] * N
        lddo = [1] * N
        luao = [0.0] * N
        ldao = [0.0] * N

    # Bias coils: held CONSTANT at GUI values for the ENTIRE cycle (no ramping).
    # Matches the sweep scripts. Bias field therefore present through MOT
    # loading too, which is fine: the quadrupole still dominates during MOT.
    biasx = [pgc_bias_x_v] * N
    biasy = [pgc_bias_y_v] * N
    biasz = [pgc_bias_z_v] * N

    # =========================================================================
    # LOAD, ARM, EXECUTE
    # =========================================================================
    camera_client.set_hardware_trigger(
        line="Line0", activation="RisingEdge",
        selector="FrameStart", overlap="ReadOut",
        acquisition_mode="Continuous")
    camera_client.set_exposure(camera_exposure_us)
    # try_set_float is best-effort: a value past the node maximum raises inside
    # SetValue and the driver swallows it, leaving the camera on its previous
    # gain. Read it back so the log states what was really applied.
    camera_client.try_set_float("Gain", gain_db)
    try:
        applied_gain = float(camera_client.get_float("Gain"))
    except Exception:
        applied_gain = gain_db
    dn_per_photon = 1.84 * 10.0 ** (applied_gain / 20.0)
    logger.info(
        "Camera: exposure %d us, gain %.1f dB requested / %.1f dB applied "
        "-> %.0f counts per photon, full well %.0f photons"
        % (camera_exposure_us, gain_db, applied_gain, dn_per_photon,
           SENSOR_MAX_MONO16 / dn_per_photon)
    )
    dataset.camera_client.start_acquisition()

    try:
        NI_card_1.finalize_clock()
    except Exception:
        pass
    h1 = None
    h2 = None

    NI_card_2.build_stack()
    NI_card_2.set_do_voltage(do_channel="dio0", value=cam, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio1", value=ado, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio2", value=cdo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio3", value=rdo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio4", value=ludo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio5", value=lddo, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio6", value=[0] * N, sample_rate=ni_rate)

    NI_card_3.build_stack()
    NI_card_3.set_ao_voltage(ao_channel="ao1", voltages=aao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao2", voltages=cao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao3", voltages=mvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao4", voltages=rvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao5", voltages=rao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=luao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=ldao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao8", voltages=biasx, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao9", voltages=biasy, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao10", voltages=biasz, sample_rate=ni_rate)

    # From here on the NI tasks own real hardware lines, so every path out of
    # this block has to release them -- see release_ni(). Without this guard,
    # anything raising before the acquisition try below (an OPX build or
    # execute failure is the usual one) left dio0..dio6 claimed and the next
    # run died at arm() with -200597.
    try:
        h1 = NI_card_2.arm(regeneration=True)
        h2 = NI_card_3.arm(regeneration=True)
        NI_card_1.arm_clock(length=N, sample_rate=ni_rate)

        OPX_client.build_stack()
        clk = OPX_client.create_new_do_elem(do_channel=1, length=500)
        M = 100000
        run_buffer = 10_000_000
        with OPX_client.for_("j", 0, M, 1):
            with OPX_client.for_("i", 0, N + 1, 1):
                OPX_client.set_digital_voltage(element=clk)
                OPX_client.delay(delay_ns)
            OPX_client.delay(run_buffer)
        OPX_client.execute(wait=False)
    except BaseException:
        # Release BEFORE propagating. BaseException, not Exception, so a stop
        # (KeyboardInterrupt / SystemExit) frees the lines too.
        logger.warn("Arming failed; releasing NI tasks so the lines are not "
                    "left claimed for the next run.")
        release_ni(NI_card_1, NI_card_2, NI_card_3, h1, h2, logger)
        raise

    def get_frame(timeout_ms=1000):
        b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    try:
        pairs_logged = 0
        logger.info("Discarding first frame pair")
        try:
            get_frame(timeout_ms=15000)
            get_frame(timeout_ms=10000)
        except Exception:
            pass

        while thread.running:
            try:
                frame1 = get_frame(timeout_ms=2000)
            except Exception:
                if not thread.running:
                    break
                continue
            if not thread.running:
                break
            try:
                frame2 = get_frame(timeout_ms=2000)
            except Exception:
                if not thread.running:
                    break
                continue
            diff = frame1.astype(np.int32) - frame2.astype(np.int32)
            # Colour range. Manual uses Display Min/Max exactly, so panels stay
            # comparable between runs; autoscale moves only the top, so the
            # background stays pinned at Display Min and does not breathe.
            hi = autoscale_top(diff, disp_lo) if disp_auto else disp_hi

            # ---------------- atom number, as in jingfastPGC ----------------
            # Fits run on the SUMMED profiles, not on single rows: collapsing
            # 2048 rows lifts the cloud 2048x while noise grows only as
            # sqrt(2048), a ~45x gain in SNR. That is what makes a fit possible
            # on a cloud too faint to see.
            img = diff.astype(np.float64)
            fx = fit_gaussian_1d(img.sum(axis=0))
            fy = fit_gaussian_1d(img.sum(axis=1))
            good = bool(fx and fy and fx['reliable'] and fy['reliable'])
            # frame1.dtype, NOT img.dtype: DN/photon is a property of the raw
            # ADC word, and img is float64, which would select the 12-bit
            # calibration and put N out by 16x. applied_gain is what the camera
            # reported back, not what was requested.
            met = (cloud_metrics(img, fx, fy, frame1.dtype,
                                 camera_exposure_us, gain_db=applied_gain)
                   if (fx and fy) else None)
            # Peak brightness per pixel, rebuilt from the two 1D fits exactly
            # as jingfastPGC does it: fx['amplitude'] is the height of the
            # x-marginal, which was already summed over y, so dividing by the
            # y extent (w * sqrt(pi/2)) puts it back into counts per pixel.
            peak_px = (fx['amplitude'] / (fy['w'] * ROOT_HALF_PI)
                       if (fx and fy and fy['w'] > 0) else float('nan'))
            # One handoff to the GUI thread: image, levels and results together,
            # so the picture and the numbers under it always describe the same
            # frame pair rather than being updated a beat apart.
            dataset.live.submit(
                diff, (disp_lo, hi),
                results_table_html(fx, fy, peak_px, met, good)
                if (fx and fy) else
                results_table_html(None, None, None, None, False))
            # N in the log too, so a run is reconstructable afterwards from the
            # logfile alone rather than only from whoever was watching.
            if met is not None:
                logger.info(
                    "N(seen)=%.3e  N(total)=%.3e (%.1f%% seen)  n0=%.2e/cm3  "
                    "sx=%.1f sy=%.1f px  peak=%.2f cnt/px%s"
                    % (met['n_atoms'], met['n_atoms_total'],
                       met['capture_fraction'] * 100.0, met['density'],
                       fx['sigma'], fy['sigma'], peak_px,
                       "" if good else "  [UNRELIABLE FIT]")
                )
            # One stats line for the first few pairs: enough to choose a manual
            # Display Max from real data instead of guessing, and it says
            # outright when the ADC itself is clipping -- no display range can
            # rescue a frame that saturated at 65532 counts, only less gain.
            if pairs_logged < 3:
                pairs_logged += 1
                sample = diff[::AUTOSCALE_STRIDE, ::AUTOSCALE_STRIDE]
                peak = int(diff.max())
                # Saturation is judged on the RAW atoms frame, not the
                # difference. Two clipped frames subtract to nearly zero, so a
                # fully saturated cloud can show up as a HOLE in diff -- dark
                # in the middle, bright at the edges -- and diff.max() would
                # look reassuringly moderate while the data is worthless.
                n_sat = int((frame1 >= SATURATION_COUNTS).sum())
                pct_sat = 100.0 * n_sat / frame1.size
                logger.info(
                    "Frame pair %d: peak=%d, 99.95%%=%.0f, 99%%=%.0f, "
                    "median=%.0f counts | showing (%.0f, %.0f) %s"
                    % (pairs_logged, peak,
                       np.percentile(sample, 99.95),
                       np.percentile(sample, 99.0),
                       np.median(sample), disp_lo, hi,
                       "auto" if disp_auto else "manual")
                )
                if n_sat:
                    logger.warn(
                        "%d pixels (%.3f%%) of the atoms frame are SATURATED "
                        "at %d counts. The image will stay two-tone whatever "
                        "the display range, because the data itself is "
                        "clipped. Lower Camera Gain (now %.1f dB, full well "
                        "%.0f photons) or shorten the exposure until this "
                        "reads zero."
                        % (n_sat, pct_sat, SATURATION_COUNTS, applied_gain,
                           SENSOR_MAX_MONO16 / dn_per_photon)
                    )
    finally:
        try:
            dataset.camera_client.stop_acquisition()
        except Exception as exc:
            logger.warn(f"Could not stop the camera acquisition: {exc}")
        # Same helper as the arming guard above, so a normal stop and a failed
        # arm release the hardware in exactly the same way.
        release_ni(NI_card_1, NI_card_2, NI_card_3, h1, h2, logger)
        # Only now, with every sequence task closed, can static writes land.
        park_outputs(NI_card_2, NI_card_3, logger,
                     lattice_am=lat_end_v, lattice_on=lattice_on)
