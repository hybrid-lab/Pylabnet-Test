# MOT atom number from fluorescence imaging.
#
# Same sequence as fastMOT (MOT -> CMOT -> release -> TOF -> image), but the
# cycle time is deliberately relaxed: frame 1 has atoms, frame 2 is taken after
# `atoms_clear_time` with the atoms gone and serves as the background. The
# subtracted image is summed to get photons, and photons are converted to atoms.
#
# ============================================================================
# PHYSICS  (Rb-87 D2, 780.241 nm)
# ============================================================================
#
# 1. Camera counts -> photons at the sensor.  The DN/photon calibration already
#    folds in the quantum efficiency (it was measured against a known photon
#    flux), so QE must NOT be applied again.  FLIR gain in dB is a linear
#    multiplier 10^(G/20), so the calibration must be rescaled if the imaging
#    gain differs from the gain it was measured at:
#
#       \Gamma_{DN} = \Gamma_{DN}^{cal} \times 10^{(G - G_{cal})/20}
#
#       N_\gamma^{sensor} = \frac{\sum_{i \in ROI} (S_i - B_i)}{\Gamma_{DN}}
#
#    where S is the atom frame, B the background frame, and \Gamma_{DN} is
#    1.84 DN/photon for a 16-bit container (Mono16 = 12-bit data left-shifted
#    by 4) or 0.115 DN/photon for raw 12-bit values.  Note 1.84 = 16 x 0.115.
#
# 2. Photons at the sensor -> photons emitted into 4\pi.  The viewport is a
#    circular aperture of radius a at distance L, subtending a half-angle
#    \theta = \arctan(a/L).  The exact solid-angle fraction of an isotropic
#    emitter it intercepts is
#
#       \frac{\Omega}{4\pi} = \frac{1}{2}\left(1 - \cos\theta\right)
#                           = \frac{1}{2}\left(1 - \frac{L}{\sqrt{L^2+a^2}}\right)
#
#    For a = 19.3/2 mm and L = 100 mm:  \theta = 5.512 deg,
#    \Omega = 0.02905 sr,  \Omega/4\pi = 2.312 \times 10^{-3} = 0.231%.
#    (The small-angle form a^2/4L^2 gives 0.233% -- agrees to 1%.)
#
#    This assumes the viewport is the limiting aperture.  If the imaging lens
#    aperture is smaller as seen from the atoms, IT sets \Omega, not the window.
#
#       N_\gamma^{emit} = \frac{N_\gamma^{sensor}}{(\Omega/4\pi)\, T_{opt}}
#
#    with T_opt the transmission of window + lens + any filters.
#
# 3. Photons emitted -> atoms.  Each beam of the six-beam molasses contributes
#    independently to the saturation parameter (they are mutually incoherent on
#    the timescale of \Gamma^{-1}):
#
#       I_{beam} = \frac{P}{\pi (d/2)^2}   (flat-top)
#       I_{beam} = \frac{2P}{\pi w^2}      (Gaussian peak, w = d/2 at 1/e^2)
#
#       s = \frac{\sum_j I_j}{I_{sat}} = \frac{N_{beams} I_{beam}}{I_{sat}}
#
#    and the two-level steady-state scattering rate is
#
#       R_{sc} = \frac{\Gamma}{2}\,
#                \frac{s}{1 + s + \left(\frac{2\delta}{\Gamma}\right)^2}
#
#    with \Gamma = 2\pi \times 6.0666 MHz and \delta the detuning from
#    resonance.  I_sat = 3.577 mW/cm^2 for isotropic polarisation (the right
#    choice for a MOT); 1.669 mW/cm^2 is the |F=2,m=2> -> |F'=3,m'=3> cycling
#    value and only applies to pure \sigma^+ light on a spin-polarised sample.
#
#       N_{atoms} = \frac{N_\gamma^{emit}}{R_{sc}\, t_{exp}}
#
#    Caveats worth remembering when you read the number off:
#      - Two-level R_sc overestimates the real rate on a multilevel atom that
#        spends part of its time in F=1 between repump events.  Expect this to
#        read HIGH by a factor of order unity (~1.5-2 is typical) unless the
#        repump is strongly saturating.
#      - R_sc saturates at \Gamma/2, so at s >> 1 the atom number is insensitive
#        to intensity errors but very sensitive to \delta.
#      - t_exp is whichever is shorter, the camera exposure or the imaging light
#        pulse.  The sequence below makes the AOM pulse >= the exposure, so the
#        exposure is the right number.
#
# Fully assembled:
#
#   N_{atoms} = \frac{\sum_i (S_i - B_i)}
#                    {\Gamma_{DN}\;(\Omega/4\pi)\; T_{opt}\; R_{sc}\; t_{exp}}
#
# ============================================================================

import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

# ---------------------------------------------------------------------------
# Rb-87 D2 constants
# ---------------------------------------------------------------------------
GAMMA_HZ = 6.0666e6              # natural linewidth Gamma/2pi [Hz]
GAMMA = 2 * np.pi * GAMMA_HZ     # [rad/s]
I_SAT_ISO = 3.5779               # [mW/cm^2] isotropic polarisation, D2
I_SAT_CYC = 1.66933              # [mW/cm^2] sigma+ cycling transition

# DN per photon, from camera_photon_per_unit.py. 16-bit container = 16 x 12-bit.
DN_PER_PHOTON_16BIT = 1.84
DN_PER_PHOTON_12BIT = 0.115

# ===========================================================================
# APPARATUS PARAMETERS -- edit these, not the GUI defaults below
# ===========================================================================
# Everything the atom-number conversion depends on lives here. The GUI fields
# are seeded from these values, so changing a number in this block changes the
# default everywhere it is used. The GUI still overrides at run time for
# shot-to-shot tweaking without editing the file.

# ---- Collection geometry -------------------------------------------------
# The limiting aperture as seen from the atoms. If the imaging lens subtends a
# smaller angle than the viewport, put the LENS numbers here instead.
WINDOW_DIAMETER_MM = 19.3        # viewport clear aperture (measured)
APERTURE_DISTANCE_MM = 100.0     # chamber centre -> window
OPTICS_TRANSMISSION = 1.0        # window * lens * filters, 0-1

# ---- MOT / imaging light -------------------------------------------------
BEAM_POWER_MW = 50.0             # power in ONE beam, at the atoms
BEAM_DIAMETER_MM = 30.0          # flat-top diameter, or 1/e^2 diameter if Gaussian
N_BEAMS = 6                      # beams illuminating the atoms during imaging
BEAM_PROFILE = "flat"            # "flat" or "gaussian"
# DataTaker calls float() on every GUI field, so the profile has to reach the
# form as a number even though the code wants the name.
_PROFILE_CODE = {"flat": 0, "gaussian": 1}
IMAGING_DETUNING_MHZ = 0.0       # detuning from resonance; negative = red
I_SAT = I_SAT_ISO                # use I_SAT_CYC only for sigma+ on spin-polarised atoms

# ---- Camera calibration --------------------------------------------------
DN_PER_PHOTON_OVERRIDE = 0.0     # 0 = auto-select 16-bit vs 12-bit from dtype
CAMERA_GAIN_DB = 0.0             # gain used for the measurement
CALIB_GAIN_DB = 0.0              # gain at which DN/photon above was measured

# ---- Saturation ----------------------------------------------------------
# Gain is applied before the ADC, so turning it up eventually rails the MOT
# peak. A clipped sum is a TRUNCATED sum, so the atom number reads LOW -- and
# silently, since nothing else about the image looks wrong. Raise the gain
# until this warns, then back off.
SATURATION_LEVEL_DN = 0          # 0 = auto from the container (see below)
SATURATION_WARN_PIXELS = 0       # warn above this many railed pixels
# ===========================================================================

INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '500'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '8'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0.31'},
    'Repump_VCO': {'Repump VCO (V)': '0.2'},

    # ===================== PHASE 2: CMOT =====================
    'cmot_hold_time': {'CMOT Hold Time (ms)': '10'},
    'cmot_mot_power': {'CMOT MOT AOM Power (V)': '0.5'},
    'cmot_repump_power': {'CMOT Repump AOM Power (V)': '0.15'},
    'cmot_detuning_voltage': {'CMOT MOT VCO (V) [-40MHz]': '-0.17'},
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '6.5'},
    'cmot_repump_vco': {'CMOT Repump VCO (V) [-3MHz]': '0.26'},

    # ===================== RELEASE + IMAGING =====================
    'tof_us': {'TOF (us)': '2000'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'coils_rampback_ms': {'Coils Rampback Duration (ms)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.6'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '10'},
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Background Frame Delay (ms)': '300'},
    'cycle_delay_ms': {'Extra Delay Between Cycles (ms)': '50'},

    # ===================== BIAS COILS =====================
    'bias_x_v': {'Bias X (V)': '0.0'},
    'bias_y_v': {'Bias Y (V)': '0.0'},
    'bias_z_v': {'Bias Z (V)': '0.0'},

    # ===================== CAMERA CALIBRATION =====================
    # dn_per_photon is auto-selected from the frame dtype unless overridden:
    # uint16 -> 1.84 DN/photon, anything else -> 0.115. Set > 0 to force.
    'dn_per_photon_override': {'DN per Photon (0 = auto from dtype)': str(DN_PER_PHOTON_OVERRIDE)},
    'camera_gain_db': {'Camera Gain (dB)': str(CAMERA_GAIN_DB)},
    'calib_gain_db': {'Gain at Which DN/Photon Was Measured (dB)': str(CALIB_GAIN_DB)},
    'optics_transmission': {'Optics Transmission (window+lens+filters)': str(OPTICS_TRANSMISSION)},

    # ===================== COLLECTION GEOMETRY =====================
    'window_diameter_mm': {'Limiting Aperture Diameter (mm)': str(WINDOW_DIAMETER_MM)},
    'aperture_distance_mm': {'Atoms to Aperture Distance (mm)': str(APERTURE_DISTANCE_MM)},

    # ===================== IMAGING LIGHT =====================
    'beam_power_mw': {'Power Per MOT Beam (mW)': str(BEAM_POWER_MW)},
    'beam_diameter_mm': {'Beam Diameter (mm)': str(BEAM_DIAMETER_MM)},
    'n_beams': {'Number of MOT Beams': str(N_BEAMS)},
    'beam_profile': {'Beam Profile (0 = flat-top, 1 = Gaussian)': str(_PROFILE_CODE[BEAM_PROFILE])},
    'imaging_detuning_mhz': {'Imaging Detuning (MHz, negative = red)': str(IMAGING_DETUNING_MHZ)},
    'i_sat_mw_cm2': {'I_sat (mW/cm^2) [3.5779 iso, 1.6693 cycling]': str(I_SAT)},

    # ===================== ROI + ANALYSIS =====================
    # ROI is [x0, x1) x [y0, y1). -1 anywhere means "use the whole frame".
    'roi_x0': {'ROI x0 (-1 = full frame)': '-1'},
    'roi_x1': {'ROI x1': '-1'},
    'roi_y0': {'ROI y0': '-1'},
    'roi_y1': {'ROI y1': '-1'},
    'subtract_offset': {'Remove Residual Offset Outside ROI (1/0)': '1'},

    # ===================== DISPLAY =====================
    'display_min': {'Display Min / Background (counts)': '-5'},
    'display_max': {'Display Max (counts)': '50'},
}


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def collection_efficiency(aperture_diameter_mm, distance_mm):
    """Fraction of isotropic 4pi emission passing a circular aperture.

    Exact form (1 - cos(theta))/2, not the small-angle a^2/4L^2 -- they differ
    by ~1% at our 5.5 deg half-angle, which is below other systematics but the
    exact one costs nothing.
    """
    theta = np.arctan((aperture_diameter_mm / 2.0) / distance_mm)
    return 0.5 * (1.0 - np.cos(theta))


def beam_intensity(power_mw, diameter_mm, profile="flat"):
    """Intensity in mW/cm^2. Gaussian returns the on-axis peak (d = 1/e^2)."""
    radius_cm = diameter_mm / 2.0 / 10.0
    area_cm2 = np.pi * radius_cm ** 2
    if str(profile).lower().startswith("gauss"):
        return 2.0 * power_mw / area_cm2
    return power_mw / area_cm2


def scattering_rate(s_total, detuning_hz):
    """Two-level steady-state photon scattering rate [photons/s/atom].

    Returns rad/s-normalised Gamma/2 * s / (1 + s + (2 delta / Gamma)^2). Both
    detuning and linewidth are used in ordinary Hz so the ratio is unitless.
    """
    return (GAMMA / 2.0) * s_total / (1.0 + s_total + (2.0 * detuning_hz / GAMMA_HZ) ** 2)


def dn_per_photon_effective(dn_per_photon_calib, gain_db, calib_gain_db):
    """Rescale the DN/photon calibration to the gain actually in use.

    FLIR Gain is specified in dB as a linear multiplier 10^(G/20). fastMOT runs
    at 50 dB (a factor of 316) while camera_photon_per_unit.py calibrates at
    0 dB, so getting this wrong is a two-and-a-half order of magnitude error.
    """
    return dn_per_photon_calib * 10.0 ** ((gain_db - calib_gain_db) / 20.0)


def saturation_ceiling(dtype, dn_calib):
    """DN at which the ADC rails, for the container the camera hands back.

    Mono16 is 12-bit sensor data left-shifted by four, so it rails at
    0xFFF0 = 65520 rather than 65535; raw 12-bit rails at 4095. The bit depth
    is inferred from which DN/photon calibration was selected, so the two
    always agree. SATURATION_LEVEL_DN overrides if the camera is configured
    some other way.
    """
    if SATURATION_LEVEL_DN > 0:
        return float(SATURATION_LEVEL_DN)
    if np.dtype(dtype) == np.uint8:
        return 255.0
    if dn_calib == DN_PER_PHOTON_12BIT:
        return 4095.0
    return 65520.0


def count_saturated(frame_atoms, frame_bg, ceiling):
    """Railed pixels in each raw frame, plus the brighter frame's peak.

    Checked on the raw frames, not the difference: clipping happens in the ADC,
    and a clipped peak minus an unclipped background still looks like a
    perfectly ordinary difference image.
    """
    n_atoms = int(np.count_nonzero(frame_atoms >= ceiling))
    n_bg = int(np.count_nonzero(frame_bg >= ceiling))
    peak = float(max(frame_atoms.max(), frame_bg.max()))
    return n_atoms, n_bg, peak


def atom_number(sum_dn, dn_per_photon, eta_collect, optics_transmission,
                r_sc, exposure_s):
    """N_atoms = sum(S - B) / (DN_per_photon * Omega/4pi * T * R_sc * t_exp)."""
    photons_sensor = sum_dn / dn_per_photon
    photons_emitted = photons_sensor / (eta_collect * optics_transmission)
    return photons_emitted / (r_sc * exposure_s)


def _resolve_roi(shape, x0, x1, y0, y1):
    """Clamp a user ROI to the frame; any -1 means 'whole frame'."""
    h, w = shape
    if min(x0, x1, y0, y1) < 0:
        return 0, w, 0, h
    x0, x1 = sorted((int(np.clip(x0, 0, w)), int(np.clip(x1, 0, w))))
    y0, y1 = sorted((int(np.clip(y0, 0, h)), int(np.clip(y1, 0, h))))
    if x1 <= x0 or y1 <= y0:
        return 0, w, 0, h
    return x0, x1, y0, y1


def integrate_signal(diff, roi, subtract_offset=True):
    """Sum the difference image over the ROI, removing any residual DC offset.

    A drift of even 0.1 DN between the two frames integrates to ~5e5 DN over a
    5 MP sensor, which would swamp the atom signal. The median outside the ROI
    is a robust estimate of that pedestal; the median (not mean) keeps stray
    scattered light and hot pixels from biasing it.
    """
    x0, x1, y0, y1 = roi
    offset = 0.0
    if subtract_offset:
        mask = np.ones(diff.shape, dtype=bool)
        mask[y0:y1, x0:x1] = False
        if mask.any():
            offset = float(np.median(diff[mask]))
    region = diff[y0:y1, x0:x1]
    return float(region.sum() - offset * region.size), offset


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")
    dataset.camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']

    disp_min = float(dataset.get_input_parameter("display_min"))
    disp_max = float(dataset.get_input_parameter("display_max"))
    dataset.diff_levels = (disp_min, disp_max)

    dataset.n_atom_history = []

    dataset.img_win = pg.GraphicsLayoutWidget(show=True, title="MOT fluorescence (atoms - background)")
    dataset.img_win.resize(600, 500)
    vb = dataset.img_win.addViewBox(lockAspect=True)
    vb.invertY(True)
    dataset.img_item = pg.ImageItem(axisOrder='row-major')
    dataset.img_item.setLevels(dataset.diff_levels)
    cmap = pg.colormap.get('inferno')
    dataset.img_item.setLookupTable(cmap.getLookupTable())
    vb.addItem(dataset.img_item)
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
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    CMOT_RAMP_MS = 10
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_v = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

    tof_us = int(dataset.get_input_parameter("tof_us"))
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    coils_rampback_ms = int(dataset.get_input_parameter("coils_rampback_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))
    cycle_delay_ms = int(dataset.get_input_parameter("cycle_delay_ms"))

    bias_x_v = float(dataset.get_input_parameter("bias_x_v"))
    bias_y_v = float(dataset.get_input_parameter("bias_y_v"))
    bias_z_v = float(dataset.get_input_parameter("bias_z_v"))

    dn_override = float(dataset.get_input_parameter("dn_per_photon_override"))
    gain_db = float(dataset.get_input_parameter("camera_gain_db"))
    calib_gain_db = float(dataset.get_input_parameter("calib_gain_db"))
    optics_transmission = float(dataset.get_input_parameter("optics_transmission"))

    window_diameter_mm = float(dataset.get_input_parameter("window_diameter_mm"))
    aperture_distance_mm = float(dataset.get_input_parameter("aperture_distance_mm"))

    beam_power_mw = float(dataset.get_input_parameter("beam_power_mw"))
    beam_diameter_mm = float(dataset.get_input_parameter("beam_diameter_mm"))
    n_beams = int(dataset.get_input_parameter("n_beams"))
    beam_profile = 'gaussian' if int(float(dataset.get_input_parameter("beam_profile"))) else 'flat'
    detuning_mhz = float(dataset.get_input_parameter("imaging_detuning_mhz"))
    i_sat = float(dataset.get_input_parameter("i_sat_mw_cm2"))

    roi_x0 = int(dataset.get_input_parameter("roi_x0"))
    roi_x1 = int(dataset.get_input_parameter("roi_x1"))
    roi_y0 = int(dataset.get_input_parameter("roi_y0"))
    roi_y1 = int(dataset.get_input_parameter("roi_y1"))
    subtract_offset = bool(int(dataset.get_input_parameter("subtract_offset")))

    # =========================================================================
    # PHOTON -> ATOM CONVERSION FACTORS (fixed for the whole run)
    # =========================================================================
    eta_collect = collection_efficiency(window_diameter_mm, aperture_distance_mm)
    i_beam = beam_intensity(beam_power_mw, beam_diameter_mm, beam_profile)
    s_total = n_beams * i_beam / i_sat
    r_sc = scattering_rate(s_total, detuning_mhz * 1e6)
    exposure_s = camera_exposure_us * 1e-6

    logger.info(
        f"Collection: aperture {window_diameter_mm}mm at {aperture_distance_mm}mm "
        f"-> Omega/4pi = {eta_collect:.4e} ({100 * eta_collect:.3f}%), T_opt = {optics_transmission}"
    )
    logger.info(
        f"Light: {n_beams} x {beam_power_mw}mW, d={beam_diameter_mm}mm ({beam_profile}) "
        f"-> I_beam = {i_beam:.2f} mW/cm^2, s_tot = {s_total:.2f}, delta = {detuning_mhz}MHz "
        f"-> R_sc = {r_sc:.3e} /s ({r_sc * exposure_s:.0f} photons/atom in {camera_exposure_us}us)"
    )

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
    tof_samp = us2s(tof_us)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    coils_rampback_s = ms2s(coils_rampback_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    # =========================================================================
    # TIMELINE: MOT -> CMOT -> release -> TOF -> atom frame -> clear -> bg frame
    # =========================================================================
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    release = cmot_end
    coils_off = release - coils_lead_s
    f1 = release + tof_samp
    f2 = f1 + clear_s
    coils_restore_t = f2 + aom_pulse_s + coils_restore_s  # coils stay OFF through frame 2
    N = coils_restore_t + coils_rampback_s + 2

    logger.info(
        f"Atom number: mot={mot_loading_ms}ms, cmot={CMOT_RAMP_MS}+{cmot_hold_ms}ms, "
        f"tof={tof_us}us, bg delay={atoms_clear_ms}ms, N={N} ({N / SPM:.0f}ms/shot)"
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
    # TOF: digital off, analog stays at full power (thermal stability)
    for s in range(release, min(f1, N)):
        ado[s] = 0
        rdo[s] = 0
    # Imaging pulses. Both frames get identical light so the background frame
    # captures the same stray scatter -- that is what makes the subtraction work.
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
    for s in range(cmot_hold_start, N):
        mvco[s] = cmot_detuning
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
    for s in range(cmot_hold_start, N):
        rvco[s] = cmot_repump_vco

    # Lattice channels: all off
    lat_off_do = [1] * N
    lat_off_ao = [0.0] * N

    # =========================================================================
    # LOAD, ARM, EXECUTE
    # =========================================================================
    camera_client.set_hardware_trigger(
        line="Line0", activation="RisingEdge",
        selector="FrameStart", overlap="ReadOut",
        acquisition_mode="Continuous")
    camera_client.set_exposure(camera_exposure_us)
    camera_client.try_set_float("Gain", gain_db)
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
    NI_card_2.set_do_voltage(do_channel="dio4", value=lat_off_do, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio5", value=lat_off_do, sample_rate=ni_rate)
    NI_card_2.set_do_voltage(do_channel="dio6", value=[0] * N, sample_rate=ni_rate)

    NI_card_3.build_stack()
    NI_card_3.set_ao_voltage(ao_channel="ao1", voltages=aao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao2", voltages=cao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao3", voltages=mvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao4", voltages=rvco, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao5", voltages=rao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=lat_off_ao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=lat_off_ao, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao8", voltages=[bias_x_v] * N, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao9", voltages=[bias_y_v] * N, sample_rate=ni_rate)
    NI_card_3.set_ao_voltage(ao_channel="ao10", voltages=[bias_z_v] * N, sample_rate=ni_rate)

    h1 = NI_card_2.arm(regeneration=True)
    h2 = NI_card_3.arm(regeneration=True)
    NI_card_1.arm_clock(length=N, sample_rate=ni_rate)

    OPX_client.build_stack()
    clk = OPX_client.create_new_do_elem(do_channel=1, length=500)
    M = 100000
    # Emit the inter-shot delay in 10 ms chunks; a single huge OPX delay can
    # overflow the QUA delay argument.
    CHUNK_NS = 10_000_000
    total_buffer_ns = max(CHUNK_NS, int(cycle_delay_ms) * 1_000_000)
    n_chunks = int(np.ceil(total_buffer_ns / CHUNK_NS))
    chunk_ns = int(total_buffer_ns / n_chunks)
    with OPX_client.for_("j", 0, M, 1):
        with OPX_client.for_("i", 0, N + 1, 1):
            OPX_client.set_digital_voltage(element=clk)
            OPX_client.delay(delay_ns)
        for _ in range(n_chunks):
            OPX_client.delay(chunk_ns)
    OPX_client.execute(wait=False)

    def get_frame(timeout_ms=1000):
        b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    # Shot period sets how long to wait for the atom frame of the next cycle.
    shot_timeout_ms = int(N / SPM + cycle_delay_ms + 5000)

    try:
        logger.info("Discarding first frame pair")
        try:
            get_frame(timeout_ms=15000)
            get_frame(timeout_ms=10000)
        except Exception:
            pass

        while thread.running:
            try:
                frame1 = get_frame(timeout_ms=shot_timeout_ms)
            except Exception:
                if not thread.running:
                    break
                continue
            if not thread.running:
                break
            try:
                frame2 = get_frame(timeout_ms=shot_timeout_ms)
            except Exception:
                if not thread.running:
                    break
                continue

            diff = frame1.astype(np.float64) - frame2.astype(np.float64)

            # DN/photon depends on the container the camera hands back: Mono16
            # is 12-bit data left-shifted by 4, so it reads 16x higher per photon.
            if dn_override > 0:
                dn_calib = dn_override
            else:
                dn_calib = (DN_PER_PHOTON_16BIT if frame1.dtype == np.uint16
                            else DN_PER_PHOTON_12BIT)
            dn_per_photon = dn_per_photon_effective(dn_calib, gain_db, calib_gain_db)

            # Clipping truncates the sum, so a saturated shot reads LOW. Warn
            # rather than silently reporting a wrong number.
            ceiling = saturation_ceiling(frame1.dtype, dn_calib)
            n_sat_atoms, n_sat_bg, peak_dn = count_saturated(frame1, frame2, ceiling)
            saturated = (n_sat_atoms + n_sat_bg) > SATURATION_WARN_PIXELS

            roi = _resolve_roi(diff.shape, roi_x0, roi_x1, roi_y0, roi_y1)
            sum_dn, offset = integrate_signal(diff, roi, subtract_offset)

            photons_sensor = sum_dn / dn_per_photon
            n_atoms = atom_number(sum_dn, dn_per_photon, eta_collect,
                                  optics_transmission, r_sc, exposure_s)
            dataset.n_atom_history.append(n_atoms)

            # Fixed background minimum; max follows the atom peak.
            # 99.95th percentile is robust against hot pixels.
            frame_max = float(np.percentile(diff, 99.95))
            if frame_max <= dataset.diff_levels[0] + 1:
                frame_max = dataset.diff_levels[0] + 10  # empty frame floor
            dataset.img_item.setImage(
                diff, levels=(dataset.diff_levels[0], frame_max),
                autoLevels=False)

            hist = np.array(dataset.n_atom_history)
            logger.info(
                f"shot {hist.size}: sum={sum_dn:.3e} DN (offset {offset:+.3f} DN/px), "
                f"peak={peak_dn:.0f}/{ceiling:.0f} DN, photons={photons_sensor:.3e}, "
                f"N_atoms={n_atoms:.3e} | "
                f"running mean {hist.mean():.3e} +/- {hist.std():.2e}"
            )
            if saturated:
                logger.warn(
                    f"SATURATED at {gain_db}dB gain: {n_sat_atoms} px railed in the atom "
                    f"frame, {n_sat_bg} in the background, ceiling {ceiling:.0f} DN. "
                    f"The sum is truncated, so N_atoms={n_atoms:.3e} is an UNDERESTIMATE. "
                    f"Lower the gain or shorten the exposure."
                )
    finally:
        dataset.camera_client.stop_acquisition()
        if h1 is not None:
            try:
                NI_card_2.finalize(h1, timeout=120.0, force_finish=True)
            except Exception:
                pass
        if h2 is not None:
            try:
                NI_card_3.finalize(h2, timeout=120.0, force_finish=True)
            except Exception:
                pass
        try:
            NI_card_1.finalize_clock()
        except Exception:
            pass
