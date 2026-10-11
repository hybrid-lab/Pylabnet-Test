# CHECK copy of fastLattice.py. Identical in every respect but one: the
# cooling and repump AOMs stay OFF during FRAME 1. Everything else -- the
# sequence, the timings, the channels, the GUI parameters, the display -- is
# fastLattice.py unchanged.
#
# What that makes the two frames:
#   Frame 1  atoms present, NO imaging light. Whatever it records did not come
#            from the imaging beams: AOM leakage, room light, lattice-induced
#            fluorescence, camera dark signal.
#   Frame 2  'Atoms Clear Time' later, imaging light ON as usual, atoms gone.
#            The ordinary background frame.
#
# The displayed image is frame1 - frame2, exactly as in fastLattice.py, and
# that sign convention is what makes this a check:
#
#   POSITIVE counts (visible on the default scale) = light in frame 1 that the
#     imaging pulse did not put there. On a healthy setup there is none and
#     the picture is essentially EMPTY -- an empty image here is a PASS, not a
#     broken run. A cloud-shaped positive signal means the atoms are being lit
#     by something other than the imaging pulse; a diffuse or structured one
#     means stray light or AOM leakage.
#   NEGATIVE counts = the ordinary frame 2 imaging background, which is now
#     unopposed. Expected, and clipped to black by Display Min = 0. Set
#     Display Min negative (e.g. -2000, autoscale off) to look at it.
#
# Because frame 1 no longer carries the fluorescence, the saturation check in
# the acquisition loop watches FRAME 2 -- the frame that now has the light in
# it. Frame 1 cannot saturate here, so testing it would silently never fire.
#
# Original description follows.
#
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
    # Gap between frame 1 and frame 2, and so also the time the atoms are
    # given to leave before the background frame is taken. Free to set from
    # the GUI; 100 ms is the fastLattice default.
    'atoms_clear_time': {'Atoms Clear Time (ms) [frame 1 -> frame 2 gap]': '100'},

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
}


# Percentile used when autoscaling the top of the colour range. Not the max:
# one hot pixel would then set the scale for the whole frame and push the
# cloud back down into the dark end. 99.95% of a 2448x2048 frame still leaves
# ~2500 pixels above the ceiling, which is inside the core of any real cloud.
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
    title = ("PGC + Lattice CHECK (frame 1 dark - frame 2 lit)" if lattice_on
             else "PGC CHECK, lattice OFF (frame 1 dark - frame 2 lit)")

    dataset.img_win = pg.GraphicsLayoutWidget(show=True, title=title)
    dataset.img_win.resize(700, 600)
    cmap = pg.colormap.get('inferno')   # matches bias sweep colormap

    vb = dataset.img_win.addViewBox(lockAspect=True)
    vb.invertY(True)
    dataset.img_item = pg.ImageItem(axisOrder='row-major')
    dataset.img_item.setLookupTable(cmap.getLookupTable())
    # Starting range only; the acquisition loop re-levels every frame pair.
    dataset.img_item.setLevels(read_display_levels(dataset)[:2])
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
        f"Fast PGC{'+Lattice' if lattice_on else ' (LATTICE OFF)'} "
        f"[CHECK: no cooling/repump light during frame 1]: "
        f"mot={mot_loading_ms}ms, cmot={mot_switch_ms}+{cmot_hold_ms}ms, "
        f"dark={dark_ms}ms, pgc_cool={pgc_cool_ms}ms, "
        f"pgc_rampdown={pgc_rampdown_ms}ms, "
        f"lattice_hold={lattice_hold_ms}ms, tof={tof_us}us, "
        + (f"(lattice CONSTANT at {lat_end_v:.2f}V whole cycle), " if lattice_on
           else "(dio4/dio5 HIGH, ao6/ao7 at 0V whole cycle), ")
        + f"release={release/SPM:.0f}ms, "
        + f"frame1->frame2 gap={atoms_clear_ms}ms, N={N}"
    )
    logger.info(
        "CHECK MODE: the cooling and repump AOMs are OFF for frame 1, so the "
        "difference image should be EMPTY. Anything POSITIVE is light that "
        "reached the atoms without the imaging pulse; the frame 2 imaging "
        "background is negative and clipped at Display Min."
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
    # Imaging pulses.
    #
    # THE CHECK: frame 1 gets its camera trigger and its exposure, but no
    # light. Both TTLs are held LOW across the whole frame-1 window while the
    # analog amplitudes stay parked at full, which is the same treatment the
    # hold and the TOF get above -- the AOM crystal keeps seeing constant RF
    # drive, so its thermal state, and with it the frame 2 pulse that follows,
    # is unchanged from fastLattice.py.
    #
    # Written out explicitly rather than left to the array defaults: these
    # samples being zero is the entire point of this file, and a later edit
    # that reinstates a default would otherwise turn the check back into an
    # ordinary image with nothing to show for it.
    for s in range(aom_pulse_s):
        if f1 + s < N:
            ado[f1 + s] = 0
            rdo[f1 + s] = 0
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
            dataset.img_item.setImage(
                diff, levels=(disp_lo, hi), autoLevels=False)
            # One stats line for the first few pairs: enough to choose a manual
            # Display Max from real data instead of guessing, and it says
            # outright when the ADC itself is clipping -- no display range can
            # rescue a frame that saturated at 65532 counts, only less gain.
            if pairs_logged < 3:
                pairs_logged += 1
                sample = diff[::AUTOSCALE_STRIDE, ::AUTOSCALE_STRIDE]
                peak = int(diff.max())
                # Saturation is judged on the RAW frame that has light in it,
                # not on the difference: two clipped frames subtract to nearly
                # zero, so a saturated frame can leave diff.max() looking
                # reassuringly moderate while the data is worthless. In THIS
                # script that frame is frame 2 -- frame 1 is taken with the
                # imaging light off and cannot saturate, so testing it would
                # never fire and a clipped frame 2 would pass unnoticed.
                n_sat = int((frame2 >= SATURATION_COUNTS).sum())
                pct_sat = 100.0 * n_sat / frame2.size
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
                        "%d pixels (%.3f%%) of the LIT frame (frame 2) are "
                        "SATURATED "
                        "at %d counts. The image will stay two-tone whatever "
                        "the display range, because the data itself is "
                        "clipped. Lower Camera Gain (now %.1f dB, full well "
                        "%.0f photons) or shorten the exposure until this "
                        "reads zero."
                        % (n_sat, pct_sat, SATURATION_COUNTS, applied_gain,
                           SENSOR_MAX_MONO16 / dn_per_photon)
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
