# Lattice TRANSPORT (move) sweep -- RF OUTPUT TEST, COILS NEVER ENERGISED.
#
# This is latticeMoveSweep.py with (a) every coil forced off and (b) the
# DIM-3000 segment chain rearranged so flag 8 lands on a sentinel instead of
# on the deceleration. Everything else -- the phases, the durations, the
# anchors, the sample count, the camera triggers -- is untouched, so the RF
# the lattice drivers put out is the RF of a real transport run and can be
# looked at on a scope or spectrum analyser in its true timing context.
#
# THE SEGMENT CHAIN (the second change; see N_DIM_SEGMENTS below)
#   Wseg:1 flag 1  base -> move   accelerate
#   Wseg:2 flag 1  move -> move   plateau
#   Wseg:3 flag 1  move -> base   DECELERATE
#   Wseg:4 flag 8  base -> base   sentinel, nothing to play
# The parent script marks the DECEL with flag 8. latticeRFtest.py records
# that a flag-8 segment does not get played, which would mean the parent
# never decelerates and leaves the beams detuned by the full move Df. Putting
# the marker on a segment that does nothing makes the profile correct either
# way. This costs a FOURTH dio6 trigger, fired inside the decel ramp; it
# moves no anchor and adds no samples.
#
# WHAT IS OFF (see COILS_ENABLED / BIAS_COILS_ENABLED below)
#   - MOT/CMOT quadrupole: dio2 held in its OFF state (1) for the whole
#     buffer, ao2 held at 0 V. The MOT Coils / CMOT Coils GUI voltages are
#     read but not used.
#   - Bias X/Y/Z (ao8/ao9/ao10): held at 0 V. The Bias GUI values are read
#     but not used.
#
# WHAT IS UNCHANGED, ON PURPOSE
#   - The timeline. Coils TTL Lead and Coils Restore still enter the anchor
#     arithmetic exactly as before -- Coils Restore sets the tail of each
#     sub-sequence and therefore its length, so touching it WOULD move the
#     timing. Both are left alone.
#   - The cooling/repump light, the VCOs, the lattice AM, and the dio0
#     camera-trigger waveform. Nothing is listening to dio0 any more, but the
#     pulses are part of the timeline and dropping them would change what the
#     buffer looks like on a scope.
#
# THE ONE EXCEPTION: the lattice TTL is no longer held on for the whole
# cycle. dio4/dio5 are driven HIGH (beam OFF) for the pre-move hold and go
# LOW (beam ON) on the first sample of the acceleration, so the RF has a hard
# edge exactly where the profile starts. No anchor moves and no sample is
# added -- it is a level change inside a window that already existed. See
# LATTICE_OFF_DURING_PRE_HOLD; set it False to get the never-switching
# lattice back.
#   - Every anchor and the sample count, the fourth segment trigger included:
#     it lands seg_lead_ms before decel_end, inside a ramp that is already
#     playing, which is the same relationship triggers 2 and 3 have to the
#     legs they follow.
#
# NO CAMERA, NO IMAGES, NO ANALYSIS. The camera client, the frame grabbing,
# the Gaussian fits, the atom-number and density calculations, the scan-grid
# panels and the temperature window are all gone -- with no quadrupole there
# is no MOT, so there was nothing in the frames but background. Roughly a
# thousand lines of the parent script went with them. The run loop is paced
# by the CLOCK instead: the buffer length and the sample rate are exact, so
# each burst takes a known time and the loop simply waits it out.
#
# ONE DISTANCE, REPEATING. The three-point sweep is gone too. Move Distance
# sets a single plateau detuning, the DIM segment chain is written ONCE at
# startup, and the same move plays over and over with nothing changing
# between repeats -- which is what a scope needs in order to average.
#
# To put the coils back and turn this into the ordinary move sweep again, set
# COILS_ENABLED = True and BIAS_COILS_ENABLED = True. To go back to the
# parent's 3-segment chain with flag 8 on the decel -- which is the A/B test
# that settles what flag 8 really does -- set N_DIM_SEGMENTS = 3.
#
# ---------------------------------------------------------------------------
# Original description follows.
# ---------------------------------------------------------------------------
#
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
#   6. Frame 1 imaging pulse   (light + dio0 trigger; nothing is recording)
#   7. Atoms clear
#   8. Frame 2 imaging pulse   (light + dio0 trigger; nothing is recording)
#
# PGC (polarization gradient cooling / optical molasses):
#   - No B-field (coils already off)
#   - Cooling beam: very far red-detuned (~-100 MHz), low power (~10-20%)
#   - Repump: very low (~3-5%) or off
#   - Typical duration: 5-20 ms
#   - Cools atoms well below Doppler limit via Sisyphus effect

import numpy as np
import dim_state
import serial
import time

# The pylabnet dataset layer still touches these NumPy aliases, so the shims
# stay even though nothing in this file fits or plots any more.
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float


# =============================================================================
# COILS OFF -- the one and only difference from latticeMoveSweep.py
# =============================================================================
# False drives the coil channels to their off state for every sample of the
# buffer instead of playing the MOT/CMOT profile:
#
#   COILS_ENABLED      dio2 (quadrupole TTL, 0=ON / 1=OFF) and ao2 (current
#                      setpoint).  False -> dio2 = 1 and ao2 = 0 V throughout.
#   BIAS_COILS_ENABLED ao8/ao9/ao10 (bias X/Y/Z).  False -> 0 V throughout.
#
# Nothing else keys off these flags. The anchors, the sample count and every
# other channel are computed exactly as in the parent script, so the RF timing
# is identical with the coils on or off. Set both True to get the ordinary
# move sweep back.
COILS_ENABLED = False
BIAS_COILS_ENABLED = False


# =============================================================================
# DIM-3000 SEGMENT CHAIN: flag 8 goes on a SENTINEL, not on the decel
# =============================================================================
# Type flag 8 marks the end of the segment chain. What the driver does with
# the segment carrying it is the open question, and this repo answers it two
# incompatible ways:
#
#   latticeRFtest.py       the flag-8 segment is NOT played -- it appends a
#                          throwaway leg to absorb the flag
#   latticeAccelSweep.py   the flag-8 segment IS played and only the NEXT
#   latticemove&back.py    trigger wraps the chain back to segment 1
#
# latticeMoveSweep.py puts flag 8 on Wseg:3, the DECELERATION. If the first
# reading is right, that script never decelerates: the driver stops at the
# plateau frequency, the two beams stay detuned by the full move Df, and the
# standing wave keeps walking for the rest of the cycle. That is exactly the
# state latticeHoldSweep_fastLattice.py has to clean up before it can trust
# its own positions ("parked at the PLATEAU frequency ... since only segment 3
# walks it back to base").
#
# This script follows latticeRFtest.py: three real legs on flag 1, plus a
# fourth base->base segment that exists only to carry flag 8. Nothing is
# riding on which reading is correct, because the only segment that can be
# skipped is one with nothing to play. The cost is one extra dio6 trigger,
# fired inside the decel ramp, which moves no anchor and adds no samples.
#
# HOW TO TELL WHICH READING IS RIGHT: put a spectrum analyser on the COM4
# output and watch the end of the move. Four segments -> the carrier returns
# to base. Set this to 3 and the chain goes back to the parent script's
# layout, with flag 8 on the decel; if the carrier then STAYS at the plateau
# frequency, latticeRFtest.py is right and every move script in the repo
# needs this fix.
N_DIM_SEGMENTS = 4      # 4 = sentinel scheme (latticeRFtest.py), 3 = parent


# =============================================================================
# LATTICE GATE DURING THE PRE-MOVE HOLD
# =============================================================================
# The parent script never gates the lattice: dio4/dio5 sit LOW and ao6/ao7 sit
# at lat_am_v for every sample of the cycle, MOT and CMOT and PGC included, so
# the AOM crystals stay in one thermal state and the fibre coupling does not
# drift. Nothing ever switches, so nothing on a scope marks where the move
# begins.
#
# True closes both beams for the whole pre-move hold and reopens them on the
# sample the acceleration starts, which puts a hard edge on the RF exactly at
# the start of the profile -- the same trick latticeRFtest.py uses with its
# t0 block. The gate length IS the pre-move hold, so 'Pre-move Hold (ms)'
# sets it.
#
# POLARITY, because it is the opposite of what it reads like: on dio4/dio5,
#     0 = LOW  = beam ON      (the value the parent holds all cycle)
#     1 = HIGH = beam OFF
# so "gated off" means these lines are driven to 1, not 0.
#
# The AM rails do NOT drop with the gate. The TTL is the switch; holding the
# RF drive level constant is what keeps the crystal thermally still across it.
# Dropping ao6/ao7 to 0 V as well would gate the beam the same way and undo
# that, so it is not done here.
LATTICE_OFF_DURING_PRE_HOLD = True


# =============================================================================
INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '10'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V) [UNUSED - coils off]': '6'},
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
    'cmot_coils_voltage': {'CMOT Coils Voltage (V) [UNUSED - coils off]': '8'},
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
    'move_ramp_ms': {'Move Accel/Decel Ramp (ms)': '100'},
    'move_plateau_ms': {'Move Plateau (ms)': '100'},
    'seg_lead_ms': {'DIM3000 Trigger Lead (ms)': '3'},
    # +1 or -1: which sign of detuning -> reverses lattice motion direction.
    'move_direction': {'Move Direction (+1 or -1)': '-1'},

    # ===================== MOVE DISTANCE (ONE POINT) =====================
    # A single transport distance, played over and over. It sets the plateau
    # detuning: v_peak = dist / (ramp + plateau), Df = v_peak / (lambda/2).
    # There is no sweep any more, so the DIM segment chain is written ONCE at
    # startup and never reprogrammed while the sequence runs.
    'dist_mm': {'Move Distance (mm)': '3'},

    # ===================== DIM-3000 AOM DRIVER =====================
    'dim_base_freq_mhz': {'DIM3000 Base Freq (MHz)': '100.800'},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)': '32.0'},

    # ===================== FIXED TOF after release =====================
    'tof_us': {'TOF (us)': '0'},

    # ===================== IMAGING =====================
    # These two are NOT disabled with the coils: coils_off and, through it,
    # the tail of each sub-sequence are computed from them, and Coils Restore
    # sets the sample count. Changing either would move the timeline, which is
    # the one thing this variant must not do.
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release) [timing only]': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2) [sets cycle length]': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    # Still read: it sets aom_pulse_s, the width of the two imaging light
    # pulses, which is part of the timeline whether or not a camera is
    # listening. The dio0 trigger pulses are still emitted for the same
    # reason -- see the run loop.
    'camera_exposure_us': {'Camera Exposure Time (us) [pulse width only]': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '2'},

    # ===================== BIAS COILS =====================
    # Read but NOT applied while BIAS_COILS_ENABLED is False: ao8/ao9/ao10 are
    # held at 0 V for the whole buffer.
    'bias_x_v': {'Bias X (V) [UNUSED - coils off]': '0.0'},
    'bias_y_v': {'Bias Y (V) [UNUSED - coils off]': '0.0'},
    'bias_z_v': {'Bias Z (V) [UNUSED - coils off]': '0.0'},

    # There is no display block and no warm-up count any more. Both existed to
    # serve the images: the colour range scaled the panels, and the warm-up
    # cycles let the MOT reach its cyclic steady state before any frame was
    # kept. With no camera and no atoms there is nothing to scale and nothing
    # to settle, so every cycle is identical and the first one is as good as
    # the last.

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
    """Bind the NI cards. No camera and no windows: this run produces RF, not
    images, so there is nothing to display and nothing to scale."""
    dataset = kwargs['dataset']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']
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

    logger.info(
        "RF-ONLY VARIANT: quadrupole coils %s, bias coils %s. One distance, "
        "repeating; no camera, no images, no fits. The timeline, the light, "
        "the VCOs and the lattice are as in latticeMoveSweep.py."
        % ("ON" if COILS_ENABLED else "OFF for the whole buffer",
           "ON" if BIAS_COILS_ENABLED else "0 V"))

    NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
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

    dist_mm = float(dataset.get_input_parameter("dist_mm"))
    pre_hold_ms = int(dataset.get_input_parameter("pre_hold_ms"))
    post_hold_ms = int(dataset.get_input_parameter("post_hold_ms"))
    move_ramp_ms = int(dataset.get_input_parameter("move_ramp_ms"))
    move_plateau_ms = int(dataset.get_input_parameter("move_plateau_ms"))
    seg_lead_ms = int(dataset.get_input_parameter("seg_lead_ms"))
    dim_base_freq_mhz = float(dataset.get_input_parameter("dim_base_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))
    move_direction = int(dataset.get_input_parameter("move_direction"))
    tof_us = int(dataset.get_input_parameter("tof_us"))

    # ============ Move physics ============
    # v = lambda/2 * df. For 795nm: v[mm/s] = 0.0003975 * df[Hz]
    LATTICE_WAVELENGTH = 795e-9
    v_per_Hz_mmps = LATTICE_WAVELENGTH / 2 * 1e3  # mm/s per Hz of detuning
    base_freq_hz = int(round(dim_base_freq_mhz * 1e6))

    # Trapezoid: distance = v_peak * (move_ramp_ms + move_plateau_ms).
    # ONE distance now, so one plateau velocity and one plateau frequency --
    # computed here, written into the segment chain once at startup, and never
    # touched again while the sequence runs.
    move_total_ms = move_ramp_ms + move_plateau_ms  # ramp + plateau (one side)
    v_peak_mmps = dist_mm / (move_total_ms * 1e-3)         # mm/s
    df_hz = int(round(v_peak_mmps / v_per_Hz_mmps))        # Hz of detuning
    move_freq_hz = base_freq_hz + move_direction * df_hz

    # Two triggers on one sample merge into a single pulse and the driver is
    # a trigger short for the rest of the run -- the chain then stalls and
    # every later cycle plays one segment out of phase. Each trigger also
    # fires seg_lead_ms before its segment, so a leg SHORTER than the lead
    # would have its trigger arrive before the previous leg had even started.
    # Neither is possible with the defaults; both become possible the moment
    # someone shortens a ramp or lengthens the lead, and neither shows up in a
    # segment readback.
    _legs_ms = [('accel', move_ramp_ms), ('plateau', move_plateau_ms),
                ('decel', move_ramp_ms)]
    if N_DIM_SEGMENTS == 4:
        # Trigger 4 sits inside the decel; the sentinel itself is played
        # during the post-move hold, so that is the gap that must clear it.
        _legs_ms.append(('post-move hold', post_hold_ms))
    _short = [n for n, ms in _legs_ms if 0 < ms < seg_lead_ms]
    if _short:
        logger.error(
            f"Leg(s) {', '.join(_short)} are shorter than the "
            f"{seg_lead_ms}ms trigger lead, so a segment trigger fires before "
            f"the segment ahead of it has started. Shorten DIM3000 Trigger "
            f"Lead or lengthen those legs.")
    _zero = [n for n, ms in _legs_ms if ms <= 0]
    if _zero:
        logger.error(
            f"Leg(s) {', '.join(_zero)} have zero length, so two segment "
            f"triggers land on the same sample and merge into one pulse. The "
            f"DIM is then one trigger short for the rest of the run.")

    logger.info(
        f"Move physics: d={dist_mm}mm, ramp={move_ramp_ms}ms, "
        f"plateau={move_plateau_ms}ms -> v_peak={v_peak_mmps:.2f}mm/s, "
        f"Df={move_direction * df_hz / 1e3:+.3f}kHz "
        f"(plateau carrier {move_freq_hz / 1e6:.6f} MHz)")

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

        FOUR segments, not three, and the difference is the whole point of
        this variant -- see N_DIM_SEGMENTS above. The three real legs all
        carry type flag 1 and a fourth, base->base, no-op segment carries the
        flag 8 end-of-chain marker:

            Wseg:1  flag 1   base -> move    accelerate   (real)
            Wseg:2  flag 1   move -> move    plateau      (real)
            Wseg:3  flag 1   move -> base    DECELERATE   (real)
            Wseg:4  flag 8   base -> base    sentinel     (nothing to play)

        The parent script puts flag 8 on Wseg:3, so if the marked segment is
        skipped the deceleration never plays and the driver is left sitting at
        the plateau frequency with the lattice still walking. Here the only
        thing that can be skipped is a segment that does nothing, so the
        profile comes out right under EITHER reading of the flag.

        Same ramp time across calls; only the endpoint changes per scan point.
        Re-asserts Mseg:1 at the end so the DIM stays in segment-trigger mode.
        """
        freq_range = abs(move_freq - base_freq_hz)
        max_step_us = 256.0
        min_steps = max(1, int(np.ceil(move_ramp_ms * 1000.0 / max_step_us)))
        num_steps = max(min_steps, 10)
        step_freq = max(1, freq_range // num_steps) if freq_range > 0 else 1
        step_time_us = (move_ramp_ms * 1000.0) / num_steps
        step_time_arb = max(1, min(65535, int(round(step_time_us * 1000.0 / 3.90625))))
        decel_flag = 1 if N_DIM_SEGMENTS == 4 else 8
        _dim_send(ser, f'Wseg:1;1;{base_freq_hz};{move_freq};{step_freq};{step_time_arb};0')
        _dim_send(ser, f'Wseg:2;1;{move_freq};{move_freq};0;1;0')
        _dim_send(ser, f'Wseg:3;{decel_flag};{move_freq};{base_freq_hz};{step_freq};{step_time_arb};0')
        if N_DIM_SEGMENTS == 4:
            # The sentinel. base->base with a zero step, so it holds the
            # frequency the decel just arrived at no matter whether the driver
            # plays it or wraps straight past it.
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
        ns, sf, st = _dim_program_segments(ser_sweep, move_freq_hz)
        # Neutralise anything a LONGER chain left behind. latticemove&back.py
        # writes up to Wseg:7, and a stale segment holding another script's
        # detuning is exactly what would move the lattice without appearing
        # anywhere in this run's log. Flag 8 should wrap the chain before it
        # ever reaches these, so this is insurance -- and it is done ONCE here
        # rather than in _dim_program_segments(), which runs between scan
        # points where every serial write costs 50 ms of dead time.
        for _i in range(N_DIM_SEGMENTS + 1, 8):
            _dim_send(ser_sweep, f'Wseg:{_i};1;{base_freq_hz};{base_freq_hz};0;1;0')
        _dim_send(ser_sweep, 'Mseg:1')
        _dim_send(ser_sweep, f'FRQ:{base_freq_hz}')
        logger.info(
            f"DIM-3000 sweep: {move_freq_hz/1e6:.6f} MHz plateau, "
            f"{ns} steps, step_freq={sf}, step_time_arb={st} (~{move_ramp_ms}ms ramp)")
        if N_DIM_SEGMENTS == 4:
            logger.info(
                "DIM-3000 chain: 4 segments -- accel(1) / plateau(1) / "
                "DECEL(1) / sentinel(8, base->base). Flag 8 is on a segment "
                "with nothing to play, so the deceleration cannot be lost to "
                "it. Wseg 5-7 rewritten base->base.")
        else:
            logger.warn(
                "DIM-3000 chain: 3 segments, flag 8 on the DECELERATION -- "
                "the parent script's layout, kept here only for the A/B test. "
                "If the driver skips the flag-8 segment the carrier will stay "
                "at the plateau frequency and the lattice will keep walking. "
                "Watch COM4 on a spectrum analyser.")
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
        + ("" if COILS_ENABLED else
           f" | COILS OFF for every sample (the coils_off anchor at "
           f"{coils_off/SPM:.0f}ms is still computed, just unused)")
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

        # Coils DO: 0=ON, 1=OFF -- and coils AO, the current setpoint.
        # With COILS_ENABLED False the TTL sits in its OFF state from sample 0
        # and the setpoint never leaves 0 V, so the quadrupole is dead for the
        # entire buffer instead of switching off `coils_lead_ms` before the
        # release. coils_off is still computed above because it is part of the
        # shared anchor arithmetic; it simply has nothing to gate here.
        if COILS_ENABLED:
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
        else:
            cdo = [1] * N          # TTL parked in the OFF state
            cao = [0.0] * N        # current setpoint at zero

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

        # Bias coils: normally held CONSTANT at the GUI values for the ENTIRE
        # sub-sequence. With BIAS_COILS_ENABLED False they stay at 0 V, so no
        # coil on the apparatus carries current during this run.
        if BIAS_COILS_ENABLED:
            biasx = [bias_x_v] * N
            biasy = [bias_y_v] * N
            biasz = [bias_z_v] * N
        else:
            biasx = [0.0] * N
            biasy = [0.0] * N
            biasz = [0.0] * N

        # Lattice: AM constant at lat_am_v for the entire sub-sequence
        # (matches the fast lattice code for AOM thermal stability). TTL LOW
        # is BEAM ON, so the default is a beam that never switches.
        ludo = [0] * N
        lddo = [0] * N
        luao = [lat_am_v] * N
        ldao = [lat_am_v] * N
        if LATTICE_OFF_DURING_PRE_HOLD:
            # Both beams gated OFF (TTL driven HIGH) for the whole pre-move
            # hold, pgc_end -> accel_start, and back ON on the first sample of
            # the acceleration. The AM rails stay up; see the flag above.
            #
            # Segment trigger 1 fires seg_lead_ms BEFORE accel_start, so it
            # lands inside this gated window. That is harmless: dio6 drives
            # the DIM's segment engine, which is running regardless of whether
            # the AOM is passing light, and the lead is exactly what makes the
            # ramp start on time when the beams come back.
            for s in range(pgc_end, min(pre_hold_end, N)):
                ludo[s] = 1
                lddo[s] = 1

        # DIM-3000 segment trigger (dio6): FOUR pulses, each ~seg_lead_ms
        # BEFORE the corresponding segment start (DIM internal latency).
        # Trigger 1: start of accel ramp   (seg1)
        # Trigger 2: start of plateau      (seg2)
        # Trigger 3: start of decel ramp   (seg3)
        # Trigger 4: end of decel          (seg4, the flag-8 sentinel)
        #
        # The fourth is what makes the sentinel scheme work, and it is also
        # why the pulse count has to match the chain length exactly: a segment
        # with no trigger stalls the chain until the NEXT cycle's first
        # trigger, which then plays every later cycle one segment out of
        # phase. Four segments, four triggers.
        #
        # Trigger 4 lands INSIDE the decel ramp, seg_lead_ms before it ends --
        # the same relationship triggers 2 and 3 already have to the legs they
        # follow. It changes no anchor and adds no samples: decel_end and
        # post_hold_end are where they were, so the timeline is untouched.
        # For 0mm reference, no triggers fire and DIM stays at base_freq.
        seg_do = [0] * N
        if do_move:
            trig_times = [accel_start, accel_end, plateau_end]
            if N_DIM_SEGMENTS == 4:
                trig_times.append(decel_end)
            for trig_t in trig_times:
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
    # ONE sub-sequence. The parent script concatenated the warm-up cycles and
    # the three scan points into a single long buffer; there is one distance
    # now, so the buffer IS the cycle. The NI cards are armed with
    # regeneration and the OPX clocks it repeatedly, so the same move plays
    # over and over with nothing changing between repeats -- which is what a
    # scope needs to average on.
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

    # =========================================================================
    # TIMING SUMMARY -- the numbers to set the scope up with
    # =========================================================================
    # Playback is TIMED, not watched. The parent script paced itself on camera
    # frames; with no camera there is nothing to wait on, but the buffer
    # length and the sample rate are both exact, so a burst of M cycles takes
    # M * (total + 1) / ni_rate seconds plus M OPX gaps.
    clock_s = (total + 1) / ni_rate
    M = 4                       # cycles per arming
    run_buffer = 10_000_000     # OPX gap between cycles [ns]
    gap_s = run_buffer * 1e-9
    period_s = clock_s + gap_s
    n_trig = int(np.sum(seg_p))
    logger.info(
        f"Buffer: {total} samples = {total / SPM:.2f} ms per cycle, "
        f"{n_trig} segment trigger(s) on dio6, {N_DIM_SEGMENTS} segments in "
        f"the chain.")
    if n_trig != N_DIM_SEGMENTS:
        logger.error(
            f"{n_trig} trigger(s) for a {N_DIM_SEGMENTS}-segment chain. A "
            f"segment with no trigger stalls the chain until the next cycle's "
            f"first trigger, and every cycle after that plays one segment out "
            f"of phase. Check Move Distance is non-zero and that no leg has "
            f"collapsed to zero length.")
    logger.info(
        f"PERIOD = {period_s * 1e3:.2f} ms  ({clock_s * 1e3:.2f} ms clocked + "
        f"{gap_s * 1e3:.2f} ms OPX gap, during which the clock is frozen and "
        f"every output holds its last sample)  ->  {1.0 / period_s:.3f} Hz")
    logger.info(
        f"Burst: {M} periods = {M * period_s:.3f} s, then the tasks are torn "
        f"down, the outputs parked and {wait_time:g}s waited before "
        f"re-arming. The RF is periodic WITHIN a burst; the gap between "
        f"bursts is not deterministic, so trigger the scope on dio6 rather "
        f"than free-running.")
    logger.info(
        f"Move starts {(pgc_end + pre_hold_s) / SPM:.2f} ms into each cycle. "
        f"Segment trigger 1 fires {seg_lead_ms} ms before that, at "
        f"{(pgc_end + pre_hold_s - seg_lead_s) / SPM:.2f} ms.")
    if LATTICE_OFF_DURING_PRE_HOLD:
        logger.info(
            f"Lattice GATED OFF (dio4/dio5 HIGH) for the pre-move hold, "
            f"{pgc_end / SPM:.2f}-{(pgc_end + pre_hold_s) / SPM:.2f} ms "
            f"({pre_hold_ms} ms), then ON for the rest of the cycle. ao6/ao7 "
            f"stay at {lat_am_v:g} V throughout. The RF reappears on the "
            f"first sample of the acceleration, so that edge marks the start "
            f"of the profile on a scope. Segment trigger 1 fires "
            f"{seg_lead_ms} ms earlier, inside the gated window.")
        if pre_hold_ms <= 0:
            logger.error(
                "Pre-move Hold is 0 ms, so the gate has zero width and "
                "nothing is switched. Give it a few ms or set "
                "LATTICE_OFF_DURING_PRE_HOLD = False.")
    else:
        logger.info(
            f"Lattice ON for the whole cycle at {lat_am_v:g} V, never gated "
            f"(the parent script's behaviour). Nothing on the RF marks where "
            f"the move begins.")

    # =========================================================================
    # MAIN LOOP -- upload, play M cycles, tear down, repeat
    # =========================================================================
    # No camera, no frames, no fits, no plots. The dio0 camera-trigger
    # waveform is still written: it is part of the timeline, and leaving it
    # out would change what the buffer looks like on a scope. Nothing is
    # listening to it.
    STOP_SLICE_S = 0.2

    def sleep_watching(seconds):
        """Wait, but notice Stop within STOP_SLICE_S rather than at the end."""
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            if not thread.running:
                return False
            time.sleep(min(STOP_SLICE_S, max(0.0, deadline - time.perf_counter())))
        return thread.running

    while thread.running:
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
        with OPX_client.for_("j", 0, M, 1):
            with OPX_client.for_("i", 0, total + 1, 1):
                OPX_client.set_digital_voltage(element=clk)
                OPX_client.delay(delay_ns)
            OPX_client.delay(run_buffer)

        OPX_client.execute(wait=False)
        logger.info(f"Playing {M} cycles ({M * period_s:.3f} s)")

        # Half a second of margin on top of the computed burst length, so a
        # little jitter in the OPX start does not have the tasks torn down
        # while the last cycle is still playing.
        sleep_watching(M * period_s + 0.5)

        logger.info(f"Cycle over {time.perf_counter_ns()}")
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
        if not sleep_watching(wait_time):
            break
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
