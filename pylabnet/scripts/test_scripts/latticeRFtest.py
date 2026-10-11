# Lattice RF / TIMING TEST -- generates the transport RF, measures nothing.
#
# WHAT THIS IS
# A signal generator with a self-check. It drives the two lattice AOM drivers
# and the NI lines exactly as a real transport would, so the resulting RF can
# be looked at on a scope or a spectrum analyser. There are no atoms, no
# camera and no data: MOT, CMOT, Dark MOT, PGC, the holds, the release, the
# time of flight and both frames have all been removed.
#
# THE PROFILE, SET LEG BY LEG
#     accelerate at a1 for t1  ->  hold that velocity for t2  ->  decelerate
#     at a2 for t3
# One DIM segment per leg, one dio6 trigger per segment. Nothing is derived
# from a distance or a target speed; the five numbers are what you set, and
# v_peak, v_end and the distance are computed from them and reported.
#
# WHAT IT CHECKS BY ITSELF (all of this lands in the log)
#   - both drivers answer *IDN?
#   - every segment is read back with Rseg after being written, and an ERROR
#     is raised naming any the driver did not accept -- the failure that
#     otherwise looks healthy right up until the RF does the wrong thing
#   - the arithmetic: v_peak = a1*t1, v_end = v_peak - a2*t3, the distance,
#     and the staircase each ramp will really be played as (step size, step
#     period, velocity change per step)
#   - that your five settings are self-consistent. They are one number more
#     than the profile needs, so v_end is only zero when a2*t3 == a1*t1; when
#     it is not, the script says so, gives the a2 and t3 that would fix it,
#     and warns that the carrier will jump at every cycle boundary
#   - that no two segment triggers collide on one sample, and that no leg is
#     shorter than the trigger lead
#
# WHAT IT CANNOT CHECK -- THIS IS WHY YOU POINT AN INSTRUMENT AT IT
#   - whether the RF actually follows the programmed profile. Everything above
#     is what the script BELIEVES it commanded; only a measurement on the port
#     shows what the driver did with it.
#   - whether the 3 ms trigger lead matches the driver's real latency. Too
#     long and each trigger truncates the ramp before it in mid-flight.
#   - whether the end-of-chain flag really wraps back to segment 1. Watch
#     whether the pattern repeats every cycle or plays once and then sits flat.
#   - whether any RF reaches the AOM at all, or at what power.
#
# It also tests nothing whatsoever about atoms. A profile that looks perfect
# here can still be far too violent for the trap to hold.
#
# HOW THE MOVE IS MADE
# ONE lattice beam is detuned (up, via DIM-3000 COM4 in segment mode) while
# the other (down, COM5) stays at the base frequency. A detuning Df between
# the two beams walks the standing wave at v = (lambda/2)*Df; at 795 nm that
# is 3.975 mm/s per 10 kHz. Units throughout are mm, mm/s and mm/s^2 --
# 1000 mm/s^2 is 1 m/s^2, and gravity is 9810 mm/s^2.
#
# WHAT IS ON THE NI CARDS DURING A RUN
# Only four lines are driven: dio4 and dio5 gate both lattice beams (HIGH is
# on) and ao6/ao7 carry the AM. Both go to their off state together for the t0
# block that opens every cycle -- TTLs LOW and AM at 0 V, so nothing is gated
# on and no RF is being made -- and both come back for the rest of it. dio6
# carries the segment triggers. Everything else -- cooling and repump light, the MOT coils, both
# VCOs, the bias coils, the camera trigger -- is driven to a safe value by
# park_outputs() between passes and when the run stops, because an NI line
# nobody writes keeps whatever the previous script left on it.

import numpy as np
import dim_state
import serial
import time

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

# =============================================================================
INIT_DICT = {
    # ===================== NO ATOM PREPARATION =====================
    # MOT, CMOT, Dark MOT and PGC have been removed. The sub-sequence begins
    # at the lattice step: t = 0 is the start of the pre-move hold, with the
    # lattice already on and stationary.
    #
    # What that costs, so it is not discovered later: nothing loads the
    # lattice, and nothing photographs it. There are no atoms and no camera.
    # This is a TIMING AND RF TEST -- it exercises the DIM segment chain, the
    # dio6 trigger train and the NI waveform so they can be watched on a scope
    # or a spectrum analyser.
    #
    # During a run only four NI lines are driven: dio4/dio5 hold the lattice
    # beams on and ao6/ao7 hold their AM level, while dio6 carries the segment
    # triggers. Everything else -- cooling and repump light, the MOT coils,
    # both VCOs, the bias coils, the camera trigger -- is left to
    # park_outputs(), which drives every one of them to a safe value between
    # points and when the run stops.

    # ===================== LATTICE AM =====================
    # DIM-3000 set to AMP=320 (+32dBm) via dim_rf_on.py; AM=1.0V => full max.
    # AOM thermal stability: lattice TTL stays ON & AM stays at lat_am_v
    # for the ENTIRE cycle.
    'lat_am_v': {'Lattice AM (V)': '1.0'},

    # ===================== THE MOVE, LEG BY LEG =====================
    # Set explicitly. Nothing here is derived from a distance or a velocity:
    # accelerate at a1 for t1, hold whatever velocity that reached for t2,
    # then decelerate at a2 for t3.
    #
    #   v_peak = a1 * t1            <- what the acceleration leg reaches
    #   v_end  = v_peak - a2 * t3   <- where the deceleration leaves it
    #
    # THAT IS ONE NUMBER MORE THAN THE PROFILE NEEDS. Only a2*t3 == a1*t1
    # brings the lattice back to rest. Any other combination ends the
    # sub-sequence with the standing wave still moving, and it keeps moving
    # until the next cycle's first trigger -- which is a perfectly reasonable
    # thing to want to look at, so it is allowed rather than silently
    # corrected. The log prints v_end at every start, warns when it is not
    # zero, and gives the a2 and the t3 that would stop it exactly.
    #
    # Units are mm/s^2 and ms. 9810 mm/s^2 is 1 g. Times are snapped to the
    # 50 us NI sample grid, so fractional ms are fine.
    'accel_1_mmps2': {'a1 -- Acceleration (mm/s^2)': '5000'},
    'accel_time_ms': {'t1 -- Acceleration Time (ms)': '1000'},
    'plateau_time_ms': {'t2 -- Constant Velocity Time (ms) [0 = no plateau]': '1000'},
    'decel_2_mmps2': {'a2 -- Deceleration (mm/s^2)': '5000'},
    'decel_time_ms': {'t3 -- Deceleration Time (ms)': '1000'},

    # t4: how long the lattice sits STOPPED at the end of each cycle, before
    # the chain wraps and the next acceleration begins. This is the only part
    # of the cycle where the standing wave is at rest, so it is what you look
    # at on the analyser to confirm the carrier really came back to base.
    #
    # It used to be hardcoded at 1 ms, which was 12% of a 33 ms cycle when the
    # legs were 10 ms each -- and 0.03% of a 3 s cycle once they were 1000 ms,
    # i.e. invisible. It does not scale itself, so it is set here. Keep it at
    # or above the trigger lead: a leg shorter than the lead cannot absorb the
    # lead of the trigger that follows it.
    #
    # 150 ms is 5% of the 3000 ms move set above -- enough to see on an
    # analyser, and enough to measure the carrier against base. Scale it with
    # the legs: the log prints the at-rest share of every period and warns
    # below 2%.
    'rest_time_ms': {'t4 -- Stop Time, Lattice At Rest (ms)': '150'},

    # Fires each segment trigger this far ahead of its leg, to absorb the
    # driver's internal latency between seeing the edge and starting to play.
    'seg_lead_ms': {'DIM3000 Trigger Lead (ms)': '3'},

    # ===================== DRIVER OFF BEFORE EACH CYCLE =====================
    # t0: every cycle opens with the lattice beams gated OFF for this long,
    # via the dio4/dio5 TTLs (HIGH = off), before anything else happens. The
    # lead-in and the legs then run behind it exactly as before -- the whole
    # rest of the sub-sequence is simply shifted later by t0.
    #
    # Both the gate and the RF go off together: dio4/dio5 drop to
    # LATTICE_TTL_OFF and ao6/ao7 drop to 0 V for the window, so there is no
    # beam and no drive. Set this to 0 for the old behaviour, where the beams
    # were on for the whole buffer and only the segment chain moved.
    't0_off_ms': {'t0 -- Lattice Beams OFF Before Cycle (ms)': '2'},
    # +1 or -1: sign of the detuning, i.e. which way the standing wave runs.
    'move_direction': {'Move Direction (+1 or -1)': '-1'},

    # ===================== DIM-3000 AOM DRIVER =====================
    'dim_base_freq_mhz': {'DIM3000 Base Freq (MHz)': '100.800'},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)': '32.0'},

    # ===================== NOTHING AFTER THE MOVE =====================
    # The release, the time of flight, both camera frames and the gap between
    # them are gone, and with them every parameter that only served them:
    # exposure, gap, end-of-cycle pad, pixel size, imaging AOM powers, both
    # VCO levels and the display scaling. The sub-sequence now ENDS when the
    # deceleration ends.

    # ===================== HOW LONG TO PLAY IT =====================
    # The buffer is one move; this is how many times it repeats before the
    # tasks are torn down and rebuilt. With no camera to pace the run it is
    # what sets how long the pattern stays up on an analyser between rearms.
    'cycles_per_run': {'Cycles per Run': '20'},

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

# GATE POLARITY for the two lattice beam TTLs (dio4, dio5).
#
# HIGH drives the beams ON; LOW gates them OFF. Every consumer in this file --
# the waveform, the safe state, park_outputs -- reads these two names instead
# of a bare 0 or 1, because a polarity that is right in the waveform and wrong
# in the park state is invisible until the beams sit off between passes.
#
# Flipping the wiring means editing these two lines and nothing else. NOTE
# that the other lattice scripts in this directory (fastLattice.py,
# latticeMoveSweep.py, latticeHoldSweep_fastLattice.py, and jingfastMOT's
# park state) still assume the OPPOSITE convention, LOW = on. If this file is
# right about the hardware then those are inverted too and are parking the
# lattice off between runs.
LATTICE_TTL_ON = 1
LATTICE_TTL_OFF = 0

# THE LATTICE IS DELIBERATELY LEFT ON, which is where this differs from
# jingfastMOT (it parks dio4/dio5 HIGH and ao6/ao7 at 0 V, i.e. lattice off).
# These scripts drive the lattice AOMs at a constant RF level for the entire
# cycle precisely so the crystals stay in one thermal state and the fibre
# coupling does not drift. Parking the lattice off between runs throws that
# away: the crystal cools, and the next run starts with the coupling walking
# back to equilibrium for minutes. So ao6/ao7 stay at the AM voltage the run
# used and dio4/dio5 stay at LATTICE_TTL_ON.
#
# dio2 = 1 is "coils off": the state the sequence drives from just before
# release through imaging. The coil analog goes to 0 V regardless, so the
# current setpoint is zero either way.
SAFE_DO_LATTICE_ON = {
    'dio0': 0,      # camera trigger idle
    'dio1': 0,      # MOT AOM enable -- light off
    'dio2': 1,      # coils TTL: the sequence's "coils off" state
    'dio3': 0,      # repump AOM enable -- light off
    'dio4': LATTICE_TTL_ON,   # lattice up   -- left running on purpose
    'dio5': LATTICE_TTL_ON,   # lattice down -- left running on purpose
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
    :param lattice_on: False parks the lattice off (ao6/ao7 = 0, TTL at
        LATTICE_TTL_OFF), for a run that had the lattice disabled to begin with
    """
    safe_ao = dict(SAFE_AO_BASE)
    safe_do = dict(SAFE_DO_LATTICE_ON)
    if lattice_on:
        safe_ao['ao6'] = float(lattice_am)
        safe_ao['ao7'] = float(lattice_am)
    else:
        safe_ao['ao6'] = 0.0
        safe_ao['ao7'] = 0.0
        safe_do['dio4'] = LATTICE_TTL_OFF
        safe_do['dio5'] = LATTICE_TTL_OFF
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
    """No windows: this script displays nothing, because it measures nothing.

    The scan grid, the frame viewer and the camera client all went with the
    imaging. What is left is the three NI cards, which experiment() needs.
    """
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

    NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
    OPX_client = kwargs["OPX_OPX"]

    # =========================================================================
    # READ GUI PARAMETERS
    # =========================================================================
    # Lattice
    lat_am_v = float(dataset.get_input_parameter("lat_am_v"))

    a1_mmps2 = float(dataset.get_input_parameter("accel_1_mmps2"))
    t1_ms = float(dataset.get_input_parameter("accel_time_ms"))
    t2_ms = float(dataset.get_input_parameter("plateau_time_ms"))
    a2_mmps2 = float(dataset.get_input_parameter("decel_2_mmps2"))
    t3_ms = float(dataset.get_input_parameter("decel_time_ms"))
    t4_ms = max(0.0, float(dataset.get_input_parameter("rest_time_ms")))
    seg_lead_ms = float(dataset.get_input_parameter("seg_lead_ms"))
    t0_off_ms = max(0.0, float(dataset.get_input_parameter("t0_off_ms")))
    dim_base_freq_mhz = float(dataset.get_input_parameter("dim_base_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))
    move_direction = int(dataset.get_input_parameter("move_direction"))
    cycles_per_run = max(1, int(dataset.get_input_parameter("cycles_per_run")))

    # ============ The move, exactly as specified ============
    # v = lambda/2 * df. For 795nm: v[mm/s] = 0.0003975 * df[Hz]
    LATTICE_WAVELENGTH = 795e-9
    v_per_Hz_mmps = LATTICE_WAVELENGTH / 2 * 1e3  # mm/s per Hz of detuning
    base_freq_hz = int(round(dim_base_freq_mhz * 1e6))

    if t1_ms <= 0 or t3_ms <= 0:
        logger.error("t1 and t3 must both be greater than 0 ms: without an "
                     "acceleration and a deceleration leg there is no move.")
    if a1_mmps2 <= 0 or a2_mmps2 <= 0:
        logger.error("a1 and a2 must both be greater than 0 mm/s^2.")
    if t4_ms <= 0:
        logger.warn(
            "t4 = 0: there is no at-rest leg. The chain then wraps straight "
            "off the end of the deceleration, so the lattice is never "
            "stationary and the last leg can be cut short by the wrap. Set t4 "
            "to at least the trigger lead unless you specifically want that.")

    v_peak_mmps = a1_mmps2 * (t1_ms * 1e-3)
    v_end_mmps = v_peak_mmps - a2_mmps2 * (t3_ms * 1e-3)
    df_peak_hz = int(round(v_peak_mmps / v_per_Hz_mmps))
    df_end_hz = int(round(v_end_mmps / v_per_Hz_mmps))
    f_peak_hz = base_freq_hz + move_direction * df_peak_hz
    f_end_hz = base_freq_hz + move_direction * df_end_hz
    # Area under the velocity profile: half of each ramp plus the plateau,
    # with the last leg finishing at v_end rather than at zero.
    distance_mm = (0.5 * v_peak_mmps * t1_ms * 1e-3
                   + v_peak_mmps * t2_ms * 1e-3
                   + 0.5 * (v_peak_mmps + v_end_mmps) * t3_ms * 1e-3)
    move_total_ms = t1_ms + t2_ms + t3_ms

    def ramp_steps(duration_ms, span_hz):
        """Staircase for one linear ramp: how the driver really plays it.

        Each ramp gets its OWN step size and step period, because a1 and a2
        are independent -- the two ramps generally cover the same frequency
        span over different times, so a single shared step would be right for
        at most one of them.

        The step count is the fewest that keeps step_time inside its 16-bit
        field (256 us full scale), which is also the coarsest staircase the
        hardware permits, so dv per step is worth reporting.
        """
        if duration_ms <= 0:
            return 0, 1, 1, 0.0
        num_steps = max(int(np.ceil(duration_ms * 1000.0 / 256.0)), 10)
        step_freq = max(1, int(span_hz // num_steps)) if span_hz > 0 else 1
        step_time_us = duration_ms * 1000.0 / num_steps
        step_time_arb = max(1, min(65535, int(round(step_time_us * 1000.0 / 3.90625))))
        return num_steps, step_freq, step_time_arb, step_time_us

    def build_legs():
        """The three legs, in play order, one DIM segment each.

        ONE definition, two consumers -- the segment writer and the dio6
        trigger builder -- so the trigger count cannot disagree with the chain
        length. A zero-length plateau is dropped rather than written: an empty
        segment would still consume a trigger and leave the driver one short
        for every cycle afterwards.
        """
        legs = [
            {'name': 'accelerate', 'dur_ms': t1_ms,
             'f0': base_freq_hz, 'f1': f_peak_hz, 'accel': a1_mmps2},
            {'name': 'constant velocity', 'dur_ms': t2_ms,
             'f0': f_peak_hz, 'f1': f_peak_hz, 'accel': 0.0},
            {'name': 'decelerate', 'dur_ms': t3_ms,
             'f0': f_peak_hz, 'f1': f_end_hz, 'accel': -a2_mmps2},
            # The at-rest leg. It carries flag 8 (end of chain), and it is
            # also what guarantees the deceleration in front of it is played
            # to completion rather than being cut off by the chain wrapping.
            # Its length is t4 rather than a hardcoded 1 ms: at 1 ms it was
            # 12% of a 33 ms cycle but 0.03% of a 3 s one, so the only
            # stationary part of the trace disappeared the moment the legs
            # were lengthened.
            {'name': 'at rest', 'dur_ms': t4_ms,
             'f0': f_end_hz, 'f1': f_end_hz, 'accel': 0}
        ]
        legs = [lg for lg in legs if lg['dur_ms'] > 0]
        for lg in legs:
            if lg['f0'] == lg['f1']:
                # A hold is written as step 0 / time 1 and sits there until
                # the next trigger, so it has no staircase to describe. Give
                # it steps=0 rather than whatever ramp_steps() would invent
                # for a zero-wide span -- the log reads off this field, and
                # "40 steps of 1 Hz" would be a plateau that does not step.
                lg.update(steps=0, step_freq=0, step_arb=1, step_us=0.0)
            else:
                ns, sf, sa, su = ramp_steps(lg['dur_ms'], abs(lg['f1'] - lg['f0']))
                lg.update(steps=ns, step_freq=sf, step_arb=sa, step_us=su)
        return legs

    legs = build_legs()
    n_legs = len(legs)

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

    def _dim_program_segments(ser):
        """Write the three legs as the segment chain and arm it.

        One Wseg per leg, in play order, the LAST carrying type flag 8 -- the
        end-of-chain marker that sends the next trigger back to segment 1 --
        and every earlier leg carrying 1. That flag on the wrong leg is the
        one mistake a readback still looks healthy for, so it is derived from
        position rather than typed.

        Each ramp carries its OWN step_freq and step_time, because a1 and a2
        are set independently. Hold legs get step 0 and sit at their frequency
        until the next trigger, so the plateau's length is set by the NI
        trigger train and not here.

        Segments past the chain are rewritten base->base. The flag 8 makes the
        chain wrap before reaching them, so this is insurance -- but a stale
        segment left by an earlier, longer profile is exactly the thing that
        would detune a beam without appearing anywhere in this run's log.
        """
        for i, leg in enumerate(legs, start=1):
            flag = 8 if i == n_legs else 1
            if leg['f0'] == leg['f1']:
                _dim_send(ser, f"Wseg:{i};{flag};{leg['f0']};{leg['f1']};0;1;0")
            else:
                _dim_send(ser, f"Wseg:{i};{flag};{leg['f0']};{leg['f1']};"
                               f"{leg['step_freq']};{leg['step_arb']};0")
        for i in range(n_legs + 1, 4):
            _dim_send(ser, f"Wseg:{i};1;{base_freq_hz};{base_freq_hz};0;1;0")
        _dim_send(ser, 'Mseg:1')
        return n_legs

    # Open both ports once and hold them for the run. The chain is written
    # a single time here: unlike the sweep scripts there is only one profile,
    # so nothing needs reprogramming while the sequence plays.
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
        _dim_send(ser_fixed, 'Mseg:0')
        _dim_send(ser_fixed, f'FRQ:{base_freq_hz}')
        _dim_send(ser_fixed, f'AMP:{amplitude_x10}')
        _dim_send(ser_fixed, 'OUT_on')
        logger.info(f"DIM-3000 fixed: {base_freq_hz/1e6:.3f} MHz, "
                    f"+{amplitude_x10/10:.1f} dBm, ON")
        # Sweep driver: base, full amp, on, then the chain, then armed
        _dim_send(ser_sweep, f'FRQ:{base_freq_hz}')
        _dim_send(ser_sweep, f'AMP:{amplitude_x10}')
        _dim_send(ser_sweep, 'OUT_on')
        _dim_program_segments(ser_sweep)
        logger.info(
            f"DIM-3000 sweep: {n_legs} segment(s) written and armed, "
            f"+{amplitude_x10/10:.1f} dBm")
        for lg in legs:
            if lg['steps']:
                logger.info(
                    f"    {lg['name']:<18} {lg['dur_ms']:6.2f}ms  "
                    f"{lg['f0']/1e6:.6f} -> {lg['f1']/1e6:.6f} MHz  "
                    f"{lg['steps']} steps of {lg['step_freq']} Hz every "
                    f"{lg['step_us']:.1f}us  (dv {lg['step_freq']*v_per_Hz_mmps:.4f} mm/s/step)")
            else:
                logger.info(
                    f"    {lg['name']:<18} {lg['dur_ms']:6.2f}ms  hold at "
                    f"{lg['f0']/1e6:.6f} MHz")
        # Read every segment back. A driver that silently ignores one leaves
        # the move half-programmed while every other line here still prints as
        # healthy, and the only symptom is RF that does not do what the log
        # says it does.
        missing = []
        for i in range(1, n_legs + 1):
            reply = _dim_query(ser_sweep, f'Rseg:{i}')
            logger.info(f"  Rseg:{i} ({legs[i-1]['name']}) -> "
                        f"{reply or '<no reply>'}")
            if not reply:
                missing.append(i)
        if missing:
            logger.error(
                f"DIM-3000 did not read back segment(s) {missing} of "
                f"{n_legs}. The chain is not fully programmed -- the RF will "
                f"not follow the profile in this log. Do not trust a trace "
                f"taken now.")
    else:
        logger.info("DIM-3000: ports unavailable, will run without serial control")
        if ser_sweep is not None:
            ser_sweep.close()
            ser_sweep = None
        if ser_fixed is not None:
            ser_fixed.close()
            ser_fixed = None

    wait_time = float(dataset.get_input_parameter("wait_time"))

    # =========================================================================
    # CHANNELS
    # =========================================================================
    # Only three of these carry anything that changes: the two lattice TTLs
    # and the segment trigger. The rest are written as constants so a value
    # left behind by whatever ran before cannot persist through this run --
    # an unwritten NI line holds its last state, and coils left energised or
    # cooling light left on would not be visible anywhere in this script.
    LATTICE_UP_do, LATTICE_DOWN_do = "dio4", "dio5"
    SEG_TRIG_do = "dio6"
    LATTICE_UP_ao, LATTICE_DOWN_ao = "ao6", "ao7"
    opx_trigger_do = 1

    # =========================================================================
    # TIMING
    # =========================================================================
    ni_rate = 20000
    SPM = ni_rate // 1000
    delay_ns = int(round(1e9 / ni_rate)) - 500
    # The OPX pauses this long after each pass through the buffer. It is
    # the sample clock, so during the pause the NI cards do not advance:
    # every line HOLDS its last sample and the RF sits at whatever the
    # final segment left it on. It is dead time inside the period, not
    # between periods, so it belongs in any rate this script reports.
    OPX_GAP_NS = 10_000_000
    # Rounds, because the derived ramp/plateau durations are floats (a
    # 14.14ms ramp is 283 samples). The 50us sample grid is 20x finer than
    # whole ms, so snapping here costs ~0.3% of a ramp instead of ~7%.
    def ms2s(t): return int(round(t * SPM))

    seg_lead_s = ms2s(seg_lead_ms)
    seg_pulse_s = max(1, SPM // 20)   # ~50us trigger pulse
    # The driver-off block that opens every cycle. Snapped to the same 50 us
    # grid as everything else, so a t0 below half a sample rounds to nothing
    # rather than to one stray sample.
    t0_off_s = ms2s(t0_off_ms)

    # =========================================================================
    # TIMELINE
    # The sub-sequence is [beams off t0] then [pad] then the move, and ends
    # when the deceleration ends. There is nothing before it and nothing
    # after it.
    # =========================================================================

    logger.warn(
        "This script only makes the RF. MOT, CMOT, Dark MOT, PGC, the holds, "
        "the release, the time of flight and both camera frames have all been "
        "deleted -- there are no atoms, no camera and no measurement. What is "
        "real is the DIM segment chain, the dio6 trigger train and the "
        "lattice drive; watch those on a scope or an analyser.")
    logger.info(f"Time at start {time.perf_counter_ns()}")
    logger.info(
        "Profile: "
        + " -> ".join("{} {:.2f}ms".format(lg['name'], lg['dur_ms'])
                      for lg in legs)
        + f"   ({move_total_ms:.2f}ms total, {n_legs} segment(s))")
    logger.info(
        f"  a1={a1_mmps2:g} mm/s^2 ({a1_mmps2/9810:.3f} g) for t1={t1_ms:g}ms "
        f"-> v_peak={v_peak_mmps:.3f} mm/s "
        f"(detuning {move_direction*df_peak_hz:+d} Hz, "
        f"{f_peak_hz/1e6:.6f} MHz)")
    logger.info(
        f"  t2={t2_ms:g}ms at constant velocity"
        + ("  [no plateau segment]" if t2_ms <= 0 else ""))
    logger.info(
        f"  a2={a2_mmps2:g} mm/s^2 ({a2_mmps2/9810:.3f} g) for t3={t3_ms:g}ms "
        f"-> v_end={v_end_mmps:.3f} mm/s "
        f"(detuning {move_direction*df_end_hz:+d} Hz, "
        f"{f_end_hz/1e6:.6f} MHz)")
    logger.info(f"  distance travelled {distance_mm:.4f} mm")

    # The five settings are one more than the profile needs, and this is the
    # consequence. A non-zero v_end is not a rounding error: the standing wave
    # is still translating when the sub-sequence ends, and goes on translating
    # until the next cycle's first trigger resets the chain to base.
    if abs(v_end_mmps) < 1e-9:
        logger.info("  v_end is 0: the lattice comes to rest. a2*t3 == a1*t1.")
    else:
        need_a2 = v_peak_mmps / (t3_ms * 1e-3) if t3_ms > 0 else float('nan')
        need_t3 = (v_peak_mmps / a2_mmps2 * 1e3) if a2_mmps2 > 0 else float('nan')
        logger.warn(
            f"v_end = {v_end_mmps:+.3f} mm/s, NOT zero: a2*t3 "
            f"({a2_mmps2 * t3_ms * 1e-3:.3f}) does not equal a1*t1 "
            f"({v_peak_mmps:.3f}). The lattice is still moving when the move "
            f"ends and keeps moving until the next cycle starts. To stop it "
            f"exactly, use a2 = {need_a2:.1f} mm/s^2 at the present t3, or "
            f"t3 = {need_t3:.3f} ms at the present a2.")
        logger.warn(
            f"  Also: the chain ends at {f_end_hz/1e6:.6f} MHz but segment 1 "
            f"starts at {base_freq_hz/1e6:.6f} MHz, so the next trigger jumps "
            f"the frequency by {abs(f_end_hz-base_freq_hz)} Hz. That step is "
            f"real RF and will show on the analyser.")

    # Two triggers on the same sample merge into one pulse, leaving the driver
    # a trigger short for the rest of the run -- the failure that produces a
    # plausible-looking trace rather than an error, so it is checked.
    _starts = np.cumsum([0] + [ms2s(lg['dur_ms']) for lg in legs[:-1]])
    if len(set(_starts)) < len(_starts):
        logger.error(
            f"Two segment triggers would land on the same sample (leg starts "
            f"{list(_starts)}). Lengthen the shortest leg.")
    _short = [lg['name'] for lg in legs if lg['dur_ms'] < seg_lead_ms]
    if _short:
        logger.warn(
            f"Leg(s) {_short} are shorter than the {seg_lead_ms}ms trigger "
            f"lead, so a trigger arrives before the driver has started the "
            f"previous segment. Shorten Trigger Lead or lengthen those legs.")
    # The buffer opens with a seg_lead_ms lead-in so the first trigger gets
    # the same head start as the others; without it that trigger clamps to
    # sample 0 and the acceleration leg is played short by exactly the lead.
    if seg_lead_ms > 0:
        logger.info(
            f"  Lead-in: {seg_lead_ms:g}ms of stationary lattice before the "
            f"move, so every trigger fires its full {seg_lead_ms:g}ms ahead "
            f"and each trigger-to-trigger gap equals its leg duration.")

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence():
        """The buffer for one cycle of the move. Nothing else is in it.

        Only the lattice lines and the trigger train are written. The trigger
        train is built from the same `legs` list the segment chain was written
        from, so the pulse count cannot disagree with the chain length.
        """
        # LEAD-IN. Every trigger fires seg_lead_ms before its leg, so leg 1
        # needs that much room in front of it or its trigger gets clamped to
        # sample 0 -- and then the gap from trigger 1 to trigger 2 is
        # (t1 - seg_lead) instead of t1, which cuts the acceleration ramp
        # short and leaves the plateau below the velocity this script reports.
        # Starting the first leg at seg_lead_s costs a few ms of dead time and
        # makes every trigger-to-trigger interval equal its leg duration.
        #
        # DRIVER OFF FIRST. The whole timeline starts t0_off_s later so the
        # cycle opens with both beams gated off; the lead-in and the legs keep
        # their existing spacing behind it. Leg 1's trigger therefore lands on
        # the very sample the beams come back on, which leaves it exactly the
        # same seg_lead_ms of stationary lattice it had before.
        leg_starts = []
        t = t0_off_s + seg_lead_s
        for leg in legs:
            leg_starts.append(t)
            t += ms2s(leg['dur_ms'])
        # One trailing sample so the buffer never ends on a trigger edge.
        N = t + 1

        # EVERYTHING OFF for the opening t0 block: the gate TTLs go to
        # LATTICE_TTL_OFF *and* the AM rails go to 0 V, so during that window
        # the beams are gated and the drivers are making no RF either. Written
        # as one explicit run rather than by slicing, so it stays obvious that
        # everything after t0_off_s is the untouched original waveform.
        #
        # Dropping the AM is a deliberate departure from the rest of these
        # scripts, which hold ao6/ao7 at a constant level all cycle to keep the
        # AOM crystals in one thermal state. Here the crystal loses its drive
        # for t0 out of every period, so if the fibre coupling starts wandering
        # between passes this is the first thing to put back.
        ludo = [LATTICE_TTL_ON] * N
        lddo = [LATTICE_TTL_ON] * N
        luao = [lat_am_v] * N
        ldao = [lat_am_v] * N
        for k in range(min(t0_off_s, N)):
            ludo[k] = LATTICE_TTL_OFF
            lddo[k] = LATTICE_TTL_OFF
            luao[k] = 0.0
            ldao[k] = 0.0

        seg_do = [0] * N
        for trig_t in leg_starts:
            t0 = max(0, trig_t - seg_lead_s)
            for k in range(seg_pulse_s):
                if t0 + k < N:
                    seg_do[t0 + k] = 1

        return ludo, lddo, luao, ldao, seg_do, N

    # =========================================================================
    # RUN LOOP -- one profile, played over and over
    # =========================================================================
    # There is a single buffer and a single segment chain, so nothing has to
    # be reprogrammed while the sequence plays. Each pass uploads it, plays it
    # cycles_per_run times, tears the tasks down and parks the outputs.
    #
    # Playback is timed rather than watched: with no camera there is nothing
    # to wait on, but the buffer length and the sample rate are both known
    # exactly, so the run takes cycles_per_run * N / ni_rate seconds. The wait
    # is that plus a margin, in slices so Stop stays responsive.
    STOP_SLICE_S = 0.2

    def sleep_watching(seconds):
        """Wait, but notice Stop within STOP_SLICE_S rather than at the end."""
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            if not thread.running:
                return False
            time.sleep(min(STOP_SLICE_S, max(0.0, deadline - time.perf_counter())))
        return thread.running

    ludo_p, lddo_p, luao_p, ldao_p, seg_p, total = build_sub_sequence()
    # PERIOD. The clock runs total+1 edges to play `total` samples, then the
    # OPX waits OPX_GAP_NS before the next pass. Both are inside one period,
    # so quoting only the buffer length would overstate the repetition rate.
    clock_s = (total + 1) / ni_rate
    gap_s = OPX_GAP_NS * 1e-9
    period_s = clock_s + gap_s
    logger.info(
        f"Buffer: {total} samples = {total / SPM:.2f} ms, "
        f"{int(np.sum(seg_p))} segment trigger(s).")
    if t0_off_s > 0:
        logger.info(
            f"  t0: everything OFF for the first {t0_off_s / SPM:.2f} ms of "
            f"every cycle -- dio4/dio5 = {LATTICE_TTL_OFF} and ao6/ao7 = 0 V "
            f"-- then ON (dio4/dio5 = {LATTICE_TTL_ON}, ao6/ao7 = "
            f"{lat_am_v:g} V) for the remaining "
            f"{(total - t0_off_s) / SPM:.2f} ms. Segment trigger 1 fires at "
            f"{t0_off_s / SPM:.2f} ms, on the sample the beams return.")
        off_frac = 100.0 * t0_off_s / total
        if off_frac < 1.0:
            logger.warn(
                f"The t0 window is only {off_frac:.3f}% of the buffer "
                f"({t0_off_s / SPM:.2f} ms in {total / SPM:.2f} ms), so it is "
                f"easy to miss on an analyser -- trigger on the dio4 falling "
                f"edge rather than on the cycle, or raise t0.")
    else:
        logger.info(
            f"  t0 = 0: beams on (dio4/dio5 = {LATTICE_TTL_ON}) and ao6/ao7 at "
            f"{lat_am_v:g} V for the whole buffer; nothing gates the lattice.")
    logger.info(
        f"PERIOD = {period_s * 1e3:.2f} ms  ({clock_s * 1e3:.2f} ms clocked "
        f"+ {gap_s * 1e3:.2f} ms OPX gap, clock frozen and every output "
        f"holding its last sample)  ->  {1.0 / period_s:.2f} Hz")
    # The stationary window is what you point the analyser at to confirm the
    # carrier came home to base. It does not scale with the legs, so its share
    # of the period is printed rather than left to be discovered on a scope.
    still_ms = seg_lead_ms + t4_ms + gap_s * 1e3
    still_frac = 100.0 * still_ms / (period_s * 1e3)
    logger.info(
        f"AT REST = {still_ms:.2f} ms of the {period_s * 1e3:.2f} ms period "
        f"({still_frac:.2f}%): {seg_lead_ms:g} ms lead-in + {t4_ms:g} ms t4 + "
        f"{gap_s * 1e3:g} ms OPX gap. The rest of the period the standing "
        f"wave is moving.")
    if still_frac < 2.0:
        logger.warn(
            f"Only {still_frac:.2f}% of each period has the lattice stopped, "
            f"so the at-rest stretch is easy to miss on an analyser and the "
            f"carrier's return to {base_freq_hz / 1e6:.6f} MHz is hard to "
            f"confirm. Raise t4 (now {t4_ms:g} ms) if you are looking for it: "
            f"{0.05 * move_total_ms:.0f}-{0.2 * move_total_ms:.0f} ms would be "
            f"5-20% of this {move_total_ms:g} ms move.")
    logger.info(
        f"Burst: {cycles_per_run} periods = {cycles_per_run * period_s:.3f} s, "
        f"then the tasks are torn down, the outputs parked and "
        f"{wait_time:g}s waited before re-arming. The RF pattern is periodic "
        f"WITHIN a burst; the gap between bursts is not deterministic.")

    while thread.running:
        try:
            NI_card_1.finalize_clock()
        except Exception:
            pass

        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(do_channel=LATTICE_UP_do, value=ludo_p,
                                 sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=LATTICE_DOWN_do, value=lddo_p,
                                 sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=SEG_TRIG_do, value=seg_p,
                                 sample_rate=ni_rate)
        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel=LATTICE_UP_ao, voltages=luao_p,
                                 sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel=LATTICE_DOWN_ao, voltages=ldao_p,
                                 sample_rate=ni_rate)

        h1 = NI_card_2.arm(regeneration=True)
        h2 = NI_card_3.arm(regeneration=True)
        NI_card_1.arm_clock(length=total, sample_rate=ni_rate)

        OPX_client.build_stack()
        clk = OPX_client.create_new_do_elem(do_channel=opx_trigger_do,
                                            length=500)
        with OPX_client.for_("j", 0, cycles_per_run, 1):
            with OPX_client.for_("i", 0, total + 1, 1):
                OPX_client.set_digital_voltage(element=clk)
                OPX_client.delay(delay_ns)
            OPX_client.delay(OPX_GAP_NS)
        OPX_client.execute(wait=False)

        play_s = cycles_per_run * period_s
        sleep_watching(play_s + 0.5)

        for card, h, name in ((NI_card_2, h1, "2"), (NI_card_3, h2, "3")):
            try:
                card.finalize(h, timeout=120.0, force_finish=True)
            except Exception as e:
                logger.info(f"NI card {name} finalize: {e!r}")
        try:
            NI_card_1.finalize_clock()
        except Exception as e:
            logger.info(f"Clock finalize: {e!r}")

        # Sequence tasks are closed, so static writes land. The lattice is
        # deliberately left ON and stationary between passes; see
        # park_outputs() for why parking it off would cost fibre coupling.
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
            # Mseg:0 FIRST, then the frequency: leaving the driver armed means
            # the next stray edge on the rear Seg input moves the carrier, and
            # the next script to run inherits a driver that does not sit still.
            # Order matters -- setting the frequency while still in segment
            # mode lets the segment engine overwrite it on that same edge.
            _dim_send(ser_sweep, 'Mseg:0')
            _dim_send(ser_sweep, f'FRQ:{base_freq_hz}')
            logger.info(f"DIM sweep parked at base {base_freq_hz/1e6:.3f} MHz, "
                        f"segment mode OFF (lattice on, stationary)")
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
