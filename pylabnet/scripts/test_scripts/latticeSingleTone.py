# ---------------------------------------------------------------------------
# ONE RF TONE THROUGH ONE LATTICE AOM.
# ---------------------------------------------------------------------------
# The smallest thing that puts RF on a lattice AOM and leaves it there: pick a
# beam, pick a frequency, press Run. No atoms, no camera, no sequence, no
# sweep, nothing that changes with time. It is a signal generator with a front
# panel, for aligning a beam, checking diffraction efficiency, measuring the
# gate suppression, or confirming a driver is alive before a real run.
#
# WHICH BEAM -- 'Beam' on the GUI, the one thing this script is configurable
# about:
#
#            serial     gate TTL    AM analog
#   up       COM4       dio4        ao6
#   down     COM5       dio5        ao7
#
# The beam that is NOT selected is gated off (its TTL driven to the off level,
# its AM rail to 0 V) and its serial port is never opened, so its driver keeps
# whatever the last script left on it. Note that this leaves the unselected
# AOM crystal with no drive: the other lattice scripts hold BOTH at a constant
# RF level all cycle precisely so the crystals stay in one thermal state and
# the fibre coupling does not drift. Running this script for a long time will
# therefore cost some coupling on the other beam while it cools, and it will
# walk back to equilibrium over the first minutes of the next real run.
#
# HOW IT DRIVES THE NI CARDS
# With single-sample immediate writes (write_static), NOT a buffered waveform.
# There is nothing periodic here, so there is no sample clock to arm, no
# master/slave routing to negotiate on PXI_Trig7, no OPX and no regenerating
# buffer to tear down. That also means this script cannot leave a clock
# reservation behind for the next one to trip over, which is the usual way a
# quick test breaks the run after it.
#
# Card 2 (DO) and card 3 (AO) are the only clients it needs. Every line on
# them is written, not just the two lattice ones -- an NI line nobody writes
# keeps whatever the previous script left on it, so the cooling and repump
# light, the MOT coils, both VCOs, the bias coils and the camera trigger are
# all driven to a safe value here as well.
#
# WHAT IT DOES NOT DO
# It does not measure anything. Nothing in the log is evidence that RF reached
# the AOM, that the AOM diffracted, or at what power -- only that the driver
# accepted the commands and the NI lines were written. Point a power meter, a
# scope or a spectrum analyser at it; that is the whole purpose.
# ---------------------------------------------------------------------------

import time
import dim_state

try:
    import serial
except ImportError:      # no pyserial -> nothing can be programmed; see below
    serial = None


# ---------------------------------------------------------------------------
# Hardware map
# ---------------------------------------------------------------------------
# One entry per beam, so the GUI's 'Beam' field selects a port, a TTL and an
# analog rail together and they cannot be mixed up individually.
BEAMS = {
    'up': {'port': 'COM4', 'gate_do': 'dio4', 'am_ao': 'ao6'},
    'down': {'port': 'COM5', 'gate_do': 'dio5', 'am_ao': 'ao7'},
}

BAUD = 19200
# Manual: the driver executes about 10 commands/sec. 0.12 s leaves margin.
CMD_GAP_S = 0.12

# DIM-3000 limits, from the manual. Checked here rather than left to the
# driver, which accepts an out-of-range value and quietly does something else
# with it.
FREQ_MIN_MHZ, FREQ_MAX_MHZ = 10.0, 400.0
AMP_MIN_DBM, AMP_MAX_DBM = 14.0, 34.0

# How often Stop is noticed while idling.
STOP_SLICE_S = 0.1


INIT_DICT = {
    # ===================== WHICH BEAM =====================
    # 'up' or 'down' (also accepted: COM4/COM5, 4/5, and the dio number).
    # Anything else stops the run rather than guessing, because guessing wrong
    # means RF on the beam you are not looking at.
    'beam': {'Beam (up = COM4/dio4/ao6, down = COM5/dio5/ao7)': 'up'},

    # ===================== THE TONE =====================
    # One frequency, held for as long as the script runs. 100 MHz is the base
    # both lattice beams normally sit at; a transport run detunes one of them
    # away from it, and this script never does.
    'freq_mhz': {'RF Frequency (MHz) [10 to 400]': '100.800'},

    # Driver output amplitude. +32 dBm is what the lattice scripts assume, and
    # the AM voltage below is referenced to it.
    'amplitude_dbm': {'RF Amplitude (dBm) [14 to 34]': '32.0'},

    # ===================== AM INPUT =====================
    # The analog rail (ao6/ao7) into the driver's AM input, which scales the
    # amplitude set above.
    #
    # THE TWO CONVENTIONS IN THIS DIRECTORY DISAGREE, so read this before
    # typing a number. aomdriverinit.py -- written about these exact drivers
    # -- says 0.0 V gives the amplitude set above, -1.0 V the minimum, and
    # anything above 0 V pushes past the setpoint until it clips at 34 dBm.
    # fastLattice.py and the sweeps instead treat +1.0 V as full scale and
    # -1.0 V as parked off, and run at 0.6 V.
    #
    # 0.0 V is the default here because it is the one value both readings
    # agree is a sane, non-clipping operating point, and because it makes the
    # output equal to the Amplitude field above -- so the number on the GUI is
    # the number at the port. Put 0.6 V in to reproduce what the lattice
    # scripts actually run at.
    #
    # Either way the AM is only meaningful if the driver's Ramoffs is 0. The
    # startup readback below checks that and says so if it is not.
    'am_voltage': {'AM Voltage (V) [0 = exactly the amplitude above]': '0.0'},

    # ===================== GATE =====================
    # 0 gates the selected beam off while leaving the driver making RF, which
    # is how the >40 dB gate suppression gets measured: run once at 1, once at
    # 0, take the difference.
    'beam_on': {'Beam ON? (1 = yes, 0 = gate it off)': '1'},

    # WHICH TTL LEVEL MEANS "ON". This is a real ambiguity in this directory,
    # not a preference: aomdriverinit.py, fastLattice.py, the bias sweeps and
    # jingfastMOT's park state all treat LOW as on, while latticeRFtest.py
    # defines LATTICE_TTL_ON = 1 and its own comments flag that everything
    # else assumes the opposite. One of them is wrong about the wiring.
    #
    # 0 here follows the majority, including aomdriverinit.py, which is the
    # file that documents the drivers themselves ("dio4/dio5 LOW = beam on,
    # HIGH = beam off (>40 dB suppression)").
    #
    # This script is the cheapest place to settle it: set Beam ON to 1, run it
    # once with this at 0 and once at 1, and the power meter says which is
    # right. Then fix whichever scripts disagree.
    'gate_on_level': {'Gate TTL Level That Turns the Beam ON (0 or 1)': '0'},

    # ===================== ON STOP =====================
    # What happens to the driver's RF output when the run is stopped. The gate
    # TTL goes to its off level either way, so the beam is suppressed
    # regardless; this only decides whether the crystal keeps seeing drive.
    #
    # 0 (leave it on) matches every other lattice script and keeps the AOM in
    # one thermal state between runs. Set it to 1 when the next thing that
    # happens is someone putting a hand or a card in the beam path.
    'rf_off_on_stop': {'Turn DIM RF Output OFF on Stop? (1 = yes, 0 = leave on)': '0'},

    # ===================== LOGGING =====================
    # Seconds between driver readbacks while idling. Each one costs a couple
    # of serial round trips and prints the internal temperature, which is the
    # only thing here that changes with time -- a driver heating up is worth
    # seeing during a long alignment. 0 reads back once at the start and then
    # stays quiet.
    'status_period_s': {'Status Readback Period (s) [0 = once at start]': '30'},
}


# ---------------------------------------------------------------------------
# Safe state for everything this script is not using.
# ---------------------------------------------------------------------------
# An NI line nobody writes holds whatever the previous script left on it, and
# the script before this one was very likely a sequence that was force-stopped
# mid-buffer -- so the coils could be at full current and the VCOs at an
# arbitrary detuning. Both cards are therefore written in full, not just the
# two lattice channels.
#
# dio2 = 1 is "coils off": the state the sequences drive from just before
# release through imaging. The coil analog goes to 0 V regardless, so the
# current setpoint is zero either way.
SAFE_AO = {
    'ao1': 0.8,     # MOT AOM drive -- held up for AOM thermal stability
    'ao5': 0.8,     # repump AOM drive
    'ao2': 0.0,     # MOT coils -- current setpoint to zero
    'ao3': 0.0,     # MOT VCO
    'ao4': 0.0,     # repump VCO
    'ao8': 0.0,     # bias X
    'ao9': 0.0,     # bias Y
    'ao10': 0.0,    # bias Z
}

SAFE_DO = {
    'dio0': 0,      # camera trigger idle
    'dio1': 0,      # MOT AOM enable -- cooling light off
    'dio2': 1,      # coils TTL: the sequences' "coils off" state
    'dio3': 0,      # repump AOM enable -- repump off
    'dio6': 0,      # DIM-3000 segment trigger idle: no segment is ever played
}


def idle_until_stop(thread):
    """Block until Stop is pressed, noticing it within STOP_SLICE_S.

    WHY AN ERROR PATH HAS TO WAIT HERE INSTEAD OF JUST RETURNING.
    DataTaker does not call experiment() once. ExperimentWorker.run() is

        while self.running:
            self.experiment(...)

    so returning hands control straight back to a loop that calls it again
    immediately. For a sequence script that costs nothing, because the next
    call blocks for a whole MOT cycle -- but an experiment() that returns in
    a millisecond turns that loop into a spin at thousands of iterations a
    second, each one logging an error over the network and, for the paths
    below the try, re-parking both NI cards over RPC.

    The Stop button is what that starves: it only sets worker.running = False
    from the GUI thread, and the GUI thread is the one drowning in the log
    traffic. The run then cannot be stopped, which is the exact symptom.

    So every exit from experiment() -- the successful one and all six error
    paths -- goes through a wait that ends only when Stop is pressed. The
    error stays on screen, nothing repeats, and the button works.
    """
    while thread.running:
        time.sleep(STOP_SLICE_S)


def parse_beam(raw):
    """'up'/'down' from whatever was typed, or None if it is not a beam.

    Accepts the port, the dio number and the bare digit as well as the name,
    because those are the three other ways this beam gets referred to in the
    surrounding scripts and in conversation at the table.
    """
    key = str(raw).strip().lower()
    aliases = {
        'up': 'up', 'u': 'up', 'com4': 'up', 'dio4': 'up', '4': 'up',
        'ao6': 'up', 'lattice up': 'up', 'latticeup': 'up',
        'down': 'down', 'd': 'down', 'com5': 'down', 'dio5': 'down',
        '5': 'down', 'ao7': 'down', 'lattice down': 'down',
        'latticedown': 'down',
    }
    return aliases.get(key)


def dim_send(ser, cmd):
    """One command, LF-terminated, at a rate the driver can keep up with."""
    ser.write((cmd + '\n').encode())
    time.sleep(CMD_GAP_S)


def dim_query(ser, cmd):
    ser.reset_input_buffer()
    dim_send(ser, cmd)
    time.sleep(0.1)
    return ser.readline().decode(errors='replace').strip()


def report_gpar(ser, logger, first_time):
    """Read the driver's parameter dump and log what matters in it.

    Two fields are worth pulling out. Ramoffs must be 0 or the AM input sits
    on a shifted baseline and the output power is altered even with no AM
    signal -- so the AM Voltage field on the GUI stops meaning what it says,
    silently. Rpcbtemp is the only quantity here that changes while the script
    idles, and a driver climbing in temperature during a long alignment is
    worth seeing before the output starts drifting with it.

    :param first_time: full report on the first call, temperature only after,
        so a 30 s readback does not print the same warning forever.
    :return: (bool) False if the driver did not answer at all
    """
    try:
        par = dim_query(ser, 'Gpar')
    except Exception as exc:
        logger.warn(f"Could not read driver parameters: {exc!r}")
        return False
    if not par:
        logger.warn(
            "Driver did not answer Gpar. It was answering earlier, so either "
            "the cable came out or something else has grabbed the port.")
        return False
    for field in par.split('|'):
        if field.startswith('Rpcbtemp:'):
            try:
                temp_c = int(field.split(':', 1)[1]) / 100.0
            except ValueError:
                continue
            logger.info(f"Driver internal temperature {temp_c:.1f} C")
        elif field.startswith('Ramoffs:') and first_time:
            offs = field.split(':', 1)[1]
            if offs != '0':
                logger.error(
                    f"Driver AM offset (Ramoffs) is {offs}, not 0. The output "
                    f"power is altered by that offset even with no AM signal, "
                    f"so the AM Voltage on the GUI does not mean what it says "
                    f"and the amplitude at the port is NOT the Amplitude "
                    f"field. Zero it with the Adv button on the front panel "
                    f"before trusting any power you measure now.")
    return True


def park_outputs(NI_do, NI_ao, logger, gate_off_level):
    """Drive every line on both cards to a defined level. Never raises.

    This runs from a finally block, where an exception would mask whatever
    actually ended the run and would also skip every channel after the one
    that failed.

    write_static() is preferred because it forces the driver out of stack mode
    first. Older servers do not have it, so fall back to per-channel writes,
    each in its own try so one bad line cannot leave the rest at random.
    """
    safe_ao = dict(SAFE_AO)
    safe_ao['ao6'] = 0.0
    safe_ao['ao7'] = 0.0
    safe_do = dict(SAFE_DO)
    safe_do['dio4'] = gate_off_level
    safe_do['dio5'] = gate_off_level
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
        f"Outputs parked: both lattice gates at {gate_off_level} (beams off), "
        f"ao6/ao7 at 0 V, VCOs 0 V, coils 0 V, bias 0 V, AOM drive 0.8 V, "
        f"cooling/repump light off.")


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    """No windows: this script displays nothing, because it measures nothing.

    Only the two NI cards are needed -- no camera, no card 1, no OPX -- so
    those are the only clients to select in DataTaker.
    """
    dataset = kwargs['dataset']
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

    NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    NI_card_3 = kwargs["nidaqmx_ni_daq_3"]

    # =========================================================================
    # READ AND CHECK THE GUI PARAMETERS
    # =========================================================================
    # Everything is validated before a single line is driven. A bad value that
    # is caught here costs a log line; the same value caught halfway through
    # means RF is already on the wrong beam, or at the wrong power, while the
    # error is being read.
    beam = parse_beam(dataset.get_input_parameter("beam"))
    if beam is None:
        logger.error(
            f"Beam {dataset.get_input_parameter('beam')!r} is not one of "
            f"'up' (COM4/dio4/ao6) or 'down' (COM5/dio5/ao7). Nothing was "
            f"driven -- guessing here would put RF on whichever beam you are "
            f"not looking at.")
        idle_until_stop(thread)
        return
    hw = BEAMS[beam]
    other = 'down' if beam == 'up' else 'up'
    other_hw = BEAMS[other]

    freq_mhz = float(dataset.get_input_parameter("freq_mhz"))
    amp_dbm = float(dataset.get_input_parameter("amplitude_dbm"))
    am_v = float(dataset.get_input_parameter("am_voltage"))
    beam_on = int(float(dataset.get_input_parameter("beam_on"))) != 0
    gate_on_level = 1 if int(float(
        dataset.get_input_parameter("gate_on_level"))) else 0
    gate_off_level = 1 - gate_on_level
    rf_off_on_stop = int(float(
        dataset.get_input_parameter("rf_off_on_stop"))) != 0
    status_period_s = max(0.0, float(
        dataset.get_input_parameter("status_period_s")))

    if not FREQ_MIN_MHZ <= freq_mhz <= FREQ_MAX_MHZ:
        logger.error(
            f"Frequency {freq_mhz} MHz is outside the driver's "
            f"{FREQ_MIN_MHZ:g} to {FREQ_MAX_MHZ:g} MHz range. Nothing was "
            f"driven.")
        idle_until_stop(thread)
        return
    if not AMP_MIN_DBM <= amp_dbm <= AMP_MAX_DBM:
        logger.error(
            f"Amplitude {amp_dbm} dBm is outside the driver's "
            f"+{AMP_MIN_DBM:g} to +{AMP_MAX_DBM:g} dBm range. Nothing was "
            f"driven.")
        idle_until_stop(thread)
        return
    if abs(am_v) > 1.0:
        logger.warn(
            f"AM Voltage {am_v:+.3f} V is outside the +/-1 V the AM input "
            f"expects. Above 0 V the driver is already being pushed past the "
            f"{amp_dbm:g} dBm setpoint and clips at {AMP_MAX_DBM:g} dBm, so "
            f"the extra volts buy no power and the tone is no longer clean.")

    if serial is None:
        logger.error(
            "pyserial is not installed, so the DIM-3000 cannot be programmed. "
            "The NI lines could still be driven, but the driver would be left "
            "on whatever frequency and amplitude the last script set -- which "
            "is exactly the silent wrong answer this script exists to avoid. "
            "Install pyserial, or set the driver by hand with "
            "aomdriverinit.py.")
        idle_until_stop(thread)
        return

    freq_hz = int(round(freq_mhz * 1e6))
    amp_x10 = int(round(amp_dbm * 10))

    logger.info(
        f"Lattice {beam} beam: {freq_mhz:.6f} MHz at {amp_dbm:.1f} dBm on "
        f"{hw['port']}, gate {hw['gate_do']} = "
        f"{gate_on_level if beam_on else gate_off_level} "
        f"({'ON' if beam_on else 'GATED OFF'}), AM {hw['am_ao']} = "
        f"{am_v:+.3f} V. Lattice {other} beam gated off "
        f"({other_hw['gate_do']} = {gate_off_level}, "
        f"{other_hw['am_ao']} = 0 V), its driver untouched.")

    ser = None
    try:
        # =====================================================================
        # PROGRAM THE DRIVER
        # =====================================================================
        try:
            ser = serial.Serial(port=hw['port'], baudrate=BAUD, bytesize=8,
                                parity='N', stopbits=1, timeout=1)
        except Exception as exc:
            logger.error(
                f"Could not open {hw['port']} for the {beam} beam: {exc!r}. "
                f"Either another program still holds it -- DataTaker running "
                f"a lattice script, or a serial terminal left open -- or the "
                f"driver is not on that port. No tone was set up; both beams "
                f"are being gated off and every other line parked, below.")
            idle_until_stop(thread)
            return

        # Record the drivers BEFORE anything is written to them, so the
        # wrapper around this function can put them back exactly as found
        # when the run stops, however it stops (dim_state.py).
        kwargs['dim_snapshots'].append(dim_state.snapshot(ser, hw['port'], logger))
        idn = dim_query(ser, '*IDN?')
        if not idn:
            logger.error(
                f"{hw['port']} opened but the driver did not answer *IDN?. "
                f"No tone was set up -- a driver that will not identify "
                f"itself will not reliably take a frequency either, and a "
                f"frequency that did not land is the one failure that looks "
                f"exactly like a misaligned beam. Both beams are being gated "
                f"off and every other line parked, below.")
            idle_until_stop(thread)
            return
        logger.info(f"DIM-3000 {hw['port']} ({beam}): {idn}")

        # Order matters. Mseg:0 FIRST, then the frequency: a driver left in
        # segment-play mode by a transport run jumps to a segment frequency on
        # the next rising edge at its rear Seg connector, and dio6 going
        # high-impedance when a task closes is such an edge. Setting FRQ while
        # still armed lets the segment engine overwrite it on that same edge,
        # and nothing in the log would say why the tone moved.
        dim_send(ser, 'Mseg:0')
        dim_send(ser, f'FRQ:{freq_hz}')
        dim_send(ser, f'AMP:{amp_x10}')
        dim_send(ser, 'FM_off')
        dim_send(ser, 'RFp_off')
        dim_send(ser, 'OUT_on')
        logger.info(
            f"Driver set: {freq_mhz:.6f} MHz, {amp_dbm:.1f} dBm, segment mode "
            f"off, FM off, pulse off, RF output on.")
        report_gpar(ser, logger, first_time=True)

        # =====================================================================
        # DRIVE THE NI LINES
        # =====================================================================
        # Single-sample immediate writes, so there is no buffered task, no
        # sample clock and nothing for the next script to collide with.
        ao = dict(SAFE_AO)
        do = dict(SAFE_DO)
        ao[hw['am_ao']] = am_v
        ao[other_hw['am_ao']] = 0.0
        do[hw['gate_do']] = gate_on_level if beam_on else gate_off_level
        do[other_hw['gate_do']] = gate_off_level
        try:
            NI_card_3.write_static(ao=ao)
            NI_card_2.write_static(do=do)
        except Exception as exc:
            logger.warn(
                f"write_static unavailable ({exc}); writing channel by "
                f"channel instead")
            for channel, value in ao.items():
                NI_card_3.set_ao_voltage(ao_channel=channel, voltages=value)
            for channel, value in do.items():
                NI_card_2.set_do_voltage(do_channel=channel, value=bool(value))

        logger.info(
            f"RF is on the {beam} AOM"
            + ("" if beam_on else " driver, but the beam is GATED OFF")
            + ". Nothing in this log says it arrived, diffracted, or at what "
              "power -- that is what the power meter is for. Press Stop to "
              "end.")

        # =====================================================================
        # HOLD IT
        # =====================================================================
        # Nothing changes from here on. The loop exists only to notice Stop
        # and, if asked, to report the driver temperature as it settles.
        next_status = time.perf_counter() + status_period_s
        while thread.running:
            time.sleep(STOP_SLICE_S)
            if status_period_s > 0 and time.perf_counter() >= next_status:
                next_status = time.perf_counter() + status_period_s
                # Re-check first: a readback is up to three serial round trips
                # and about 1.3 s in the worst case, and starting one on the
                # sample Stop arrives adds all of that to how long the button
                # appears stuck.
                if not thread.running:
                    break
                report_gpar(ser, logger, first_time=False)
    finally:
        # Teardown is announced step by step. Everything below talks to
        # hardware over a network RPC or a serial port, so any of it can be
        # slow or hang, and when that happens the run looks exactly like a
        # Stop that did nothing. These lines say which step it got stuck on
        # instead of leaving it to be guessed at.
        logger.info("Stop: parking NI outputs...")
        # Gate both beams off and put every other line back to a defined
        # level. The driver's own RF output is left running unless asked
        # otherwise -- see the GUI field for why.
        park_outputs(NI_card_2, NI_card_3, logger, gate_off_level)
        if ser is not None:
            logger.info(f"Stop: finishing with the driver on {hw['port']}...")
            try:
                if rf_off_on_stop:
                    dim_send(ser, 'OUT_off')
                    logger.info(
                        f"DIM {hw['port']} RF output OFF; the driver keeps "
                        f"{freq_mhz:.6f} MHz / {amp_dbm:.1f} dBm for the next "
                        f"OUT_on.")
                else:
                    logger.info(
                        f"DIM {hw['port']} left ON at {freq_mhz:.6f} MHz, "
                        f"{amp_dbm:.1f} dBm, segment mode off. The beam is "
                        f"gated off at the NI card, so no light -- but the "
                        f"crystal keeps its drive and its thermal state.")
            except Exception as exc:
                logger.warn(f"Could not finish with the driver: {exc!r}")
            try:
                # Always close the port: an open handle keeps the driver in
                # REMOTE CONTROL on its front panel and blocks the next script
                # from opening it at all.
                ser.close()
            except Exception:
                pass
        logger.info("Stopped. Nothing is being driven by this script.")
