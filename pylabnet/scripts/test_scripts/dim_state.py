"""Put a DIM-3000 AOM driver back exactly the way a run found it.

Every lattice script reprograms the drivers: frequency, amplitude, output,
the segment table, segment-play mode. Before this module, stopping a script
left them however the run happened to leave them -- usually still in segment
mode, sometimes mid-move at a detuned frequency -- and the next thing to use
them inherited that.

USE
    snap = dim_state.snapshot(ser, 'COM4', logger)   # BEFORE the first write
    ...
    dim_state.restore(ser, snap, logger)             # on the way out, always

The scripts do this through a thin wrapper around experiment(), so restore()
runs in a `finally:` -- on Stop, and equally when the run dies with an
exception half way through. It cannot run if the DataTaker process itself is
killed; aomdriverinit.py is the manual recovery for that.

WHAT IS RESTORED
Everything the driver reports through Gpar that a run can change:
    Rfreq   output frequency        -> FRQ:<Hz>
    Rampl   output amplitude        -> AMP:<dBm*10>
    Rout    RF output on/off        -> OUT_on, or Sout:0
    Rfmon   FM on/off               -> Sfmon:<0|1>    only if it changed
    Rswpm   sweep mode              -> Sswpm:<0..4>   only if it changed
    Rplson  pulse mode on/off       -> Splson:<0|1>   only if it changed
The first three use the same commands the scripts already use, so they are
known to work on this firmware. The last three use the manual's documented
"S prefix writes an R parameter" form; they are sent only when the value
actually differs, so a firmware that ignored them would cost nothing in the
normal case.

WHAT CANNOT BE RESTORED, AND WHAT HAPPENS INSTEAD
Segment-play mode (Mseg) does not appear in Gpar, so there is no way to read
what it was. restore() always leaves it OFF (Mseg:0). That is the only safe
answer: a driver left in segment mode moves its carrier on any stray edge at
the rear Seg input, which is the exact failure this module exists to stop.
Mseg:0 goes out FIRST, before the frequency, because setting the frequency
while still armed lets the segment engine overwrite it on that same edge.

The segment table itself is not restored either. With segment mode off it is
inert, and every script writes a complete fresh chain before using it.

The result is VERIFIED: Gpar is read again afterwards and any field that did
not come back is logged as a warning, naming the field and both values.
"""

import time

# Manual: about ten commands a second is the ceiling. 0.12 s leaves margin.
_GAP_S = 0.12

# Gpar fields a run can change, in the order they are restored.
_FIELDS = ('Rfreq', 'Rampl', 'Rout', 'Rfmon', 'Rswpm', 'Rplson')
# Of those, the ones restored with the "S prefix" write -- only on change.
_S_FIELDS = ('Rfmon', 'Rswpm', 'Rplson')


def _send(ser, cmd):
    ser.write((cmd + '\n').encode())
    time.sleep(_GAP_S)


def _query(ser, cmd):
    try:
        ser.reset_input_buffer()
    except Exception:
        try:
            ser.flushInput()
        except Exception:
            pass
    _send(ser, cmd)
    time.sleep(0.1)
    return ser.readline().decode(errors='replace').strip()


def read_params(ser):
    """Gpar, parsed to {field: int} for the fields this module cares about.

    Missing or unparseable fields are simply absent from the result, so the
    caller can tell "not reported" from "reported as 0".
    """
    raw = _query(ser, 'Gpar')
    out = {}
    for item in raw.split('|'):
        if ':' not in item:
            continue
        key, val = item.split(':', 1)
        key = key.strip()
        if key in _FIELDS:
            try:
                out[key] = int(val.strip())
            except ValueError:
                pass
    return out


def set_frequency(ser, freq_hz, port='', logger=None):
    """Set a driver's output frequency and PROVE it took.

    WHY THIS EXISTS. A DIM-3000 silently ignores commands it does not like,
    and this firmware turned out to ignore a whole family of them: the
    manual's sweep commands (SWPm:, SWPs:, ...) do nothing, and only the
    Gpar-style "S prefix" writes (Sswpm:, Sswps:, ...) are acted on. A base
    frequency typed into a GUI and quietly dropped is the worst version of
    that: both beams stay wherever the last run left them, every distance in
    the log is wrong, and nothing looks broken.

    So the frequency is written, read back through Gpar, and -- if the driver
    did not take it -- written again in the S-prefix form (Sfreq:) and read
    back once more. Returns whether the driver ended up on the asked-for
    frequency, and says so in the log either way.

    :param freq_hz: (float|int) wanted output frequency [Hz]
    :param port: (str) COM port, for the log only
    :return: (bool) True if Gpar reports the frequency asked for
    """
    target = int(round(freq_hz))
    got = None
    for cmd in ('FRQ', 'Sfreq'):
        try:
            _send(ser, f'{cmd}:{target}')
            got = read_params(ser).get('Rfreq')
        except Exception as exc:
            if logger:
                logger.warn(f"DIM-3000 {port}: could not set/verify the "
                            f"frequency ({exc!r}).")
            return False
        if got == target:
            if cmd != 'FRQ' and logger:
                logger.warn(
                    f"DIM-3000 {port}: FRQ: was ignored; {target / 1e6:.6f} MHz "
                    f"took with {cmd}: instead. Verified by Gpar.")
            return True
    if logger:
        seen = 'nothing' if got is None else f"{got / 1e6:.6f} MHz"
        logger.error(
            f"DIM-3000 {port}: asked for {target / 1e6:.6f} MHz but Gpar "
            f"reports {seen} -- neither FRQ: nor Sfreq: took. The beams are "
            f"NOT where this run thinks they are; do not trust its distances.")
    return False


def snapshot(ser, port, logger=None):
    """Record a driver's settings. Call BEFORE the run writes anything to it.

    :return: (dict) the snapshot, or None if the driver did not report the
        three fields a restore needs -- in which case restore() does nothing
        rather than guess.
    """
    if ser is None:
        return None
    try:
        params = read_params(ser)
    except Exception as exc:
        if logger:
            logger.warn(f"DIM-3000 {port}: could not read settings to "
                        f"snapshot ({exc!r}); it will NOT be restored on stop.")
        return None
    if not all(k in params for k in ('Rfreq', 'Rampl', 'Rout')):
        if logger:
            logger.warn(f"DIM-3000 {port}: Gpar did not report frequency, "
                        f"amplitude and output state ({params}); it will NOT "
                        f"be restored on stop.")
        return None
    snap = dict(params)
    snap['_port'] = port
    snap['_ser'] = ser
    if logger:
        logger.info(
            f"DIM-3000 {port}: settings recorded to restore on stop -- "
            f"{snap['Rfreq'] / 1e6:.6f} MHz, {snap['Rampl'] / 10:.1f} dBm, "
            f"RF {'on' if snap['Rout'] else 'OFF'}"
            + (f", FM {'on' if snap['Rfmon'] else 'off'}" if 'Rfmon' in snap else "")
            + (f", sweep mode {snap['Rswpm']}" if 'Rswpm' in snap else "")
            + (f", pulse {'on' if snap['Rplson'] else 'off'}" if 'Rplson' in snap else ""))
    return snap


def restore(snap, logger=None):
    """Put the driver back as snapshot() found it, then close its port.

    Reopens the port if the run already closed it, and closes it at the end
    either way: on an exception path the run never reaches its own close(),
    and a COM port left open blocks the next run from opening it at all.
    Never raises -- this runs from a `finally:`, where an exception would hide
    whatever actually ended the run.
    """
    if not snap:
        return
    ser, port = snap['_ser'], snap['_port']
    try:
        if not ser.is_open:
            ser.open()
        _send(ser, 'Mseg:0')                          # FIRST: disarm
        _send(ser, f"FRQ:{snap['Rfreq']}")
        _send(ser, f"AMP:{snap['Rampl']}")
        _send(ser, 'OUT_on' if snap['Rout'] else 'Sout:0')
        now = read_params(ser)
        for key in _S_FIELDS:
            if key in snap and now.get(key) != snap[key]:
                _send(ser, f"S{key[1:]}:{snap[key]}")
        # Verify rather than assume.
        now = read_params(ser)
        wrong = [f"{k} is {now.get(k)} (was {snap[k]})"
                 for k in _FIELDS if k in snap and now.get(k) != snap[k]]
        if logger:
            if wrong:
                logger.warn(f"DIM-3000 {port}: restored, but not every setting "
                            f"came back: {'; '.join(wrong)}. Segment mode is "
                            f"OFF regardless.")
            else:
                logger.info(
                    f"DIM-3000 {port}: restored to "
                    f"{snap['Rfreq'] / 1e6:.6f} MHz, {snap['Rampl'] / 10:.1f} dBm, "
                    f"RF {'on' if snap['Rout'] else 'OFF'}; segment mode OFF. "
                    f"Verified by Gpar readback.")
    except Exception as exc:
        if logger:
            logger.warn(f"DIM-3000 {port}: could not restore settings on stop "
                        f"({exc!r}). Run aomdriverinit.py to reset it by hand.")
    finally:
        try:
            ser.close()
        except Exception:
            pass


def restore_all(snaps, logger=None):
    """restore() each snapshot, in reverse order. One failing does not stop
    the rest."""
    for snap in reversed([s for s in snaps if s]):
        restore(snap, logger)
