"""Bench test of the DIM-3000 frequency sweep, with no NI card and no Sync.

Question it answers: does this driver's sweep engine sweep at all? If mode 1
(continuous triangle, INTERNAL trigger) shows a band on the spectrum
analyser, the sweep works and the problem is the external trigger path. If it
does not, no amount of Sync wiring will help -- ask Time-Base.

BEFORE RUNNING
  * Stop any DataTaker script using this port (a COM port has one owner).
  * In modes 1 and 3 the rear "Sync" SMA is an OUTPUT. Unplug dio6 from it
    first, or the NI line and the driver drive the same wire.

USAGE (from a terminal, in the pylabnet env)
  python dim_sweep_test.py                      # mode 1, 99.0 -> 99.2 MHz
  python dim_sweep_test.py --start 99e6 --stop 98.8e6    # try a DOWNWARD sweep
  python dim_sweep_test.py --mode 2             # mode 2: then drive Sync by hand
  python dim_sweep_test.py --query              # just print Gpar and exit

Spectrum analyser: centre on the middle of the sweep, span ~2x the sweep
width, RBW <= 1 kHz, MAX HOLD. A working sweep fills in a flat band between
start and stop.

On Enter (or Ctrl+C) the sweep is switched off and the driver is put back on
the frequency it had before.
"""

import argparse
import time

import serial


def send(ser, cmd):
    ser.write((cmd + '\n').encode())
    time.sleep(0.12)                # manual: ~10 commands/s max


def gpar(ser):
    ser.reset_input_buffer()
    send(ser, 'Gpar')
    time.sleep(0.1)
    raw = ser.readline().decode(errors='replace').strip()
    out = {}
    for item in raw.split('|'):
        if ':' in item:
            k, v = item.split(':', 1)
            out[k.strip()] = v.strip()
    return raw, out


def show(label, d):
    keys = ('Rfreq', 'Rampl', 'Rout', 'Rswpm', 'Rswps', 'Rswpp', 'Rswpf',
            'Rswpt', 'Rffreq', 'Rfampl', 'Ramoffs', 'Rfmon')
    print(f"{label}: " + "  ".join(f"{k}={d.get(k, '?')}" for k in keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='COM4')
    ap.add_argument('--mode', type=int, default=1, choices=(1, 2, 3, 4),
                    help='1 = continuous triangle, internal trigger (default)')
    ap.add_argument('--start', type=float, default=99.0e6, help='SWPs, Hz')
    ap.add_argument('--stop', type=float, default=99.2e6, help='SWPp, Hz')
    ap.add_argument('--step', type=int, default=100, help='SWPf, Hz per step')
    ap.add_argument('--time', type=int, default=1000, help='SWPt, ns per step')
    ap.add_argument('--query', action='store_true', help='print Gpar and exit')
    a = ap.parse_args()

    ser = serial.Serial(port=a.port, baudrate=19200, bytesize=8, parity='N',
                        stopbits=1, timeout=1)
    before = {}
    armed = False                   # --query must stay read-only
    try:
        send(ser, '*IDN?')
        print('IDN:', ser.readline().decode(errors='replace').strip())
        raw, before = gpar(ser)
        print('Gpar raw:', raw)
        show('before', before)
        if a.query:
            return

        start, stop = int(a.start), int(a.stop)
        ramp_ms = abs(stop - start) / a.step * a.time * 1e-6
        print(f"\nSweep mode {a.mode}: {start} -> {stop} Hz, {a.step} Hz x "
              f"{a.time} ns = {ramp_ms:.3f} ms per ramp")
        armed = True
        send(ser, 'Mseg:0')
        send(ser, 'SWPm:0')
        send(ser, f'FRQ:{start}')
        send(ser, f'SWPs:{start}')
        send(ser, f'SWPp:{stop}')
        send(ser, f'SWPf:{a.step}')
        send(ser, f'SWPt:{a.time}')
        send(ser, f'SWPm:{a.mode}')
        _, after = gpar(ser)
        show('armed ', after)
        wrong = [f"{k}={after.get(k)} (sent {v})" for k, v in
                 (('Rswpm', a.mode), ('Rswps', start), ('Rswpp', stop),
                  ('Rswpf', a.step), ('Rswpt', a.time))
                 if after.get(k) != str(v)]
        if wrong:
            print('DRIVER DID NOT TAKE:', '; '.join(wrong))
        if a.mode in (2, 4):
            print('Mode 2/4: output sits at SWPs until Sync goes HIGH. Drive '
                  'Sync with a 3.3-5 V level by hand (or the NI line).')
        input('\nLook at the analyser now. Press Enter to switch the sweep off... ')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if armed:
                send(ser, 'SWPm:0')
                if 'Rfreq' in before:
                    send(ser, f"FRQ:{before['Rfreq']}")
                _, end = gpar(ser)
                show('\nafter ', end)
        finally:
            ser.close()


if __name__ == '__main__':
    main()
