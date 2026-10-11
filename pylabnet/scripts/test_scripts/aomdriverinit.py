"""Put both lattice DIM-3000 drivers into the state the fast lattice script assumes.

Run this from a normal Python prompt BEFORE starting fastLatticePGC.py in
DataTaker. That script only writes an AM voltage (ao6/ao7) and a TTL level
(dio4/dio5) -- it never opens a serial port, so frequency, amplitude, segment
mode and RF-on all have to be set here.

The Mseg:0 is the important one. If the transport sweep ran last, COM4 is
still in segment-play mode and parked at whatever frequency its last played
segment left behind, so the two beams start out detuned and the standing wave
runs continuously. Nothing in the fast lattice log would say why the atoms
left.

Usage:
    python dim_setup_lattice.py
    python dim_setup_lattice.py --amp 30.0 --freq 100.0
"""

import argparse
import sys
import time

import serial

PORTS = {'COM4': 'lattice up', 'COM5': 'lattice down'}

BAUD = 19200
# Manual: max execute speed is about 10 commands/sec. 0.12 s leaves margin.
CMD_GAP_S = 0.12


def send(ser, cmd):
    """One command, LF-terminated, at a rate the driver can keep up with."""
    ser.write((cmd + '\n').encode())
    time.sleep(CMD_GAP_S)


def query(ser, cmd):
    ser.reset_input_buffer()
    send(ser, cmd)
    time.sleep(0.1)
    return ser.readline().decode(errors='replace').strip()


def setup(port, label, freq_hz, amp_x10):
    with serial.Serial(port=port, baudrate=BAUD, bytesize=8,
                       parity='N', stopbits=1, timeout=1) as ser:
        idn = query(ser, '*IDN?')
        print(f"{port} ({label}): {idn or '<no reply>'}")
        if not idn:
            print(f"  {port}: no response -- wrong port, or another program "
                  f"still holds it. Nothing was configured.")
            return False

        # Order matters: leave segment mode BEFORE setting the frequency, or
        # the segment engine can overwrite it on the next stray trigger.
        send(ser, 'Mseg:0')
        send(ser, f'FRQ:{freq_hz}')
        send(ser, f'AMP:{amp_x10}')
        send(ser, 'FM_off')
        send(ser, 'RFp_off')
        send(ser, 'OUT_on')

        par = query(ser, 'Gpar')
        print(f"  set to {freq_hz/1e6:.3f} MHz, {amp_x10/10:.1f} dBm, "
              f"segment mode off, RF on")
        # Ramoffs must be 0 or the displayed amplitude is not the real output,
        # and the AM input sits on a shifted baseline.
        for field in par.split('|'):
            if field.startswith('Ramoffs:'):
                offs = field.split(':', 1)[1]
                if offs != '0':
                    print(f"  WARNING: AM offset is {offs}, not 0. The output "
                          f"power is altered by the offset even with no AM "
                          f"signal, so ao6/ao7 will not mean what you think. "
                          f"Zero it with the Adv button on the front panel.")
            if field.startswith('Rpcbtemp:'):
                print(f"  internal temperature "
                      f"{int(field.split(':', 1)[1]) / 100:.1f} C")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freq', type=float, default=100.8,
                    help='carrier frequency in MHz (default 100.8)')
    ap.add_argument('--amp', type=float, default=32.0,
                    help='output amplitude in dBm, 14 to 34 (default 32.0)')
    args = ap.parse_args()

    if not 14.0 <= args.amp <= 34.0:
        sys.exit(f"Amplitude {args.amp} dBm is outside the driver's "
                 f"+14 to +34 dBm range.")
    if not 10.0 <= args.freq <= 400.0:
        sys.exit(f"Frequency {args.freq} MHz is outside the driver's "
                 f"10 to 400 MHz range.")

    freq_hz = int(round(args.freq * 1e6))
    amp_x10 = int(round(args.amp * 10))

    ok = True
    for port, label in PORTS.items():
        try:
            ok &= setup(port, label, freq_hz, amp_x10)
        except Exception as exc:
            ok = False
            print(f"{port} ({label}): FAILED -- {exc!r}")

    print()
    if ok:
        print("Both drivers ready. Note the AM convention before you set "
              "ao6/ao7 in the GUI:")
        print(f"  ao = 0.0 V  -> the full {args.amp:.1f} dBm set above")
        print(f"  ao = -1.0 V -> minimum output")
        print(f"  ao > 0.0 V  -> above the set amplitude, clipped at 34 dBm")
        print("dio4/dio5 LOW = beam on, HIGH = beam off (>40 dB suppression).")
    else:
        print("At least one driver was not configured. Do not start the "
              "sequence until both report an IDN.")


if __name__ == '__main__':
    main()
