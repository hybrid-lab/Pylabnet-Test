# DIM-3000 AOM Driver Setup
# Run this ONCE after power cycling the DIM-3000 devices.
# Configures both drivers, writes segments, enables output.
# After running, the devices are ready for the experiment.

import serial
import time
import sys

# =========================================================================
# CONFIGURATION -- edit these to match your setup
# =========================================================================
COM_SWEEP = 'COM4'      # DIM-3000 with frequency sweep (lattice up)
COM_FIXED = 'COM5'      # DIM-3000 at fixed frequency (lattice down)

BASE_FREQ = 100000000   # 100 MHz -- both start here
AMPLITUDE = 340         # dBm * 10 -> +34.0 dBm
MOVE_FREQ = 101677000   # 101.677 MHz -- sweep target

# Segment ramp parameters
# Step freq: (MOVE_FREQ - BASE_FREQ) / num_steps
NUM_STEPS = 76
STEP_FREQ = (MOVE_FREQ - BASE_FREQ) // NUM_STEPS  # ~22064 Hz
STEP_TIME = 67328       # arb units (1 unit = 3.90625 ns)
# 67328 * 3.90625 ns = 263 us/step
# total ramp = 76 * 263 us = ~20 ms

# =========================================================================


def open_port(port):
    try:
        ser = serial.Serial(
            port=port, baudrate=19200,
            bytesize=8, parity='N', stopbits=1, timeout=1
        )
        return ser
    except Exception as e:
        print(f"ERROR: cannot open {port}: {e}")
        return None


def send(ser, cmd):
    ser.write((cmd + '\n').encode())
    time.sleep(0.1)


def query(ser, cmd):
    ser.flushInput()
    send(ser, cmd)
    time.sleep(0.15)
    return ser.readline().decode().strip()


def setup_device(ser, name, freq, amp):
    """Basic setup: set frequency, amplitude, enable output."""
    idn = query(ser, '*IDN?')
    print(f"  [{name}] ID: {idn}")
    if not idn:
        print(f"  [{name}] WARNING: no response to *IDN? -- check connection")
        return False

    send(ser, f'FRQ:{freq}')
    send(ser, f'AMP:{amp}')
    send(ser, 'OUT_on')

    f = query(ser, 'FRQ?')
    a = query(ser, 'AMP?')
    print(f"  [{name}] Freq: {f} Hz, Amp: {a} (x10 dBm), RF: ON")
    return True


def setup_segments(ser, name):
    """Write sweep segments for lattice move."""
    print(f"  [{name}] Writing segments...")

    # Segment 1: Ramp UP  base -> move freq
    cmd1 = f'Wseg:1;1;{BASE_FREQ};{MOVE_FREQ};{STEP_FREQ};{STEP_TIME};0'
    send(ser, cmd1)
    r1 = query(ser, 'Rseg:1')
    print(f"    Seg1 (ramp up):   {r1}")

    # Segment 2: HOLD at move freq (step=0)
    cmd2 = f'Wseg:2;1;{MOVE_FREQ};{MOVE_FREQ};0;1;0'
    send(ser, cmd2)
    r2 = query(ser, 'Rseg:2')
    print(f"    Seg2 (hold):      {r2}")

    # Segment 3: Ramp DOWN  move freq -> base (last segment, resets to 1)
    cmd3 = f'Wseg:3;8;{MOVE_FREQ};{BASE_FREQ};{STEP_FREQ};{STEP_TIME};0'
    send(ser, cmd3)
    r3 = query(ser, 'Rseg:3')
    print(f"    Seg3 (ramp down): {r3}")

    # Enable segment play mode
    send(ser, 'Mseg:1')
    print(f"  [{name}] Segment mode ON (waiting for triggers on rear Seg)")


# =========================================================================
# MAIN
# =========================================================================
print("=" * 60)
print("DIM-3000 AOM Driver Setup")
print("=" * 60)
print()

# Open both ports
print(f"Opening {COM_SWEEP} (sweep)...")
ser1 = open_port(COM_SWEEP)
if ser1 is None:
    sys.exit(1)

print(f"Opening {COM_FIXED} (fixed)...")
ser2 = open_port(COM_FIXED)
if ser2 is None:
    ser1.close()
    sys.exit(1)

print()

# Setup COM_FIXED (simple: fixed frequency, output on)
print(f"--- {COM_FIXED}: Fixed frequency ---")
ok2 = setup_device(ser2, COM_FIXED, BASE_FREQ, AMPLITUDE)

print()

# Setup COM_SWEEP (frequency + segments + segment mode)
print(f"--- {COM_SWEEP}: Sweep + segments ---")
ok1 = setup_device(ser1, COM_SWEEP, BASE_FREQ, AMPLITUDE)
if ok1:
    setup_segments(ser1, COM_SWEEP)

print()

# Summary
ramp_time_ms = NUM_STEPS * STEP_TIME * 3.90625e-9 * 1e3
print("=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
print(f"  {COM_FIXED}: {BASE_FREQ/1e6:.3f} MHz, +{AMPLITUDE/10:.1f} dBm, RF ON")
print(f"  {COM_SWEEP}: {BASE_FREQ/1e6:.3f} MHz, +{AMPLITUDE/10:.1f} dBm, RF ON")
print(f"  Segments: {BASE_FREQ/1e6:.3f} -> {MOVE_FREQ/1e6:.3f} MHz")
print(f"  Ramp: {NUM_STEPS} steps, ~{ramp_time_ms:.1f} ms per ramp")
print(f"  Segment mode: ON (triggers via rear Seg SMA)")
print()
print("Wiring:")
print(f"  NI dio4 -> {COM_SWEEP} front TTL (lattice up on/off)")
print(f"  NI dio5 -> {COM_FIXED} front TTL (lattice down on/off)")
print(f"  NI dio6 -> {COM_SWEEP} rear Seg  (segment triggers)")
print()
print("Ready for experiment. You can close this window.")

# Close ports (DIM-3000 retains settings)
ser1.close()
ser2.close()
