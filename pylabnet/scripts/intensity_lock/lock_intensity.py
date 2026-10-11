import time
import pyrpl

#Set NI voltage to 0.45V
# --- tuning knobs -----------------------------------------------------
G = 10        # V/V, plant slope from the fit (positive)
F0 = 10e3       # target crossover [Hz] before backoff
K = 1         # backoff factor: actual crossover = F0/K
V_OP = -0.85      # DAC volts at the operating point   <-- SET FROM FIT
SETPOINT = -0.015
VMIN = -1.0       # anti-windup / AM-port safe range
VMAX = -0.55   #The actual voltage is 1V higher than what is set on the computer because the DC offset was removed from red pitaya. So -0.55 V corresponds to 0.45 V
# ----------------------------------------------------------------------


def compute_gains(G, F0, K):
    sign = -1.0 if G > 0 else 1.0          # negative feedback lives in C's sign
    p_gain = sign / (abs(G) * K)
    i_gain = sign * F0 / (abs(G) * K)
    return p_gain, i_gain


p_gain, i_gain = compute_gains(G, F0, K)

print(f"p = {p_gain:+.4f}   i = {i_gain:+.1f} Hz")
print(f"crossover ~ {F0/K:.0f} Hz, P/I corner {abs(i_gain/p_gain):.0f} Hz")

# Connect to the Red Pitaya
rp = pyrpl.Pyrpl(hostname='rp-f0f5f6.local', config='intensity_lock', gui=False).rp

# Make sure nothing else is summing onto OUT1
for m in [rp.pid1, rp.pid2, rp.iq0, rp.iq1, rp.iq2, rp.asg0, rp.asg1]:
    m.output_direct = 'off'

pid = rp.pid0
pid.input = 'in1'
pid.output_direct = 'off'      # stay open while configuring
pid.setpoint = SETPOINT
pid.min_voltage = VMIN
pid.max_voltage = VMAX
pid.p = 0.0
pid.i = 0.0
pid.ival = V_OP       # preload integrator so the loop starts in range

pid.output_direct = 'out1'     # close the loop
time.sleep(0.2)
pid.p = p_gain
pid.i = i_gain
time.sleep(0.5)


def report_status():
    err = rp.sampler.in1 - SETPOINT
    print(f"\nIN1 = {rp.sampler.in1:+.4f} V   setpoint {SETPOINT:+.4f} V"
          f"   error {err*1e3:+.2f} mV")
    print(f"integrator (DAC out) = {pid.ival:+.4f} V")
    if abs(pid.ival - VMIN) < 0.01 or abs(pid.ival - VMAX) < 0.01:
        print("  -> RAILED. Flip the sign, or the setpoint is unreachable.")
    else:
        print("  -> lock engaged.")


report_status()

# --- live tuning REPL ---------------------------------------------------
# Connection to the Red Pitaya stays open, so changes below take effect
# immediately -- no need to re-run the script (that's what costs ~15s).
print("\nLive tuning. Commands: 'g <value>', 'f0 <value>', 'reset [value]', 'status', 'quit'")
while True:
    try:
        raw = input("> ").strip()
    except EOFError:
        break
    if not raw:
        continue
    cmd, *rest = raw.split()
    cmd = cmd.lower()

    if cmd in ('q', 'quit', 'exit'):
        break
    elif cmd == 'status':
        report_status()
    elif cmd == 'reset':
        if rest:
            try:
                reset_to = float(rest[0])
            except ValueError:
                print(f"  couldn't parse '{rest[0]}' as a number")
                continue
        else:
            reset_to = V_OP
        pid.ival = reset_to
        print(f"integrator reset to {reset_to:+.4f} V")
        report_status()
    elif cmd in ('g', 'f0') and rest:
        try:
            value = float(rest[0])
        except ValueError:
            print(f"  couldn't parse '{rest[0]}' as a number")
            continue
        if cmd == 'g':
            G = value
        else:
            F0 = value
        p_gain, i_gain = compute_gains(G, F0, K)
        pid.p = p_gain
        pid.i = i_gain
        print(f"G = {G:.4g}   F0 = {F0:.4g} Hz   ->   "
              f"p = {p_gain:+.4f}   i = {i_gain:+.1f} Hz")
        report_status()
    else:
        print("  unknown command. Use 'g <value>', 'f0 <value>', 'status', or 'quit'")

print("Exiting REPL, connection to Red Pitaya will close.")
