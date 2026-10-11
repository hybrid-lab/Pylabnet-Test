# DIM-3000 SWEEP MODE 2 CHECK -- is COM4 actually changing frequency?
#
# A slow, deliberate version of what latticeMoveSweep2_only.py does, built to
# be WATCHED rather than used:
#
#     dio6 LOW  for "Base Time"            COM4 sits at SWPs
#     dio6 HIGH for ramp + "Hold Time"     COM4 ramps SWPs -> SWPp, holds
#     dio6 LOW  for ramp + "Base Time"     COM4 ramps back to SWPs
#     ... repeated until Stop
#
# The ramp takes SECONDS, so on the spectrum analyser (normal sweep, NOT max
# hold) the peak visibly walks from SWPs to SWPp and back. That is the real
# test.
#
# Meanwhile the script asks COM4 for its frequency (FRQ?) as fast as the serial
# link allows (~5/s) and plots the answers over the expected trapezoid. If the
# answers follow the trapezoid, the driver is reporting its live sweep
# frequency and the sweep is confirmed from software too. If they stay flat,
# that is NOT proof the sweep failed: the manual never says whether FRQ?
# reports the live sweep frequency or only the set value -- trust the
# analyser in that case. The log says which of the two happened.
#
# dio6 is driven by single static writes (no OPX, no buffered task), so
# nothing else on the NI card is touched. Wiring as for the move scripts:
# dio6 -> COM4 rear "Sync". COM5 is not touched at all.

import time

import numpy as np
import pyqtgraph as pg
import serial
from pyqtgraph.Qt import QtCore, QtWidgets

import dim_state
from latticeMoveSweep2_only import sweep_steps

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float


INIT_DICT = {
    # SWPs is where COM4 sits while dio6 is LOW; SWPp is where the ramp ends.
    # Stop below start tests a DOWNWARD sweep, as Move Direction -1 uses.
    # Pick a span wide enough to see clearly on the analyser.
    'start_mhz': {'Sweep Start SWPs (MHz)': '99.000'},
    'stop_mhz': {'Sweep Stop SWPp (MHz)': '99.200'},
    'ramp_s': {'Ramp Time (s, each way)': '3'},
    'hold_s': {'Hold at Stop (s)': '2'},
    'base_s': {'Time at Start between sweeps (s)': '2'},
    'amplitude_dbm': {'DIM3000 Amplitude (dBm)': '32.0'},
    # FSK TEST. Sets COM4's FSK frequency (FFRQ) for this run, put back on
    # Stop. If the frequency Sync HIGH jumps to MOVES to this value, the
    # Sync input is acting as an FSK key, not a sweep trigger. Blank = leave
    # FFRQ as it is.
    'fsk_test_mhz': {'FSK Freq FFRQ for this run (MHz, blank = leave)': '99.5'},
}

DIM_PORT = 'COM4'
GATE_LINE = 'dio6'


def set_gate(NI_do, level, logger):
    """dio6 to a fixed level NOW. write_static first; per-channel fallback."""
    try:
        NI_do.write_static(do={GATE_LINE: int(level)})
    except Exception as exc:
        try:
            NI_do.set_do_voltage(do_channel=GATE_LINE, value=bool(level))
        except Exception as e:
            logger.error(f"Could not set {GATE_LINE} {'HIGH' if level else 'LOW'}: "
                         f"write_static: {exc!r}; set_do_voltage: {e!r}")


class CheckView:
    """Numbers on top; polled vs expected frequency; dio6 below.
    Worker thread only stores data; a GUI-thread QTimer draws it."""

    POLL_MS = 200

    def __init__(self):
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("DIM-3000 sweep mode 2 check (COM4)")
        self.win.setStyleSheet("background-color: black;")
        box = QtWidgets.QVBoxLayout(self.win)
        box.setContentsMargins(6, 6, 6, 6)
        self.label = QtWidgets.QLabel('connecting to COM4...')
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setStyleSheet("color: #DDDDDD; font-size: 10pt;")
        box.addWidget(self.label)
        canvas = pg.GraphicsLayoutWidget()
        box.addWidget(canvas, stretch=1)
        self.win.resize(900, 700)
        self.win.show()

        self.f_plot = canvas.addPlot(row=0, col=0)
        self.f_plot.setLabel('left', 'COM4 frequency', units='Hz')
        self.f_plot.showGrid(x=True, y=True, alpha=0.3)
        self.f_plot.addLegend(offset=(10, 10))
        self.exp_curve = self.f_plot.plot(
            [], [], pen=pg.mkPen('#4DD0E1', width=2, style=QtCore.Qt.DashLine),
            name='expected')
        self.meas_curve = self.f_plot.plot(
            [], [], pen=None, symbol='o', symbolSize=6,
            symbolBrush='#FFB74D', symbolPen=None, name='FRQ? reply')
        g_plot = canvas.addPlot(row=1, col=0)
        g_plot.setMaximumHeight(130)
        g_plot.setLabel('left', 'dio6')
        g_plot.setLabel('bottom', 'time in sweep', units='s')
        g_plot.setYRange(-0.1, 1.1)
        self.g_curve = g_plot.plot([], [], pen=pg.mkPen('w', width=2))
        self.f_plot.setXLink(g_plot)

        self._pending = None
        self._version = 0
        self._drawn = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(self.POLL_MS)

    def update(self, html=None, expected=None, measured=None, gate=None):
        prev = self._pending or {}
        self._version += 1
        self._pending = {
            'html': html if html is not None else prev.get('html'),
            'expected': expected if expected is not None else prev.get('expected'),
            'measured': measured if measured is not None else prev.get('measured'),
            'gate': gate if gate is not None else prev.get('gate'),
            'v': self._version}

    def _poll(self):
        p = self._pending
        if p is None or p['v'] <= self._drawn:
            return
        self._drawn = p['v']
        try:
            if p['html'] is not None:
                self.label.setText(p['html'])
            if p['expected'] is not None:
                self.exp_curve.setData(*p['expected'])
            if p['measured'] is not None:
                self.meas_curve.setData(*p['measured'])
            if p['gate'] is not None:
                self.g_curve.setData(*p['gate'])
        except Exception:
            pass


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.live = CheckView()
    dataset.graph.hide()


def experiment(**kwargs):
    """Run, then gate LOW, sweep OFF and COM4 restored, however it ends."""
    snaps = []
    NI_do = kwargs['nidaqmx_ni_daq_2']
    logger = kwargs['dataset'].log
    try:
        _run(dim_snapshots=snaps, **kwargs)
    finally:
        set_gate(NI_do, 0, logger)
        for snap in snaps:
            if not snap:
                continue
            try:
                if not snap['_ser'].is_open:
                    snap['_ser'].open()
                snap['_ser'].write(b'SWPm:0\n')
                time.sleep(0.12)
                if ffrq_found[0] is not None:
                    snap['_ser'].write(f'FFRQ:{ffrq_found[0]}\n'.encode())
                    time.sleep(0.12)
                    ffrq_found[0] = None
            except Exception:
                pass
        dim_state.restore_all(snaps, logger)


def _run(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    logger = dataset.log
    NI_do = kwargs['nidaqmx_ni_daq_2']
    g = dataset.get_input_parameter

    start_hz = int(round(float(g('start_mhz')) * 1e6))
    stop_hz = int(round(float(g('stop_mhz')) * 1e6))
    ramp_s = float(g('ramp_s'))
    hold_s = float(g('hold_s'))
    base_s = float(g('base_s'))
    amp_x10 = int(round(float(g('amplitude_dbm')) * 10))
    span = abs(stop_hz - start_hz)
    if span < 1:
        raise ValueError("Start and stop frequency are the same.")
    if min(ramp_s, hold_s, base_s) <= 0:
        raise ValueError("Ramp, hold and base times must be > 0.")

    st = sweep_steps(span, ramp_s * 1e3)
    # The driver plays n * SWPf, so SWPp is set to exactly what it will reach.
    direction = 1 if stop_hz > start_hz else -1
    stop_hz = start_hz + direction * st['span_hz']
    ramp_real_s = st['ramp_ms'] * 1e-3
    if st['clipped']:
        logger.warn(f"Ramp shortened to {ramp_real_s:.3f} s: {span} Hz in 1 Hz "
                    f"steps at the longest step time (262 us) is all it can do.")

    # ---- COM4 --------------------------------------------------------------
    set_gate(NI_do, 0, logger)          # LOW before arming: start at SWPs
    try:
        ser = serial.Serial(port=DIM_PORT, baudrate=19200, bytesize=8,
                            parity='N', stopbits=1, timeout=1)
    except Exception as exc:
        raise RuntimeError(f"Cannot open {DIM_PORT} ({exc}). Is another "
                           f"script still holding it?")
    kwargs['dim_snapshots'].append(dim_state.snapshot(ser, DIM_PORT, logger))

    def send(cmd):
        ser.write((cmd + '\n').encode())
        time.sleep(0.05)

    def query(cmd):
        ser.reset_input_buffer()
        send(cmd)
        time.sleep(0.05)
        return ser.readline().decode(errors='replace').strip()

    def gpar():
        out = {}
        for item in query('Gpar').split('|'):
            if ':' in item:
                k, v = item.split(':', 1)
                try:
                    out[k.strip()] = int(v.strip())
                except ValueError:
                    pass
        return out

    logger.info(f"{DIM_PORT}: {query('*IDN?')}")
    found = gpar()
    logger.info(f"{DIM_PORT} as found: FRQ={found.get('Rfreq')} "
                f"FFRQ={found.get('Rffreq')} FAMP={found.get('Rfampl')} "
                f"AMP={found.get('Rampl')} SWPm={found.get('Rswpm')}")
    fsk_raw = str(g('fsk_test_mhz')).strip()
    fsk_hz = int(round(float(fsk_raw) * 1e6)) if fsk_raw else None
    if fsk_hz is not None:
        if 'Rffreq' in found:
            ffrq_found[0] = found['Rffreq']
        send(f'FFRQ:{fsk_hz}')
        time.sleep(0.07)
        logger.info(f"{DIM_PORT}: FSK frequency set to {fsk_hz} Hz for this run "
                    f"(was {found.get('Rffreq')}). If dio6 HIGH now jumps to "
                    f"{fsk_hz / 1e6:.3f} MHz instead of 100 MHz, Sync is "
                    f"switching to FSK, not sweeping.")
    for cmd in ('Mseg:0', 'SWPm:0', f'FRQ:{start_hz}', f'AMP:{amp_x10}',
                f'FAMP:{amp_x10}', 'OUT_on', f'SWPs:{start_hz}',
                f'SWPp:{stop_hz}', f"SWPf:{st['fstep_hz']}",
                f"SWPt:{st['tstep_ns']}", 'SWPm:2'):
        send(cmd)
        time.sleep(0.07)                 # ~8 commands/s, inside the manual's 10
    got = gpar()
    want = {'Rswpm': 2, 'Rswps': start_hz, 'Rswpp': stop_hz,
            'Rswpf': st['fstep_hz'], 'Rswpt': st['tstep_ns']}
    bad = [f"{k}={got.get(k, '<missing>')} (sent {v})"
           for k, v in want.items() if got.get(k) != v]
    logger.info(f"{DIM_PORT} armed: SWPm={got.get('Rswpm')} SWPs={got.get('Rswps')} "
                f"SWPp={got.get('Rswpp')} SWPf={got.get('Rswpf')} "
                f"SWPt={got.get('Rswpt')} FFRQ={got.get('Rffreq')} "
                f"FAMP={got.get('Rfampl')}")
    if bad:
        logger.error(f"{DIM_PORT} did not take: {'; '.join(bad)}")

    # ---- Expected trapezoid for one sweep ----------------------------------
    t_hi = base_s                       # gate rises
    t_lo = base_s + ramp_real_s + hold_s   # gate falls
    t_end = t_lo + ramp_real_s + base_s
    exp_t = np.array([0, t_hi, t_hi + ramp_real_s, t_lo, t_lo + ramp_real_s, t_end])
    exp_f = np.array([start_hz, start_hz, stop_hz, stop_hz, start_hz, start_hz],
                     dtype=float)
    gate_t = np.array([0, t_hi, t_hi, t_lo, t_lo, t_end])
    gate_v = np.array([0, 0, 1, 1, 0, 0], dtype=float)

    def header(n_sweeps, verdict):
        rows = [
            ("SWPs (start, dio6 LOW)", f"{start_hz / 1e6:.6f} MHz",
             "SWPp (stop)", f"{stop_hz / 1e6:.6f} MHz"),
            ("ramp each way", f"{ramp_real_s:.3f} s",
             "SWPf x SWPt", f"{st['fstep_hz']} Hz x {st['tstep_ns']} ns "
                            f"({st['n_steps']} steps)"),
            ("hold at stop / base", f"{hold_s:.1f} / {base_s:.1f} s",
             "sweeps done", f"{n_sweeps}"),
            ("driver readback", "OK" if not bad else "MISMATCH -- see log",
             "FRQ? verdict", verdict),
        ]
        body = "".join(
            f"<tr><td>{a}</td><td><b><span style='color:#4DD0E1'>{b}</span></b></td>"
            f"<td>&nbsp;&nbsp;&nbsp;{c}</td>"
            f"<td><b><span style='color:#4DD0E1'>{e}</span></b></td></tr>"
            for a, b, c, e in rows)
        return f"<div style='font-family:monospace'><table>{body}</table></div>"

    live = dataset.live
    live.update(html=header(0, 'waiting for first sweep'),
                expected=(exp_t, exp_f), measured=(np.array([]), np.array([])),
                gate=(gate_t, gate_v))
    logger.info(f"Sweep check: {start_hz} -> {stop_hz} Hz, {ramp_real_s:.3f} s "
                f"ramps, hold {hold_s} s, base {base_s} s. Watch the analyser "
                f"(normal sweep, not max hold): the peak should walk between "
                f"the two frequencies.")

    # ---- Sweep, polling FRQ? all the way through ----------------------------
    n_sweeps = 0
    while thread.running:
        ts, fs = [], []
        t0 = time.perf_counter()
        gate_high = gate_done = False
        while thread.running:
            t = time.perf_counter() - t0
            if t >= t_end:
                break
            if not gate_high and t >= t_hi:
                set_gate(NI_do, 1, logger)
                gate_high = True
            if gate_high and not gate_done and t >= t_lo:
                set_gate(NI_do, 0, logger)
                gate_done = True
            reply = query('FRQ?')
            try:
                fs.append(float(reply.split(':')[-1]))
                ts.append(time.perf_counter() - t0)
            except ValueError:
                pass
            if len(ts) % 5 == 0:
                live.update(measured=(np.array(ts), np.array(fs)))
        set_gate(NI_do, 0, logger)
        if not thread.running:
            break
        n_sweeps += 1
        live.update(measured=(np.array(ts), np.array(fs)))

        # Verdict: did FRQ? move across most of the span while dio6 was HIGH?
        if not fs:
            verdict = "no FRQ? replies"
        else:
            moved = max(abs(f - start_hz) for f in fs)
            frac = moved / st['span_hz']
            if frac > 0.5:
                verdict = f"FOLLOWS the sweep ({frac * 100:.0f}% of span)"
            else:
                verdict = (f"flat at {np.median(fs) / 1e6:.6f} MHz -- "
                           f"trust the analyser")
        live.update(html=header(n_sweeps, verdict))
        logger.info(f"Sweep {n_sweeps}: {len(fs)} FRQ? replies, "
                    f"range {min(fs, default=0):.0f}-{max(fs, default=0):.0f} Hz "
                    f"-> {verdict}")

    try:
        send('SWPm:0')
        send(f'FRQ:{start_hz}')
    finally:
        ser.close()


# FFRQ as found, restored in experiment()'s finally (dim_state does not
# restore it). One-element list so _run can fill it in.
ffrq_found = [None]
