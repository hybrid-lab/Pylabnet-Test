# Lattice MOVE ONLY -- DIM-3000 frequency sweep (mode 2), repeated forever.
#
# fastLatticeMoveSweep2.py with everything except the lattice move removed:
# no MOT, CMOT, Dark MOT, PGC, imaging, camera or fits. The NI card drives the
# lattice lines ONLY (dio4/dio5 TTL, ao6/ao7 AM, dio6 sweep gate). Every other
# NI line is left alone.
#
# One cycle, repeated back to back until Stop:
#
#     pre-move time    both AOMs at base (standing wave stationary)
#     dio6 HIGH        COM4 ramps base -> Fstop (T_r), holds at Fstop (T_p)
#     dio6 LOW         COM4 ramps Fstop -> base (T_r)
#     post-move time   both AOMs at base again
#     -> next cycle
#
# The sweep is set one of two ways ("Sweep from" input):
#   1 (default)  SWPs / SWPp / SWPf / SWPt typed in directly, plus a hold time
#   0            derived from a move distance and durations, exactly as in
#                fastLatticeMoveSweep2
# The resulting settings, sweep duration and a frequency-vs-time plot are
# shown in the window and written to the log.
#
# COMMAND FORM: this driver's firmware ignores the manual's SWPm:/SWPs:/...
# commands and only acts on the S-prefix writes (Sswpm:2, Sswps:..., found on
# the bench), so every sweep command here uses that form.
#
# WIRING: the same as fastLatticeMoveSweep2 -- dio6 on COM4's rear "Sync" SMA.
#
# The NI buffer holds ONE cycle and regenerates. The OPX clocks it for
# "Moves per Arm" cycles, then the cards are re-armed, which leaves a short gap
# (the lattice sits at base during it). Lattice TTL and AM are constant the
# whole time.

import time

import numpy as np
import pyqtgraph as pg
import serial
from pyqtgraph.Qt import QtCore, QtWidgets

import dim_state

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float


# =============================================================================
INIT_DICT = {
    # ===================== WHICH SWEEP SETTINGS =====================
    # 1 = use the SWP values typed directly below (default).
    # 0 = work them out from the move distance and durations further down.
    'sweep_input': {'Sweep from (1 = direct SWP values, 0 = move distance)': '1'},

    # ===================== DIRECT SWEEP SETTINGS (sweep_input = 1) =====================
    # Sent to COM4 as-is:
    #   Mseg:0  Sswpm:0  FRQ:<start>  AMP:<amp>  OUT_on
    #   Sswps:<start>  Sswpp:<stop>  Sswpf:<step>  Sswpt:<step time>  Sswpm:2
    # Ramp time (each way) = |stop - start| / step * step time   (manual p.8).
    # Defaults: 99.8 -> 100.0 MHz, 1 Hz x 15000 ns = 3 s per ramp.
    # COM5 sits at the START frequency, so the lattice is at rest between moves.
    'swp_start_mhz': {'SWPs Start Freq (MHz) [COM4 + COM5 rest]': '99.800'},
    'swp_stop_mhz': {'SWPp Stop Freq (MHz)': '100.000'},
    'swp_step_hz': {'SWPf Step (Hz)': '1'},
    'swp_step_ns': {'SWPt Step Time (ns) [4..262000, multiple of 4]': '15000'},
    'swp_hold_ms': {'Hold at Stop Freq (ms) [dio6 HIGH after ramp]': '1000'},

    # ===================== MOVE (sweep_input = 0) =====================
    # v = d / (T_a/2 + T_p + T_d/2). Mode 2 has one step rate for both ramps,
    # so accel and decel each run for (T_a + T_d)/2. 0 distance = no move:
    # both AOMs just sit at base.
    'dist_mm': {'Move Distance (mm, one way)': '0.5'},
    'accel_ms': {'Accel Time T_a (ms)': '5.0'},
    'plat_ms': {'Plateau (constant velocity) Time T_p (ms)': '15.0'},
    'decel_ms': {'Decel Time T_d (ms)': '5.0'},
    # Gate raised this much early to absorb COM4's Sync latency. Must not be
    # longer than the pre-move time.
    'seg_lead_ms': {'Sweep Gate Lead (ms) [Sync latency]': '0'},
    'move_direction': {'Move Direction (+1 or -1)': '-1'},

    # ===================== TIME AROUND EACH MOVE =====================
    'premove_ms': {'Pre-move Time (ms) [both AOMs at base]': '5'},
    'postmove_ms': {'Post-move Time (ms) [both AOMs at base]': '5'},

    # ===================== DIM-3000 / LATTICE =====================
    # Used only when sweep_input = 0; direct mode rests both AOMs at SWPs.
    'dim_base_freq_mhz': {'DIM3000 Base Freq (MHz) [both AOMs, move mode]': '99.800'},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)': '32.0'},
    # AM input (ao6/ao7). Manual: 0 V = exactly the DIM3000 Amplitude above,
    # positive raises it (capped at +34 dBm), -1 V = minimum. Set 0 for the
    # output to BE the typed amplitude.
    'lat_am_v': {'Lattice AM (V) [0 = exactly DIM3000 Amplitude]': '1.0'},

    # ===================== RUN =====================
    # Cycles the OPX clocks per NI arm. More = fewer re-arm gaps. The default
    # direct sweep is a 7 s cycle, so 5 moves = ~35 s per arm.
    'reps_per_arm': {'Moves per Arm (then re-arm)': '5'},
    'wait_time': {'Wait Time Between Arms (s)': '0.3'},
}


# Lattice lines only. LOW = beam ON on dio4/dio5; dio6 LOW = COM4 at base.
LATTICE_DO = ('dio4', 'dio5', 'dio6')
LATTICE_AO = ('ao6', 'ao7')


def park_lattice(NI_do, NI_ao, logger, lattice_am):
    """Lattice ON and stationary: TTLs low, AM at the run's value, gate low.

    Touches ONLY the lattice lines. Never raises -- it runs from finally.
    """
    safe_do = {'dio4': 0, 'dio5': 0, 'dio6': 0}
    safe_ao = {'ao6': float(lattice_am), 'ao7': float(lattice_am)}
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
    logger.info(f"Lattice parked: ON at {float(lattice_am):.2f} V, gate LOW "
                f"(COM4 at base, stationary).")


def sweep_html(m, moves_done):
    """The move as the driver will play it, as a two-column table."""
    nan = float('nan')
    rows = [
        ("SWEEP START  SWPs", f"{m.get('f_start_hz', nan) / 1e6:.6f} MHz",
         "SWEEP STOP  SWPp", f"{m.get('f_stop_hz', nan) / 1e6:.6f} MHz"),
        ("SWEEP DURATION", f"{m.get('total_ms', nan):.3f} ms",
         "COM4 detuning at Fstop", f"{m.get('df_khz_signed', nan):+.3f} kHz"),
        ("accel = decel ramp T_r", f"{m.get('ramp_ms', nan):.3f} ms",
         "acceleration (= decel)", f"{m.get('a_mps2', nan):.2f} m/s&sup2;"),
        ("plateau at Fstop T_p", f"{m.get('plat_real_ms', nan):.3f} ms",
         "peak velocity", f"{m.get('v_mmps', nan):.2f} mm/s"),
        ("Sync gate (dio6 HIGH)", f"{m.get('gate_ms', nan):.3f} ms",
         "SWPf x SWPt",
         f"{m.get('fstep_hz', 0)} Hz x {m.get('tstep_ns', 0)} ns "
         f"({m.get('n_steps', 0)} steps/ramp)"),
        ("distance", f"{m.get('d_mm', nan):.3f} mm "
                     f"(realised {m.get('d_real_mm', nan):.4f})",
         "cycle period", f"{m.get('cycle_ms', nan):.2f} ms"),
        ("pre / post-move time",
         f"{m.get('pre_ms', nan):.2f} / {m.get('post_ms', nan):.2f} ms",
         "moves played", f"{moves_done}"),
    ]
    body = "".join(
        f"<tr><td>{a}</td><td><b><span style='color:#4DD0E1'>{b}</span></b></td>"
        f"<td>&nbsp;&nbsp;&nbsp;{c}</td>"
        f"<td><b><span style='color:#4DD0E1'>{e}</span></b></td></tr>"
        for a, b, c, e in rows)
    return (f"<div style='font-family:monospace'><table cellspacing=2 "
            f"cellpadding=1>{body}</table></div>")


class SweepInfoView:
    """The numbers on top, then each AOM's frequency vs time over one cycle,
    then the dio6 gate on the same time axis.

    Callers only store data here (set_html / set_trace); a QTimer on the GUI
    thread draws it, because pyqtgraph is not thread safe and experiment()
    runs on a worker thread.
    """

    POLL_MS = 200

    def __init__(self, title):
        # The numbers are a plain QLabel ABOVE the plot canvas, not a label
        # inside it: inside, the table's width became the plot column's width
        # and pushed the end of the cycle off the right edge of the window.
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle(title)
        self.win.setStyleSheet("background-color: black;")
        box = QtWidgets.QVBoxLayout(self.win)
        box.setContentsMargins(6, 6, 6, 6)
        self.label = QtWidgets.QLabel('programming DIM-3000...')
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setStyleSheet("color: #DDDDDD; font-size: 10pt;")
        box.addWidget(self.label)
        self.canvas = pg.GraphicsLayoutWidget()
        box.addWidget(self.canvas, stretch=1)
        self.win.resize(860, 720)
        self.win.show()

        self.f_plot = self.canvas.addPlot(row=0, col=0)
        self.f_plot.setLabel('left', 'RF frequency', units='Hz')
        self.f_plot.showGrid(x=True, y=True, alpha=0.3)
        self.f_plot.addLegend(offset=(10, 10))
        # Plotted in Hz so pyqtgraph picks the SI prefix and keeps enough
        # digits for a sweep of a few kHz on a 100 MHz carrier.
        self.f4_curve = self.f_plot.plot(
            [], [], pen=pg.mkPen('#4DD0E1', width=2), name='COM4 (swept)')
        self.f5_curve = self.f_plot.plot(
            [], [], pen=pg.mkPen('#FFB74D', width=2, style=QtCore.Qt.DashLine),
            name='COM5 (fixed)')

        self.g_plot = self.canvas.addPlot(row=1, col=0)
        self.g_plot.setMaximumHeight(130)
        self.g_plot.setLabel('left', 'dio6 gate')
        self.g_plot.setLabel('bottom', 'time in cycle', units='s')
        self.g_plot.setYRange(-0.1, 1.1)
        self.g_plot.showGrid(x=True, y=False, alpha=0.3)
        self.g_curve = self.g_plot.plot([], [], pen=pg.mkPen('w', width=2))
        self.f_plot.setXLink(self.g_plot)

        self._pending = None
        self._rendered = 0
        self._pending_trace = None
        self._rendered_trace = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(self.POLL_MS)

    def set_html(self, html):
        self._pending = (html, self._rendered + 1)

    def set_trace(self, trace):
        """trace: frequency_trace() output (plain numpy, no Qt)."""
        self._pending_trace = (trace, self._rendered_trace + 1)

    def _poll(self):
        entry = self._pending
        if entry is not None and entry[1] > self._rendered:
            self._rendered = entry[1]
            try:
                self.label.setText(entry[0])
            except Exception:
                pass
        entry = self._pending_trace
        if entry is not None and entry[1] > self._rendered_trace:
            self._rendered_trace = entry[1]
            tr = entry[0]
            try:
                self.f4_curve.setData(tr['t_ms'] * 1e-3, tr['f4_mhz'] * 1e6)
                self.f5_curve.setData(tr['t5_ms'] * 1e-3, tr['f5_mhz'] * 1e6)
                self.g_curve.setData(tr['gate_t_ms'] * 1e-3, tr['gate'])
                # A flat line autoscales to a zero-height range; give it some.
                lo = min(tr['f4_mhz'].min(), tr['f5_mhz'].min()) * 1e6
                hi = max(tr['f4_mhz'].max(), tr['f5_mhz'].max()) * 1e6
                pad = max((hi - lo) * 0.1, 1e3)
                self.f_plot.setYRange(lo - pad, hi + pad)
                self.f_plot.setXRange(0, tr['t_ms'][-1] * 1e-3)
            except Exception:
                pass


# =============================================================================
# MOVE PHYSICS -> DIM-3000 SWEEP SETTINGS (same as fastLatticeMoveSweep2)
# =============================================================================
# Module level so configure() (to plot the move before running) and
# experiment() (to play it) work from ONE copy of the arithmetic.

LATTICE_WAVELENGTH = 795e-9
V_PER_HZ_MMPS = LATTICE_WAVELENGTH / 2 * 1e3    # mm/s per Hz of detuning

NI_RATE = 20000
SPM = NI_RATE // 1000                           # samples per ms
SAMPLE_MS = 1000.0 / NI_RATE

SWEEP_STEP_NS_QUANTUM = 4
SWEEP_STEP_NS_MAX = 262000
SWEEP_STEP_NS_MIN = 1000
SWEEP_FSTEP_SEARCH = 20000


def ceil_samples(t_ms):
    """ms -> whole NI samples, rounding UP (1e-9 absorbs float fuzz)."""
    return int(np.ceil(t_ms * SPM - 1e-9))


def ms2s(t):
    return int(round(t * SPM))


def read_params(dataset):
    """Every GUI input, parsed and checked. Raises ValueError on a bad one."""
    g = dataset.get_input_parameter
    p = {
        'dist_mm': float(g("dist_mm")),
        'accel_ms': float(g("accel_ms")),
        'plat_ms': float(g("plat_ms")),
        'decel_ms': float(g("decel_ms")),
        'seg_lead_ms': float(g("seg_lead_ms")),
        'move_direction': int(g("move_direction")),
        'premove_ms': float(g("premove_ms")),
        'postmove_ms': float(g("postmove_ms")),
        'dim_base_freq_mhz': float(g("dim_base_freq_mhz")),
        'dim_amplitude_dbm': float(g("dim_amplitude_dbm")),
        'lat_am_v': float(g("lat_am_v")),
        'reps_per_arm': int(g("reps_per_arm")),
        'wait_time': float(g("wait_time")),
        'direct': int(float(g("sweep_input"))) == 1,
        'swp_start_hz': int(round(float(g("swp_start_mhz")) * 1e6)),
        'swp_stop_hz': int(round(float(g("swp_stop_mhz")) * 1e6)),
        'swp_step_hz': int(float(g("swp_step_hz"))),
        'swp_step_ns': int(float(g("swp_step_ns"))),
        'swp_hold_ms': float(g("swp_hold_ms")),
    }
    if p['direct']:
        # Both AOMs rest at SWPs: that is where mode 2 sits while dio6 is LOW.
        p['base_freq_hz'] = p['swp_start_hz']
        if p['swp_stop_hz'] == p['swp_start_hz']:
            raise ValueError("SWPs and SWPp are the same frequency: nothing to sweep.")
        if p['swp_step_hz'] < 1:
            raise ValueError(f"SWPf must be >= 1 Hz, got {p['swp_step_hz']}.")
        if not (4 <= p['swp_step_ns'] <= SWEEP_STEP_NS_MAX):
            raise ValueError(
                f"SWPt must be 4..{SWEEP_STEP_NS_MAX} ns, got {p['swp_step_ns']}.")
        if p['swp_step_ns'] % 4:
            raise ValueError(
                f"SWPt must be a multiple of 4 ns (the driver rounds down), "
                f"got {p['swp_step_ns']}.")
        if p['swp_hold_ms'] < 0:
            raise ValueError(f"Hold must be >= 0 ms, got {p['swp_hold_ms']}.")
        for key in ('swp_start_hz', 'swp_stop_hz'):
            if not (10_000_000 <= p[key] <= 400_000_000):
                raise ValueError(f"{key} = {p[key]} Hz is outside 10-400 MHz.")
    else:
        p['base_freq_hz'] = int(round(p['dim_base_freq_mhz'] * 1e6))
    if min(p['accel_ms'], p['plat_ms'], p['decel_ms']) < 0:
        raise ValueError(
            f"Move durations must be >= 0 (got T_a={p['accel_ms']}, "
            f"T_p={p['plat_ms']}, T_d={p['decel_ms']} ms).")
    if p['dist_mm'] < 0:
        raise ValueError(
            "Move Distance must be >= 0; the direction is set by Move "
            "Direction (+1 or -1), not by the sign of the distance.")
    if p['move_direction'] not in (1, -1):
        raise ValueError(
            f"Move Direction must be +1 or -1, got {p['move_direction']}.")
    if min(p['premove_ms'], p['postmove_ms'], p['seg_lead_ms']) < 0:
        raise ValueError("Pre-move time, post-move time and gate lead must be >= 0.")
    if p['seg_lead_ms'] > p['premove_ms']:
        # The gate would have to rise before the cycle starts.
        raise ValueError(
            f"Sweep Gate Lead ({p['seg_lead_ms']} ms) is longer than the "
            f"Pre-move Time ({p['premove_ms']} ms). Raise the Pre-move Time.")
    if p['reps_per_arm'] < 1:
        raise ValueError(f"Moves per Arm must be >= 1, got {p['reps_per_arm']}.")
    return p


def sweep_steps(span_hz, ramp_ms):
    """SWPf and SWPt for a ramp covering span_hz in (at most) ramp_ms.

    Finest steps first; the first within 2e-4 of the span and 2e-3 of the
    ramp time is taken, else the best-scoring one. SWPt is rounded DOWN,
    so the ramp never outlasts the gate.
    """
    q = SWEEP_STEP_NS_QUANTUM
    target_ns = ramp_ms * 1e6
    if target_ns < SWEEP_STEP_NS_MIN:
        t_ns = max(q, int(target_ns // q) * q)
        return {'fstep_hz': span_hz, 'tstep_ns': t_ns, 'n_steps': 1,
                'span_hz': span_hz, 'ramp_ms': t_ns * 1e-6,
                'clipped': False}
    n_min = max(1, int(np.ceil(target_ns / SWEEP_STEP_NS_MAX)))
    n_max = int(target_ns // SWEEP_STEP_NS_MIN)
    if span_hz < n_min:
        t_ns = (SWEEP_STEP_NS_MAX // q) * q
        return {'fstep_hz': 1, 'tstep_ns': t_ns, 'n_steps': span_hz,
                'span_hz': span_hz, 'ramp_ms': span_hz * t_ns * 1e-6,
                'clipped': True}
    best = None
    f_lo = max(1, span_hz // max(n_max, 1))
    for f in range(f_lo, f_lo + SWEEP_FSTEP_SEARCH):
        n = int(round(span_hz / f))
        if n < n_min:
            break
        if n > n_max:
            continue
        t_ns = int((target_ns / n) // q) * q
        if t_ns < q or t_ns > SWEEP_STEP_NS_MAX:
            continue
        span_err = abs(n * f - span_hz) / span_hz
        ramp_err = abs(n * t_ns - target_ns) / target_ns
        score = 10.0 * span_err + ramp_err
        if best is None or score < best[0]:
            best = (score, f, t_ns, n)
        if span_err <= 2e-4 and ramp_err <= 2e-3:
            best = (score, f, t_ns, n)
            break
    if best is None:
        raise ValueError(
            f"No DIM sweep step settings cover {span_hz} Hz in "
            f"{ramp_ms:.3f} ms. Change the ramp time or the distance.")
    _, f, t_ns, n = best
    return {'fstep_hz': f, 'tstep_ns': t_ns, 'n_steps': n,
            'span_hz': n * f, 'ramp_ms': n * t_ns * 1e-6,
            'clipped': False}


def schedule_for_distance(p):
    """Driver settings and gate width for the move (see fastLatticeMoveSweep2)."""
    d_mm = p['dist_mm']
    base_freq_hz = p['base_freq_hz']
    if d_mm <= 0:
        return {'d_mm': float(d_mm), 'd_real_mm': 0.0, 'v_mmps': 0.0,
                'ramp_req_ms': 0.0, 'ramp_ms': 0.0, 'plat_real_ms': 0.0,
                'gate_s': 0, 'gate_ms': 0.0, 'total_ms': 0.0,
                'a_mps2': 0.0, 'f_start_hz': base_freq_hz,
                'f_stop_hz': base_freq_hz, 'df_hz': 0,
                'fstep_hz': 0, 'tstep_ns': 0, 'n_steps': 0,
                'clipped': False}
    ramp_req = 0.5 * (p['accel_ms'] + p['decel_ms'])
    if ramp_req + p['plat_ms'] <= 0:
        raise ValueError(
            f"Distance {d_mm} mm needs a non-zero move time, but "
            f"T_a/2 + T_p + T_d/2 = 0. Set at least one duration.")
    gate_s = ceil_samples(ramp_req + p['plat_ms'])
    gate_ms = gate_s * SAMPLE_MS
    span_target = int(round(d_mm / (gate_ms * 1e-3) / V_PER_HZ_MMPS))
    if span_target < 1:
        raise ValueError(
            f"{d_mm} mm in {gate_ms:.3f} ms is under 1 Hz of detuning, "
            f"below the driver's resolution.")
    st = sweep_steps(span_target, ramp_req)
    span = st['span_hz']
    v_mmps = span * V_PER_HZ_MMPS
    ramp_ms = st['ramp_ms']
    return {'d_mm': float(d_mm),
            'd_real_mm': v_mmps * gate_ms * 1e-3,
            'v_mmps': v_mmps,
            'ramp_req_ms': ramp_req,
            'ramp_ms': ramp_ms,
            'plat_real_ms': gate_ms - ramp_ms,
            'gate_s': gate_s,
            'gate_ms': gate_ms,
            'total_ms': gate_ms + ramp_ms,
            'a_mps2': (v_mmps * 1e-3) / (ramp_ms * 1e-3),
            'f_start_hz': base_freq_hz,
            'f_stop_hz': base_freq_hz + p['move_direction'] * span,
            'df_hz': span,
            'fstep_hz': st['fstep_hz'],
            'tstep_ns': st['tstep_ns'],
            'n_steps': st['n_steps'],
            'clipped': st['clipped']}


def schedule_direct(p):
    """The typed SWP values as a schedule, in the same shape as
    schedule_for_distance() so everything downstream is shared.

    Ramp time is the manual's T = |Fstop - Fstart| / Fstep * Tstep. The gate
    is ramp + hold, rounded UP to the NI grid so the ramp always completes
    before the gate falls. Distance is what that sweep does to the lattice:
    d = (lambda/2) * |Fstop - Fstart| * gate.
    """
    start, stop = p['swp_start_hz'], p['swp_stop_hz']
    span = abs(stop - start)
    fstep, tstep = p['swp_step_hz'], p['swp_step_ns']
    ramp_ms = span / fstep * tstep * 1e-6
    gate_s = ceil_samples(ramp_ms + p['swp_hold_ms'])
    gate_ms = gate_s * SAMPLE_MS
    v_mmps = span * V_PER_HZ_MMPS
    return {'d_mm': v_mmps * gate_ms * 1e-3,
            'd_real_mm': v_mmps * gate_ms * 1e-3,
            'v_mmps': v_mmps,
            'ramp_req_ms': ramp_ms,
            'ramp_ms': ramp_ms,
            'plat_real_ms': gate_ms - ramp_ms,
            'gate_s': gate_s,
            'gate_ms': gate_ms,
            'total_ms': gate_ms + ramp_ms,
            'a_mps2': (v_mmps * 1e-3) / (ramp_ms * 1e-3) if ramp_ms > 0 else float('inf'),
            'f_start_hz': start,
            'f_stop_hz': stop,
            'df_hz': span,
            'fstep_hz': fstep,
            'tstep_ns': tstep,
            'n_steps': span // fstep,
            # A span that is not a whole number of steps: the manual does not
            # say whether the last step stops short of or overshoots SWPp.
            'uneven_steps': span % fstep != 0,
            'clipped': False}


def schedule(p):
    """Direct SWP values or move-derived ones, per the Sweep-from switch."""
    return schedule_direct(p) if p['direct'] else schedule_for_distance(p)


def cycle_layout(p, sc):
    """One cycle in NI samples. The waveform AND the plot are built from this,
    so the plot shows exactly what is played.

    :return: (dict) move_start, gate_on, gate_s, move_end, N (all samples)
    """
    move_start = ms2s(p['premove_ms'])
    if sc['d_mm'] > 0:
        gate_s = sc['gate_s']
        move_end = move_start + gate_s + ceil_samples(sc['ramp_ms'])
    else:
        gate_s = 0
        move_end = move_start
    # At least 1 ms, so the buffer is never empty.
    N = max(move_end + ms2s(p['postmove_ms']), SPM)
    # >= 0 because read_params() refuses a lead longer than the pre-move time.
    gate_on = move_start - ms2s(p['seg_lead_ms'])
    return {'move_start': move_start, 'gate_on': gate_on, 'gate_s': gate_s,
            'move_end': move_end, 'N': N}


def move_summary(p, sc, cyc):
    """What the header table shows."""
    info = dict(sc)
    info.update({'df_khz_signed': (sc['f_stop_hz'] - p['base_freq_hz']) / 1e3,
                 'cycle_ms': cyc['N'] / SPM,
                 'pre_ms': cyc['move_start'] / SPM,
                 'post_ms': (cyc['N'] - cyc['move_end']) / SPM})
    return info


def frequency_trace(p, sc, cyc):
    """Expected RF frequency of each AOM, and the dio6 gate, over one cycle.

    Piecewise linear, so a handful of corner points draws the exact
    trapezoid. It assumes the Gate Lead equals the real Sync latency, so the
    RF starts ramping at move_start. The gate is drawn where the NI card
    actually puts it, i.e. Gate Lead earlier.

    :return: (dict) t_ms, f4_mhz (COM4, swept), f5_mhz (COM5, fixed),
        gate_t_ms, gate (0/1)
    """
    base = p['base_freq_hz'] / 1e6
    end_ms = cyc['N'] / SPM
    t0 = cyc['move_start'] / SPM
    if sc['d_mm'] > 0:
        stop = sc['f_stop_hz'] / 1e6
        ramp, gate = sc['ramp_ms'], sc['gate_ms']
        t = [0.0, t0, t0 + ramp, t0 + gate, t0 + gate + ramp, end_ms]
        f4 = [base, base, stop, stop, base, base]
        g_on = cyc['gate_on'] / SPM
        g_off = g_on + cyc['gate_s'] / SPM
        gt = [0.0, g_on, g_on, g_off, g_off, end_ms]
        gv = [0, 0, 1, 1, 0, 0]
    else:
        t, f4 = [0.0, end_ms], [base, base]
        gt, gv = [0.0, end_ms], [0, 0]
    return {'t_ms': np.array(t), 'f4_mhz': np.array(f4),
            'f5_mhz': np.array([base, base]),
            't5_ms': np.array([0.0, end_ms]),
            'gate_t_ms': np.array(gt), 'gate': np.array(gv, dtype=float)}


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    dataset.live = SweepInfoView("Lattice move ONLY (COM4 sweep mode 2)")
    dataset.graph.hide()
    # Show the move as soon as it is configured, before anything is armed.
    # Nothing here touches hardware: it is the same arithmetic experiment()
    # will use, drawn from the current inputs.
    try:
        p = read_params(dataset)
        sc = schedule(p)
        cyc = cycle_layout(p, sc)
        dataset.live.set_html(sweep_html(move_summary(p, sc, cyc), 0))
        dataset.live.set_trace(frequency_trace(p, sc, cyc))
    except Exception as exc:
        dataset.live.set_html(
            f"<span style='color:#ff8a80'>Cannot compute the move from these "
            f"inputs: {exc}</span>")


def experiment(**kwargs):
    """Run, then turn sweep mode off and restore the drivers however it ends.

    This firmware ignores the manual's SWPm:/SWPs:/... commands and only acts
    on the Gpar-style S-prefix writes (Sswpm:, Sswps:, ...), found on the
    bench. Every sweep command in this file uses that form.
    """
    snaps = []
    try:
        _run_experiment(dim_snapshots=snaps, **kwargs)
    finally:
        for snap in snaps:
            if not snap:
                continue
            try:
                if snap['_ser'].is_open:
                    snap['_ser'].write(b'Sswpm:0\n')
                    time.sleep(0.12)
            except Exception:
                pass
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
    # READ GUI PARAMETERS AND WORK OUT THE MOVE (module-level helpers)
    # =========================================================================
    p = read_params(dataset)
    accel_ms, plat_ms_in, decel_ms = p['accel_ms'], p['plat_ms'], p['decel_ms']
    dim_amplitude_dbm = p['dim_amplitude_dbm']
    lat_am_v = p['lat_am_v']
    reps_per_arm = p['reps_per_arm']
    wait_time = p['wait_time']
    base_freq_hz = p['base_freq_hz']
    ni_rate = NI_RATE
    if lat_am_v != 0:
        logger.warn(
            f"Lattice AM is {lat_am_v} V, not 0 V: the DIM AM input adds to "
            f"the set amplitude, so the output is NOT {dim_amplitude_dbm} dBm "
            f"(higher for positive AM, capped at +34 dBm). It is constant, but "
            f"set Lattice AM to 0 for the output to equal the typed amplitude.")

    sc = schedule(p)
    do_move = sc['d_mm'] > 0
    if p['direct']:
        logger.info(
            f"Direct sweep (COM4 mode 2): SWPs={sc['f_start_hz']} Hz, "
            f"SWPp={sc['f_stop_hz']} Hz, SWPf={sc['fstep_hz']} Hz, "
            f"SWPt={sc['tstep_ns']} ns -> {sc['n_steps']} steps, ramp "
            f"{sc['ramp_ms']:.3f} ms each way, hold {sc['plat_real_ms']:.3f} ms, "
            f"gate {sc['gate_ms']:.3f} ms, sweep {sc['total_ms']:.3f} ms; "
            f"lattice v={sc['v_mmps']:.2f} mm/s, d={sc['d_real_mm']:.3f} mm")
        if sc.get('uneven_steps'):
            logger.warn(
                f"|SWPp - SWPs| = {sc['df_hz']} Hz is not a whole number of "
                f"{sc['fstep_hz']} Hz steps; the last step may stop short of or "
                f"overshoot SWPp.")
    elif do_move:
        logger.info(
            f"Move (COM4 sweep mode 2): d={sc['d_mm']:.3f}mm, typed "
            f"T_a={accel_ms:.3f} / T_p={plat_ms_in:.3f} / T_d={decel_ms:.3f} ms "
            f"-> ramp {sc['ramp_ms']:.4f} ms each way, plateau "
            f"{sc['plat_real_ms']:.4f} ms, gate {sc['gate_ms']:.3f} ms, sweep "
            f"{sc['total_ms']:.4f} ms; v={sc['v_mmps']:.3f}mm/s, "
            f"a={sc['a_mps2']:.3f}m/s^2, SWPs={sc['f_start_hz']} Hz, "
            f"SWPp={sc['f_stop_hz']} Hz, SWPf={sc['fstep_hz']} Hz, "
            f"SWPt={sc['tstep_ns']} ns, {sc['n_steps']} steps")
        if abs(accel_ms - decel_ms) > 1e-9:
            logger.warn(
                f"T_a ({accel_ms} ms) != T_d ({decel_ms} ms): sweep mode 2 runs "
                f"BOTH ramps for their mean, {sc['ramp_req_ms']:.3f} ms.")
        if accel_ms + decel_ms <= 0:
            logger.warn("Accel and decel times are 0: the lattice velocity "
                        "jumps in a single step.")
        if sc['clipped']:
            logger.warn(
                f"The ramp is only {sc['ramp_ms']:.3f} ms, not the "
                f"{sc['ramp_req_ms']:.3f} ms asked for (1 Hz steps at the "
                f"driver's longest step time run out first). Distance unaffected.")
    else:
        logger.info("Move distance 0: no move -- both AOMs sit at base.")

    # =========================================================================
    # CONFIGURE DIM-3000 DRIVERS
    # =========================================================================
    DIM_COM_SWEEP = 'COM4'   # lattice up -- swept, rear "Sync" on dio6
    DIM_COM_FIXED = 'COM5'   # lattice down -- fixed at base
    DIM_MAX_SEGMENTS = 7     # only to neutralise segment tables other scripts leave
    amplitude_x10 = int(round(dim_amplitude_dbm * 10))

    def _dim_open(port):
        try:
            return serial.Serial(port=port, baudrate=19200,
                                 bytesize=8, parity='N', stopbits=1, timeout=1)
        except Exception:
            return None

    def _dim_send(ser, cmd):
        ser.write((cmd + '\n').encode())
        time.sleep(0.12)          # manual: ~10 commands/s at most

    def _dim_query(ser, cmd):
        ser.flushInput()
        _dim_send(ser, cmd)
        time.sleep(0.1)
        return ser.readline().decode().strip()

    def _dim_gpar(ser):
        out = {}
        for item in _dim_query(ser, 'Gpar').split('|'):
            if ':' in item:
                key, val = item.split(':', 1)
                try:
                    out[key.strip()] = int(val.strip())
                except ValueError:
                    pass
        return out

    def _dim_neutralise_segments(ser):
        for i in range(1, DIM_MAX_SEGMENTS + 1):
            flag = 8 if i == DIM_MAX_SEGMENTS else 1
            _dim_send(ser, f'Wseg:{i};{flag};{base_freq_hz};{base_freq_hz};0;1;0')

    # ---- Output power lock -------------------------------------------------
    # Every setting the driver has that sets its output power, and the value
    # it must hold for the output to be exactly the typed amplitude:
    #   Rampl    normal amplitude (AMP)
    #   Rfampl   FSK amplitude (FAMP) -- a second amplitude the driver may
    #            switch to; manual example 15.6 dBm, never set by older scripts
    #   Ramoffs  AM offset -- the manual: the display is the true output power
    #            "only in the case when the AM offset is set to 0"
    # The AM INPUT voltage is the one thing this cannot fix from here; see
    # the lat_am_v warning below.
    POWER_WANT = {'Rampl': amplitude_x10, 'Rfampl': amplitude_x10, 'Ramoffs': 0}
    # FAMP and AM offset as found, per port, restored on exit (dim_state
    # already restores AMP).
    power_found = {}

    def _dim_lock_power(ser, port):
        if port not in power_found:
            g = _dim_gpar(ser)
            power_found[port] = {k: g[k] for k in ('Rfampl', 'Ramoffs') if k in g}
            logger.info(
                f"DIM-3000 {port} power settings found: AMP={g.get('Rampl')} "
                f"FAMP={g.get('Rfampl')} (dBm*10), AM offset={g.get('Ramoffs')}, "
                f"FSK freq FFRQ={g.get('Rffreq')} Hz, sweep mode={g.get('Rswpm')}")
        _dim_send(ser, f'AMP:{amplitude_x10}')
        # S-prefix forms, like the sweep commands on this firmware.
        _dim_send(ser, f'Sfampl:{amplitude_x10}')
        _dim_send(ser, 'Samoffs:0')

    def _dim_check_power(ser, port):
        """Gpar readback of the power settings. Re-locks and logs an ERROR if
        any has moved. :return: True if all were right."""
        g = _dim_gpar(ser)
        bad = [f"{k} is {g[k]} (want {v})"
               for k, v in POWER_WANT.items() if k in g and g[k] != v]
        missing = [k for k in POWER_WANT if k not in g]
        if missing:
            logger.warn(f"DIM-3000 {port}: Gpar did not report {missing}; "
                        f"those cannot be verified.")
        if bad:
            logger.error(f"DIM-3000 {port}: output power settings not as set: "
                         f"{'; '.join(bad)}. Re-sending them.")
            _dim_lock_power(ser, port)
        return not bad

    def _dim_restore_power(ser, port):
        found = power_found.get(port, {})
        if 'Rfampl' in found:
            _dim_send(ser, f"Sfampl:{found['Rfampl']}")
        if 'Ramoffs' in found:
            _dim_send(ser, f"Samoffs:{found['Ramoffs']}")

    def _dim_make_deaf(ser, port):
        """Base frequency, plain single-frequency mode: sweep OFF, segments OFF."""
        _dim_send(ser, 'Sswpm:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_lock_power(ser, port)
        _dim_send(ser, 'OUT_on')
        _dim_neutralise_segments(ser)
        _dim_send(ser, 'Mseg:0')
        # Verified, not just sent: a base frequency the driver quietly ignored
        # would leave both beams wherever the last run left them.
        dim_state.set_frequency(ser, base_freq_hz, port, logger)
        logger.info(f"DIM-3000 {port}: {base_freq_hz/1e6:.6f} MHz, "
                    f"+{amplitude_x10/10:.1f} dBm, ON, sweep OFF, segments OFF.")

    def _dim_program_sweep(ser):
        """Load the sweep into COM4 and arm mode 2, then verify.

        The bench-tested sequence, in order:
            Mseg:0  Sswpm:0  FRQ:<start>  AMP:<amp>  OUT_on
            Sswps:<start>  Sswpp:<stop>  Sswpf:<step>  Sswpt:<time>  Sswpm:2
        plus Sfampl / Samoffs from the power lock, next to AMP.
        """
        _dim_send(ser, 'Mseg:0')
        _dim_send(ser, 'Sswpm:0')
        dim_state.set_frequency(ser, base_freq_hz, DIM_COM_SWEEP, logger)
        _dim_lock_power(ser, DIM_COM_SWEEP)
        _dim_send(ser, 'OUT_on')
        _dim_send(ser, f"Sswps:{sc['f_start_hz']}")
        _dim_send(ser, f"Sswpp:{sc['f_stop_hz']}")
        _dim_send(ser, f"Sswpf:{sc['fstep_hz']}")
        _dim_send(ser, f"Sswpt:{sc['tstep_ns']}")
        _dim_send(ser, 'Sswpm:2')
        time.sleep(0.1)
        got = _dim_gpar(ser)
        logger.info(
            f"DIM-3000 {DIM_COM_SWEEP} after arming: FRQ={got.get('Rfreq')} "
            f"SWPm={got.get('Rswpm')} SWPs={got.get('Rswps')} "
            f"SWPp={got.get('Rswpp')} SWPf={got.get('Rswpf')} "
            f"SWPt={got.get('Rswpt')} FFRQ={got.get('Rffreq')}")
        want = dict(POWER_WANT)
        want.update({'Rswpm': 2, 'Rswps': sc['f_start_hz'],
                     'Rswpp': sc['f_stop_hz'], 'Rswpf': sc['fstep_hz'],
                     'Rswpt': sc['tstep_ns']})
        return [f"{k} is {got.get(k, '<missing>')} (sent {v})"
                for k, v in want.items() if got.get(k) != v]

    ser_sweep = _dim_open(DIM_COM_SWEEP)
    ser_fixed = _dim_open(DIM_COM_FIXED)
    kwargs['dim_snapshots'].append(dim_state.snapshot(ser_sweep, DIM_COM_SWEEP, logger))
    kwargs['dim_snapshots'].append(dim_state.snapshot(ser_fixed, DIM_COM_FIXED, logger))
    if ser_sweep is not None and ser_fixed is not None:
        logger.info(f"DIM-3000 sweep ({DIM_COM_SWEEP}): {_dim_query(ser_sweep, '*IDN?')}")
        logger.info(f"DIM-3000 fixed ({DIM_COM_FIXED}): {_dim_query(ser_fixed, '*IDN?')}")
        _dim_make_deaf(ser_fixed, DIM_COM_FIXED)
        if not do_move:
            _dim_make_deaf(ser_sweep, DIM_COM_SWEEP)
        else:
            _dim_neutralise_segments(ser_sweep)
            bad = _dim_program_sweep(ser_sweep)
            if bad:
                logger.error(
                    f"DIM-3000 {DIM_COM_SWEEP} did not take the sweep settings: "
                    f"{'; '.join(bad)}. The move will not be the one shown.")
            else:
                logger.info(f"DIM-3000 {DIM_COM_SWEEP}: mode 2 armed, settings "
                            f"verified by Gpar readback.")
        # COM4's power is already in the sweep readback when it is moving.
        ok = _dim_check_power(ser_fixed, DIM_COM_FIXED)
        if not do_move:
            ok = _dim_check_power(ser_sweep, DIM_COM_SWEEP) and ok
        if ok:
            logger.info(f"Output power settings verified on both drivers: "
                        f"AMP = FAMP = {amplitude_x10 / 10:.1f} dBm, AM offset 0.")
    else:
        logger.error("DIM-3000: COM4/COM5 not available -- running the NI "
                     "pattern without serial control of the drivers.")
        for ser in (ser_sweep, ser_fixed):
            if ser is not None:
                ser.close()
        ser_sweep = ser_fixed = None

    # =========================================================================
    # ONE CYCLE:  pre-move -> gate (accel + plateau) -> decel -> post-move
    # =========================================================================
    cyc = cycle_layout(p, sc)
    N = cyc['N']
    seg_do = [0] * N
    for s in range(cyc['gate_s']):
        seg_do[cyc['gate_on'] + s] = 1
    ludo = [0] * N                       # lattice up TTL:   LOW = ON
    lddo = [0] * N                       # lattice down TTL: LOW = ON
    luao = [lat_am_v] * N
    ldao = [lat_am_v] * N
    total = N

    info = move_summary(p, sc, cyc)
    logger.info(f"Cycle: pre-move {info['pre_ms']:.2f} ms, move "
                f"{sc['total_ms']:.3f} ms, post-move {info['post_ms']:.2f} ms "
                f"-> period {N / SPM:.2f} ms ({N} samples), "
                f"{reps_per_arm} moves per arm.")
    moves_done = 0
    dataset.live.set_html(sweep_html(info, moves_done))
    dataset.live.set_trace(frequency_trace(p, sc, cyc))

    # =========================================================================
    # MAIN LOOP: arm, let the OPX clock reps_per_arm cycles, re-arm
    # =========================================================================
    delay_ns = int(round(1e9 / ni_rate)) - 500      # 500 ns tick + delay = 1 sample
    run_buffer = 10_000_000                          # ns of OPX idle after the pass
    opx_trigger_do = 1
    pass_s = reps_per_arm * total / ni_rate + run_buffer * 1e-9

    try:
        while thread.running:
            try:
                NI_card_1.finalize_clock()
            except Exception:
                pass

            NI_card_2.build_stack()
            NI_card_2.set_do_voltage(do_channel="dio4", value=ludo, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel="dio5", value=lddo, sample_rate=ni_rate)
            NI_card_2.set_do_voltage(do_channel="dio6", value=seg_do, sample_rate=ni_rate)
            NI_card_3.build_stack()
            NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=luao, sample_rate=ni_rate)
            NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=ldao, sample_rate=ni_rate)

            # Data cards before the clock, as in fastLatticeMoveSweep2.
            h1 = NI_card_2.arm(regeneration=True)
            h2 = NI_card_3.arm(regeneration=True)
            NI_card_1.arm_clock(length=total, sample_rate=ni_rate)

            OPX_client.build_stack()
            clk = OPX_client.create_new_do_elem(do_channel=opx_trigger_do, length=500)
            with OPX_client.for_("j", 0, reps_per_arm, 1):
                with OPX_client.for_("i", 0, total, 1):
                    OPX_client.set_digital_voltage(element=clk)
                    OPX_client.delay(delay_ns)
            OPX_client.delay(run_buffer)
            OPX_client.execute(wait=False)

            # Nothing to read back, so just wait out the pass -- in short
            # slices, so Stop takes effect within a fraction of a second.
            t_end = time.time() + pass_s + 0.2
            while thread.running and time.time() < t_end:
                time.sleep(0.1)
            if thread.running:
                moves_done += reps_per_arm if do_move else 0
                dataset.live.set_html(sweep_html(info, moves_done))

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
            # A stop can land mid-gate, leaving dio6 HIGH and COM4 parked at
            # Fstop (lattice walking). Parking drops the gate straight away.
            park_lattice(NI_card_2, NI_card_3, logger, lat_am_v)
            # Re-check the power settings between arms -- the NI tasks are
            # closed, so nothing is moving while the serial port is busy.
            for ser, port in ((ser_sweep, DIM_COM_SWEEP), (ser_fixed, DIM_COM_FIXED)):
                if ser is not None and thread.running:
                    _dim_check_power(ser, port)
            logger.info(f"Arm done: {moves_done} moves so far.")
            time.sleep(wait_time)
    finally:
        if ser_sweep is not None:
            try:
                _dim_send(ser_sweep, 'Sswpm:0')
                dim_state.set_frequency(ser_sweep, base_freq_hz, DIM_COM_SWEEP, logger)
                _dim_restore_power(ser_sweep, DIM_COM_SWEEP)
                logger.info(f"DIM sweep OFF and parked at base "
                            f"{base_freq_hz/1e6:.3f} MHz (lattice stationary)")
            except Exception as e:
                logger.warn(f"Could not park the DIM at base frequency: {e!r}")
            try:
                ser_sweep.close()
            except Exception:
                pass
        if ser_fixed is not None:
            try:
                _dim_restore_power(ser_fixed, DIM_COM_FIXED)
            except Exception as e:
                logger.warn(f"Could not restore COM5 FAMP / AM offset: {e!r}")
            try:
                ser_fixed.close()
            except Exception:
                pass
