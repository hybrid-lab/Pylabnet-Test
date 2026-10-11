# LATTICE TO BOTTOM -- SPINVIEW VERSION: transport, then hold with the light ON.
#
# lattice_to_Bottom.py with the camera taken out of the script entirely, so
# SpinView can own the bottom camera while this sequence runs. Use it to WATCH
# the cloud arrive, not to measure it: there are no frames, no fits and no
# atom number here.
#
# WHAT IS DIFFERENT FROM lattice_to_Bottom.py
#   * dio8 is NEVER driven. The camera trigger is not in the waveform and not
#     written to the NI card, so nothing this script does reaches the camera.
#     SpinView holds the device on its own, free-running or on its own trigger.
#   * No camera client. The bottom-chamber server does not need to be running
#     -- in fact it must NOT be, because SpinView and the pylabnet server
#     cannot both own the camera.
#   * The cooling and repump light stay ON for a long hold at the far point,
#     instead of one 200 us pulse. That is what makes the cloud visible as a
#     live picture: the atoms fluoresce for as long as the light is on.
#   * No TOF, no second frame, no atoms-clear wait.
#
# THE LIGHT IS DESTRUCTIVE. Resonant light heats the atoms and pushes them out
# of the lattice within milliseconds to tens of milliseconds. This sequence is
# for alignment and for confirming that transport works, not for data. Keep
# "Imaging Power" as low as SpinView can still see, and expect the cloud to
# fade during the hold.
#
# One cycle:
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> pre-move hold
#   -> leg 1 (Move Mode 1: COM4 / 2: COM5 / 3: First Driver, then the other)
#   -> post-move hold
#   -> BOTTOM HOLD: lattice stationary at the far point, imaging light ON,
#      MOT VCO at the imaging detuning, coils off   <-- watch this in SpinView
#   -> coils back on, next cycle
#
# WIRING: unchanged apart from dio8. dio6 -> COM4 rear "Sync", dio7 -> COM5
# rear "Sync". dio8 stays wired to the camera but is only ever parked low.
#
# COMMANDS: this firmware only acts on the S-prefix form of the sweep commands
# (Sswpm:2, Sswps:..., Sswpp:..., Sswpf:..., Sswpt:...), not the manual's
# SWPm:/SWPs:/... -- found on the bench.
#
# HOW MODE 2 BEHAVES (manual v9, p.8-9). Sync is a GATE:
#     Sync LOW            output sits at Fstart (SWPs = base)
#     Sync goes HIGH      ramps Fstart -> Fstop (SWPp), then HOLDS at Fstop
#     Sync goes LOW       ramps back to Fstart at the same rate
# so one gate pulse is one whole leg: accel, plateau, decel.
#
#     ramp time   T_r = (|Fstop - Fstart| / SWPf) * SWPt
#     gate width  T_g = T_r + T_p
#     distance    d   = v * T_g,   v = (lambda/2) * |Fstop - Fstart|
#
# The "RF signals" window draws the sub-sequence as it is sent: both AOM
# frequencies, the lattice velocity they imply, and the dio6 / dio7 gates
# taken from the actual NI buffers.

from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import dim_state
import serial
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D
import time
import pyqtgraph as pg

try:
    from scipy.optimize import curve_fit
except ImportError:      # fall back to moments; see fit_gaussian_1d
    curve_fit = None

# A 1/e^2 radius is 2 sigma, and integrating a 1/e^2-parameterised Gaussian
# gives w*sqrt(pi/2) -- both used to turn the two 1D fits back into a 2D peak.
ROOT_HALF_PI = np.sqrt(np.pi / 2.0)

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

# Brightest a single pixel can report on this camera. Mono16 is 12-bit data
# left-shifted by four plus a 12 count pedestal, so the ceiling is
# 16*4095 + 12, NOT the 65535 the container suggests. A typed display range
# is clamped to +/- this, so the scale can never open on an impossible value.


def rf_signal_trace(base_hz, sc, legs, gates, gate_line, spm, v_per_hz_mmps):
    """One sub-sequence of the RF drive, as plottable piecewise-linear lines.

    :param base_hz: (int) rest frequency of both drivers (SWPs)
    :param sc: (dict) schedule entry -- f_stop_hz, ramp_ms, gate_ms
    :param legs: (list) leg_layout() output: [(port, start_sample, gate_s)];
        each driver's RF starts ramping at its start sample
    :param gates: (dict) NI line -> list of 0/1 samples, EXACTLY as sent
    :param gate_line: (dict) port -> NI line, e.g. {'COM4': 'dio6', ...}
    :param spm: (int) NI samples per ms
    :param v_per_hz_mmps: (float) lattice velocity per Hz of detuning
    :return: (dict) t_ms, f4_hz, f5_hz, v_mps (all on t_ms), and
        gate_<line> -> (t_ms, 0/1) step points for each line
    """
    any_line = next(iter(gates.values()))
    end_ms = len(any_line) / spm

    def corners(port):
        """(t_ms, f_hz) corners for one driver: base, then a trapezoid per leg."""
        pts = [(0.0, float(base_hz))]
        for p, start, _ in legs:
            if p != port:
                continue
            t0 = start / spm
            stop = float(sc['f_stop_hz'])
            ramp, g_ms = sc['ramp_ms'], sc['gate_ms']
            pts += [(t0, base_hz), (t0 + ramp, stop), (t0 + g_ms, stop),
                    (t0 + g_ms + ramp, base_hz)]
        pts.append((end_ms, float(base_hz)))
        return np.array(pts, dtype=float)

    c4, c5 = corners('COM4'), corners('COM5')
    # A common time axis holding every corner of both drivers. Both lines are
    # linear between corners, so interpolating onto it is exact, and the
    # velocity -- a difference of the two -- is exact on it too.
    t = np.unique(np.concatenate([c4[:, 0], c5[:, 0]]))
    f4 = np.interp(t, c4[:, 0], c4[:, 1])
    f5 = np.interp(t, c5[:, 0], c5[:, 1])
    # v = (lambda/2) * (f_COM4 - f_COM5): "+" while COM4 is up, "-" while
    # COM5 is up. In m/s so pyqtgraph picks the SI prefix.
    v_mps = v_per_hz_mmps * (f4 - f5) * 1e-3
    out = {'t_ms': t, 'f4_hz': f4, 'f5_hz': f5, 'v_mps': v_mps}
    for port, line in gate_line.items():
        g = np.asarray(gates[line], dtype=np.int8)
        edges = np.flatnonzero(np.diff(g)) + 1
        gt, gv = [0.0], [float(g[0]) if g.size else 0.0]
        for e in edges:
            gt += [e / spm, e / spm]
            gv += [gv[-1], float(g[e])]
        gt.append(end_ms)
        gv.append(gv[-1])
        out[f'gate_{port}'] = (np.array(gt), np.array(gv))
    return out


class RFSignalView:
    """Second window: the RF drive over one sub-sequence.

    Three plots on one time axis -- both AOM frequencies, the lattice velocity
    they imply, and the two sweep gates (dio6 -> COM4, dio7 -> COM5) as they
    are in the NI buffers -- with the sequence's phase boundaries marked. It
    opens zoomed on the move; drag or right-click -> "View All" for the whole
    cycle.

    Same threading rule as MoveLiveView: callers only store data (set_trace),
    a GUI-thread QTimer draws it.
    """

    POLL_MS = 200
    COLOURS = {'COM4': '#4DD0E1', 'COM5': '#FFB74D'}

    def __init__(self, title="RF signals (COM4 + COM5)"):
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle(title)
        self.win.setStyleSheet("background-color: black;")
        box = QtWidgets.QVBoxLayout(self.win)
        box.setContentsMargins(6, 6, 6, 6)
        self.label = QtWidgets.QLabel('RF signals appear when the run starts.')
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setStyleSheet("color: #DDDDDD; font-size: 10pt;")
        box.addWidget(self.label)
        canvas = pg.GraphicsLayoutWidget()
        box.addWidget(canvas, stretch=1)
        self.win.resize(900, 860)
        self.win.show()

        self.f_plot = canvas.addPlot(row=0, col=0)
        self.f_plot.setLabel('left', 'RF frequency', units='Hz')
        self.f_plot.showGrid(x=True, y=True, alpha=0.3)
        self.f_plot.addLegend(offset=(10, 10))
        self.f4_curve = self.f_plot.plot(
            [], [], pen=pg.mkPen(self.COLOURS['COM4'], width=2), name='COM4')
        self.f5_curve = self.f_plot.plot(
            [], [], pen=pg.mkPen(self.COLOURS['COM5'], width=2,
                                 style=QtCore.Qt.DashLine), name='COM5')

        self.v_plot = canvas.addPlot(row=1, col=0)
        self.v_plot.setLabel('left', 'lattice velocity', units='m/s')
        self.v_plot.showGrid(x=True, y=True, alpha=0.3)
        self.v_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen('#616161', width=1)))
        self.v_curve = self.v_plot.plot([], [], pen=pg.mkPen('#81C784', width=2))

        self.g_plot = canvas.addPlot(row=2, col=0)
        self.g_plot.setMaximumHeight(150)
        self.g_plot.setLabel('left', 'Sync gates')
        self.g_plot.setLabel('bottom', 'time in sub-sequence', units='s')
        self.g_plot.setYRange(-0.1, 1.2)
        self.g_plot.showGrid(x=True, y=False, alpha=0.3)
        self.g_plot.addLegend(offset=(10, 5))
        # COM5's gate is drawn slightly lower so the two never hide each other.
        self.g_curves = {
            'COM4': self.g_plot.plot([], [], pen=pg.mkPen(self.COLOURS['COM4'], width=2),
                                     name='dio6 -> COM4'),
            'COM5': self.g_plot.plot([], [], pen=pg.mkPen(self.COLOURS['COM5'], width=2),
                                     name='dio7 -> COM5'),
        }
        self.G_OFFSET = {'COM4': 0.0, 'COM5': -0.04}

        self.v_plot.setXLink(self.g_plot)
        self.f_plot.setXLink(self.g_plot)
        self._markers = []

        self._pending = None
        self._drawn = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(self.POLL_MS)

    def set_trace(self, trace, markers, html, zoom_ms):
        """Called from the acquisition thread; stores data, touches no Qt.

        :param markers: (list of (t_ms, text)) phase boundaries to mark
        :param zoom_ms: (lo, hi) initial x range in ms
        """
        self._pending = (trace, markers, html, zoom_ms, self._drawn + 1)

    def _poll(self):
        entry = self._pending
        if entry is None or entry[-1] <= self._drawn:
            return
        trace, markers, html, zoom_ms, version = entry
        self._drawn = version
        try:
            self.label.setText(html)
            s = 1e-3                                      # ms -> s
            self.f4_curve.setData(trace['t_ms'] * s, trace['f4_hz'])
            self.f5_curve.setData(trace['t_ms'] * s, trace['f5_hz'])
            self.v_curve.setData(trace['t_ms'] * s, trace['v_mps'])
            for port, curve in self.g_curves.items():
                gt, gv = trace[f'gate_{port}']
                curve.setData(gt * s, gv + self.G_OFFSET[port])
            # Flat lines autoscale to a zero-height range; give them some.
            lo = min(trace['f4_hz'].min(), trace['f5_hz'].min())
            hi = max(trace['f4_hz'].max(), trace['f5_hz'].max())
            pad = max((hi - lo) * 0.1, 1e3)
            self.f_plot.setYRange(lo - pad, hi + pad)
            vlo, vhi = trace['v_mps'].min(), trace['v_mps'].max()
            vpad = max((vhi - vlo) * 0.1, 1e-4)
            self.v_plot.setYRange(vlo - vpad, vhi + vpad)
            for plot, line in self._markers:
                plot.removeItem(line)
            self._markers = []
            for t_ms, text in markers:
                for plot in (self.f_plot, self.v_plot, self.g_plot):
                    line = pg.InfiniteLine(
                        pos=t_ms * s, angle=90,
                        pen=pg.mkPen('#9E9E9E', width=1, style=QtCore.Qt.DotLine),
                        label=text if plot is self.f_plot else None,
                        labelOpts={'position': 0.95, 'color': '#BDBDBD'})
                    plot.addItem(line)
                    self._markers.append((plot, line))
            self.g_plot.setXRange(zoom_ms[0] * s, zoom_ms[1] * s, padding=0)
        except Exception:
            pass          # a display problem must never stop the run


# =============================================================================
INIT_DICT = {
    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '2000'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '6'},
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
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '8'},
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
    'pgc_rampdown_ms': {'PGC Rampdown Duration (ms)': '0'},

    # ===================== LATTICE AM =====================
    # DIM-3000 set to AMP=320 (+32dBm) via dim_rf_on.py; AM=1.0V => full max.
    # AOM thermal stability: lattice TTL stays ON & AM stays at lat_am_v
    # for the ENTIRE cycle.
    'lat_am_v': {'Lattice AM (V)': '0.6'},

    # ===================== MOVE MODE =====================
    #   1 = COM4 swept, one way ("+")      2 = COM5 swept, one way ("-")
    #   3 = there and back: First Driver out, the other driver back
    # COM5 drives the BOTTOM lattice beam, COM4 the TOP one, so COM5 carries
    # the atoms DOWN toward the bottom chamber and COM4 brings them back UP.
    # Mode 2 is the default here: this script exists to watch them arrive.
    # (If a run shows the opposite, flip LEG_DIR in _run_experiment -- it is
    # the one place the direction words are defined.)
    'move_mode': {'Move Mode (1 = COM4 up, 2 = COM5 down, 3 = there & back: down then up)': '2'},
    # Mode 3 only: which driver sweeps on the way OUT (the other brings the
    # cloud back). 4 = COM4 first ("+" then "-"), 5 = COM5 first.
    # 5 = COM5 first, i.e. DOWN then UP (the default: down to the bottom
    # chamber, then back up). 4 = COM4 first, i.e. UP then DOWN.
    'first_driver': {'There & Back: First Driver (5 = COM5 down first, 4 = COM4 up first)': '5'},
    # Mode 3 only: lattice stationary at the far point between the two legs.
    'far_hold_ms': {'There & Back: Far-point Hold (ms)': '5'},

    # ===================== MOVE (each leg; same for both drivers) =====================
    # Same inputs as fastLatticeMove. Velocity is derived exactly as there:
    #
    #     v = d / (T_a/2 + T_p + T_d/2)
    #
    # but the sweep has ONE step rate for both ramps, so accel and decel both
    # run for (T_a + T_d)/2. With T_a == T_d (the defaults) nothing changes.
    # The start/stop frequencies and the sweep duration the driver will
    # actually play are printed at the top of the panel. The DIRECTION is set
    # by which driver sweeps (Move Mode), not by a sign: sweeps only go up.
    #
    # 0 distance is a stationary reference: no gate opens, sweep mode is left
    # OFF on both drivers, and the cloud is imaged where it was loaded.
    'dist_mm': {'Move Distance (mm, each leg)': '256'},
    # T_p = 0 is a triangle (the gate closes the moment Fstop is reached).
    # T_a = T_d = 0 is a velocity STEP, which the log warns about.
    'accel_ms': {'Accel Time T_a (ms)': '1.0'},
    'plat_ms': {'Plateau (constant velocity) Time T_p (ms)': '160'},
    'decel_ms': {'Decel Time T_d (ms)': '1.0'},
    # How far AHEAD of each leg its gate is raised, to absorb the driver's
    # Sync-input latency (one value for both drivers). The manual documents
    # none for the sweep, so 0 until measured. Fractional ms allowed; snapped
    # to the 50 us NI grid.
    'seg_lead_ms': {'Sweep Gate Lead (ms) [Sync latency]': '0'},

    # ===================== FIXED LATTICE HOLDS (before + after move) =====================
    'pre_hold_ms': {'Pre-move Hold (ms)': '5'},
    # After the last decel the lattice is stationary -- at the far point in
    # modes 1/2, back at the start in mode 3. How long before release.
    'post_hold_ms': {'Post-move Hold (ms)': '5'},

    # ===================== DIM-3000 AOM DRIVER =====================
    'dim_base_freq_mhz': {'DIM3000 Base Freq (MHz)': '99.800'},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)': '32.0'},

    # ===================== BOTTOM HOLD, LIGHT ON (watch this in SpinView) =====================
    # After the move the lattice is stationary at the far point and the cooling
    # and repump light come ON and STAY on for this long, so the cloud
    # fluoresces continuously and SpinView shows it live.
    #
    # The light is resonant and destructive: the atoms heat and leave the
    # lattice in milliseconds to tens of milliseconds. Start long enough to
    # find the cloud, then shorten it. A hold longer than the cloud survives
    # costs nothing but cycle time.
    'bottom_hold_ms': {'Bottom Hold with Light ON (ms)': '500'},
    # The light opens the moment the post-move hold ENDS, so the sequence is:
    # lattice comes to rest -> post_hold_ms of darkness while the cloud
    # settles -> light on for exactly bottom_hold_ms. The lit time IS
    # bottom_hold_ms; nothing else adds to it.
    # Drive for the imaging light during that hold. Same channels as the MOT
    # (ao1/ao5); lower is gentler on the atoms and still visible if SpinView's
    # gain and exposure are up.
    'img_mot_power': {'Imaging Cooling Power (V)': '0.8'},
    'img_repump_power': {'Imaging Repump Power (V)': '0.8'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before Dark MOT)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after the hold)': '10'},

    # ===================== BIAS COILS =====================
    # Held CONSTANT at these values for the ENTIRE sub-sequence
    # (no ramping, on through MOT loading too). Matches sweep scripts.
    'bias_x_v': {'Bias X (V)': '0.0'},
    'bias_y_v': {'Bias Y (V)': '0.0'},
    'bias_z_v': {'Bias Z (V)': '0.0'},

    # ===================== RUN =====================
    # Cycles the OPX clocks per NI arm. The cards are re-armed between passes,
    # which leaves a gap of a second or so with the lattice at base.
    'reps_per_arm': {'Cycles per Arm (then re-arm)': '200'},

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
    'dio8': 0,      # bottom camera trigger idle
    'dio1': 0,      # MOT AOM enable -- light off
    'dio2': 1,      # coils TTL: the sequence's "coils off" state
    'dio3': 0,      # repump AOM enable -- light off
    'dio4': 0,      # lattice up   -- LOW = ON, left running on purpose
    'dio5': 0,      # lattice down -- LOW = ON, left running on purpose
    'dio6': 0,      # COM4 sweep gate LOW -> COM4 back at Fstart (= base)
    'dio7': 0,      # COM5 sweep gate LOW -> COM5 back at Fstart (= base)
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
        "cooling/repump light off, camera triggers low, lattice %s."
        % (f"ON at {float(lattice_am):.2f} V (stationary)" if lattice_on
           else "OFF")
    )


# The bottom camera is NOT touched by this script: no client, and dio8 is
# never driven. SpinView owns the device. dio8 is still parked low on stop so
# the line cannot float into the camera Line0 input.
CAMERA_TRIGGER_DO = None


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    # The RF signals window is the only display: there are no frames to draw.
    dataset.rf_view = RFSignalView()
    dataset.graph.hide()


def experiment(**kwargs):
    """Run the sequence, then put the RF drivers back the way they were found.

    DataTaker calls this; the sequence itself is _run_experiment(). This
    wrapper exists for its `finally:` alone -- whatever ends the run (Stop, an
    exception, a device dropping out), every DIM-3000 driver the run opened is
    restored to the settings dim_state.snapshot() recorded before the run
    wrote anything to it, and left OUT of segment mode. See dim_state.py.

    Sweep mode is switched off here FIRST, with Sswpm:0. A driver left in
    mode 2 walks the lattice to Fstop on any stray HIGH at its Sync input.

    COMMAND FORM: this firmware ignores the manual's SWPm:/SWPs:/SWPp:/SWPf:/
    SWPt: commands and only acts on the Gpar-style S-prefix writes (Sswpm:,
    Sswps:, ...), found on the bench. Every sweep command in this file uses
    that form.
    """
    snaps = []
    FSK_FOUND.clear()
    try:
        _run_experiment(dim_snapshots=snaps, **kwargs)
    finally:
        for snap in snaps:
            if not snap:
                continue
            ser = snap['_ser']
            try:
                if not ser.is_open:
                    ser.open()
                ser.write(b'Sswpm:0\n')
                time.sleep(0.12)
                # FSK frequency / amplitude as found (dim_state does not
                # restore them; this run set both to base / AMP).
                found = FSK_FOUND.get(snap['_port'], {})
                if 'Rffreq' in found:
                    ser.write(f"Sffreq:{found['Rffreq']}\n".encode())
                    time.sleep(0.12)
                if 'Rfampl' in found:
                    ser.write(f"Sfampl:{found['Rfampl']}\n".encode())
                    time.sleep(0.12)
            except Exception:
                pass
        dim_state.restore_all(snaps, kwargs['dataset'].log)


# FSK frequency / amplitude of each driver as the run found them, keyed by
# port; filled by _run_experiment, restored by experiment()'s finally.
FSK_FOUND = {}


def _run_experiment(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    logger = dataset.log

    NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
    OPX_client = kwargs["OPX_OPX"]      # no camera client: SpinView owns it

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

    # ONE transport distance, kept as a one-entry list so the per-point
    # structures below index the same way fastLatticeMove's do.
    dist_values_mm = [float(dataset.get_input_parameter("dist_mm"))]
    n_scan = 1
    pre_hold_ms = int(dataset.get_input_parameter("pre_hold_ms"))
    post_hold_ms = int(dataset.get_input_parameter("post_hold_ms"))
    accel_ms = float(dataset.get_input_parameter("accel_ms"))
    plat_ms_in = float(dataset.get_input_parameter("plat_ms"))
    decel_ms = float(dataset.get_input_parameter("decel_ms"))
    # Float, not int as in fastLatticeMove: the Sync latency may well be
    # sub-millisecond, and int("0.2") would stop the run before it started.
    seg_lead_ms = float(dataset.get_input_parameter("seg_lead_ms"))
    if min(accel_ms, plat_ms_in, decel_ms) < 0:
        raise ValueError(
            f"Move durations must be >= 0 (got T_a={accel_ms}, "
            f"T_p={plat_ms_in}, T_d={decel_ms} ms).")
    if seg_lead_ms < 0:
        raise ValueError(f"Sweep Gate Lead must be >= 0, got {seg_lead_ms} ms.")
    if float(dataset.get_input_parameter("dist_mm")) < 0:
        raise ValueError(
            "Move Distance must be >= 0; the direction is set by Move Mode "
            "(which driver sweeps), not by the sign of the distance.")
    dim_base_freq_mhz = float(dataset.get_input_parameter("dim_base_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))
    # Sweeps only go UP on this driver, so every leg is base -> base + df and
    # the direction comes from WHICH driver sweeps. schedule_for_distance()
    # still takes a direction; it is always +1 here.
    move_direction = 1
    move_mode = int(float(dataset.get_input_parameter("move_mode")))
    if move_mode not in (1, 2, 3):
        raise ValueError(f"Move Mode must be 1, 2 or 3, got {move_mode}.")
    first_driver = int(float(dataset.get_input_parameter("first_driver")))
    if move_mode == 3 and first_driver not in (4, 5):
        raise ValueError(f"First Driver must be 4 (COM4) or 5 (COM5), got {first_driver}.")
    far_hold_ms = float(dataset.get_input_parameter("far_hold_ms"))
    if move_mode == 3 and far_hold_ms < 0:
        raise ValueError(f"Far-point Hold must be >= 0 ms, got {far_hold_ms}.")
    # The legs, in play order, as the COM port each one sweeps.
    if move_mode == 1:
        leg_ports = ['COM4']
    elif move_mode == 2:
        leg_ports = ['COM5']
    else:
        leg_ports = ['COM4', 'COM5'] if first_driver == 4 else ['COM5', 'COM4']
    # "+" = COM4 above COM5, "-" = COM5 above COM4 (v = lambda/2 * (f4 - f5)).
    LEG_SIGN = {'COM4': +1, 'COM5': -1}
    # WHICH WAY EACH DRIVER SENDS THE ATOMS. COM5 drives the BOTTOM lattice
    # beam and COM4 the TOP one, so sweeping COM5 up carries the cloud DOWN
    # toward the bottom chamber and sweeping COM4 up carries it back UP.
    #
    # THE ONE PLACE TO CHANGE IF THE LAB SAYS OTHERWISE. A moving lattice
    # travels along the propagation direction of the HIGHER-frequency beam, so
    # which label is right depends on which way each beam points -- something
    # no part of this script can check. Run one leg, watch the cloud in
    # SpinView, and if it goes the other way, swap the two words here.
    LEG_DIR = {'COM5': 'DOWN', 'COM4': 'UP'}
    if move_mode == 3:
        mode_desc = (f"there & back: {leg_ports[0]} {LEG_DIR[leg_ports[0]]}, "
                     f"then {leg_ports[1]} {LEG_DIR[leg_ports[1]]}")
    else:
        port = leg_ports[0]
        mode_desc = f"{port} one way ({LEG_DIR[port]})"
    # The lit hold at the far point: what SpinView watches.
    bottom_hold_ms = float(dataset.get_input_parameter("bottom_hold_ms"))
    if bottom_hold_ms <= 0:
        raise ValueError(
            f"Bottom Hold must be > 0 ms, got {bottom_hold_ms}: with no hold the "
            f"light never comes on and there is nothing to see in SpinView.")
    img_mot_power = float(dataset.get_input_parameter("img_mot_power"))
    img_repump_power = float(dataset.get_input_parameter("img_repump_power"))
    reps_per_arm = max(1, int(float(dataset.get_input_parameter("reps_per_arm"))))

    # ============ Move physics ============
    # v = lambda/2 * df. For 795nm: v[mm/s] = 0.0003975 * df[Hz]
    LATTICE_WAVELENGTH = 795e-9
    v_per_Hz_mmps = LATTICE_WAVELENGTH / 2 * 1e3  # mm/s per Hz of detuning
    base_freq_hz = int(round(dim_base_freq_mhz * 1e6))

    # NI sample clock. Needed this early because the gate width -- and through
    # it the stop frequency -- is quantised to it.
    ni_rate = 20000
    SPM = ni_rate // 1000
    SAMPLE_MS = 1000.0 / ni_rate

    def ceil_samples(t_ms):
        """ms -> whole NI samples, rounding UP (1e-9 absorbs float fuzz)."""
        return int(np.ceil(t_ms * SPM - 1e-9))

    # DIM-3000 sweep limits (manual p.4 and p.13). SWPt is in ns, 4..262000,
    # and the firmware rounds it DOWN to a multiple of 4 -- so only multiples
    # of 4 are ever sent, and what is sent is what plays.
    #
    # SWEEP_STEP_NS_MIN is ours, not the manual's: the driver accepts 4 ns,
    # but 1 us steps already make a 5 ms ramp 5000 steps long, far smoother
    # than the lattice can resolve, and nothing is gained by asking the DDS
    # for more.
    SWEEP_STEP_NS_QUANTUM = 4
    SWEEP_STEP_NS_MAX = 262000
    SWEEP_STEP_NS_MIN = 1000
    SWEEP_FSTEP_SEARCH = 20000   # Fstep candidates tried, from the finest up

    def sweep_steps(span_hz, ramp_ms):
        """SWPf and SWPt for a ramp covering span_hz in (at most) ramp_ms.

        The ramp is n steps of SWPf Hz, each SWPt ns long, and the driver
        plays exactly that, so n * SWPf is the span it actually covers.
        Candidates are tried from the FINEST step upward, and the first one
        within 2e-4 of the span (the span IS the velocity, and so the
        distance) and 2e-3 of the ramp time is taken -- so the ramp is as
        smooth as the step-time floor allows. If none gets that close, the
        best-scoring one is used. SWPt is always rounded DOWN, so the ramp is
        never longer than asked for and therefore always fits inside the gate.

        :return: (dict) fstep_hz, tstep_ns, n_steps, span_hz (as played),
            ramp_ms (as played), clipped (True if the ramp had to be made
            SHORTER than asked because even 1 Hz steps at the longest step
            time cannot fill it)
        """
        q = SWEEP_STEP_NS_QUANTUM
        target_ns = ramp_ms * 1e6
        if target_ns < SWEEP_STEP_NS_MIN:
            # A ramp shorter than one step -- including 0, a velocity step.
            # One step straight to Fstop at the shortest time the driver takes.
            t_ns = max(q, int(target_ns // q) * q)
            return {'fstep_hz': span_hz, 'tstep_ns': t_ns, 'n_steps': 1,
                    'span_hz': span_hz, 'ramp_ms': t_ns * 1e-6,
                    'clipped': False}
        n_min = max(1, int(np.ceil(target_ns / SWEEP_STEP_NS_MAX)))
        n_max = int(target_ns // SWEEP_STEP_NS_MIN)
        if span_hz < n_min:
            # Too slow a ramp for the span: 1 Hz steps at 262 us each run out
            # of frequency before they run out of time. Take the slowest ramp
            # there is and say so.
            t_ns = (SWEEP_STEP_NS_MAX // q) * q
            return {'fstep_hz': 1, 'tstep_ns': t_ns, 'n_steps': span_hz,
                    'span_hz': span_hz, 'ramp_ms': span_hz * t_ns * 1e-6,
                    'clipped': True}
        best = None
        f_lo = max(1, span_hz // max(n_max, 1))
        for f in range(f_lo, f_lo + SWEEP_FSTEP_SEARCH):
            n = int(round(span_hz / f))
            if n < n_min:
                break          # steps only get coarser from here
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

    def schedule_for_distance(d_mm):
        """Everything the sweep needs for one one-way move of distance d_mm.

        Velocity comes from the typed durations exactly as in fastLatticeMove,
        v = d / (T_a/2 + T_p + T_d/2). Mode 2 then forces both ramps to one
        length, T_r = (T_a + T_d)/2, which keeps T_r + T_p -- and so v and
        the total time -- unchanged.

        The gate is T_r + T_p, rounded UP to the NI grid so the ramp always
        completes inside it (a gate that closed early would turn round below
        Fstop, like the second pulse in the manual's mode-2 diagram). Since
        d = v * gate for this profile, the detuning is then recomputed from
        the gate as it will actually be played, so the rounding costs a few
        tens of microseconds of plateau rather than any distance.

        :return: (dict) with f_start_hz / f_stop_hz / fstep_hz / tstep_ns
            for the driver, gate_s for the NI card, and every derived
            quantity the panel and log print.
        """
        stationary = {'d_mm': float(d_mm), 'd_real_mm': 0.0, 'v_mmps': 0.0,
                      'ramp_req_ms': 0.0, 'ramp_ms': 0.0,
                      'plat_ms': 0.0, 'plat_real_ms': 0.0,
                      'gate_s': 0, 'gate_ms': 0.0, 'total_ms': 0.0,
                      'a_mps2': 0.0, 'f_start_hz': base_freq_hz,
                      'f_stop_hz': base_freq_hz, 'df_hz': 0,
                      'fstep_hz': 0, 'tstep_ns': 0, 'n_steps': 0,
                      'clipped': False}
        if d_mm <= 0:
            return stationary
        ramp_req = 0.5 * (accel_ms + decel_ms)
        if ramp_req + plat_ms_in <= 0:
            raise ValueError(
                f"Distance {d_mm} mm needs a non-zero move time, but "
                f"T_a/2 + T_p + T_d/2 = 0. Set at least one duration.")
        gate_s = ceil_samples(ramp_req + plat_ms_in)
        gate_ms = gate_s * SAMPLE_MS
        span_target = int(round(d_mm / (gate_ms * 1e-3) / v_per_Hz_mmps))
        if span_target < 1:
            raise ValueError(
                f"{d_mm} mm in {gate_ms:.3f} ms is under 1 Hz of detuning, "
                f"below the driver's resolution.")
        st = sweep_steps(span_target, ramp_req)
        span = st['span_hz']
        v_mmps = span * v_per_Hz_mmps
        ramp_ms = st['ramp_ms']
        return {'d_mm': float(d_mm),
                'd_real_mm': v_mmps * gate_ms * 1e-3,
                'v_mmps': v_mmps,
                'ramp_req_ms': ramp_req,
                'ramp_ms': ramp_ms,
                'plat_ms': plat_ms_in,
                'plat_real_ms': gate_ms - ramp_ms,
                'gate_s': gate_s,
                'gate_ms': gate_ms,
                # Sync rising edge -> back at Fstart: up, hold, down.
                'total_ms': gate_ms + ramp_ms,
                'a_mps2': (v_mmps * 1e-3) / (ramp_ms * 1e-3),
                'f_start_hz': base_freq_hz,
                'f_stop_hz': base_freq_hz + move_direction * span,
                'df_hz': span,
                'fstep_hz': st['fstep_hz'],
                'tstep_ns': st['tstep_ns'],
                'n_steps': st['n_steps'],
                'clipped': st['clipped']}

    schedule = [schedule_for_distance(d) for d in dist_values_mm]
    v_peak_mmps_per_point = [sc['v_mmps'] for sc in schedule]

    sc0 = schedule[0]
    do_move_any = sc0['d_mm'] > 0
    if do_move_any:
        logger.info(
            f"Move ({mode_desc}), EACH LEG: d={sc0['d_mm']:.3f}mm, "
            f"typed T_a={accel_ms:.3f} / T_p={plat_ms_in:.3f} / "
            f"T_d={decel_ms:.3f} ms -> played: ramp {sc0['ramp_ms']:.4f} ms "
            f"each way, plateau {sc0['plat_real_ms']:.4f} ms, gate "
            f"{sc0['gate_ms']:.3f} ms, sweep {sc0['total_ms']:.4f} ms; "
            f"v={sc0['v_mmps']:.3f}mm/s, a={sc0['a_mps2']:.3f}m/s^2, "
            f"SWPs={sc0['f_start_hz']} Hz, SWPp={sc0['f_stop_hz']} Hz "
            f"(+{sc0['df_hz'] / 1e3:.3f} kHz), "
            f"SWPf={sc0['fstep_hz']} Hz, SWPt={sc0['tstep_ns']} ns, "
            f"{sc0['n_steps']} steps; realised distance {sc0['d_real_mm']:.4f}mm")
        if abs(accel_ms - decel_ms) > 1e-9:
            logger.warn(
                f"T_a ({accel_ms} ms) != T_d ({decel_ms} ms). Sweep mode 2 has "
                f"one step rate for both ramps, so BOTH run for their mean, "
                f"{sc0['ramp_req_ms']:.3f} ms. Distance, peak velocity and leg "
                f"time are as typed; the accelerations are not.")
        if accel_ms + decel_ms <= 0:
            logger.warn(
                f"Accel and decel times are 0: the lattice jumps to "
                f"{sc0['v_mmps']:.1f} mm/s in a single step and stops the same "
                f"way. That is an unbounded acceleration and will shake most "
                f"atoms out.")
        if sc0['clipped']:
            logger.warn(
                f"The ramp is only {sc0['ramp_ms']:.3f} ms, not the "
                f"{sc0['ramp_req_ms']:.3f} ms asked for: {sc0['df_hz']} Hz of "
                f"detuning in 1 Hz steps at the longest step time the driver "
                f"has (262 us) cannot take longer. The plateau is longer to "
                f"match, so the distance is unaffected.")
    else:
        logger.info("Move distance 0: STATIONARY reference -- no gate opens, "
                    "sweep mode OFF on both drivers, cloud imaged where it "
                    "was loaded.")

    # =========================================================================
    # CONFIGURE DIM-3000 DRIVERS (called once per experiment start)
    # =========================================================================
    # Each driver, the NI line wired to its rear "Sync", and which way the
    # lattice goes when it is the one swept up.
    DIM_PORTS = ('COM4', 'COM5')
    GATE_LINE = {'COM4': 'dio6', 'COM5': 'dio7'}
    swept_ports = set(leg_ports) if do_move_any else set()

    # Only used to neutralise segment tables other scripts leave behind; this
    # script never plays a segment.
    DIM_MAX_SEGMENTS = 7
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
        """Gpar parsed to {field: int}; unparseable fields left out."""
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

    def _dim_fsk_safe(ser, port):
        """FSK frequency = base and FSK amplitude = AMP, for the run.

        In sweep mode 0 a HIGH at Sync jumps this firmware to its FSK
        frequency and amplitude (seen on the bench). The idle driver's Sync
        line stays LOW in the waveform, but if it ever did see a pulse, the
        jump is now to where it already is. The values found are recorded in
        FSK_FOUND and put back by experiment() on the way out.
        """
        found = _dim_gpar(ser)
        FSK_FOUND[port] = {k: found[k] for k in ('Rffreq', 'Rfampl') if k in found}
        _dim_send(ser, f'Sffreq:{base_freq_hz}')
        _dim_send(ser, f'Sfampl:{amplitude_x10}')

    def _dim_make_deaf(ser, port):
        """Park a driver at base in plain single-frequency operation: sweep OFF
        and segment mode OFF, so neither its Sync nor its Seg input moves it."""
        _dim_send(ser, 'Sswpm:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_send(ser, f'AMP:{amplitude_x10}')
        _dim_send(ser, 'OUT_on')
        _dim_neutralise_segments(ser)
        _dim_send(ser, 'Mseg:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_fsk_safe(ser, port)
        logger.info(
            f"DIM-3000 {port}: {base_freq_hz/1e6:.6f} MHz, "
            f"+{amplitude_x10/10:.1f} dBm, ON, sweep OFF (Sswpm:0), segment "
            f"mode OFF (Mseg:0), FSK = base -- not swept this run.")

    def _dim_program_sweep(ser, port, sc):
        """Load one leg into this driver's sweep engine and arm mode 2.

        The bench-tested sequence (S-prefix forms; see experiment()):
            Mseg:0  Sswpm:0  FRQ  AMP  OUT_on  Sswps  Sswpp  Sswpf  Sswpt  Sswpm:2
        with the segment scrub and FSK safeguard before it. Both drivers get
        identical settings, so a COM4 leg and a COM5 leg are mirror images.

        :return: (list of str) mismatches found on Gpar readback; empty = OK
        """
        _dim_neutralise_segments(ser)
        _dim_fsk_safe(ser, port)
        _dim_send(ser, 'Mseg:0')
        _dim_send(ser, 'Sswpm:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_send(ser, f'AMP:{amplitude_x10}')
        _dim_send(ser, 'OUT_on')
        _dim_send(ser, f"Sswps:{sc['f_start_hz']}")
        _dim_send(ser, f"Sswpp:{sc['f_stop_hz']}")
        _dim_send(ser, f"Sswpf:{sc['fstep_hz']}")
        _dim_send(ser, f"Sswpt:{sc['tstep_ns']}")
        _dim_send(ser, 'Sswpm:2')
        time.sleep(0.1)
        got = _dim_gpar(ser)
        logger.info(
            f"DIM-3000 {port} after arming: FRQ={got.get('Rfreq')} "
            f"SWPm={got.get('Rswpm')} SWPs={got.get('Rswps')} "
            f"SWPp={got.get('Rswpp')} SWPf={got.get('Rswpf')} "
            f"SWPt={got.get('Rswpt')} FFRQ={got.get('Rffreq')} "
            f"FAMP={got.get('Rfampl')}")
        want = {'Rswpm': 2, 'Rswps': sc['f_start_hz'],
                'Rswpp': sc['f_stop_hz'], 'Rswpf': sc['fstep_hz'],
                'Rswpt': sc['tstep_ns'], 'Rffreq': base_freq_hz,
                'Rfampl': amplitude_x10}
        return [f"{k} is {got.get(k, '<missing>')} (sent {v})"
                for k, v in want.items() if got.get(k) != v]

    # Open both ports ONCE and keep them open for the whole run.
    sers = {port: _dim_open(port) for port in DIM_PORTS}
    # Record the drivers BEFORE anything is written to them, so the
    # wrapper around this function can put them back exactly as found
    # when the run stops, however it stops (dim_state.py).
    for port in DIM_PORTS:
        kwargs['dim_snapshots'].append(dim_state.snapshot(sers[port], port, logger))
    if all(sers[port] is not None for port in DIM_PORTS):
        for port in DIM_PORTS:
            logger.info(f"DIM-3000 {port}: {_dim_query(sers[port], '*IDN?')}")
        for port in DIM_PORTS:
            ser = sers[port]
            if port not in swept_ports:
                _dim_make_deaf(ser, port)
                continue
            bad = _dim_program_sweep(ser, port, sc0)
            logger.info(
                f"DIM-3000 {port}: mode 2 armed on {GATE_LINE[port]} -- SWPs "
                f"{sc0['f_start_hz']} Hz, SWPp {sc0['f_stop_hz']} Hz, SWPf "
                f"{sc0['fstep_hz']} Hz, SWPt {sc0['tstep_ns']} ns.")
            if bad:
                logger.error(
                    f"DIM-3000 {port} did not take the sweep settings: "
                    f"{'; '.join(bad)}. The move will not be the one on the "
                    f"panel. Do not trust this run's images.")
            else:
                logger.info(f"DIM-3000 {port}: sweep settings verified by "
                            f"Gpar readback.")
    else:
        logger.info("DIM-3000: ports unavailable, will run without serial control")
        for port in DIM_PORTS:
            if sers[port] is not None:
                sers[port].close()
                sers[port] = None

    # Imaging (the lit hold)
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    wait_time = float(dataset.get_input_parameter("wait_time"))

    bias_x_v = float(dataset.get_input_parameter("bias_x_v"))
    bias_y_v = float(dataset.get_input_parameter("bias_y_v"))
    bias_z_v = float(dataset.get_input_parameter("bias_z_v"))

    # =========================================================================
    # CHANNELS
    # =========================================================================
    # No camera trigger channel: dio8 is never written (see CAMERA_TRIGGER_DO).
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
    # ni_rate and SPM are set with the move physics above: the gate width,
    # and through it the stop frequency, is quantised to this clock.
    SPU = ni_rate / 1e6
    delay_ns = int(round(1e9 / ni_rate)) - 500
    ttl_up = max(1, SPM // 20)
    # Rounds, because the derived ramp/plateau durations are floats (a
    # 14.14ms ramp is 283 samples). The 50us sample grid is 20x finer than
    # whole ms, so snapping here costs ~0.3% of a ramp instead of ~7%.
    def ms2s(t): return int(round(t * SPM))
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
    seg_lead_s = ms2s(seg_lead_ms)
    bottom_hold_s = ms2s(bottom_hold_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)

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
        f"pre_hold={pre_hold_ms}ms, "
        f"move {mode_desc}: {len(leg_ports)} leg(s) of {schedule[0]['total_ms']:.3f}ms "
        f"(gate {schedule[0]['gate_ms']:.3f}ms + decel {schedule[0]['ramp_ms']:.3f}ms)"
        + (f" with {far_hold_ms:g}ms far-point hold" if move_mode == 3 else "")
        + f", post_hold={post_hold_ms}ms (DARK), then light ON for the "
        + f"{bottom_hold_ms:g}ms bottom hold"
    )
    # The gate rises seg_lead_ms before the move is meant to start, and the
    # move starts pre_hold_ms after PGC ends. A lead longer than the pre-move
    # hold would push it back into the PGC ramp-down, where it would start the
    # atoms moving while the cooling light is still coming down.
    if seg_lead_ms >= pre_hold_ms:
        logger.warn(
            f"Gate Lead ({seg_lead_ms}ms) is not shorter than the pre-move "
            f"hold ({pre_hold_ms}ms), so the sweep gate opens during PGC "
            f"ramp-down instead of in the hold. Raise Pre-move Hold above "
            f"{seg_lead_ms}ms.")

    far_hold_s = ms2s(far_hold_ms) if move_mode == 3 else 0

    def leg_layout(sc):
        """Every leg of the move in samples -- ONE definition, used by both the
        waveform and the RF signals window, so they cannot disagree.

        A leg is one gate pulse on its driver's Sync line: HIGH for gate_s
        (accel + plateau), then the driver decelerates on its own for the
        ramp time after the falling edge. Legs run back to back, with the
        far-point hold between them in mode 3.

        :return: (legs, move_end). legs = [(port, start, gate_s), ...] where
            start is the sample at which that driver's RF starts ramping (its
            gate rises seg_lead_s earlier); move_end is when the last leg is
            back at base. No legs for a 0 mm reference.
        """
        t = pgc_end + pre_hold_s
        if sc['d_mm'] <= 0:
            return [], t
        leg_s = sc['gate_s'] + ceil_samples(sc['ramp_ms'])
        legs = []
        for i, port in enumerate(leg_ports):
            if i:
                t += far_hold_s
            legs.append((port, t, sc['gate_s']))
            t += leg_s
        return legs, t

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence(idx):
        sc = schedule[idx]
        # Sequence after PGC end:
        #   pre_hold -> leg(s) from leg_layout() -> post_hold -> LIT HOLD
        # The lit hold is the point of this script: the lattice sits still at
        # the far point while the cooling and repump light stay on, so the
        # cloud fluoresces for as long as SpinView needs to see it.
        # For 0mm reference (dist_mm==0) there are no legs, no gate opens, and
        # the atoms are lit where they were loaded.
        legs, move_end = leg_layout(sc)
        post_hold_end = move_end + post_hold_s
        # The light opens the moment the post-move hold ENDS. The whole move
        # and the hold after it are dark, so the cloud has post_hold_ms to
        # settle in a stationary lattice before any resonant light touches it,
        # and the lit window is exactly bottom_hold_ms -- what the input says
        # and nothing more.
        lit_start = post_hold_end
        lit_end = lit_start + bottom_hold_s
        crt = lit_end + coils_restore_s      # coils back on after the hold
        N = crt + ttl_up + 1

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
        # Lattice hold and the whole move: TTLs off (no cooling light), analog
        # stays at full for AOM thermal stability. The lattice does the
        # trapping from the end of PGC until the light comes back on.
        for s in range(pgc_end, min(lit_start, N)):
            ado[s] = 0
            rdo[s] = 0

        # THE LIT WINDOW. Cooling and repump ON continuously at the imaging
        # drive, from the end of the post-move hold until the bottom
        # hold runs out. This is what SpinView sees; it is also what destroys
        # the cloud, so keep it only as long as needed.
        for s in range(lit_start, min(lit_end, N)):
            ado[s] = 1
            aao[s] = img_mot_power
            rdo[s] = 1
            rao[s] = img_repump_power
        # After the hold: light off again for the tail of the cycle.
        for s in range(min(lit_end, N), N):
            ado[s] = 0
            rdo[s] = 0

        # Coils DO: 0=ON, 1=OFF
        cdo = [0] * N
        for s in range(coils_off, min(crt, N)):
            cdo[s] = 1

        # Coils AO
        cao = [mot_coils_v] * N
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            cao[t] = mot_coils_v + frac * (cmot_coils_v - mot_coils_v)
        for s in range(cmot_hold_start, N):
            cao[s] = cmot_coils_v

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
        # Imaging VCO override: on for the whole lit hold, switched vco_lead_us
        # early so the light is already at the imaging detuning when it opens.
        vlo = max(0, lit_start - vco_lead_s)
        vhi = min(N, lit_end)
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

        # Bias coils: held CONSTANT at GUI values for the ENTIRE sub-sequence
        # (no ramping, on through MOT loading too). Matches sweep scripts.
        biasx = [bias_x_v] * N
        biasy = [bias_y_v] * N
        biasz = [bias_z_v] * N

        # Lattice: TTL ON and AM constant at lat_am_v for the entire sub-seq
        # (matches the fast lattice code for AOM thermal stability).
        ludo = [0] * N
        lddo = [0] * N
        luao = [lat_am_v] * N
        ldao = [lat_am_v] * N

        # DIM-3000 sweep gates, one NI line per driver (dio6 -> COM4 Sync,
        # dio7 -> COM5 Sync). Each leg is HIGH on ITS driver's line for the
        # whole accel + plateau, raised seg_lead_ms early to absorb the Sync
        # latency. The lead moves both edges, so the gate width -- which is
        # what sets the distance -- is untouched by it. The other driver's
        # line stays LOW for that leg, so only the intended beam moves.
        gates = {GATE_LINE[port]: [0] * N for port in DIM_PORTS}
        for port, start, gate_s in legs:
            line = gates[GATE_LINE[port]]
            t0 = max(0, start - seg_lead_s)
            for s in range(gate_s):
                if t0 + s < N:
                    line[t0 + s] = 1

        return (ado, aao, rdo, rao, cdo, cao, mvco, rvco, biasx, biasy, biasz,
                ludo, lddo, luao, ldao, gates, N)

    # ONE sub-sequence per buffer. There are no camera frames to keep in step
    # with, so there is nothing for warm-up cycles to protect: the buffer
    # regenerates and every cycle is identical.
    (ado_p, aao_p, rdo_p, rao_p, cdo_p, cao_p, mvco_p, rvco_p,
     biasx_p, biasy_p, biasz_p, ludo_p, lddo_p, luao_p, ldao_p,
     gates_p, N) = build_sub_sequence(0)
    gates_w = gates_p

    total = N
    legs0, move_end0 = leg_layout(schedule[0])
    net_sign = sum(LEG_SIGN[port] for port, _, _ in legs0)   # +1, -1 or 0
    where = {1: "the FAR point (UP)", -1: "the FAR point (DOWN)",
             0: "its ORIGINAL position (there and back)"}[net_sign]
    # Same two anchors build_sub_sequence() uses, so the log, the markers and
    # the waveform cannot disagree about when the light is on.
    lit_start0 = move_end0 + post_hold_s
    lit_end0 = lit_start0 + bottom_hold_s
    logger.info(
        f"Lattice MOVE ({mode_desc}), SpinView version: "
        f"d={dist_values_mm[0]}mm per leg, "
        f"T_a/T_p/T_d = {accel_ms:.2f}/{plat_ms_in:.2f}/{decel_ms:.2f}ms, "
        f"v={schedule[0]['v_mmps']:.2f}mm/s, atoms lit at {where} from "
        f"{lit_start0 / SPM:.1f}ms ({post_hold_ms:g}ms of dark hold after the move) "
        f"for {(lit_end0 - lit_start0) / SPM:.1f}ms "
        f"(cooling {img_mot_power:.2f}V, repump {img_repump_power:.2f}V, "
        f"VCO {vco_imaging:.2f}V), cycle {N / SPM:.1f}ms, {total} samples")
    logger.info(
        "dio8 is NOT driven by this script -- open the bottom camera in "
        "SpinView. The pylabnet bottom-camera server must be stopped, or it "
        "and SpinView will fight over the device.")

    # RF signals window: the only display here. The gates are the real dio6 /
    # dio7 buffers, and the frequency lines come from the same leg_layout()
    # the waveform was built from, so the picture cannot drift from what is
    # played.
    sc0 = schedule[0]
    move_start0 = pgc_end + pre_hold_s
    try:
        rf_trace = rf_signal_trace(base_freq_hz, sc0, legs0, gates_w,
                                   GATE_LINE, SPM, v_per_Hz_mmps)
        rf_markers = [(pgc_end / SPM, 'PGC end')]
        for i, (port, start, _) in enumerate(legs0, start=1):
            rf_markers.append((start / SPM, f'leg {i}: {port}'))
        rf_markers += [(move_end0 / SPM, 'move end'),
                       (lit_start0 / SPM, 'light ON'),
                       (lit_end0 / SPM, 'light off')]
        legs_desc = " then ".join(
            f"{port} ({GATE_LINE[port]}, {LEG_DIR[port]})"
            for port, _, _ in legs0) or "none"
        rf_html = (
            f"<span style='font-family:monospace'>"
            f"<b>{mode_desc}</b> | legs: {legs_desc} | "
            f"net {net_sign * sc0['d_real_mm']:+.4f} mm<br>"
            f"each leg: SWPs <b>{sc0['f_start_hz'] / 1e6:.6f} MHz</b> &rarr; SWPp "
            f"<b>{sc0['f_stop_hz'] / 1e6:.6f} MHz</b>, "
            f"{sc0['fstep_hz']} Hz x {sc0['tstep_ns']} ns, sweep "
            f"{sc0['total_ms']:.3f} ms, gate {sc0['gate_ms']:.3f} ms, "
            f"peak v {sc0['v_mmps']:.2f} mm/s<br>"
            f"<b>light ON {(lit_end0 - lit_start0) / SPM:.1f} ms</b> from "
            f"{lit_start0 / SPM:.1f} ms, after a {post_hold_ms:g} ms dark hold "
            f"at the far point (cooling {img_mot_power:.2f} V, repump "
            f"{img_repump_power:.2f} V, VCO {vco_imaging:.2f} V) | cycle "
            f"{N / SPM:.1f} ms | dio8 not driven &mdash; SpinView owns the "
            f"camera</span>")
        margin = 5.0
        zoom = ((move_start0 - seg_lead_s) / SPM - pre_hold_ms - margin,
                lit_end0 / SPM + margin)
        dataset.rf_view.set_trace(rf_trace, rf_markers, rf_html, zoom)
    except Exception as exc:
        logger.warn(f"Could not draw the RF signals window: {exc!r}")

    # =========================================================================
    # MAIN LOOP: arm, let the OPX clock reps_per_arm cycles, re-arm
    # =========================================================================
    # No frames to wait on, so each pass is paced by its own length. Stop is
    # checked in short slices, so it takes effect in a fraction of a second
    # rather than at the end of a pass.
    run_buffer = 10_000_000                     # ns of OPX idle after the pass
    pass_s = reps_per_arm * total / ni_rate + run_buffer * 1e-9
    logger.info(f"Each arm plays {reps_per_arm} cycles = {pass_s:.1f} s, then "
                f"re-arms (lattice sits at base during the gap).")
    cycles_done = 0

    while thread.running:
        try:
            NI_card_1.finalize_clock()
        except Exception:
            pass

        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(do_channel=MOT_AOM_do, value=ado_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=mot_coils_do, value=cdo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel=Repump_AOM_do, value=rdo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio4", value=ludo_p, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio5", value=lddo_p, sample_rate=ni_rate)
        # Sweep gates: dio6 -> COM4 Sync, dio7 -> COM5 Sync. dio8 is absent on
        # purpose -- the camera belongs to SpinView.
        for line, samples in gates_p.items():
            NI_card_2.set_do_voltage(do_channel=line, value=samples, sample_rate=ni_rate)
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
        NI_card_1.arm_clock(length=total, sample_rate=ni_rate)

        OPX_client.build_stack()
        clk = OPX_client.create_new_do_elem(do_channel=opx_trigger_do, length=500)
        with OPX_client.for_("j", 0, reps_per_arm, 1):
            with OPX_client.for_("i", 0, total, 1):
                OPX_client.set_digital_voltage(element=clk)
                OPX_client.delay(delay_ns)
        OPX_client.delay(run_buffer)
        OPX_client.execute(wait=False)
        logger.info(f"Running {reps_per_arm} cycles -- watch SpinView.")

        t_end = time.time() + pass_s + 0.2
        while thread.running and time.time() < t_end:
            time.sleep(0.1)
        if thread.running:
            cycles_done += reps_per_arm

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
        # Sequence tasks are closed now, so static writes can land. A stop can
        # land mid-gate, which would leave a driver parked at Fstop with the
        # lattice walking; parking drops both gates at once.
        park_outputs(NI_card_2, NI_card_3, logger, lattice_am=lat_am_v)
        logger.info(f"Arm done: {cycles_done} cycles so far.")
        time.sleep(wait_time)

    # After the while loop ends (thread stopped): close DIM serial ports
    # (release REMOTE CONTROL display, free the COM port for next run)
    # Park BOTH drivers before letting go of the ports. Left in mode 2, a
    # driver would ramp to Fstop and STAY there on any HIGH at its Sync -- a
    # stop landing mid-gate, or the next script's dio6/dio7 pattern -- and the
    # standing wave would keep walking. Sweep off first, then base, so the
    # lattice is on but stationary, which is what park_outputs() assumes.
    for port in DIM_PORTS:
        ser = sers[port]
        if ser is None:
            continue
        try:
            _dim_send(ser, 'Sswpm:0')
            _dim_send(ser, f'FRQ:{base_freq_hz}')
            logger.info(f"DIM {port} sweep OFF (Sswpm:0) and parked at base "
                        f"{base_freq_hz/1e6:.3f} MHz (lattice on, stationary)")
        except Exception as e:
            logger.warn(f"Could not park DIM {port} at base frequency: {e!r}")
        try:
            ser.close()
        except Exception:
            pass
