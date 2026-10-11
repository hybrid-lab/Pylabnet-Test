# Lattice TRANSPORT ONE WAY, DIM-3000 SWEEP MODE 2, BOTH AOMs SWEEPING.
#
# fastLatticeMoveSweep2_bothBeams.py cut down to ONE leg -- accelerate, drift,
# decelerate, then HOLD at the far point -- and extended so that BOTH drivers
# sweep at once, one up in frequency and one down, which doubles the lattice
# velocity for the same excursion per AOM. MOT/CMOT/Dark MOT/PGC/lattice
# loading, imaging, camera, fits and atom number are unchanged.
#
# THE PHYSICS. Only the beat between the two beams moves the standing wave:
#
#     v = (lambda/2) * (f_up - f_down)
#
# so sweeping one beam by df, or both by df/2 in opposite directions, gives
# the SAME transport. What halves is how far either AOM is from the frequency
# it was aligned at, which is what lets the velocity cap be set twice as high.
#
# THE PROBLEM. The DIM-3000 sweep engine only ramps UP from SWPs to SWPp
# (bench-tested: a stop frequency below the start does not sweep). Sweeping
# BOTH drivers up would leave the beat unchanged and the lattice at rest.
#
# THE TRICK. Manual v9 p.9, mode 2 diagram: Sync is a level.
#     Sync HIGH   ->  ramp SWPs -> SWPp at SWPf per SWPt, then HOLD at SWPp
#     Sync LOW    ->  ramp back DOWN to SWPs at the same rate, then hold there
# The falling edge IS a downward sweep. So the "down" driver is given a sweep
# window that ENDS at base, and its gate is run INVERTED:
#
#     UP driver    SWPs = base          SWPp = base + df/2
#                  gate LOW at rest, HIGH for accel + plateau
#     DOWN driver  SWPs = base - df/2   SWPp = base
#                  gate HIGH at rest (sits at SWPp = base), LOW for the move
#
# One NI window [t0, t0 + gate) is written HIGH on the up driver's line and
# LOW on the down driver's line. At t0 the up driver ramps up and the down
# driver ramps down -- both at the same SWPf/SWPt, so the beat opens at twice
# the per-driver rate. At t0 + gate both ramp back to base and the lattice
# comes to rest, displaced by d. Both then sit at base -- one with Sync LOW at
# its SWPs, the other with Sync HIGH at its SWPp -- which is single-frequency
# operation in everything but name. That is the hold at the far point.
#
# WHY THE HOLD IS NOT "OUTSIDE SWEEP MODE". Mode changes go over serial at
# about 10 commands a second with no hardware timing, so Sswpm:0 cannot be
# placed at a defined point in a 20 kHz NI sequence, and it is not needed: at
# rest both drivers are at base and neither Sync line moves. Sweep mode is
# switched off, over serial, only when the run stops.
#
# WIRING (both drivers' rear "Sync" SMA, each on its own NI line):
#   dio6 -> COM4 Sync        dio7 -> COM5 Sync
# NOT a tee: the two gates are complements of each other during the move.
# As a safeguard the FSK frequency and amplitude of BOTH drivers are set to
# base / AMP for the run (restored on stop), so a HIGH at Sync in sweep mode
# 0 -- which jumps this firmware to its FSK frequency -- changes nothing.
#
# REST STATE OF THE DOWN DRIVER'S LINE IS HIGH. It is written HIGH at every
# sample of the sub-sequence except the move window, so there is no edge at
# the buffer boundary, and park_outputs() holds it HIGH between passes so the
# lattice stays stationary while the buffer is re-armed. At the very start of
# the run it is raised as soon as the driver is armed, so the down driver
# ramps SWPs -> base once, before any atoms exist. Both lines go LOW only
# after sweep mode has been switched off at the end of the run.
#
# COMMANDS: this firmware only acts on the S-prefix form of the sweep commands
# (Sswpm:2, Sswps:..., Sswpp:..., Sswpf:..., Sswpt:..., and Sffreq/Sfampl for
# FSK), not the manual's SWPm:/SWPs:/... -- found on the bench.
#
#     ramp time   T_r = (df/2 / SWPf) * SWPt          (per driver)
#     gate width  T_g = T_r + T_p
#     distance    d   = v * T_g,   v = (lambda/2) * df
#
# One step rate is used both ways, so accel and decel are always the same
# length: T_a and T_d are read but both ramps run for their MEAN.
#
# Full sequence:
#   MOT -> CMOT -> Dark MOT -> PGC_cool -> PGC_rampdown(+lattice loading)
#   -> pre-move hold
#   -> ACCELERATE   (both ramps,   T_r)
#   -> DRIFT        (both hold,    T_p)
#   -> DECELERATE   (both ramp back, T_r)
#   -> HOLD AT FAR POINT (both at base, sweep mode still armed, Sync idle)
#   -> release -> TOF -> image at the FAR point
#
# 'Both AOMs Sweep' = 0 falls back to the single-driver move of the
# bothBeams script: the up driver carries the whole df, the other driver is
# parked at base with sweep and segment mode OFF and its line held LOW.
#
# DIRECTION. The atoms move toward the beam with the LOWER frequency.
#   +1  COM4 up, COM5 down   ->  the "+" direction of the bothBeams script
#   -1  COM5 up, COM4 down   ->  the "-" direction
# Which of those is physically "down" is fixed by which beam comes from
# where; check it once on the camera and leave the default set to it.
#
# A second window, "RF signals", draws the sub-sequence as it is sent: both
# AOM frequencies, the lattice velocity they imply, and the dio6 / dio7 gates
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


def rf_signal_trace(port_freqs, start_s, ramp_ms, gate_ms, gates, gate_line,
                    spm, v_per_hz_mmps):
    """One sub-sequence of the RF drive, as plottable piecewise-linear lines.

    :param port_freqs: (dict) port -> (rest_hz, moved_hz). A driver that does
        not sweep has moved_hz == rest_hz and draws a flat line.
    :param start_s: (int) sample at which BOTH drivers start ramping
    :param ramp_ms, gate_ms: (float) per-driver ramp and the NI gate width
    :param gates: (dict) NI line -> list of 0/1 samples, EXACTLY as sent
    :param gate_line: (dict) port -> NI line, e.g. {'COM4': 'dio6', ...}
    :param spm: (int) NI samples per ms
    :param v_per_hz_mmps: (float) lattice velocity per Hz of beat
    :return: (dict) t_ms, f4_hz, f5_hz, v_mps (all on t_ms), and
        gate_<port> -> (t_ms, 0/1) step points for each line
    """
    any_line = next(iter(gates.values()))
    end_ms = len(any_line) / spm

    def corners(port):
        rest, moved = (float(v) for v in port_freqs[port])
        pts = [(0.0, rest)]
        if start_s is not None and moved != rest:
            t0 = start_s / spm
            pts += [(t0, rest), (t0 + ramp_ms, moved), (t0 + gate_ms, moved),
                    (t0 + gate_ms + ramp_ms, rest)]
        pts.append((end_ms, rest))
        return np.array(pts, dtype=float)

    c4, c5 = corners('COM4'), corners('COM5')
    t = np.unique(np.concatenate([c4[:, 0], c5[:, 0]]))
    f4 = np.interp(t, c4[:, 0], c4[:, 1])
    f5 = np.interp(t, c5[:, 0], c5[:, 1])
    # v = (lambda/2) * (f_COM4 - f_COM5), in m/s so pyqtgraph picks the prefix.
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

    Both AOM frequencies, the lattice velocity they imply, and the two sweep
    gates (dio6 -> COM4, dio7 -> COM5) as they are in the NI buffers. Same
    threading rule as MoveLiveView.
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
        """Called from the acquisition thread; stores data, touches no Qt."""
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
    'pgc_rampdown_ms': {'PGC Rampdown Duration (ms)': '0'},

    # ===================== LATTICE AM =====================
    # AOM thermal stability: lattice TTL stays ON & AM stays at lat_am_v
    # for the ENTIRE cycle.
    'lat_am_v': {'Lattice AM (V)': '0.6'},

    # ===================== MOVE DIRECTION AND SWEEP MODE =====================
    # +1: COM4 sweeps UP, COM5 sweeps DOWN -- the "+" direction of the
    #     bothBeams script (COM4 above COM5).
    # -1: COM5 sweeps UP, COM4 sweeps DOWN -- the "-" direction.
    # The atoms move toward the beam with the LOWER frequency. Which sign is
    # physically "down" depends on which beam comes from where: check it once
    # on the camera and set the default to it.
    'move_direction': {'Move Direction (+1: COM4 up/COM5 down, -1: COM5 up/COM4 down)': '-1'},
    # 1 = both drivers sweep, each carrying HALF the beat: the down driver
    #     uses the falling-edge ramp of sweep mode 2 (see header). Its Sync
    #     line idles HIGH.
    # 0 = single driver: the up driver carries the whole beat, the other is
    #     parked at base with sweep and segment mode OFF, its line LOW.
    'dual_sweep_enabled': {'Both AOMs Sweep? (1 = yes, 0 = up driver only)': '1'},

    # ===================== MOVE (one leg, one way) =====================
    #     v = d / (T_a/2 + T_p + T_d/2)
    # The sweep has ONE step rate for both ramps, so accel and decel both run
    # for (T_a + T_d)/2. With T_a == T_d (the defaults) nothing changes.
    # 0 distance is a stationary reference: no gate opens, sweep mode stays
    # OFF on both drivers, and the cloud is imaged where it was loaded.
    'dist_mm': {'Move Distance (mm, one way)': '256.0'},
    'accel_ms': {'Accel Time T_a (ms)': '1.0'},
    'plat_ms': {'Drift (constant velocity) Time T_p (ms)': '80.0'},
    'decel_ms': {'Decel Time T_d (ms)': '1.0'},
    # How far AHEAD of the move both gates switch, to absorb the driver's
    # Sync-input latency. The manual documents none for the sweep, so 0
    # until measured. Fractional ms allowed; snapped to the 50 us NI grid.
    'seg_lead_ms': {'Sweep Gate Lead (ms) [Sync latency]': '0'},

    # ===================== HOLDS (before + after move) =====================
    'pre_hold_ms': {'Pre-move Hold (ms)': '5'},
    # After the decel both drivers sit at base with their Sync lines idle:
    # the lattice is stationary at the far point for this long, then release.
    'post_hold_ms': {'Hold at Far Point After Move (ms)': '5'},

    # ===================== DIM-3000 AOM DRIVER =====================
    'dim_base_freq_mhz': {'DIM3000 Base Freq (MHz)': '99.800'},
    'dim_amplitude_dbm': {'DIM3000 Amplitude (dBm)': '32.0'},

    # ===================== LIT HOLD (watch this in SpinView) =====================
    # After the move the lattice is stationary at the far point and the
    # cooling and repump light come ON and STAY on, so the cloud fluoresces
    # continuously and SpinView shows it live. No camera trigger is produced
    # by this script at all.
    #
    # The light opens the moment the hold at the far point ENDS, so the
    # sequence is: lattice comes to rest -> post_hold_ms of darkness while the
    # cloud settles -> light on for exactly bottom_hold_ms. The lit time IS
    # bottom_hold_ms; nothing else adds to it.
    #
    # The light is resonant and destructive: the atoms heat and leave the
    # lattice in milliseconds to tens of milliseconds. Start long enough to
    # find the cloud, then shorten it.
    'bottom_hold_ms': {'Bottom Hold with Light ON (ms)': '500'},
    'img_mot_power': {'Imaging Cooling Power (V)': '0.8'},
    'img_repump_power': {'Imaging Repump Power (V)': '0.8'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before Dark MOT)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after the hold)': '10'},

    # ===================== BIAS COILS =====================
    'bias_x_v': {'Bias X (V)': '0.0'},
    'bias_y_v': {'Bias Y (V)': '0.0'},
    'bias_z_v': {'Bias Z (V)': '0.0'},

    # ===================== RUN =====================
    # Cycles the OPX clocks per NI arm. The cards are re-armed between passes,
    # which leaves a gap of a second or so with the lattice at base.
    'reps_per_arm': {'Cycles per Arm (then re-arm)': '200'},

    # ===================== GENERAL =====================
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},
}


# ---------------------------------------------------------------------------
# Safe state applied when the run stops.
# ---------------------------------------------------------------------------
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

# THE LATTICE IS DELIBERATELY LEFT ON so the AOM crystals stay in one thermal
# state. dio6/dio7 are the two sweep gates; their idle value is LOW, but the
# DOWN driver's line is overridden to HIGH for as long as that driver is in
# sweep mode 2 (its rest state is Sync HIGH at SWPp = base). See park_outputs.
SAFE_DO_LATTICE_ON = {
    'dio8': 0,      # bottom camera trigger idle
    'dio1': 0,      # MOT AOM enable -- light off
    'dio2': 1,      # coils TTL: the sequence's "coils off" state
    'dio3': 0,      # repump AOM enable -- light off
    'dio4': 0,      # lattice up   -- LOW = ON, left running on purpose
    'dio5': 0,      # lattice down -- LOW = ON, left running on purpose
    'dio6': 0,      # COM4 sweep gate
    'dio7': 0,      # COM5 sweep gate
}


def park_do(NI_do, safe_do, logger):
    """Drive every DO line to the given value. Best effort, never raises."""
    try:
        NI_do.write_static(do=safe_do)
        return
    except Exception as exc:
        logger.warn(f"write_static unavailable ({exc}); parking DO channel by channel")
    for channel, value in safe_do.items():
        try:
            NI_do.set_do_voltage(do_channel=channel, value=bool(value))
        except Exception as e:
            logger.warn(f"Could not park {channel} at {value}: {e}")


# The bottom camera is NOT touched by this script: no client, and dio8 is
# never driven. SpinView owns the device. dio8 is still parked low on stop so
# the line cannot float into the camera Line0 input.
CAMERA_TRIGGER_DO = None


def park_outputs(NI_do, NI_ao, logger, lattice_am=0.0, lattice_on=True,
                 do_overrides=None):
    """Drive every line to its safe value, lattice left ON and stationary.

    :param lattice_am: AM voltage the run used, held on ao6/ao7
    :param lattice_on: False parks the lattice off (ao6/ao7 = 0, TTL high)
    :param do_overrides: (dict) DO lines whose parked value differs from
        SAFE_DO_LATTICE_ON -- the down driver's gate, held HIGH while that
        driver is in sweep mode 2 so it sits at base rather than at SWPs.
    """
    safe_ao = dict(SAFE_AO_BASE)
    safe_do = dict(SAFE_DO_LATTICE_ON)
    if do_overrides:
        safe_do.update(do_overrides)
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
    except Exception as exc:
        logger.warn(f"write_static unavailable ({exc}); parking AO channel by channel")
        for channel, value in safe_ao.items():
            try:
                NI_ao.set_ao_voltage(ao_channel=channel, voltages=value)
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value} V: {e}")
    park_do(NI_do, safe_do, logger)
    logger.info(
        "Outputs parked: VCOs 0 V, coils 0 V, bias 0 V, AOM drive 0.8 V, "
        "cooling/repump light off, lattice %s, gates dio6=%d dio7=%d."
        % ((f"ON at {float(lattice_am):.2f} V (stationary)" if lattice_on
            else "OFF"), safe_do['dio6'], safe_do['dio7'])
    )


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


# FSK frequency / amplitude of each driver as the run found them, keyed by
# port; filled by _run_experiment, restored by experiment()'s finally.
FSK_FOUND = {}


def experiment(**kwargs):
    """Run the sequence, then put the RF drivers back the way they were found.

    Order on the way out matters:
      1. both gate lines LOW  -- a driver in mode 2 with Sync HIGH is sitting
         at SWPp; dropping the line ramps it back to SWPs. And a driver put
         into mode 0 while its Sync is HIGH jumps to its FSK frequency, which
         is about to be restored to whatever it was before the run.
      2. Sswpm:0 on both, FSK put back, then dim_state restores the rest.
    """
    snaps = []
    FSK_FOUND.clear()
    try:
        _run_experiment(dim_snapshots=snaps, **kwargs)
    finally:
        try:
            park_do(kwargs['nidaqmx_ni_daq_2'], dict(SAFE_DO_LATTICE_ON),
                    kwargs['dataset'].log)
        except Exception:
            pass
        time.sleep(0.3)      # longest ramp back to SWPs is a few ms; be generous
        for snap in snaps:
            if not snap:
                continue
            ser = snap['_ser']
            try:
                if not ser.is_open:
                    ser.open()
                ser.write(b'Sswpm:0\n')
                time.sleep(0.12)
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
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    CMOT_RAMP_MS = int(dataset.get_input_parameter("mot_switch_time"))
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_v = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

    dark_enabled = int(dataset.get_input_parameter("dark_mot_enabled")) != 0
    dark_ms = int(dataset.get_input_parameter("dark_mot_duration"))
    dark_power = float(dataset.get_input_parameter("dark_mot_power"))
    dark_detuning = float(dataset.get_input_parameter("dark_mot_detuning"))
    dark_rep_power = float(dataset.get_input_parameter("dark_mot_repump_power"))
    dark_rep_vco = float(dataset.get_input_parameter("dark_mot_repump_vco"))

    pgc_cool_ms = int(dataset.get_input_parameter("pgc_cool_ms"))
    pgc_rampdown_ms = int(dataset.get_input_parameter("pgc_rampdown_ms"))
    pgc_mot_power = float(dataset.get_input_parameter("pgc_mot_power"))
    pgc_detuning = float(dataset.get_input_parameter("pgc_detuning"))
    pgc_rep_power = float(dataset.get_input_parameter("pgc_repump_power"))
    pgc_rep_vco = float(dataset.get_input_parameter("pgc_repump_vco"))
    lat_am_v = float(dataset.get_input_parameter("lat_am_v"))

    dist_mm_in = float(dataset.get_input_parameter("dist_mm"))
    pre_hold_ms = int(dataset.get_input_parameter("pre_hold_ms"))
    post_hold_ms = int(dataset.get_input_parameter("post_hold_ms"))
    accel_ms = float(dataset.get_input_parameter("accel_ms"))
    plat_ms_in = float(dataset.get_input_parameter("plat_ms"))
    decel_ms = float(dataset.get_input_parameter("decel_ms"))
    seg_lead_ms = float(dataset.get_input_parameter("seg_lead_ms"))
    if min(accel_ms, plat_ms_in, decel_ms) < 0:
        raise ValueError(
            f"Move durations must be >= 0 (got T_a={accel_ms}, "
            f"T_p={plat_ms_in}, T_d={decel_ms} ms).")
    if seg_lead_ms < 0:
        raise ValueError(f"Sweep Gate Lead must be >= 0, got {seg_lead_ms} ms.")
    if dist_mm_in < 0:
        raise ValueError(
            "Move Distance must be >= 0; the direction is set by Move "
            "Direction, not by the sign of the distance.")
    dim_base_freq_mhz = float(dataset.get_input_parameter("dim_base_freq_mhz"))
    dim_amplitude_dbm = float(dataset.get_input_parameter("dim_amplitude_dbm"))
    move_direction = int(float(dataset.get_input_parameter("move_direction")))
    if move_direction not in (1, -1):
        raise ValueError(f"Move Direction must be +1 or -1, got {move_direction}.")
    dual_sweep = int(float(dataset.get_input_parameter("dual_sweep_enabled"))) != 0

    # The lit hold: what SpinView watches.
    bottom_hold_ms = float(dataset.get_input_parameter("bottom_hold_ms"))
    if bottom_hold_ms <= 0:
        raise ValueError(
            f"Bottom Hold must be > 0 ms, got {bottom_hold_ms}: with no hold the "
            f"light never comes on and there is nothing to see in SpinView.")
    img_mot_power = float(dataset.get_input_parameter("img_mot_power"))
    img_repump_power = float(dataset.get_input_parameter("img_repump_power"))
    reps_per_arm = max(1, int(float(dataset.get_input_parameter("reps_per_arm"))))

    # Which driver goes UP and which goes DOWN. v = (lambda/2)*(f4 - f5):
    # +1 puts COM4 above COM5.
    UP_PORT = 'COM4' if move_direction > 0 else 'COM5'
    DN_PORT = 'COM5' if move_direction > 0 else 'COM4'
    DIM_PORTS = ('COM4', 'COM5')
    GATE_LINE = {'COM4': 'dio6', 'COM5': 'dio7'}
    mode_desc = (f"one way, {'+' if move_direction > 0 else '-'}: "
                 f"{UP_PORT} up" + (f", {DN_PORT} down" if dual_sweep else ""))
    sweep_desc = ("both AOMs (down driver on the falling-edge ramp)"
                  if dual_sweep else f"{UP_PORT} only, {DN_PORT} parked at base")

    # ============ Move physics ============
    LATTICE_WAVELENGTH = 795e-9
    v_per_Hz_mmps = LATTICE_WAVELENGTH / 2 * 1e3  # mm/s per Hz of BEAT
    base_freq_hz = int(round(dim_base_freq_mhz * 1e6))

    ni_rate = 20000
    SPM = ni_rate // 1000
    SAMPLE_MS = 1000.0 / ni_rate

    def ceil_samples(t_ms):
        """ms -> whole NI samples, rounding UP (1e-9 absorbs float fuzz)."""
        return int(np.ceil(t_ms * SPM - 1e-9))

    # DIM-3000 sweep limits (manual p.4 and p.13). SWPt is in ns, 4..262000,
    # rounded DOWN by the firmware to a multiple of 4.
    SWEEP_STEP_NS_QUANTUM = 4
    SWEEP_STEP_NS_MAX = 262000
    SWEEP_STEP_NS_MIN = 1000
    SWEEP_FSTEP_SEARCH = 20000

    def sweep_steps(span_hz, ramp_ms):
        """SWPf and SWPt for a ramp covering span_hz in (at most) ramp_ms.

        Candidates are tried from the FINEST step upward, and the first one
        within 2e-4 of the span and 2e-3 of the ramp time is taken. SWPt is
        always rounded DOWN, so the ramp is never longer than asked for and
        therefore always fits inside the gate.

        :return: (dict) fstep_hz, tstep_ns, n_steps, span_hz (as played),
            ramp_ms (as played), clipped
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

    def schedule_for_distance(d_mm):
        """Everything both sweeps need for one one-way move of d_mm.

        v = d / (T_a/2 + T_p + T_d/2) as before. The gate is T_r + T_p rounded
        UP to the NI grid; d = v * gate, so the BEAT is recomputed from the
        gate as played. With both drivers sweeping each carries half the beat
        with IDENTICAL SWPf/SWPt, so the two ramps are the same length by
        construction and the beat ramps at twice the per-driver rate.

        :return: (dict) per-driver windows f_up_start/stop, f_dn_start/stop,
            the shared fstep/tstep, gate_s for the NI card, and every derived
            quantity the panel and log print.
        """
        stationary = {'d_mm': float(d_mm), 'd_real_mm': 0.0, 'v_mmps': 0.0,
                      'ramp_req_ms': 0.0, 'ramp_ms': 0.0,
                      'plat_ms': 0.0, 'plat_real_ms': 0.0,
                      'gate_s': 0, 'gate_ms': 0.0, 'total_ms': 0.0,
                      'a_mps2': 0.0, 'df_hz': 0, 'df_drv_hz': 0,
                      'f_up_start': base_freq_hz, 'f_up_stop': base_freq_hz,
                      'f_dn_start': base_freq_hz, 'f_dn_stop': base_freq_hz,
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
        beat_target = int(round(d_mm / (gate_ms * 1e-3) / v_per_Hz_mmps))
        if beat_target < (2 if dual_sweep else 1):
            raise ValueError(
                f"{d_mm} mm in {gate_ms:.3f} ms is under the driver's 1 Hz "
                f"resolution (per driver).")
        if dual_sweep:
            # Half each, EXACTLY the same on both drivers: the sweep engine
            # is programmed once and copied, so any 1 Hz remainder is
            # dropped rather than given to one driver only.
            st = sweep_steps(beat_target // 2, ramp_req)
            df_drv = st['span_hz']
            beat = 2 * df_drv
        else:
            st = sweep_steps(beat_target, ramp_req)
            df_drv = st['span_hz']
            beat = df_drv
        v_mmps = beat * v_per_Hz_mmps
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
                'total_ms': gate_ms + ramp_ms,      # edge -> back at rest
                'a_mps2': (v_mmps * 1e-3) / (ramp_ms * 1e-3),
                'df_hz': beat,
                'df_drv_hz': df_drv,
                # UP driver: rest at SWPs = base, moved at SWPp = base + df_drv.
                'f_up_start': base_freq_hz,
                'f_up_stop': base_freq_hz + df_drv,
                # DOWN driver: rest at SWPp = base (Sync HIGH), moved at
                # SWPs = base - df_drv (Sync LOW). Equal to base when single.
                'f_dn_start': base_freq_hz - (df_drv if dual_sweep else 0),
                'f_dn_stop': base_freq_hz,
                'fstep_hz': st['fstep_hz'],
                'tstep_ns': st['tstep_ns'],
                'n_steps': st['n_steps'],
                'clipped': st['clipped']}

    sc0 = schedule_for_distance(dist_mm_in)
    do_move = sc0['d_mm'] > 0
    for f in (sc0['f_up_start'], sc0['f_up_stop'], sc0['f_dn_start'], sc0['f_dn_stop']):
        if not (10_000_000 <= f <= 400_000_000):
            raise ValueError(f"Sweep frequency {f} Hz is outside the DIM-3000's "
                             f"10-400 MHz range.")

    if do_move:
        logger.info(
            f"Move ({mode_desc}), {sweep_desc}: d={sc0['d_mm']:.3f}mm, "
            f"typed T_a={accel_ms:.3f} / T_p={plat_ms_in:.3f} / "
            f"T_d={decel_ms:.3f} ms -> played: ramp {sc0['ramp_ms']:.4f} ms "
            f"each way, drift {sc0['plat_real_ms']:.4f} ms, gate "
            f"{sc0['gate_ms']:.3f} ms, edge-to-rest {sc0['total_ms']:.4f} ms; "
            f"v={sc0['v_mmps']:.3f}mm/s, a={sc0['a_mps2']:.3f}m/s^2, "
            f"BEAT {sc0['df_hz']/1e3:.3f} kHz = {sc0['df_drv_hz']/1e3:.3f} kHz "
            f"per driver; {UP_PORT}: SWPs {sc0['f_up_start']} -> SWPp "
            f"{sc0['f_up_stop']} Hz"
            + (f"; {DN_PORT}: SWPs {sc0['f_dn_start']} -> SWPp "
               f"{sc0['f_dn_stop']} Hz (idles HIGH at SWPp = base)"
               if dual_sweep else f"; {DN_PORT} parked at base")
            + f"; SWPf={sc0['fstep_hz']} Hz, SWPt={sc0['tstep_ns']} ns, "
            f"{sc0['n_steps']} steps; realised distance {sc0['d_real_mm']:.4f}mm")
        if abs(accel_ms - decel_ms) > 1e-9:
            logger.warn(
                f"T_a ({accel_ms} ms) != T_d ({decel_ms} ms). Sweep mode 2 has "
                f"one step rate for both ramps, so BOTH run for their mean, "
                f"{sc0['ramp_req_ms']:.3f} ms. Distance, peak velocity and "
                f"move time are as typed; the accelerations are not.")
        if accel_ms + decel_ms <= 0:
            logger.warn(
                f"Accel and decel times are 0: the lattice jumps to "
                f"{sc0['v_mmps']:.1f} mm/s in a single step and stops the same "
                f"way. That is an unbounded acceleration and will shake most "
                f"atoms out.")
        if sc0['clipped']:
            logger.warn(
                f"The ramp is only {sc0['ramp_ms']:.3f} ms, not the "
                f"{sc0['ramp_req_ms']:.3f} ms asked for: {sc0['df_drv_hz']} Hz "
                f"per driver in 1 Hz steps at the longest step time (262 us) "
                f"cannot take longer. The drift is longer to match, so the "
                f"distance is unaffected.")
    else:
        logger.info("Move distance 0: STATIONARY reference -- no gate opens, "
                    "sweep mode OFF on both drivers, cloud imaged where it "
                    "was loaded.")

    # =========================================================================
    # CONFIGURE DIM-3000 DRIVERS (called once per experiment start)
    # =========================================================================
    swept_ports = set()
    if do_move:
        swept_ports.add(UP_PORT)
        if dual_sweep:
            swept_ports.add(DN_PORT)
    # The down driver's rest state is Sync HIGH, for as long as it is armed.
    gate_rest = {GATE_LINE[DN_PORT]: 1} if (do_move and dual_sweep) else {}

    DIM_MAX_SEGMENTS = 7          # only to neutralise other scripts' tables
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
        frequency and amplitude (seen on the bench). With FSK = base such a
        jump goes nowhere. The values found are put back by experiment().
        """
        found = _dim_gpar(ser)
        FSK_FOUND[port] = {k: found[k] for k in ('Rffreq', 'Rfampl') if k in found}
        _dim_send(ser, f'Sffreq:{base_freq_hz}')
        _dim_send(ser, f'Sfampl:{amplitude_x10}')

    def _dim_make_deaf(ser, port):
        """Park a driver at base in plain single-frequency operation."""
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

    def _dim_program_sweep(ser, port, f_start, f_stop):
        """Load one sweep window into this driver and arm mode 2.

        Bench-tested sequence (S-prefix forms):
            Mseg:0  Sswpm:0  FRQ  AMP  OUT_on  Sswps  Sswpp  Sswpf  Sswpt  Sswpm:2
        SWPf/SWPt are the same on both drivers, so the two ramps are the
        same length whichever way each one goes.

        :return: (list of str) mismatches found on Gpar readback; empty = OK
        """
        _dim_neutralise_segments(ser)
        _dim_fsk_safe(ser, port)
        _dim_send(ser, 'Mseg:0')
        _dim_send(ser, 'Sswpm:0')
        _dim_send(ser, f'FRQ:{base_freq_hz}')
        _dim_send(ser, f'AMP:{amplitude_x10}')
        _dim_send(ser, 'OUT_on')
        _dim_send(ser, f"Sswps:{f_start}")
        _dim_send(ser, f"Sswpp:{f_stop}")
        _dim_send(ser, f"Sswpf:{sc0['fstep_hz']}")
        _dim_send(ser, f"Sswpt:{sc0['tstep_ns']}")
        _dim_send(ser, 'Sswpm:2')
        time.sleep(0.1)
        got = _dim_gpar(ser)
        logger.info(
            f"DIM-3000 {port} after arming: FRQ={got.get('Rfreq')} "
            f"SWPm={got.get('Rswpm')} SWPs={got.get('Rswps')} "
            f"SWPp={got.get('Rswpp')} SWPf={got.get('Rswpf')} "
            f"SWPt={got.get('Rswpt')} FFRQ={got.get('Rffreq')} "
            f"FAMP={got.get('Rfampl')}")
        want = {'Rswpm': 2, 'Rswps': f_start, 'Rswpp': f_stop,
                'Rswpf': sc0['fstep_hz'], 'Rswpt': sc0['tstep_ns'],
                'Rffreq': base_freq_hz, 'Rfampl': amplitude_x10}
        return [f"{k} is {got.get(k, '<missing>')} (sent {v})"
                for k, v in want.items() if got.get(k) != v]

    # Both gate lines LOW before any driver is touched: a driver that is
    # still in mode 0 from a previous run must not see a HIGH (FSK jump), and
    # the down driver must enter mode 2 sitting at SWPs before its line is
    # raised, so that its one ramp up to base happens now, with no atoms.
    park_do(NI_card_2, dict(SAFE_DO_LATTICE_ON), logger)

    sers = {port: _dim_open(port) for port in DIM_PORTS}
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
            if port == UP_PORT:
                f_start, f_stop = sc0['f_up_start'], sc0['f_up_stop']
                rest = f"LOW at SWPs = base; HIGH ramps to SWPp"
            else:
                f_start, f_stop = sc0['f_dn_start'], sc0['f_dn_stop']
                rest = f"HIGH at SWPp = base; LOW ramps down to SWPs"
            bad = _dim_program_sweep(ser, port, f_start, f_stop)
            logger.info(
                f"DIM-3000 {port}: mode 2 armed on {GATE_LINE[port]} -- SWPs "
                f"{f_start} Hz, SWPp {f_stop} Hz, SWPf {sc0['fstep_hz']} Hz, "
                f"SWPt {sc0['tstep_ns']} ns. Sync {rest}.")
            if bad:
                logger.error(
                    f"DIM-3000 {port} did not take the sweep settings: "
                    f"{'; '.join(bad)}. The move will not be the one on the "
                    f"panel. Do not trust this run's images.")
            else:
                logger.info(f"DIM-3000 {port}: sweep settings verified by "
                            f"Gpar readback.")
        if gate_rest:
            # Raise the down driver's line NOW: it ramps SWPs -> base once,
            # over ramp_ms, long before the first atoms are loaded, and then
            # sits at base until the first move window pulls the line LOW.
            park_do(NI_card_2, {**SAFE_DO_LATTICE_ON, **gate_rest}, logger)
            time.sleep(0.05 + sc0['ramp_ms'] * 1e-3)
            logger.info(
                f"{GATE_LINE[DN_PORT]} raised: {DN_PORT} has ramped up to base "
                f"and idles there with Sync HIGH. It goes LOW only for the move "
                f"window, and is held HIGH by park_outputs() between passes.")
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
    SPU = ni_rate / 1e6
    delay_ns = int(round(1e9 / ni_rate)) - 500
    ttl_up = max(1, SPM // 20)
    def ms2s(t): return int(round(t * SPM))
    def us2s(t): return int(round(t * SPU))

    mot_s = ms2s(mot_loading_ms)
    cmot_ramp_s = ms2s(CMOT_RAMP_MS)
    cmot_hold_s = ms2s(cmot_hold_ms)
    dark_s = ms2s(dark_ms) if dark_enabled else 0
    pgc_cool_s = ms2s(pgc_cool_ms)
    pgc_rampdown_s = ms2s(pgc_rampdown_ms)
    pgc_s = pgc_cool_s + pgc_rampdown_s
    pre_hold_s = ms2s(pre_hold_ms)
    post_hold_s = ms2s(post_hold_ms)
    seg_lead_s = ms2s(seg_lead_ms)
    bottom_hold_s = ms2s(bottom_hold_ms)
    vco_lead_s = us2s(vco_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)

    # =========================================================================
    # TIMELINE ANCHORS
    # =========================================================================
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    dark_start = cmot_end
    dark_end = dark_start + dark_s
    pgc_start = dark_end
    pgc_cool_end = pgc_start + pgc_cool_s
    pgc_end = pgc_cool_end + pgc_rampdown_s

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
        f"pre_hold={pre_hold_ms}ms, move {mode_desc}: {sc0['total_ms']:.3f}ms "
        f"(gate {sc0['gate_ms']:.3f}ms + decel {sc0['ramp_ms']:.3f}ms), "
        f"hold at far point={post_hold_ms}ms (DARK), then light ON for the "
        f"{bottom_hold_ms:g}ms bottom hold"
    )
    if seg_lead_ms >= pre_hold_ms:
        logger.warn(
            f"Gate Lead ({seg_lead_ms}ms) is not shorter than the pre-move "
            f"hold ({pre_hold_ms}ms), so the gates switch during PGC "
            f"ramp-down instead of in the hold. Raise Pre-move Hold above "
            f"{seg_lead_ms}ms.")

    def leg_layout():
        """The one move window in samples -- ONE definition, used by both the
        waveform and the RF signals window.

        :return: (start, gate_s, move_end). start is the sample at which both
            drivers begin ramping (the gates switch seg_lead_s earlier);
            move_end is when both are back at base. start is None for a
            0 mm reference.
        """
        t = pgc_end + pre_hold_s
        if not do_move:
            return None, 0, t
        return t, sc0['gate_s'], t + sc0['gate_s'] + ceil_samples(sc0['ramp_ms'])

    # =========================================================================
    # WAVEFORM CONSTRUCTION
    # =========================================================================
    def build_sub_sequence():
        start, gate_s, move_end = leg_layout()
        post_hold_end = move_end + post_hold_s
        # The light opens the moment the hold at the far point ENDS. The whole
        # move and the hold after it are dark, so the cloud has post_hold_ms to
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

        for s in range(cmot_start):
            ado[s] = 1
            aao[s] = mot_aom_power
            rdo[s] = 1
            rao[s] = repump_aom_power
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            ado[t] = 1
            aao[t] = mot_aom_power + frac * (cmot_mot_power - mot_aom_power)
            rdo[t] = 1
            rao[t] = repump_aom_power + frac * (cmot_repump_power - repump_aom_power)
        for s in range(cmot_hold_s):
            t = cmot_hold_start + s
            ado[t] = 1
            aao[t] = cmot_mot_power
            rdo[t] = 1
            rao[t] = cmot_repump_power
        if dark_enabled:
            for s in range(dark_s):
                t = dark_start + s
                ado[t] = 1
                aao[t] = dark_power
                rdo[t] = 1
                rao[t] = dark_rep_power
        for s in range(pgc_cool_s):
            t = pgc_start + s
            ado[t] = 1
            aao[t] = pgc_mot_power
            rdo[t] = 1
            rao[t] = pgc_rep_power
        for s in range(pgc_rampdown_s):
            t = pgc_cool_end + s
            frac = s / max(pgc_rampdown_s - 1, 1)
            ado[t] = 1
            aao[t] = pgc_mot_power * (1.0 - frac)
            rdo[t] = 1
            rao[t] = pgc_rep_power * (1.0 - frac)
        # Lattice hold and the whole move: TTLs off (no cooling light), analog
        # stays at full for AOM thermal stability.
        for s in range(pgc_end, min(lit_start, N)):
            ado[s] = 0
            rdo[s] = 0
        # THE LIT WINDOW. Cooling and repump ON continuously at the imaging
        # drive, from the end of the hold at the far point until the bottom
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

        cdo = [0] * N
        for s in range(coils_off, min(crt, N)):
            cdo[s] = 1

        cao = [mot_coils_v] * N
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            cao[t] = mot_coils_v + frac * (cmot_coils_v - mot_coils_v)
        for s in range(cmot_hold_start, N):
            cao[s] = cmot_coils_v

        mvco = [vco_loading] * N
        for s in range(cmot_ramp_s):
            t = cmot_start + s
            frac = s / max(cmot_ramp_s - 1, 1)
            mvco[t] = vco_loading + frac * (cmot_detuning - vco_loading)
        for s in range(cmot_hold_start, cmot_end):
            mvco[s] = cmot_detuning
        if dark_enabled:
            for s in range(dark_s):
                mvco[dark_start + s] = dark_detuning
        for s in range(pgc_s):
            mvco[pgc_start + s] = pgc_detuning
        for s in range(pgc_end, N):
            mvco[s] = pgc_detuning
        # Imaging VCO override: on for the whole lit window, switched
        # vco_lead_us early so the light is already at the imaging detuning.
        vlo = max(0, lit_start - vco_lead_s)
        vhi = min(N, lit_end)
        for s in range(vlo, vhi):
            mvco[s] = vco_imaging

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

        biasx = [bias_x_v] * N
        biasy = [bias_y_v] * N
        biasz = [bias_z_v] * N

        ludo = [0] * N
        lddo = [0] * N
        luao = [lat_am_v] * N
        ldao = [lat_am_v] * N

        # ---- DIM-3000 sweep gates ----
        # UP driver's line:   LOW everywhere, HIGH for [t0, t0 + gate).
        # DOWN driver's line: HIGH everywhere, LOW for the SAME window (dual)
        #                     or LOW everywhere (single: driver is deaf).
        # The lead moves both edges of both lines equally, so the gate width,
        # which sets the distance, is untouched by it. The down line is HIGH
        # at sample 0 and at sample N-1, so the buffer boundary has no edge.
        gates = {GATE_LINE[port]: [0] * N for port in DIM_PORTS}
        if do_move and dual_sweep:
            gates[GATE_LINE[DN_PORT]] = [1] * N
        if start is not None:
            t0 = max(0, start - seg_lead_s)
            up_line = gates[GATE_LINE[UP_PORT]]
            dn_line = gates[GATE_LINE[DN_PORT]]
            for s in range(gate_s):
                if t0 + s < N:
                    up_line[t0 + s] = 1
                    if dual_sweep:
                        dn_line[t0 + s] = 0

        return (ado, aao, rdo, rao, cdo, cao, mvco, rvco, biasx, biasy, biasz,
                ludo, lddo, luao, ldao, gates, N)

    # ONE sub-sequence per buffer. There are no camera frames to keep in step
    # with, so there is nothing for warm-up cycles to protect: the buffer
    # regenerates and every cycle is identical.
    (ado_p, aao_p, rdo_p, rao_p, cdo_p, cao_p, mvco_p, rvco_p,
     biasx_p, biasy_p, biasz_p, ludo_p, lddo_p, luao_p, ldao_p,
     gates_p, N) = build_sub_sequence()
    gates_w = gates_p

    total = N
    start0, gate_s0, move_end0 = leg_layout()
    net_sign = move_direction if do_move else 0
    lit_start0 = move_end0 + post_hold_s
    lit_end0 = lit_start0 + bottom_hold_s
    logger.info(
        f"Lattice MOVE ONE WAY ({mode_desc}), {sweep_desc}, SpinView version: "
        f"d={dist_mm_in}mm, T_a/T_p/T_d = {accel_ms:.2f}/{plat_ms_in:.2f}/"
        f"{decel_ms:.2f}ms, v={sc0['v_mmps']:.2f}mm/s, atoms lit at the FAR "
        f"point from {lit_start0 / SPM:.1f}ms ({post_hold_ms:g}ms of dark hold "
        f"after the move) for {(lit_end0 - lit_start0) / SPM:.1f}ms "
        f"(cooling {img_mot_power:.2f}V, repump {img_repump_power:.2f}V, "
        f"VCO {vco_imaging:.2f}V), cycle {N / SPM:.1f}ms, {total} samples")
    logger.info(
        "No camera trigger is produced by this script -- open the camera in "
        "SpinView. The pylabnet camera server must be stopped, or it and "
        "SpinView will fight over the device.")

    move_start0 = pgc_end + pre_hold_s
    # RF signals window: the only display here. The gates are the real dio6 /
    # dio7 buffers, so the picture cannot drift from what is played.
    try:
        port_freqs = {
            UP_PORT: (sc0['f_up_start'], sc0['f_up_stop']),
            DN_PORT: (sc0['f_dn_stop'], sc0['f_dn_start']),   # rest, moved
        }
        rf_trace = rf_signal_trace(port_freqs, start0, sc0['ramp_ms'],
                                   sc0['gate_ms'], gates_w, GATE_LINE, SPM,
                                   v_per_Hz_mmps)
        rf_markers = [(pgc_end / SPM, 'PGC end')]
        if start0 is not None:
            rf_markers.append((start0 / SPM, 'move'))
        rf_markers += [(lit_start0 / SPM, 'light ON'),
                       (move_end0 / SPM, 'at rest (far point)'),
                       (lit_end0 / SPM, 'light off')]
        rf_html = (
            f"<span style='font-family:monospace'>"
            f"<b>{mode_desc}</b> | {sweep_desc}<br>"
            f"{UP_PORT}: {sc0['f_up_start']/1e6:.6f} -> {sc0['f_up_stop']/1e6:.6f} MHz | "
            + (f"{DN_PORT}: {sc0['f_dn_stop']/1e6:.6f} &larr; {sc0['f_dn_start']/1e6:.6f} MHz"
               if dual_sweep else f"{DN_PORT}: parked at base") +
            f" | {sc0['fstep_hz']} Hz x {sc0['tstep_ns']} ns<br>"
            f"ramp {sc0['ramp_ms']:.3f} + drift {sc0['plat_real_ms']:.3f} + ramp "
            f"{sc0['ramp_ms']:.3f} ms | gate {sc0['gate_ms']:.3f} ms | beat "
            f"{sc0['df_hz']/1e3:.3f} kHz | peak v {sc0['v_mmps']:.2f} mm/s | "
            f"a {sc0['a_mps2']:.2f} m/s^2<br>"
            f"<b>light ON {(lit_end0 - lit_start0) / SPM:.1f} ms</b> from "
            f"{lit_start0 / SPM:.1f} ms, after a {post_hold_ms:g} ms dark hold at "
            f"the far point (cooling {img_mot_power:.2f} V, repump {img_repump_power:.2f} V, "
            f"VCO {vco_imaging:.2f} V) | cycle {N / SPM:.1f} ms | no camera "
            f"trigger &mdash; SpinView owns the camera</span>")
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
        # Sweep gates: dio6 -> COM4 Sync, dio7 -> COM5 Sync. No camera line.
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
        # Sequence tasks are closed now, so static writes can land. The gate
        # idle states matter here: the DOWN driver rests with its Sync HIGH.
        park_outputs(NI_card_2, NI_card_3, logger, lattice_am=lat_am_v,
                     do_overrides=gate_rest)
        logger.info(f"Arm done: {cycles_done} cycles so far.")
        time.sleep(wait_time)

    # After the while loop ends (thread stopped). Order:
    #   1. both gate lines LOW  (down driver ramps SWPp -> SWPs, a few ms)
    #   2. Sswpm:0 + FRQ base on both, so nothing at Sync moves them again
    #   3. close the ports; experiment() then restores the snapshots.
    # 1 before 2 because a driver put into mode 0 with Sync HIGH jumps to its
    # FSK frequency, and FSK is about to be restored to whatever it was.
    park_do(NI_card_2, dict(SAFE_DO_LATTICE_ON), logger)
    time.sleep(0.05 + sc0['ramp_ms'] * 1e-3)
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
