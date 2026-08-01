"""
MOT fluorescence imaging script (release-and-image with TOF sweep).

For each cycle, a sweep of time-of-flight (TOF) values is run.
The TOF sweep is currently HARDCODED to [50, 100, 150, 200] us (edit
`tof_values_us` inside `experiment()` and `tof_values` inside `configure()`
to change it).

Each TOF point produces one signal/background frame pair concatenated
back-to-back into a single NI waveform. The OPX repeats the concatenated
waveform `number_of_runs` times for averaging.

Sub-sequence (per TOF point), parameters arranged in time order from t=0:

    MOT_AOM_start ... MOT_AOM_end       MOT AOM long pulse (loading)
    mot_loading_time                    MOT coils DO switches 0 -> 1 (off)
    + tof                               Frame 1 trigger (signal)
    + atoms_clear_time                  Frame 2 trigger (background)

At each camera frame the MOT AOM also pulses on briefly for
aom_imaging_pulse_ms (derived from the camera exposure). The two matched
pulses ensure stray-light contributions cancel in (Frame 1 - Frame 2).

The MOT coils analog setpoint stays at on-voltage for the entire cycle;
only the digital gate switches.

Plots: two rows of N_tof difference images.
    Top row    — most recent run for each TOF point.
    Bottom row — running average across runs for each TOF point.

Hardware overview:
    - BFS-U3-51S5M camera (hardware triggered, "FrameStart" on a rising edge)
    - NI DAQ card 1: clock source / timing master
    - NI DAQ card 2: digital outputs (camera trigger, MOT AOM gate, MOT coils gate)
    - NI DAQ card 3: analog outputs (MOT AOM amplitude, MOT coils current setpoint)
    - OPX: generates the 1 kHz sample clock as a 500 ns TTL train fanned out to the NI cards
"""

import numpy as np
from pylabnet.scripts.data_center.datasets import Plot2DWithAvg, Plot2D  # noqa: F401
import time
from qt_plotting import QtMatplotlibFrameViewer
import pyqtgraph as pg


# -----------------------------------------------------------------------------
# 1D Gaussian width estimator (pure numpy — no scipy, no threading hazards)
# -----------------------------------------------------------------------------


def _sigma_from_moment(profile):
    """Pure-numpy moment-based estimate of the Gaussian width of a 1D profile.
    Returns sigma in samples, or NaN if the profile has no usable signal.

    Steps:
      1. Subtract the median as a baseline offset
      2. Clip negative values (so a bright cloud on dark background dominates)
      3. Compute the first moment (centroid) and second central moment (variance)
      4. sigma = sqrt(variance)
    """
    n = profile.size
    if n < 5:
        return float('nan')
    y = profile.astype(np.float64, copy=True)
    y -= np.median(y)
    np.clip(y, 0, None, out=y)
    w_sum = y.sum()
    if w_sum <= 0:
        return float('nan')
    x = np.arange(n, dtype=np.float64)
    mu = (y * x).sum() / w_sum
    var = (y * (x - mu) ** 2).sum() / w_sum
    if var <= 0 or not np.isfinite(var):
        return float('nan')
    return float(np.sqrt(var))


# -----------------------------------------------------------------------------
# Standalone TOF grid window
# -----------------------------------------------------------------------------
# pylabnet's plot panel uses a vertical layout, which forces every Plot2D
# to stack on top of the previous one. To get a real 2 x N grid of square
# tiles we open our own pyqtgraph window — completely independent of the
# pylabnet plot panel — and write the per-TOF images into it directly.
# -----------------------------------------------------------------------------
class TofGridWindow:
    """A standalone pyqtgraph window with a 2 x n_tof grid of ImageItems.
    Top row = current-run diff, bottom row = running-average diff. Sigma_x
    and sigma_y are computed but not displayed on screen (Qt TextItem
    cross-thread writes crash the process) — they're returned by
    set_average() and logged by the experiment loop."""

    def __init__(self, tof_values, diff_levels, avg_levels,
                 title="TOF grid", tile_px=320):
        self.tof_values = list(tof_values)
        self.n_tof = len(self.tof_values)
        self.diff_levels = diff_levels
        self.avg_levels = avg_levels

        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(tile_px * self.n_tof, tile_px * 2 + 60)

        # Build the grid: column headers in row 0, current images in row 1,
        # average images in row 2.
        self.current_imgs = []
        self.avg_imgs = []
        for col, tof in enumerate(self.tof_values):
            self.win.addLabel(f"<b>tof = {tof} us</b>",
                              row=0, col=col, size="11pt")

        for col in range(self.n_tof):
            vb = self.win.addViewBox(row=1, col=col, lockAspect=True)
            vb.invertY(True)
            img = pg.ImageItem(axisOrder='row-major')
            img.setLevels(self.diff_levels)
            vb.addItem(img)
            self.current_imgs.append(img)

        for col in range(self.n_tof):
            vb = self.win.addViewBox(row=2, col=col, lockAspect=True)
            vb.invertY(True)
            img = pg.ImageItem(axisOrder='row-major')
            img.setLevels(self.avg_levels)
            vb.addItem(img)
            self.avg_imgs.append(img)

        # Row labels on the far left
        self.win.addLabel("current", row=1, col=-1, angle=-90, size="10pt")
        self.win.addLabel("average", row=2, col=-1, angle=-90, size="10pt")

    def set_current(self, tof_idx, diff_image):
        try:
            self.current_imgs[tof_idx].setImage(
                diff_image, levels=self.diff_levels, autoLevels=False
            )
        except Exception:
            pass

    def set_average(self, tof_idx, avg_image):
        """Update the average tile and compute sigma_x, sigma_y by the
        moment method. Returns (sigma_x, sigma_y) in pixels — printed to
        the log by the caller. We do NOT touch any Qt text widgets from
        this method (which runs in the experiment thread); writing to a
        Qt TextItem from a non-GUI thread crashes the process."""
        try:
            self.avg_imgs[tof_idx].setImage(
                avg_image, levels=self.avg_levels, autoLevels=False
            )
        except Exception:
            pass

        sigma_x = float('nan')
        sigma_y = float('nan')
        try:
            # Copy first so pyqtgraph can't mutate the buffer under us.
            img = np.asarray(avg_image, dtype=np.float64).copy()
            x_proj = img.sum(axis=0)
            y_proj = img.sum(axis=1)
            sigma_x = _sigma_from_moment(x_proj)
            sigma_y = _sigma_from_moment(y_proj)
        except Exception:
            pass

        return sigma_x, sigma_y


# -----------------------------------------------------------------------------
# NumPy compatibility shims
# -----------------------------------------------------------------------------
# Some upstream pylabnet/qt_plotting code still uses np.int / np.bool / np.float,
# which were removed in NumPy 1.20+. Re-alias them to the built-in types so the
# legacy code keeps working without modification.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Display levels (counts / counts-difference)
# -----------------------------------------------------------------------------
# Min/max contrast limits for each on-screen plot. Adjust these to match the
# brightness of your MOT image without re-running the experiment.
FRAME1_LEVELS = (0, 20)       # Raw frame 1 (signal)
FRAME2_LEVELS = (0, 20)       # Raw frame 2 (background)
DIFF_LEVELS = (0, 20)         # Single-shot difference image
AVG_DIFF_LEVELS = (0, 20)     # Running-average difference image


# -----------------------------------------------------------------------------
# Experiment parameter dictionary
# -----------------------------------------------------------------------------
# Editable parameters exposed to the GUI. All times in ms unless otherwise
# noted. To disable a scan, set its parameters to -1.
INIT_DICT = {
    # Timing parameters are arranged in the order they occur in the sequence,
    # from t=0 forward. Each derives the time of the next event:
    #     MOT_AOM_start -> MOT_AOM_end -> mot_loading_time
    #         -> tof (hardcoded sweep in the script) -> frame_1
    #         -> atoms_clear_time -> frame_2

    # --- MOT AOM gating ---
    # Long imaging pulse during the MOT loading phase. A short matched pulse
    # is added automatically at each frame trigger to illuminate the released
    # atoms (frame 1) and the background (frame 2).
    'MOT_AOM_start': {'MOT AOM Start Time (ms)': '0'},
    'MOT_AOM_end': {'MOT AOM End Time (ms)': '1000'},

    # --- MOT loading ---
    # Duration of MOT loading. The coils switch off (DO 0 -> 1) at the end
    # of this window, releasing the atoms.
    'mot_loading_time': {'MOT Loading Time (ms)': '1000'},

    # --- Release-and-image timing ---
    # TOF sweep: 4 values in microseconds, individually editable from the GUI.
    # Each value is the delay between MOT coils-off and the Frame 1 trigger.
    # atoms_clear_time: delay from Frame 1 to Frame 2 (background), long
    # enough for the released atoms to disperse out of the field of view.
    'tof_1_us': {'TOF 1 (us)': '50'},
    'tof_2_us': {'TOF 2 (us)': '5000'},
    'tof_3_us': {'TOF 3 (us)': '10000'},
    'tof_4_us': {'TOF 4 (us)': '20000'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '190'},

    # --- Camera exposure ---
    'camera_exposure_us': {'Camera Exposure Time (us)': '1000'},

    # Idle time between successive experiment cycles.
    'wait_time': {'Wait Time Between Cycles (s)': '0.3'},

    # --- MOT coil current setpoint ---
    # Held at this voltage for the whole sequence. Digital gate (hardcoded to
    # dio2) controls when the coil driver is enabled.
    'mot_coils_on_voltage': {'MOT Coils ON Analog Voltage (V)': '7.0'},

    # --- VCO frequency control ---
    # MOT laser VCO sits at the loading voltage for most of the sequence.
    # Around each imaging pulse (frame 1 and frame 2) it briefly switches to
    # the imaging voltage, starting `vco_lead_us` microseconds BEFORE the
    # MOT AOM TTL rises and returning when the AOM falls. Repump VCO is held
    # at a single constant voltage.
    # (NI channels: MOT VCO on ao3, Repump VCO on ao4 — hardcoded in script.)
    'MOT_freq_VCO_loading_voltage': {'MOT VCO Loading Voltage (V)': '0.4'},
    'MOT_freq_VCO_imaging_voltage': {'MOT VCO Imaging Voltage (V)': '0.55'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '10'},
    'Repump_freq_VCO_voltage': {'Repump Freq VCO Voltage (V)': '0.0'},

}


def define_dataset():
    """Specifies the type of plot to use for the data."""
    return 'Dataset'


# =============================================================================
# CONFIGURE: hardware handles and plot windows (runs once at script load)
# =============================================================================
def configure(**kwargs):
    """Sets up the hardware and the plot before the experiment runs."""
    dataset = kwargs['dataset']

    # -------------------------------------------------------------------------
    # Auxiliary live viewer (separate window) for the most recent raw frame.
    # -------------------------------------------------------------------------
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")

    # -------------------------------------------------------------------------
    # Grab references to hardware clients from the pylabnet container so the
    # `experiment` function can reuse them without re-resolving each cycle.
    # -------------------------------------------------------------------------
    camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.camera_client = camera_client

    NI_card_1 = kwargs['nidaqmx_ni_daq_1']   # Clock master
    dataset.NI_card_1 = NI_card_1
    NI_card_2 = kwargs['nidaqmx_ni_daq_2']   # Digital outputs
    dataset.NI_card_2 = NI_card_2
    NI_card_3 = kwargs['nidaqmx_ni_daq_3']   # Analog outputs
    dataset.NI_card_3 = NI_card_3

    # -------------------------------------------------------------------------
    # Open a standalone pyqtgraph window with a real 2 x n_tof grid of square
    # tiles. We do NOT use pylabnet's plot panel for the TOF images — its
    # vertical layout would force every plot to be a stretched horizontal
    # strip. The grid window is independent of pylabnet's GUI.
    # -------------------------------------------------------------------------
    tof_values = [
        int(dataset.get_input_parameter("tof_1_us")),
        int(dataset.get_input_parameter("tof_2_us")),
        int(dataset.get_input_parameter("tof_3_us")),
        int(dataset.get_input_parameter("tof_4_us")),
    ]
    dataset.tof_values = tof_values

    dataset.tof_grid = TofGridWindow(
        tof_values=tof_values,
        diff_levels=DIFF_LEVELS,
        avg_levels=AVG_DIFF_LEVELS,
        title="TOF sweep — current (top) / average + Gaussian widths (bottom)",
    )

    # Hide the parent dataset graph; the TOF grid window is separate.
    dataset.graph.hide()


# =============================================================================
# EXPERIMENT: runs the acquisition loop until the GUI thread is stopped
# =============================================================================
def experiment(**kwargs):
    """Run one experiment cycle and exit."""
    dataset = kwargs['dataset']
    thread = kwargs['thread']

    logger = dataset.log

    # -------------------------------------------------------------------------
    # Re-resolve hardware handles each time `experiment` is invoked. This is
    # defensive — the GUI may reload clients between runs.
    # -------------------------------------------------------------------------
    dataset.NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    dataset.NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    dataset.NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
    dataset.camera_client = kwargs["fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber"]
    dataset.OPX_client = kwargs["OPX_OPX"]

    OPX_client = dataset.OPX_client
    NI_card_1 = dataset.NI_card_1
    NI_card_2 = dataset.NI_card_2
    NI_card_3 = dataset.NI_card_3
    camera_client = dataset.camera_client

    # -------------------------------------------------------------------------
    # Pull GUI parameters into local variables with the right Python types.
    # -------------------------------------------------------------------------
    # MOT AOM window
    MOT_AOM_start = int(dataset.get_input_parameter("MOT_AOM_start"))
    MOT_AOM_end = int(dataset.get_input_parameter("MOT_AOM_end"))

    # Camera
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))

    # -------------------------------------------------------------------------
    # Forward-flowing sequence timing.
    # The MOT loading phase is fixed; the TOF delay sweeps over a hardcoded
    # list (one sub-sequence per TOF point, all concatenated into one NI
    # waveform). Frame 1 and Frame 2 trigger times are TOF-dependent and
    # computed inside the build loop below.
    #
    # Per TOF point:
    #   t = 0                                start of sub-sequence
    #   t = MOT_AOM_start ... _end           MOT AOM long pulse
    #   t = mot_loading_time                 MOT coils switch OFF
    #   t = mot_loading_time + tof           Frame 1 (signal, after release)
    #   t = frame_1 + atoms_clear_time       Frame 2 (background, atoms gone)
    # -------------------------------------------------------------------------
    mot_loading_time = int(dataset.get_input_parameter("mot_loading_time"))
    atoms_clear_time = int(dataset.get_input_parameter("atoms_clear_time"))

    # TOF sweep — values in microseconds, read from the GUI.
    tof_values_us = [
        int(dataset.get_input_parameter("tof_1_us")),
        int(dataset.get_input_parameter("tof_2_us")),
        int(dataset.get_input_parameter("tof_3_us")),
        int(dataset.get_input_parameter("tof_4_us")),
    ]
    tof_values = tof_values_us   # alias kept for any downstream label use
    n_tof = len(tof_values_us)

    # Derived absolute times that are the same for every TOF point (in samples; set below).

    # Inter-cycle idle
    wait_time = float(dataset.get_input_parameter("wait_time"))

    # -------------------------------------------------------------------------
    # Hardcoded NI DAQ channel assignments. Change these only if the physical
    # wiring to the NI breakout boxes changes.
    # -------------------------------------------------------------------------
    # NI card 2 (digital outputs):
    camera_trigger_do = "dio0"
    MOT_AOM_do = "dio1"
    mot_coils_do = "dio2"
    # NI card 3 (analog outputs):
    MOT_AOM_ao = "ao1"
    mot_coils_ao = "ao2"
    MOT_freq_VCO_ao = "ao3"
    Repump_freq_VCO_ao = "ao4"
    # OPX digital output (the 1 kHz/20 kHz clock the NI cards slave off):
    opx_trigger_do = 1

    # Voltage setpoints (still configurable from the GUI).
    mot_coils_on_voltage = float(dataset.get_input_parameter("mot_coils_on_voltage"))
    MOT_freq_VCO_loading_voltage = float(dataset.get_input_parameter("MOT_freq_VCO_loading_voltage"))
    MOT_freq_VCO_imaging_voltage = float(dataset.get_input_parameter("MOT_freq_VCO_imaging_voltage"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    Repump_freq_VCO_voltage = float(dataset.get_input_parameter("Repump_freq_VCO_voltage"))

    # -------------------------------------------------------------------------
    # Timing constants for the OPX → NI card clocking scheme.
    # -------------------------------------------------------------------------
    # NI cards run at 20 kHz so that each waveform sample is 50 us — fine
    # enough to resolve sub-millisecond TOF values. The OPX emits one 500 ns
    # TTL pulse every 50 us; each rising edge advances the NI cards by one
    # waveform sample. All durations sourced from the GUI are in ms and are
    # converted to integer sample counts via SAMPLES_PER_MS below.
    # -------------------------------------------------------------------------
    ni_sample_rate = 20000              # Hz — NI card sample clock (20 kHz)
    SAMPLES_PER_MS = ni_sample_rate // 1000        # 20 samples per ms
    SAMPLES_PER_US = ni_sample_rate / 1_000_000.0  # 0.02 samples per us
    trigger_line = "Line0"              # Camera hardware-trigger line name
    trigger_edge = "RisingEdge"
    opx_ttl_pulse_ns = 500              # OPX pulse high-time
    sample_period_ns = int(round(1e9 / ni_sample_rate))   # 50_000 ns
    delay_ns = sample_period_ns - opx_ttl_pulse_ns        # 49_500 ns between OPX pulses
    camera_ttl_up = max(1, SAMPLES_PER_MS // 20)  # Camera trigger high-time (~50 us / 1 sample)

    # Convert GUI parameters from ms to integer NI sample counts.
    MOT_AOM_start_s = MOT_AOM_start * SAMPLES_PER_MS
    MOT_AOM_end_s = MOT_AOM_end * SAMPLES_PER_MS
    mot_loading_time_s = mot_loading_time * SAMPLES_PER_MS
    atoms_clear_time_s = atoms_clear_time * SAMPLES_PER_MS
    coils_off_time_s = mot_loading_time_s

    # TOF values converted from us to integer samples.
    def us_to_samples(t_us):
        return int(round(t_us * SAMPLES_PER_US))
    tof_samples_list = [us_to_samples(t) for t in tof_values_us]
    vco_lead_samples = us_to_samples(vco_lead_us)

    # MOT AOM pulse co-located with each frame trigger — keep it at least 1
    # sample wide (camera_ttl_up), or scale up if camera_exposure_us exceeds
    # one NI sample period.
    aom_imaging_pulse_samples = max(
        camera_ttl_up,
        int(np.ceil(camera_exposure_us * SAMPLES_PER_US))
    )

    # -------------------------------------------------------------------------
    # Parameter sanity checks — fail loudly before arming the hardware.
    # -------------------------------------------------------------------------
    if camera_exposure_us <= 0:
        raise ValueError("camera_exposure_us must be greater than 0")
    if delay_ns < 0:
        raise ValueError("ni_sample_rate is too high for a 500 ns OPX TTL pulse")
    if MOT_AOM_start < 0:
        raise ValueError("MOT_AOM_start must be non-negative")
    if MOT_AOM_end < MOT_AOM_start:
        raise ValueError("MOT_AOM_end must be greater than or equal to MOT_AOM_start")
    if mot_loading_time < 0:
        raise ValueError("mot_loading_time must be non-negative")
    if any(t < 0 for t in tof_values_us):
        raise ValueError("all tof_values_us must be non-negative")
    if atoms_clear_time_s <= camera_ttl_up:
        raise ValueError(
            f"atoms_clear_time must be greater than camera_ttl_up ({camera_ttl_up} samples) "
            "so the two camera triggers don't overlap"
        )

    logger.info(f"Time at start {time.perf_counter_ns()}")

    # =========================================================================
    # WAVEFORM CONSTRUCTION (per-TOF sub-sequences concatenated end-to-end)
    # =========================================================================
    # Strategy: build one sub-sequence per TOF value, then concatenate them
    # into a single big waveform that is written to the NI cards once. The
    # camera receives 2 triggers per TOF point (2 * n_tof frames per run);
    # the outer "runs" averaging loop replays the entire concatenated buffer.
    # Each output channel is built as a list of samples at 1 kHz (1 sample/ms).
    # -------------------------------------------------------------------------

    def build_sub_sequence(tof_samples):
        """Build (camera, aom, mot_do, mot_ao, mot_vco, length, frame1_t, frame2_t)
        for one TOF point, starting at sub-sequence sample 0. All time
        indices are in NI samples (1 sample = 50 us at 20 kHz)."""
        frame_1_local = mot_loading_time_s + tof_samples
        frame_2_local = frame_1_local + atoms_clear_time_s
        sub_end = max(frame_2_local + camera_ttl_up, MOT_AOM_end_s)

        # --- Camera trigger DO: two short rising-edge pulses per TOF point ---
        down = frame_2_local - frame_1_local - camera_ttl_up
        cam = (
            [0] * frame_1_local +
            [1] * camera_ttl_up +
            [0] * down +
            [1] * camera_ttl_up +
            [0] * max(0, sub_end - frame_2_local - camera_ttl_up)
        )

        # --- MOT AOM (DO + AO share the same waveform): long MOT-loading
        #     pulse plus two matched short pulses at frame_1 and frame_2.
        aom = [0] * sub_end
        for idx in range(MOT_AOM_start_s, min(MOT_AOM_end_s, sub_end)):
            aom[idx] = 1
        for offset in range(aom_imaging_pulse_samples):
            if frame_1_local + offset < sub_end:
                aom[frame_1_local + offset] = 1
            if frame_2_local + offset < sub_end:
                aom[frame_2_local + offset] = 1

        # --- MOT coils DO: 0 = ON during loading, 1 = OFF for the rest.
        coils_do = [0] * coils_off_time_s + [1] * (sub_end - coils_off_time_s)

        # --- MOT coils AO: held at on-voltage for the full sub-sequence.
        coils_ao = [mot_coils_on_voltage] * sub_end

        # --- MOT VCO: default to loading voltage; switch to imaging voltage
        #     starting `vco_lead_samples` BEFORE the frame_1 AOM pulse rising
        #     edge, and hold through the end of the frame_2 AOM pulse. The
        #     VCO stays at imaging voltage continuously across both frames
        #     (including the atoms_clear gap between them).
        mot_vco = [MOT_freq_VCO_loading_voltage] * sub_end
        vco_lo = max(0, frame_1_local - vco_lead_samples)
        vco_hi = min(sub_end, frame_2_local + aom_imaging_pulse_samples)
        for idx in range(vco_lo, vco_hi):
            mot_vco[idx] = MOT_freq_VCO_imaging_voltage

        return cam, aom, coils_do, coils_ao, mot_vco, sub_end, frame_1_local, frame_2_local

    # Concatenate sub-sequences into the master waveform buffers.
    camera_trigger_pulse = []
    MOT_AOM_pulse = []
    mot_coils_do_pulse = []
    mot_coils_ao_pulse = []
    MOT_freq_VCO_pulse = []
    sub_sequence_starts = []  # bookkeeping (sub-sequence start times in ms)
    sub_sequence_lengths = []

    for tof_samples in tof_samples_list:
        cam, aom, coils_do, coils_ao, mot_vco, sub_end, f1, f2 = build_sub_sequence(tof_samples)
        sub_sequence_starts.append(len(camera_trigger_pulse))
        sub_sequence_lengths.append(sub_end)
        camera_trigger_pulse += cam
        MOT_AOM_pulse += aom
        mot_coils_do_pulse += coils_do
        mot_coils_ao_pulse += coils_ao
        MOT_freq_VCO_pulse += mot_vco

    sequence_end_samples = len(camera_trigger_pulse)
    experiment_length_samples = int(sequence_end_samples)
    sequence_end_ms = sequence_end_samples / SAMPLES_PER_MS

    # Repump VCO stays constant for the whole sequence — single flat array.
    Repump_freq_VCO_pulse = [Repump_freq_VCO_voltage] * sequence_end_samples

    logger.info(
        f"Running TOF sweep: "
        f"aom_start={MOT_AOM_start} ms, aom_end={MOT_AOM_end} ms, "
        f"mot_loading={mot_loading_time} ms, "
        f"tof_values_us={tof_values_us} (n={n_tof}), "
        f"atoms_clear={atoms_clear_time} ms, "
        f"exposure={camera_exposure_us} us, "
        f"matched_aom_pulse={aom_imaging_pulse_samples} samples, "
        f"sequence_length={sequence_end_samples} samples ({sequence_end_ms:.3f} ms), "
        f"opx_delay={delay_ns} ns, "
        f"ni_rate={ni_sample_rate} Hz"
    )

    # =========================================================================
    # PERSISTENT RUNNING-AVERAGE ACCUMULATORS
    # Initialized ONCE here, outside the outer `while thread.running` loop, so
    # that subsequent outer cycles (each of N=number_of_runs rounds) keep
    # accumulating into the same averages instead of restarting from scratch.
    # =========================================================================
    diff_sums = [None] * n_tof                 # cumulative sum of diff images
    diff_counts = [0] * n_tof                  # total number of runs averaged so far

    # =========================================================================
    # MAIN ACQUISITION LOOP
    # Each iteration: configure hardware, run M sub-cycles back-to-back on the
    # OPX clock, pull frame pairs from the camera, plot, then tear down.
    # =========================================================================
    while thread.running:

        # ---------------------------------------------------------------------
        # Configure the camera for hardware-triggered continuous acquisition.
        # ---------------------------------------------------------------------
        camera_client.set_hardware_trigger(
            line=trigger_line,
            activation=trigger_edge,
            selector="FrameStart",
            overlap="ReadOut",                  # Allow exposure during prior readout
            acquisition_mode="Continuous",
        )
        camera_client.set_exposure(camera_exposure_us)
        camera_client.try_set_float("Gain", 50.0)

        logger.info("Starting acquisition")
        dataset.camera_client.start_acquisition()

        # ---------------------------------------------------------------------
        # Set up the NI master clock (card 1). The other cards slave off it.
        # ---------------------------------------------------------------------
        NI_card_1.arm_clock(length=experiment_length_samples, sample_rate=ni_sample_rate)
        logger.info("Clock configured")

        # ---------------------------------------------------------------------
        # Load digital waveforms onto NI card 2 (camera trigger, MOT AOM,
        # MOT coils gate). build_stack() clears any previous task config.
        # ---------------------------------------------------------------------
        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(
            do_channel=camera_trigger_do,
            value=camera_trigger_pulse,
            sample_rate=ni_sample_rate
        )
        NI_card_2.set_do_voltage(
            do_channel=MOT_AOM_do,
            value=MOT_AOM_pulse,
            sample_rate=ni_sample_rate
        )
        NI_card_2.set_do_voltage(
            do_channel=mot_coils_do,
            value=mot_coils_do_pulse,
            sample_rate=ni_sample_rate
        )

        # ---------------------------------------------------------------------
        # Load analog waveforms onto NI card 3 (MOT AOM amplitude, MOT
        # coil current setpoint).
        # ---------------------------------------------------------------------
        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(
            ao_channel=MOT_AOM_ao,
            voltages=MOT_AOM_pulse,
            sample_rate=ni_sample_rate
        )
        NI_card_3.set_ao_voltage(
            ao_channel=mot_coils_ao,
            voltages=mot_coils_ao_pulse,
            sample_rate=ni_sample_rate
        )
        NI_card_3.set_ao_voltage(
            ao_channel=MOT_freq_VCO_ao,
            voltages=MOT_freq_VCO_pulse,
            sample_rate=ni_sample_rate
        )
        NI_card_3.set_ao_voltage(
            ao_channel=Repump_freq_VCO_ao,
            voltages=Repump_freq_VCO_pulse,
            sample_rate=ni_sample_rate
        )

        # ---------------------------------------------------------------------
        # Build the OPX program that drives the 1 kHz clock. The OPX emits
        # N+1 pulses per sub-cycle (N = total ms of waveform), then idles
        # for `run_buffer` ns between sub-cycles. M sub-cycles per arming.
        # ---------------------------------------------------------------------
        OPX_client.build_stack()
        clock_elem = OPX_client.create_new_do_elem(
            do_channel=opx_trigger_do,
            length=500                           # 500 ns TTL pulse
        )
        logger.info(f"Experiment Length: {experiment_length_samples} samples")
        N = experiment_length_samples            # samples per sub-cycle
        number_of_runs = 4
        M = number_of_runs                       # number of sub-cycles
        run_buffer = 10_000_000                  # ns idle between sub-cycles (10 ms)
        with OPX_client.for_("j", 0, M, 1):
            with OPX_client.for_("i", 0, N + 1, 1):
                OPX_client.set_digital_voltage(element=clock_elem)
                OPX_client.delay(delay_ns)
            OPX_client.delay(run_buffer)

        logger.info(f"Time after configureing voltages before arm {time.perf_counter_ns()}")

        # ---------------------------------------------------------------------
        # Arm both NI cards (regeneration=True replays the buffer each cycle),
        # then launch the OPX program. wait=False returns immediately so this
        # thread can pull frames concurrently.
        # ---------------------------------------------------------------------
        h1 = NI_card_2.arm(regeneration=True)
        h2 = NI_card_3.arm(regeneration=True)
        OPX_client.execute(wait=False)

        logger.info(f"Time sequence done, image gathering starts {time.perf_counter_ns()}")

        # ---------------------------------------------------------------------
        # Helper: pull a single frame from the camera client and reshape it.
        # ---------------------------------------------------------------------
        def get_frame(timeout_ms=1000):
            b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
            logger.info(f"{shape}")
            return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        # TOF grid window (created in configure()) — write images directly.
        # NOTE: diff_sums / diff_counts are persistent (declared outside this
        # while loop) so the running average keeps growing across outer cycles.
        tof_grid = dataset.tof_grid

        # =====================================================================
        # FRAME PULL + PLOT LOOP
        # Outer: M runs (averaging). Inner: n_tof TOF points (one frame pair
        # each). Total camera frames pulled per cycle = M * n_tof * 2.
        # The bottom-row running average is computed manually: avg = sum / count.
        # =====================================================================
        try:
            for run_idx in range(M):
                logger.info(
                    f"Run {run_idx + 1} of {M} — pulling {n_tof} TOF points"
                )

                for tof_idx, tof_us in enumerate(tof_values_us):
                    # Frame 1 — signal (atoms released, time-of-flight = tof_us us).
                    # The first frame of the first run gets a shorter initial
                    # timeout; everything after gets a long timeout to absorb
                    # jitter from the camera readout pipeline.
                    is_very_first = (run_idx == 0 and tof_idx == 0)
                    frame1 = get_frame(timeout_ms=10000 if is_very_first else 100000)
                    logger.info(
                        f"  tof={tof_us} us frame1: shape={frame1.shape}, "
                        f"dtype={frame1.dtype}, min={frame1.min()}, max={frame1.max()}"
                    )

                    # Frame 2 — background (matched AOM pulse, no atoms).
                    frame2 = get_frame(timeout_ms=100000)
                    logger.info(
                        f"  tof={tof_us} us frame2: shape={frame2.shape}, "
                        f"dtype={frame2.dtype}, min={frame2.min()}, max={frame2.max()}"
                    )

                    # Cast to int32 before subtracting to avoid uint underflow.
                    diff = frame1.astype(np.int32) - frame2.astype(np.int32)

                    # Top row of the TOF grid: current-run diff.
                    try:
                        tof_grid.set_current(tof_idx, diff)
                    except Exception as e:
                        logger.info(f"  set_current failed: {e!r}")

                    # Bottom row: running average for this TOF column.
                    if diff_sums[tof_idx] is None:
                        diff_sums[tof_idx] = diff.astype(np.float64)
                    else:
                        diff_sums[tof_idx] += diff
                    diff_counts[tof_idx] += 1
                    avg_image = diff_sums[tof_idx] / diff_counts[tof_idx]
                    try:
                        sigma_x, sigma_y = tof_grid.set_average(tof_idx, avg_image)
                    except Exception as e:
                        logger.info(f"  set_average failed: {e!r}")
                        sigma_x, sigma_y = float('nan'), float('nan')
                    logger.info(
                        f"  tof={tof_us} us fit: sigma_x={sigma_x}, "
                        f"sigma_y={sigma_y} (n_avg={diff_counts[tof_idx]})"
                    )

                logger.info(f"Time images gathered plots updated {time.perf_counter_ns()}")

                # Throttle plot updates so the GUI stays responsive.
                time.sleep(wait_time)

                logger.info(f"Updated DataTaker plots for run {run_idx + 1} of {M}")
        finally:
            # Always stop the camera, even if a frame pull throws.
            logger.info("Stopping acquisition")
            dataset.camera_client.stop_acquisition()

        logger.info(f"Plots plotted, cycle over {time.perf_counter_ns()}")

        # ---------------------------------------------------------------------
        # Cycle teardown: idle, then disarm all three NI cards. force_finish
        # cancels any pending samples so we don't block on a stuck buffer.
        # Uncomment `thread.running = False` to make this a one-shot script.
        # ---------------------------------------------------------------------
        time.sleep(wait_time)
        # thread.running = False
        NI_card_2.finalize(h1, timeout=120.0, force_finish=True)
        NI_card_3.finalize(h2, timeout=120.0, force_finish=True)
        NI_card_1.finalize_clock()
