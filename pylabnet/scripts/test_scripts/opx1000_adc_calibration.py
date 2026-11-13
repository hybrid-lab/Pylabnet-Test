"""Calibrate an OPX1000 LF-FEM analog input: raw ADC code -> volts, using a loopback ramp.

Cable: an SMA cable from the LF-FEM analog OUTPUT (default slot 3, output 1) to the
analog INPUT being calibrated (default slot 3, input 1), nothing else connected.

What one shot does (one QUA program, so the output and the input share one clock):
    the output plays a small triangle  0 -> +A -> -A -> 0  (A = "Ramp amplitude");
    the middle part, +A down to -A, is "the ramp" used for the fit,
    while the input records a raw ADC trace that is longer than the whole triangle.
The input sees the triangle a little later than it was played (cable + converter delay).
The analysis finds that delay from the whole trace (the corners pin it down), then fits
ONLY the samples on the +A -> -A ramp. Shots repeat for "Measurement time" seconds and
the traces are averaged (each shot starts the triangle and the capture together, so the
traces line up exactly and averaging only removes noise).

Analysis, on the ramp only:   code = offset + slope * V_out
    slope (codes per volt)     -> volts = (code - offset) / slope
    sign of the slope          -> whether the input inverts
    residuals from the line    -> linearity (a curve shows up as a bow in the residual plot)
The full input range is extrapolated from that line: the converter's codes run from
-2048 to +2047, so the range is (-2048 - offset) / slope  to  (2047 - offset) / slope.
It is also given as an equation in the input gain, assuming the gain scales the input
as documented (a voltage factor G = 10**(gain_db / 20)):
    range = +-R0 * 10**(-gain_db / 20) V  =  +-R0 / G V,   R0 = range at 0 dB (expected 0.5 V)

Notes for users:
- The calibration is only as good as the output's own voltage. To check the output
  independently, look at the same ramp on a scope with a 50 ohm input.
- The ramp amplitude is capped at 0.2 V, well below the 0.5 V input limit.
- Results go to the DataTaker log and to the run's metadata; Save writes the plotted
  curves (averaged trace, code vs voltage, residuals) as .csv files.
"""

import time

import numpy as np

# Small helpers shared by the OPX1000 test scripts: reading integer parameters,
# redrawing the plot from the GUI thread, waiting for a job, reporting errors.
from pylabnet.scripts.data_center.opx1000_test_helpers import (
    integer, prepare_plot_updates, report_error, request_plot_update, wait_for_job,
)

# Hard safety cap on the ramp: the LF-FEM input should never see more than ~0.5 V.
MAX_RAMP_V = 0.2
# The driver captures at most 16000 ns of raw ADC trace per shot.
MAX_CAPTURE_NS = 16000
# Each edge of the triangle (0 -> +A at the start, -A -> 0 at the end) takes this long.
EDGE_NS = 1000
# Number of voltage bins for the "code vs voltage" and residual plots.
N_BINS = 41

# Parameters shown in DataTaker. Each value is the default the user can edit.
INIT_DICT = {
    "fem_slot": {"LF FEM slot (3 or 4)": "3"},
    "ao_channel": {"Analog output playing the ramp (1-8)": "1"},
    "ai_channel": {"Analog input being calibrated (1 or 2)": "1"},
    "ramp_mv": {"Ramp amplitude A (mV, ramp goes +A down to -A, max 200)": "200"},
    "ramp_ns": {"Ramp duration +A to -A (ns, multiple of 4)": "10000"},
    "capture_ns": {"Input capture length (ns, max 16000)": "16000"},
    "measure_s": {"Measurement time (s): shots repeat and are averaged": "1"},
}


def define_dataset():
    # ImageGrid shows several labelled line plots, each with an optional red fit line.
    return "ImageGrid"


def configure(**kwargs):
    dataset = kwargs["dataset"]
    # The OPX1000 client; DataTaker names it "<server type>_<device_id>".
    dataset.opx = kwargs["OPX1000_OPX1000"]
    # Lets the experiment thread ask the GUI thread to redraw.
    prepare_plot_updates(dataset)


def ramp_waveform(amplitude_v, ramp_ns, edge_ns=EDGE_NS):
    """The output's voltage, one sample per ns: 0 -> +A (edge), +A -> -A (ramp), -A -> 0 (edge).

    A triangle without jumps, so the input can follow it exactly. The middle part (the
    ramp) covers every voltage from +A down to -A; only it is used for the fit.
    """
    up = np.linspace(0.0, amplitude_v, edge_ns, endpoint=False)
    ramp = np.linspace(amplitude_v, -amplitude_v, ramp_ns, endpoint=False)
    back = np.linspace(-amplitude_v, 0.0, edge_ns, endpoint=False)
    return np.concatenate([up, ramp, back])


def fit_ramp(trace, waveform, ramp_ns, edge_ns=EDGE_NS):
    """Find the delay from the whole trace, then fit a line to the +A -> -A ramp only.

    trace: averaged ADC codes, one per ns.  waveform: output volts, one per ns (from
    ramp_waveform); the output is 0 V before and after it.
    Step 1 (delay): tries every delay that keeps the whole waveform inside the trace and
    keeps the one where code-vs-voltage is closest to a line. The corners of the triangle
    make this sharp.
    Step 2 (calibration): code = offset + slope * V, least squares on the ramp samples only.
    Returns slope (codes/V), offset (codes), delay_ns, the voltage seen by each sample
    (volts), a mask of the ramp samples, the fitted codes and the residuals.
    """
    trace = np.asarray(trace, dtype=float)
    n, length = trace.size, waveform.size
    if n < length + 1:
        raise ValueError("The capture must be longer than the ramp")
    best = None
    for delay in range(0, n - length + 1):
        volts = np.zeros(n)
        volts[delay:delay + length] = waveform
        # Least-squares straight line through (volts, trace)
        vm, cm = volts.mean(), trace.mean()
        dv = volts - vm
        slope = float(np.dot(dv, trace - cm) / np.dot(dv, dv))
        offset = cm - slope * vm
        err = float(np.sum((trace - offset - slope * volts) ** 2))
        if best is None or err < best[0]:
            best = (err, delay, slope, offset, volts)
    _, delay, _, _, volts = best
    # Step 2: the samples that saw the +A -> -A ramp (they come after the first edge).
    on_ramp = np.zeros(n, dtype=bool)
    on_ramp[delay + edge_ns:delay + edge_ns + ramp_ns] = True
    slope, offset = np.polyfit(volts[on_ramp], trace[on_ramp], 1)
    fitted = offset + slope * volts
    return dict(slope=float(slope), offset=float(offset), delay_ns=int(delay), volts=volts,
                on_ramp=on_ramp, fitted=fitted, residual=trace - fitted)


def linearity(volts, residual, amplitude_v, n_bins=N_BINS):
    """Bin the residuals by voltage; also fit a parabola to measure any bow.

    Returns bin centres, mean residual per bin, and the bow: the parabola's deviation from
    the straight line at the ends of the ramp (+-A), in codes.
    """
    edges = np.linspace(-amplitude_v, amplitude_v, n_bins + 1)
    which = np.clip(np.digitize(volts, edges) - 1, 0, n_bins - 1)
    centres, means = [], []
    for b in range(n_bins):
        sel = which == b
        if sel.sum() >= 20:               # skip nearly empty bins
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(float(residual[sel].mean()))
    # Parabola through the residuals: residual ~ c2 * V^2 + c1 * V + c0. The straight-line fit
    # already removed c1 and c0, so c2 * A^2 is how far the ends bend away from a line.
    c2 = np.polyfit(volts, residual, 2)[0]
    return np.array(centres), np.array(means), float(c2 * amplitude_v ** 2)


def binned_codes(volts, codes, amplitude_v, n_bins=N_BINS):
    """Mean ADC code in each voltage bin (for the code-vs-voltage plot)."""
    edges = np.linspace(-amplitude_v, amplitude_v, n_bins + 1)
    which = np.clip(np.digitize(volts, edges) - 1, 0, n_bins - 1)
    centres, means = [], []
    for b in range(n_bins):
        sel = which == b
        if sel.sum() >= 20:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(float(codes[sel].mean()))
    return np.array(centres), np.array(means)


def input_gain_db(opx, slot, channel):
    """The input gain the server is configured with (OPX1000Config.py), in dB."""
    try:
        fem = opx.get_config()["controllers"]["con1"]["fems"][slot]
        return float(fem["analog_inputs"][channel].get("gain_db", 0))
    except (KeyError, TypeError):
        return None


def experiment(**kwargs):
    dataset, thread = kwargs["dataset"], kwargs["thread"]
    opx = dataset.opx
    try:
        if not thread.running:
            return

        # ---- Read and check the parameters --------------------------------------------
        slot = integer(dataset, "fem_slot")
        ao = integer(dataset, "ao_channel")
        ai = integer(dataset, "ai_channel")
        ramp_ns = integer(dataset, "ramp_ns", minimum=16)
        capture_ns = integer(dataset, "capture_ns", minimum=16)
        amplitude_v = float(dataset.get_input_parameter("ramp_mv")) / 1000
        measure_s = float(dataset.get_input_parameter("measure_s"))
        if not 0 < amplitude_v <= MAX_RAMP_V:
            raise ValueError("Ramp amplitude must be between 0 and {:.0f} mV".format(MAX_RAMP_V * 1000))
        if ramp_ns % 4 or capture_ns % 4:
            raise ValueError("Ramp duration and capture length must be multiples of 4 ns")
        waveform = ramp_waveform(amplitude_v, ramp_ns)
        if capture_ns > MAX_CAPTURE_NS or capture_ns < waveform.size + 200:
            raise ValueError("Capture must be at least the ramp + edges + 200 ns ({} ns) and at most {} ns"
                             .format(waveform.size + 200, MAX_CAPTURE_NS))
        if not measure_s >= 0:
            raise ValueError("Measurement time must be >= 0 s")
        gain_db = input_gain_db(opx, slot, ai)

        dataset.log.info(
            "ADC calibration: slot {} output {} -> input {}; ramp +-{:.0f} mV over {} ns; "
            "repeating shots for {:g} s".format(slot, ao, ai, amplitude_v * 1000, ramp_ns, measure_s))

        # ---- Shots: ramp out + capture in, repeated for measure_s, then averaged -------
        traces = []
        started = time.monotonic()
        while thread.running:
            opx.build_stack()                     # start an empty sequence
            # Play the ramp on the output. Output and input are different elements, so the
            # OPX runs them at the same time (nothing here makes one wait for the other).
            opx.set_ao_voltage(pulse="arbitrary", ao_channel=ao, wave_function=waveform.tolist(),
                               length=waveform.size, fem_slot=slot)
            # Record the input for capture_ns, starting with the ramp.
            opx.get_ai_voltage(ai, length=capture_ns, fem_slot=slot, label="trace")
            opx.execute(wait=False)
            if not wait_for_job(opx, thread, 60):  # False = the user pressed Stop
                break
            traces.append(np.asarray(opx.get_results()["trace"]["raw_adc"], dtype=float))
            if time.monotonic() - started >= measure_s:
                break
        if not traces:
            dataset.log.info("Stopped before any shot finished")
            return

        # ---- Fit -----------------------------------------------------------------------
        trace = np.mean(traces, axis=0)           # averaged raw codes, one per ns
        fit = fit_ramp(trace, waveform, ramp_ns)
        slope, offset = fit["slope"], fit["offset"]
        # Everything below uses only the samples on the +A -> -A ramp.
        ramp = fit["on_ramp"]
        ramp_volts, ramp_codes, ramp_residual = fit["volts"][ramp], trace[ramp], fit["residual"][ramp]
        noise = float(np.std(ramp_residual))     # spread about the line, in codes
        swing = abs(slope) * 2 * amplitude_v      # codes covered by the ramp
        centres, bow_means, bow = linearity(ramp_volts, ramp_residual, amplitude_v)
        v_bins, code_bins = binned_codes(ramp_volts, ramp_codes, amplitude_v)

        # ---- Plots ---------------------------------------------------------------------
        t = np.arange(trace.size)
        dataset.set_curve("Averaged input trace", t, trace,
                          title="Input trace, {} shot(s) averaged (red: line fitted to the ramp)".format(
                              len(traces)),
                          xlabel="Time in capture (ns)", ylabel="ADC code", row=0,
                          fit=(t, fit["fitted"]))
        line_v = np.array([-amplitude_v, amplitude_v])
        dataset.set_curve("Code vs voltage", v_bins * 1000, code_bins,
                          title="ADC code vs output voltage on the ramp (red: straight-line fit)",
                          xlabel="Output voltage (mV)", ylabel="ADC code", row=1,
                          fit=(line_v * 1000, offset + slope * line_v))
        dataset.set_curve("Residual", centres * 1000, bow_means,
                          title="Residual from the line (flat = linear)",
                          xlabel="Output voltage (mV)", ylabel="ADC code", row=2,
                          fit=(line_v * 1000, np.zeros(2)))

        # ---- Report --------------------------------------------------------------------
        if not slope:
            raise RuntimeError("The fitted slope is zero: the input did not see the ramp")
        # The gain this was measured at. If the server's config could not be read, assume 0 dB.
        g = 0.0 if gain_db is None else gain_db
        g_note = "" if gain_db is not None else " (gain_db unknown, assumed 0)"
        # Full input range, extrapolated from the ramp's line to the converter's end codes.
        range_lo, range_hi = sorted([(-2048 - offset) / slope, (2047 - offset) / slope])
        span = range_hi - range_lo
        # Referred to 0 dB, assuming the gain multiplies the input voltage by G = 10**(gain_db/20):
        slope0 = slope / 10 ** (g / 20)          # codes per volt at 0 dB
        r0 = 2048 / abs(slope0)                  # half-range at 0 dB, volts (expected 0.5)
        expected = 4096 * 10 ** (g / 20)         # documented 1 V span at 0 dB
        lines = [
            "Shots averaged: {} | delay output -> input: {} ns | fit uses the {}-ns ramp "
            "+{:.0f} -> -{:.0f} mV only".format(len(traces), fit["delay_ns"], ramp_ns,
                                                amplitude_v * 1000, amplitude_v * 1000),
            "Slope: {:.1f} codes/V | offset: {:.1f} codes | sign: {} | measured at gain_db = {:g}{}".format(
                slope, offset, "inverted (negative)" if slope < 0 else "not inverted", g, g_note),
            "Conversion at this gain: volts = (code - {:.1f}) / {:.1f}   (1 code = {:.4f} mV)".format(
                offset, slope, 1000 / abs(slope)),
            "Input range at this gain (extrapolated from the ramp): {:+.4f} V to {:+.4f} V "
            "(span {:.4f} V)".format(range_lo, range_hi, span),
            "Range vs gain, with dB:  +-{:.4f} V * 10^(-gain_db/20)   ;   "
            "volts = (code - {:.1f}) / ({:.1f} * 10^(gain_db/20))".format(r0, offset, slope0),
            "Range vs gain, no dB:    +-{:.4f} V / G   ;   volts = (code - {:.1f}) / ({:.1f} * G)   "
            "where G is the input's voltage gain factor (G = 1 at the default setting, "
            "G = 2 for twice the amplification)".format(r0, offset, slope0),
            "Documented: +-0.5 V at 0 dB, i.e. {:.1f} codes/V here -> measured/documented slope = {:.3f}"
            .format(expected, abs(slope) / expected),
            "(The gain equations assume the gain scales the input as documented, and the offset "
            "may change with gain: rerun at another gain_db to check both.)",
        ]
        lines.append("Linearity: scatter about the line {:.2f} codes rms; bow at the ramp ends {:+.2f} codes "
                     "({:.2f}% of the ramp's {:.0f}-code swing)".format(
                         noise, bow, 100 * abs(bow) / swing if swing else float("nan"), swing))
        if swing < 10 * noise or swing < 20:
            lines.insert(0, "WARNING: the ramp is barely visible on the input. Is the output cabled to "
                         "this input? The results below are not meaningful.")
        for line in lines:
            dataset.log.info(line)

        # Everything also goes into the run's metadata (saved with the data).
        dataset.log.update_metadata(opx1000_adc_calibration=dict(
            fem_slot=slot, ao_channel=ao, ai_channel=ai, ramp_amplitude_v=amplitude_v, ramp_ns=ramp_ns,
            edge_ns=EDGE_NS, capture_ns=capture_ns, shots=len(traces),
            elapsed_s=time.monotonic() - started, delay_ns=fit["delay_ns"],
            slope_codes_per_v=slope, offset_codes=offset, input_range_v=[range_lo, range_hi],
            full_scale_span_v=span, input_gain_db=gain_db, slope_at_0db_codes_per_v=slope0,
            half_range_at_0db_v=r0, documented_slope_codes_per_v=expected,
            fit_samples="ramp +A -> -A only",
            residual_rms_codes=noise, bow_codes=bow, ramp_swing_codes=swing,
        ))
    except Exception as error:
        opx.close()
        report_error(kwargs, error)
    finally:
        request_plot_update(dataset)
        thread.running = False               # one measurement per press of Run
