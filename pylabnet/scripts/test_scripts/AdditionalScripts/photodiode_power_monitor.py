# Long-term power monitor: logs three NI analog inputs -- photodiode, error
# signal and servo output, in that channel order -- and reports their rolling
# stability, so we can see how the beam power drifts over long timescales.
#
# Each displayed point is a hardware-clocked burst of samples (e.g. 10000
# samples at 100 kS/s = 100 ms of gap-free data). Only one number out of the
# burst is plotted as the reading -- the first sample, or the mean if averaging
# is on -- but EVERY sample in the burst is compared against a dropout
# threshold, so a 10 us drop that no plotted point would ever show still gets
# counted and logged.
#
# The only line this script drives is ao0 on the SAME card it reads from,
# held at a constant voltage for the whole run. The AOM/AO11 hold on the
# other two cards is commented out throughout (search NI_do / NI_ao to
# re-enable it as a block).

import time
import numpy as np

INIT_DICT = {
    # Feed the commented-out AOM/AO11 hold; re-enable together with the
    # NI_do / NI_ao lines in experiment().
    # 'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    # 'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    # 'ao11_voltage': {'AO11 Hold Voltage (V)': '0.45'},
    # DataTaker casts every INIT_DICT entry to float, so the AI channel is
    # given as an index and turned into "aiN" below.
    'ai_channel_index': {'Photodiode AI Channel Index': '0'},
    'ai_channel_index_2': {'Diff Mon AI Channel Index': '1'},
    'ai_channel_index_3': {'Servo Mon AI Channel Index': '2'},
    'ai_max_range': {'AI Max Range (V)': '10.0'},
    # Constant DC level on ao0 of the SAME card the photodiodes are read from.
    # Written once at the start of the run and left there -- see experiment().
    'ao0_voltage': {'Setpoint Voltage (V)': '0.5'},
    'sample_interval_s': {'Sample Interval (s)': '0.1'},
    'averaging_on': {'Average Each Point? (1=yes, 0=no)': '0'},
    # Burst size and clock rate. 10000 samples at 100000 Hz is 100 ms of
    # gap-free data per channel per point. The three channels are read one
    # after another, so the wall-clock cost of a point is three bursts: push
    # these up and it is the loop, not the sample interval, that sets the point
    # rate (the start of the run logs a warning when that happens).
    'samples_per_point': {'Samples per Point (burst size)': '10000'},
    'sample_rate': {'AI Sample Rate (Hz)': '100000'},
    # Dropout check. Applied to every sample of every burst, not just the
    # plotted one. Set the threshold just under the normal operating level of
    # the photodiode -- the default here is only a placeholder.
    #
    # Turning this off puts the script back on exactly the path it took before
    # the check existed: averaging on takes the mean of samples_per_point reps,
    # averaging off takes one instantaneous sample and ignores the burst size.
    # The only thing the check brought with it is the clock rate above, which
    # used to be hardwired at 10000 Hz -- for a literal match of an old
    # high-rep run, set AI Sample Rate back to 10000.
    'dropout_check_on': {'Check for Dropouts? (1=yes, 0=no)': '1'},
    'dropout_threshold': {'Dropout Threshold (V)': '1.5'},
    # Stability is judged over a trailing window: for an intensity lock the
    # question is "how quiet is it now", not "how far has it drifted since
    # the run started". 0 uses every point taken so far instead.
    'stats_window': {'Stability Window (points, 0 = whole run)': '60'},
}

# Per-channel display strings, in channel order: the three AI inputs carry
# different signals, so they get their own titles rather than three copies of
# one. Each entry is (plot title, left-axis label, short name for the stability
# plot). Only what is drawn on screen -- the dataset keys below, which end up
# in the saved filenames, are left alone.
PLOT_LABELS = (
    ('Photodiode Power Monitor', 'Photodiode voltage', 'Photodiode power'),
    ('Error Signal Monitor', 'Error signal', 'Error signal'),
    ('Servo Monitor', 'Servo voltage', 'Servo'),
)

# Fixed keys for the extra plots, so configure() and experiment() can agree on
# them regardless of which channel index is selected.
CHILD_NAME = 'Photodiode 2'
CHILD_NAME_3 = 'Photodiode 3'
STAB_NAME = 'Stability'
STAB_NAME_2 = 'Stability 2'
STAB_NAME_3 = 'Stability 3'
# Burst-minimum traces. These are children of the three voltage datasets
# rather than of the root, because they share their parent's graph -- see
# configure().
MIN_NAME = 'Burst Min'
MIN_NAME_2 = 'Burst Min 2'
MIN_NAME_3 = 'Burst Min 3'

# Colour slot for the burst-minimum traces, distinct from the 0/1/2 the three
# voltage traces get in either of the GUI colour lists.
MIN_COLOR_INDEX = 7

# How often to write a stability summary to the log, in points.
STATS_LOG_EVERY = 60

# Minimum spacing between dropout warnings, in seconds. One warning per
# affected burst would flood the log when the beam misbehaves; the running
# counts in the periodic and end-of-run summaries lose nothing.
DROPOUT_WARN_INTERVAL_S = 2.0

# NOTE ON THREADING: configure() runs on the GUI thread, experiment() runs in a
# QThread worker. Qt widget calls (setTitle, setLabel, setLevels...) from the
# worker crash the application. Everything the loop displays must therefore go
# through set_data(), which only stores values -- the GUI's UpdateThread is
# what actually redraws. All widget setup belongs in configure().


def define_dataset():
    return 'Dataset'


def stability(values, window=0):
    """Mean, standard deviation and fractional RMS over a trailing window.

    Uses the sample standard deviation (ddof=1): these are measurements of a
    fluctuating quantity, not a full population, and with a short window the
    difference from ddof=0 is not negligible. Returns NaN for the deviation
    until there are at least two points to compare.

    :param values: (list) readings in volts, oldest first
    :param window: (int) trailing points to use; 0 or negative uses all
    :return: (mean_v, std_v, fractional_rms) -- fractional is std/|mean|
    """
    arr = np.asarray(values[-window:] if window > 0 else values, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan'), float('nan')
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, float('nan'), float('nan')
    std = float(arr.std(ddof=1))
    # A photodiode reading near zero makes the fractional figure meaningless
    # rather than merely large, so report it as undefined.
    frac = std / abs(mean) if mean != 0.0 else float('nan')
    return mean, std, frac


def format_stability(channel, values, window):
    """One-line stability summary for a plot title.

    Scales the fractional figure into whichever of %/ppm reads naturally, so a
    well-locked beam does not display as '0.003 %'.
    """
    mean, std, frac = stability(values, window)
    n = len(values[-window:] if window > 0 else values)

    if not np.isfinite(std):
        return f'{channel}  |  {mean:.4f} V  |  collecting...'

    if np.isfinite(frac):
        frac_str = f'{frac * 1e6:.0f} ppm' if frac < 1e-3 else f'{frac * 100:.3f} %'
    else:
        frac_str = 'n/a'

    return (f'{channel}  |  mean {mean:.4f} V  |  '
            f'sigma {std * 1e3:.3f} mV ({frac_str} RMS, n={n})')


def configure(**kwargs):
    dataset = kwargs['dataset']

    ai_channel = f"ai{int(dataset.get_input_parameter('ai_channel_index'))}"
    ai_channel_2 = f"ai{int(dataset.get_input_parameter('ai_channel_index_2'))}"
    ai_channel_3 = f"ai{int(dataset.get_input_parameter('ai_channel_index_3'))}"

    window = int(dataset.get_input_parameter('stats_window'))
    window_txt = f'{window}-point' if window > 0 else 'whole-run'

    title, axis_label, _ = PLOT_LABELS[0]
    dataset.graph.setLabel('bottom', 'Elapsed time', units='s')
    dataset.graph.setLabel('left', axis_label, units='V')
    dataset.graph.getPlotItem().setTitle(f'{title} ({ai_channel})')

    # Second and third traces, each on its own plot, sharing the same time axis.
    for name, channel, labels in ((CHILD_NAME, ai_channel_2, PLOT_LABELS[1]),
                                  (CHILD_NAME_3, ai_channel_3, PLOT_LABELS[2])):
        title, axis_label, _ = labels
        dataset.add_child(name=name, new_plot=True)
        child = dataset.children[name]
        child.graph.setLabel('bottom', 'Elapsed time', units='s')
        child.graph.setLabel('left', axis_label, units='V')
        child.graph.getPlotItem().setTitle(f'{title} ({channel})')

    # Burst minimum, overlaid on each voltage plot: new_plot=False reuses the
    # parent's graph, which is why these hang off the voltage datasets. The
    # main trace is one sample out of thousands, so without this a dropout
    # between plotted samples shows up only in the log, not on screen.
    for parent_key, min_name in ((None, MIN_NAME), (CHILD_NAME, MIN_NAME_2),
                                 (CHILD_NAME_3, MIN_NAME_3)):
        parent = dataset if parent_key is None else dataset.children[parent_key]
        parent.add_child(name=min_name, new_plot=False,
                         color_index=MIN_COLOR_INDEX)
        # Two curves on one plot are otherwise only told apart by colour.
        legend = parent.graph.getPlotItem().addLegend(offset=(-10, 10))
        legend.addItem(parent.curve, 'reading')
        legend.addItem(parent.children[min_name].curve, 'burst min')

    # Rolling stability, one plot per channel. This is the live readout for
    # tuning the lock: watch it fall.
    # No 'units' kwarg on the left axis -- pyqtgraph would then SI-prefix the
    # values into "kppm" once the beam is noisy.
    for name, channel, labels in ((STAB_NAME, ai_channel, PLOT_LABELS[0]),
                                  (STAB_NAME_2, ai_channel_2, PLOT_LABELS[1]),
                                  (STAB_NAME_3, ai_channel_3, PLOT_LABELS[2])):
        short_name = labels[2]
        dataset.add_child(name=name, new_plot=True)
        stab = dataset.children[name]
        stab.graph.setLabel('bottom', 'Elapsed time', units='s')
        stab.graph.setLabel('left', 'Fractional RMS (ppm)')
        stab.graph.getPlotItem().setTitle(
            f'{short_name} rolling stability, {window_txt} window ({channel})')


def experiment(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    logger = dataset.log

    # NI_card_1 (PXI1Slot2_1) has a free AI bank -- it's only used for the
    # OPX clock input in the fast MOT sequences, so it's not touched here.
    NI_ai = kwargs['nidaqmx_ni_daq_1']
    # Same DO/AO wiring as fastMOT.py: dio1/ao1 = MOT AOM, dio3/ao5 = repump AOM.
    # NI_do = kwargs['nidaqmx_ni_daq_2']
    # NI_ao = kwargs['nidaqmx_ni_daq_3']

    # mot_aom_power = float(dataset.get_input_parameter('mot_aom_power'))
    # repump_aom_power = float(dataset.get_input_parameter('repump_aom_power'))
    # ao11_voltage = float(dataset.get_input_parameter('ao11_voltage'))
    ai_channel = f"ai{int(dataset.get_input_parameter('ai_channel_index'))}"
    ai_channel_2 = f"ai{int(dataset.get_input_parameter('ai_channel_index_2'))}"
    ai_channel_3 = f"ai{int(dataset.get_input_parameter('ai_channel_index_3'))}"
    max_range = float(dataset.get_input_parameter('ai_max_range'))
    ao0_voltage = float(dataset.get_input_parameter('ao0_voltage'))
    sample_interval_s = float(dataset.get_input_parameter('sample_interval_s'))
    averaging_on = bool(int(dataset.get_input_parameter('averaging_on')))
    dropout_check_on = bool(int(dataset.get_input_parameter('dropout_check_on')))
    dropout_threshold = float(dataset.get_input_parameter('dropout_threshold'))
    sample_rate = float(dataset.get_input_parameter('sample_rate'))
    stats_window = int(dataset.get_input_parameter('stats_window'))
    samples_per_point = max(1, int(dataset.get_input_parameter('samples_per_point')))
    # The burst is only worth taking if something looks at more than its first
    # sample: averaging reads the whole thing, and so does the dropout check.
    # With both off, a single instantaneous reading is all that ever gets used.
    if not averaging_on and not dropout_check_on:
        samples_per_point = 1
    burst_duration = samples_per_point / sample_rate

    # Hold the MOT + repump AOMs on at the same voltage used in the MOT
    # scripts, for the whole duration of the monitor.
    # NI_do.set_do_voltage(do_channel='dio1', value=True)   # MOT AOM enable
    # NI_do.set_do_voltage(do_channel='dio3', value=True)   # Repump AOM enable
    # NI_ao.set_ao_voltage(ao_channel='ao1', voltages=mot_aom_power)
    # NI_ao.set_ao_voltage(ao_channel='ao5', voltages=repump_aom_power)
    # NI_ao.set_ao_voltage(ao_channel='ao11', voltages=ao11_voltage)

    # Constant DC level on ao0 of the read card. A single-sample write is an
    # immediate one-shot: it opens a task, writes, and closes it again. The
    # hardware holds the last value after the task closes, so this stays put
    # for the whole run -- and, note, AFTER the run stops as well. Nothing
    # here ever returns ao0 to 0 V.
    #
    # ao0 on this card is also what arm_clock() uses to export the sample
    # clock in the fast MOT sequences, so the two cannot be used at once.
    NI_ai.set_ao_voltage(ao_channel='ao0', voltages=ao0_voltage)
    logger.info(f'Holding ao0 at {ao0_voltage} V for the duration of the run.')

    logger.info(
        f'Sampling {ai_channel}, {ai_channel_2} and {ai_channel_3} '
        f'every {sample_interval_s} s: {samples_per_point} sample(s) per '
        f'channel at {sample_rate:.0f} Hz '
        f'({burst_duration * 1e3:.2f} ms per burst), '
        + ('plotting the burst mean.' if averaging_on
           else 'plotting the first sample of each burst.')
    )
    if dropout_check_on:
        logger.info(
            f'Dropout check on: every sample of every burst is compared '
            f'against {dropout_threshold:.4f} V.'
        )
    else:
        logger.info('Dropout check off.')
    # Three bursts back to back are the floor on how long a point takes; past
    # that, the sample interval is no longer what sets the point rate.
    if 3 * burst_duration > sample_interval_s:
        logger.warn(
            f'Three bursts take {3 * burst_duration * 1e3:.0f} ms, longer than '
            f'the {sample_interval_s * 1e3:.0f} ms sample interval -- the loop '
            f'will run flat out and points will be further apart than asked.'
        )

    def read_burst(channel):
        """One point's worth of samples, taken back to back off the AI clock.

        All channels go through this, so they are sampled identically -- same
        count, range and rate. The burst comes back whole rather than reduced
        to a number: the plotted point needs one sample out of it, but the
        dropout check needs all of them.
        """
        reading = NI_ai.get_ai_voltage(
            ai_channel=channel,
            num_samples=samples_per_point,
            max_range=max_range,
            sample_rate=sample_rate,
        )
        burst = np.ravel(np.asarray(reading, dtype=float))
        if burst.size == 0:
            raise ValueError(f'AI read returned no samples for {channel}')
        return burst

    def point_value(burst):
        """The one number out of a burst that gets plotted and kept.

        Averaging is about the displayed trace only -- the dropout check sees
        every sample either way.
        """
        return float(burst.mean()) if averaging_on else float(burst[0])

    times = []
    voltages = []
    voltages_2 = []
    voltages_3 = []
    sigma_ppm = []
    sigma_2_ppm = []
    sigma_3_ppm = []
    t0 = time.time()

    stab = dataset.children[STAB_NAME]
    stab_2 = dataset.children[STAB_NAME_2]
    stab_3 = dataset.children[STAB_NAME_3]
    child = dataset.children[CHILD_NAME]
    child_3 = dataset.children[CHILD_NAME_3]

    # Dropout bookkeeping, one entry per channel in plot order. 'worst' is the
    # lowest single sample ever seen on that channel, which is the number to
    # look at when deciding where the threshold belongs in the first place.
    watch = [
        {
            'label': label,
            'trace': trace,
            'mins': [],
            'low_samples': 0,
            'low_points': 0,
            'total_samples': 0,
            'worst': float('inf'),
            'first_t': None,
            'last_t': None,
        }
        for label, trace in (
            (ai_channel, dataset.children[MIN_NAME]),
            (ai_channel_2, child.children[MIN_NAME_2]),
            (ai_channel_3, child_3.children[MIN_NAME_3]),
        )
    ]
    last_warn = 0.0

    while thread.running:
        loop_start = time.time()

        # Read all three before appending anything: a failure on any channel
        # skips the whole point, so the traces stay aligned with `times`.
        try:
            bursts = (read_burst(ai_channel), read_burst(ai_channel_2),
                      read_burst(ai_channel_3))
        except Exception as e:
            logger.warn(f'AI read failed, skipping point: {e}')
            time.sleep(sample_interval_s)
            continue

        elapsed = loop_start - t0
        times.append(elapsed)
        voltages.append(point_value(bursts[0]))
        voltages_2.append(point_value(bursts[1]))
        voltages_3.append(point_value(bursts[2]))

        # The dropout check. Nothing is discarded before being counted: one
        # sample below threshold anywhere in the burst is recorded, even
        # though the plotted reading may have been taken microseconds earlier
        # at full power.
        if dropout_check_on:
            flagged = []
            for w, burst in zip(watch, bursts):
                burst_min = float(burst.min())
                w['mins'].append(burst_min)
                w['total_samples'] += int(burst.size)
                w['worst'] = min(w['worst'], burst_min)

                low = int(np.count_nonzero(burst < dropout_threshold))
                if low:
                    w['low_samples'] += low
                    w['low_points'] += 1
                    if w['first_t'] is None:
                        w['first_t'] = elapsed
                    w['last_t'] = elapsed
                    flagged.append(f"{w['label']} {low}/{burst.size} samples "
                                   f"(min {burst_min:.4f} V)")

                w['trace'].set_data(data=w['mins'], x=times)

            if flagged and (loop_start - last_warn) >= DROPOUT_WARN_INTERVAL_S:
                logger.warn(
                    f'@ {elapsed:.1f} s  below {dropout_threshold:.4f} V  ->  '
                    + '; '.join(flagged)
                )
                last_warn = loop_start

        # Rolling fractional RMS, recomputed over the trailing window each
        # point. NaN until there are two points to compare -- pyqtgraph draws
        # that as a gap rather than a spurious zero.
        _, _, frac = stability(voltages, stats_window)
        _, _, frac_2 = stability(voltages_2, stats_window)
        _, _, frac_3 = stability(voltages_3, stats_window)
        sigma_ppm.append(frac * 1e6)
        sigma_2_ppm.append(frac_2 * 1e6)
        sigma_3_ppm.append(frac_3 * 1e6)

        dataset.set_data(data=voltages, x=times)
        child.set_data(data=voltages_2, x=times)
        child_3.set_data(data=voltages_3, x=times)
        stab.set_data(data=sigma_ppm, x=times)
        stab_2.set_data(data=sigma_2_ppm, x=times)
        stab_3.set_data(data=sigma_3_ppm, x=times)

        # Periodic summary so the run leaves a record, with the whole-run
        # figure alongside the windowed one: they separate slow drift from
        # fast noise. Whole-run sigma much larger than windowed means drift.
        if len(times) % STATS_LOG_EVERY == 0:
            for label, series in ((ai_channel, voltages), (ai_channel_2, voltages_2),
                                  (ai_channel_3, voltages_3)):
                _, std_all, frac_all = stability(series, 0)
                logger.info(
                    f'@ {elapsed:.0f} s  '
                    + format_stability(label, series, stats_window)
                    + f'  |  whole run sigma {std_all * 1e3:.3f} mV '
                      f'({frac_all * 1e6:.0f} ppm)'
                )
            if dropout_check_on:
                for w in watch:
                    logger.info(
                        f"@ {elapsed:.0f} s  {w['label']} dropouts: "
                        f"{w['low_samples']}/{w['total_samples']} samples below "
                        f"{dropout_threshold:.4f} V, in {w['low_points']}/"
                        f"{len(times)} points  |  lowest sample "
                        f"{w['worst']:.4f} V"
                    )

        sleep_time = sample_interval_s - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # End-of-run verdict: the line to read after leaving the monitor running
    # overnight. Either nothing ever went below threshold, or here is how much
    # did and when.
    if dropout_check_on:
        logger.info(f'Monitor stopped after {len(times)} points '
                    f'({(time.time() - t0) / 60:.1f} min).')
        for w in watch:
            if w['total_samples'] == 0:
                continue
            if w['low_samples']:
                logger.warn(
                    f"{w['label']}: {w['low_samples']} of {w['total_samples']} "
                    f"samples below {dropout_threshold:.4f} V, in "
                    f"{w['low_points']} of {len(times)} points; first at "
                    f"{w['first_t']:.1f} s, last at {w['last_t']:.1f} s, "
                    f"lowest sample {w['worst']:.4f} V."
                )
            else:
                logger.info(
                    f"{w['label']}: no sample below {dropout_threshold:.4f} V "
                    f"in {w['total_samples']} samples "
                    f"(lowest {w['worst']:.4f} V)."
                )
