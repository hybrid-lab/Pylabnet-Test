import numpy as np

from pylabnet.scripts.data_center.datasets import (
    Plot2D,
    Dataset,
    InfiniteRollingLine,
)
from scipy.optimize import curve_fit


if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


RAW_IMAGE_LEVELS = (0, 20)
CAMERA_TRIGGER_WIDTH_MS = 1
SIGMA_HISTORY_LENGTH = 1000


INIT_DICT = {
    'mot_boot_up_time': {'MOT Boot Up Time (ms)': '100'},
    'mot_alive_time': {'MOT Alive Time (ms)': '300'},
    'expansion_time': {'Expansion Time (ms)': '20'},
    'mot_dissolution_time': {'MOT Dissolution Time (ms)': '100'},
    'mot_dead_time': {'MOT Dead Time (ms)': '10'},
    'exposure_time': {'Camera Exposure Time (us)': '501'},
    'camera_gain': {'Camera Gain': '50'},
    'camera_trigger_do': {'Camera Trigger DO Channel': '0'},
    'magnetic_coils_do': {'Magnetic Coils DO Channel': '1'},
    'magnetic_coils_ao': {'Magnetic Coils AO Channel': '1'},
    'scan_parameter_ao': {'Scan Parameter AO Channel': '2'},
    'scan_min': {'Scan Min (V)': '0.0'},
    'scan_max': {'Scan Max (V)': '1.0'},
    'scan_pts': {'Scan Points': '21'},
    'opx_trigger_do': {'OPX Trigger DO Channel': '1'},
}


def define_dataset():
    return 'Dataset'


def ensure_child_dataset(parent, name, data_type=Dataset, **kwargs):
    child = parent.children.get(name)
    if child is None:
        parent.add_child(name=name, data_type=data_type, **kwargs)
        child = parent.children[name]
    return child


def ensure_shared_trace(parent, name, data_type=Dataset, **kwargs):
    child = parent.children.get(name)
    if child is None:
        parent.add_child(name=name, data_type=data_type, new_plot=False, **kwargs)
        child = parent.children[name]
    return child


def gaussian_1d(x, A, x0, sigma, offset):
    return A * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2)) + offset


def estimate_gaussian_moments(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    offset = float(np.min(y))
    y_shift = y - offset
    y_shift[y_shift < 0] = 0

    total = np.sum(y_shift)
    if total <= 0:
        return 0.0, float(np.mean(x)), 1.0, offset

    x0 = float(np.sum(x * y_shift) / total)
    var = float(np.sum(y_shift * (x - x0) ** 2) / total)
    sigma = np.sqrt(max(var, 1e-12))
    A = float(np.max(y) - offset)
    return A, x0, sigma, offset


def fit_gaussian_1d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    A0, x00, sigma0, offset0 = estimate_gaussian_moments(x, y)

    try:
        p0 = [A0, x00, max(sigma0, 1.0), offset0]
        bounds = (
            [0.0, float(np.min(x)), 1e-6, -np.inf],
            [np.inf, float(np.max(x)), np.inf, np.inf]
        )
        popt, _ = curve_fit(
            gaussian_1d,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        A, x0, sigma, offset = popt
        sigma = abs(float(sigma))
    except Exception:
        A, x0, sigma, offset = A0, x00, sigma0, offset0

    fit_y = gaussian_1d(x, A, x0, sigma, offset)
    return {
        "A": float(A),
        "x0": float(x0),
        "sigma": float(sigma),
        "offset": float(offset),
        "fit_y": fit_y,
    }


def project_and_fit(frame):
    frame = np.asarray(frame, dtype=float)

    x_axis = np.arange(frame.shape[1], dtype=float)
    y_axis = np.arange(frame.shape[0], dtype=float)

    x_proj = np.sum(frame, axis=0)
    y_proj = np.sum(frame, axis=1)

    x_fit = fit_gaussian_1d(x_axis, x_proj)
    y_fit = fit_gaussian_1d(y_axis, y_proj)

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_fit": x_fit,
        "y_fit": y_fit,
    }


def overall_sigma_from_fit(fit):
    return 0.5 * (fit["x_fit"]["sigma"] + fit["y_fit"]["sigma"])


def get_cycle_timings(dataset):
    boot = int(dataset.get_input_parameter("mot_boot_up_time"))
    alive = int(dataset.get_input_parameter("mot_alive_time"))
    expansion = int(dataset.get_input_parameter("expansion_time"))
    dissolution = int(dataset.get_input_parameter("mot_dissolution_time"))
    dead = int(dataset.get_input_parameter("mot_dead_time"))

    t1 = boot
    coil_off = boot + alive
    t2 = coil_off + expansion
    t3 = t2 + dissolution
    cycle_length = t3 + dead + CAMERA_TRIGGER_WIDTH_MS

    return {
        "boot": boot,
        "alive": alive,
        "expansion": expansion,
        "dissolution": dissolution,
        "dead": dead,
        "t1": t1,
        "coil_off": coil_off,
        "t2": t2,
        "t3": t3,
        "cycle_length": cycle_length,
    }


def make_trigger_pulse(cycle_length, trigger_times):
    pulse = np.zeros(cycle_length, dtype=int)
    for trigger_time in trigger_times:
        start = max(int(trigger_time), 0)
        stop = min(start + CAMERA_TRIGGER_WIDTH_MS, cycle_length)
        pulse[start:stop] = 1
    return pulse.tolist()


def make_window_pulse(cycle_length, start, stop, high_value=1.0):
    pulse = np.zeros(cycle_length, dtype=float)
    pulse[max(start, 0):min(stop, cycle_length)] = high_value
    return pulse.tolist()


def make_triangle_scan_values(scan_min, scan_max, scan_pts):
    if scan_pts <= 1:
        return np.array([float(scan_min)])

    n_up = (scan_pts + 1) // 2
    n_down = scan_pts - n_up

    up = np.linspace(scan_min, scan_max, n_up)
    if n_down > 0:
        down = np.linspace(scan_max, scan_min, n_down + 1)[1:]
        return np.concatenate((up, down))
    return up


def concat_cycle_pulses(per_cycle_pulse, num_cycles):
    return np.tile(np.asarray(per_cycle_pulse), num_cycles).tolist()


def build_scan_parameter_pulse(cycle_length, scan_values):
    return np.concatenate([
        np.full(cycle_length, float(value), dtype=float)
        for value in scan_values
    ]).tolist()


def update_scan_plot(dataset, scan_values, metrics):
    scan_ds = dataset.children["Expansion Rate Scan"]
    bwd_ds = scan_ds.children["Backward Trace"]
    peak_index = int(np.argmax(scan_values))

    fwd_x = np.asarray(scan_values[:peak_index + 1], dtype=float)
    fwd_y = np.asarray(metrics[:peak_index + 1], dtype=float)
    bwd_x = np.asarray(scan_values[peak_index + 1:], dtype=float)
    bwd_y = np.asarray(metrics[peak_index + 1:], dtype=float)

    scan_ds.set_data(data=fwd_y, x=fwd_x)
    scan_ds.update()

    if len(bwd_x) > 0:
        bwd_ds.set_data(data=bwd_y, x=bwd_x)
        bwd_ds.update()
    else:
        bwd_ds.set_data(data=np.array([]), x=np.array([]))
        bwd_ds.update()


def ensure_plot_structure(dataset):
    image_names = (
        "MOT Image",
        "Expanded MOT Image",
        "Background Image",
    )
    for name in image_names:
        image_ds = ensure_child_dataset(
            dataset,
            name=name,
            data_type=Plot2D,
            min_x=0, max_x=2448, pts_x=2448,
            min_y=0, max_y=2048, pts_y=2048,
            new_plot=True
        )
        image_ds.graph.setLevels(*RAW_IMAGE_LEVELS)
        image_ds.graph.view.setAspectLocked(True)

    scan_ds = ensure_child_dataset(
        dataset,
        name="Expansion Rate Scan",
        data_type=Dataset,
        new_plot=True
    )
    ensure_shared_trace(
        scan_ds,
        name="Backward Trace",
        data_type=Dataset,
        color_index=1
    )

    ensure_child_dataset(
        dataset,
        name="Image 1 Sigma History",
        data_type=InfiniteRollingLine,
        data_length=SIGMA_HISTORY_LENGTH,
        new_plot=True
    )
    ensure_child_dataset(
        dataset,
        name="Image 2 Sigma History",
        data_type=InfiniteRollingLine,
        data_length=SIGMA_HISTORY_LENGTH,
        new_plot=True
    )


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    dataset.camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber']
    dataset.OPX_client = kwargs['OPX_OPX']

    ensure_plot_structure(dataset)
    dataset.graph.hide()


def experiment(**kwargs):
    thread = kwargs['thread']
    dataset = kwargs['dataset']
    logger = dataset.log

    dataset.NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    dataset.NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    dataset.NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
    dataset.camera_client = kwargs["fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber"]
    dataset.OPX_client = kwargs["OPX_OPX"]

    NI_card_1 = dataset.NI_card_1
    NI_card_2 = dataset.NI_card_2
    NI_card_3 = dataset.NI_card_3
    camera_client = dataset.camera_client
    OPX_client = dataset.OPX_client

    ensure_plot_structure(dataset)

    exposure_time = float(dataset.get_input_parameter("exposure_time"))
    camera_gain = float(dataset.get_input_parameter("camera_gain"))
    opx_trigger_do = int(dataset.get_input_parameter("opx_trigger_do"))
    camera_trigger_do = "dio" + str(int(dataset.get_input_parameter("camera_trigger_do")))
    magnetic_coils_do = "dio" + str(int(dataset.get_input_parameter("magnetic_coils_do")))
    magnetic_coils_ao = "ao" + str(int(dataset.get_input_parameter("magnetic_coils_ao")))
    scan_parameter_ao = "ao" + str(int(dataset.get_input_parameter("scan_parameter_ao")))

    scan_min = float(dataset.get_input_parameter("scan_min"))
    scan_max = float(dataset.get_input_parameter("scan_max"))
    scan_pts = int(dataset.get_input_parameter("scan_pts"))
    scan_values = make_triangle_scan_values(scan_min, scan_max, scan_pts)
    num_cycles = len(scan_values)

    ni_sample_rate = 1000
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"

    timings = get_cycle_timings(dataset)
    camera_trigger_pulse = make_trigger_pulse(
        timings["cycle_length"],
        (timings["t1"], timings["t2"], timings["t3"])
    )
    magnetic_coils_pulse = make_window_pulse(
        timings["cycle_length"],
        0,
        timings["coil_off"],
        high_value=1.0
    )
    full_camera_trigger_pulse = concat_cycle_pulses(camera_trigger_pulse, num_cycles)
    full_magnetic_coils_pulse = concat_cycle_pulses(magnetic_coils_pulse, num_cycles)
    full_scan_parameter_pulse = build_scan_parameter_pulse(timings["cycle_length"], scan_values)
    total_length = timings["cycle_length"] * num_cycles

    logger.info(
        f'Cycle timing: t1={timings["t1"]} ms, t2={timings["t2"]} ms, '
        f't3={timings["t3"]} ms, cycle_length={timings["cycle_length"]} ms, '
        f'num_cycles={num_cycles}, total_length={total_length} ms'
    )
    camera_client.set_hardware_trigger(
        line=trigger_line,
        activation=trigger_edge,
        selector="FrameStart",
        overlap="ReadOut",
        acquisition_mode="Continuous",
    )
    camera_client.set_exposure(exposure_time)
    camera_client.try_set_float("Gain", camera_gain)
    dataset.camera_client.start_acquisition()

    NI_card_1.arm_clock(length=total_length, sample_rate=ni_sample_rate)

    NI_card_2.build_stack()
    NI_card_2.set_do_voltage(
        do_channel=camera_trigger_do,
        value=full_camera_trigger_pulse,
        sample_rate=ni_sample_rate
    )
    NI_card_2.set_do_voltage(
        do_channel=magnetic_coils_do,
        value=full_magnetic_coils_pulse,
        sample_rate=ni_sample_rate
    )

    NI_card_3.build_stack()
    NI_card_3.set_ao_voltage(
        ao_channel=magnetic_coils_ao,
        voltages=full_magnetic_coils_pulse,
        sample_rate=ni_sample_rate
    )
    NI_card_3.set_ao_voltage(
        ao_channel=scan_parameter_ao,
        voltages=full_scan_parameter_pulse,
        sample_rate=ni_sample_rate
    )

    OPX_client.build_stack()
    clock_elem = OPX_client.create_new_do_elem(
        do_channel=opx_trigger_do,
        length=500
    )
    with OPX_client.for_("i", 0, total_length, 1):
        OPX_client.set_digital_voltage(element=clock_elem)
        OPX_client.delay(999500)

    h1 = NI_card_2.arm()
    h2 = NI_card_3.arm()
    OPX_client.execute()
    NI_card_2.finalize(h1, timeout=120.0)
    NI_card_3.finalize(h2, timeout=120.0)
    NI_card_1.finalize_clock()

    def get_frame(timeout_ms=1000):
        frame_bytes, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
        return np.frombuffer(frame_bytes, dtype=np.dtype(dtype)).reshape(shape)

    raw_frames = []
    expected_frames = 3 * num_cycles
    per_frame_timeout_ms = max(10000, 5 * timings["cycle_length"])
    try:
        for _ in range(expected_frames):
            raw_frames.append(get_frame(timeout_ms=per_frame_timeout_ms))
    finally:
        dataset.camera_client.stop_acquisition()

    sigma_1_values = []
    sigma_2_values = []
    sigma_diff_values = []

    for idx, scan_value in enumerate(scan_values):
        frame_1 = raw_frames[3 * idx]
        frame_2 = raw_frames[3 * idx + 1]
        frame_3 = raw_frames[3 * idx + 2]

        image_1_sub = frame_1.astype(np.int32) - frame_3.astype(np.int32)
        image_2_sub = frame_2.astype(np.int32) - frame_3.astype(np.int32)

        fit_1 = project_and_fit(image_1_sub)
        fit_2 = project_and_fit(image_2_sub)
        sigma_1 = overall_sigma_from_fit(fit_1)
        sigma_2 = overall_sigma_from_fit(fit_2)
        sigma_diff = sigma_2 - sigma_1

        sigma_1_values.append(sigma_1)
        sigma_2_values.append(sigma_2)
        sigma_diff_values.append(sigma_diff)

        logger.info(
            f'Scan point {idx + 1}/{num_cycles}, value={scan_value:.4f}: '
            f'image1 sigma_x={fit_1["x_fit"]["sigma"]:.2f}, sigma_y={fit_1["y_fit"]["sigma"]:.2f}, overall={sigma_1:.2f}; '
            f'image2 sigma_x={fit_2["x_fit"]["sigma"]:.2f}, sigma_y={fit_2["y_fit"]["sigma"]:.2f}, overall={sigma_2:.2f}; '
            f'delta={sigma_diff:.2f}'
        )

        dataset.children["Image 1 Sigma History"].set_data(sigma_1)
        dataset.children["Image 2 Sigma History"].set_data(sigma_2)

    dataset.children["Image 1 Sigma History"].update()
    dataset.children["Image 2 Sigma History"].update()

    last_frame_1 = raw_frames[-3]
    last_frame_2 = raw_frames[-2]
    last_frame_3 = raw_frames[-1]
    dataset.children["MOT Image"].set_data(last_frame_1)
    dataset.children["MOT Image"].update()
    dataset.children["Expanded MOT Image"].set_data(last_frame_2)
    dataset.children["Expanded MOT Image"].update()
    dataset.children["Background Image"].set_data(last_frame_3)
    dataset.children["Background Image"].update()

    update_scan_plot(dataset, scan_values, sigma_diff_values)
    thread.running = False
