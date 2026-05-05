import numpy as np
import time

from pylabnet.scripts.data_center.datasets import (
    Plot2DWithAvg,
    Plot2D,
    Dataset,
    InfiniteRollingLine,
)
from qt_plotting import QtMatplotlibFrameViewer

from scipy.optimize import curve_fit


if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

FRAME1_LEVELS = (0, 20)
FRAME2_LEVELS = (0, 20)
DIFF_LEVELS = (0, 20)
AVG_DIFF_LEVELS = (0, 20)

# Waist history trace colors:
# color 0 -> Frame 1 X
# color 1 -> Frame 1 Y
# color 2 -> Frame 2 X
# color 3 -> Frame 2 Y
WAIST_HISTORY_ROOT = "Waist History"

INIT_DICT = {
    'imaging_AOM_start': {'Imaging AOM Start Time (ms)': '1'},
    'imaging_AOM_end': {'Imaging AOM End Time (ms)': '500'},
    'frame_1': {'Camera Frame 1 Time (ms)': '400'},
    'frame_2': {'Camera Frame 2 Time (ms)': '1000'},
    'wait_time': {'Wait Time Between Cycles (s)': '3'},
    'exposure_time': {'Camera Exposure Time (us)': '501'},
    'camera_gain': {'Camera Gain': '50'},
    'camera_trigger_do': {'Camera Trigger DO Channel': '0'},
    'imaging_AOM_do': {'Imaging AOM DO Channel': '1'},
    'imaging_AOM_ao': {'Imaging AOM AO Channel': '1'},
    'opx_trigger_do': {'OPX Trigging DO Channel': '1'},
}


def define_dataset():
    return 'Dataset'


def normalize_window_state(gui, window_name):
    window = gui.windows.get(window_name)
    if window is not None and not hasattr(window, "tabs_enabled"):
        window.tabs_enabled = False


def ensure_child_dataset(parent, name, data_type=Dataset, **kwargs):
    child = parent.children.get(name)
    if child is None:
        window_name = kwargs.get("window")
        if window_name is not None:
            normalize_window_state(parent.gui, window_name)
        parent.add_child(name=name, data_type=data_type, **kwargs)
        child = parent.children[name]
    return child


def ensure_shared_trace(parent, name, data_type=Dataset, **kwargs):
    child = parent.children.get(name)
    if child is None:
        parent.add_child(name=name, data_type=data_type, new_plot=False, **kwargs)
        child = parent.children[name]
    return child


def ensure_plot_structure(dataset):
    # Main image plots
    ensure_child_dataset(
        dataset,
        name="Frame 1",
        data_type=Plot2D,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    ensure_child_dataset(
        dataset,
        name="Frame 2",
        data_type=Plot2D,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    ensure_child_dataset(
        dataset,
        name="Image Difference",
        data_type=Plot2DWithAvg,
        min_x=0, max_x=2448, pts_x=2448,
        min_y=0, max_y=2048, pts_y=2048,
        new_plot=True
    )

    # Gaussian fit plots
    for name in (
        "Frame 1 X Fit",
        "Frame 1 Y Fit",
        "Frame 2 X Fit",
        "Frame 2 Y Fit",
    ):
        ensure_child_dataset(
            dataset,
            name=name,
            data_type=Dataset,
            new_plot=True,
            window="gaussian_fits",
            window_title="Gaussian Fits"
        )

    # Waist history
    waist_root = ensure_child_dataset(
        dataset,
        name=WAIST_HISTORY_ROOT,
        data_type=InfiniteRollingLine,
        data_length=1000,
        new_plot=True,
        window="waist_history",
        window_title="Gaussian Waist History",
        color_index=0
    )
    ensure_shared_trace(
        waist_root,
        name="Frame 1 Waist Y",
        data_type=InfiniteRollingLine,
        data_length=1000,
        color_index=1
    )
    ensure_shared_trace(
        waist_root,
        name="Frame 2 Waist X",
        data_type=InfiniteRollingLine,
        data_length=1000,
        color_index=2
    )
    ensure_shared_trace(
        waist_root,
        name="Frame 2 Waist Y",
        data_type=InfiniteRollingLine,
        data_length=1000,
        color_index=3
    )

    frame1_ds = dataset.children["Frame 1"]
    frame2_ds = dataset.children["Frame 2"]
    diff_ds = dataset.children["Image Difference"]
    avg_ds = diff_ds.children["Image Differencecurrentavg"]

    frame1_ds.graph.setLevels(*FRAME1_LEVELS)
    frame2_ds.graph.setLevels(*FRAME2_LEVELS)
    diff_ds.graph.setLevels(*DIFF_LEVELS)
    avg_ds.graph.setLevels(*AVG_DIFF_LEVELS)

    frame1_ds.graph.view.setAspectLocked(True)
    frame2_ds.graph.view.setAspectLocked(True)
    diff_ds.graph.view.setAspectLocked(True)
    avg_ds.graph.view.setAspectLocked(True)


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
        "x_proj": x_proj,
        "y_proj": y_proj,
        "x_fit": x_fit,
        "y_fit": y_fit,
    }


def estimate_2d_gaussian_height(fit):
    amp_from_x = fit["x_fit"]["A"] / (np.sqrt(2.0 * np.pi) * fit["y_fit"]["sigma"])
    amp_from_y = fit["y_fit"]["A"] / (np.sqrt(2.0 * np.pi) * fit["x_fit"]["sigma"])
    return 0.5 * (amp_from_x + amp_from_y)


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.frame_viewer = QtMatplotlibFrameViewer("Most recent camera frame")

    camera_client = kwargs['fluorescence_imaging_camera_bfs_u3_51s5m']
    dataset.camera_client = camera_client

    NI_card_1 = kwargs['nidaqmx_ni_daq_1']
    dataset.NI_card_1 = NI_card_1
    NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_2 = NI_card_2
    NI_card_3 = kwargs['nidaqmx_ni_daq_3']
    dataset.NI_card_3 = NI_card_3

    ensure_plot_structure(dataset)

    dataset.graph.hide()


def experiment(**kwargs):
    thread = kwargs['thread']
    dataset = kwargs['dataset']
    logger = dataset.log

    dataset.NI_card_1 = kwargs["nidaqmx_ni_daq_1"]
    dataset.NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    dataset.NI_card_3 = kwargs["nidaqmx_ni_daq_3"]
    dataset.camera_client = kwargs["fluorescence_imaging_camera_bfs_u3_51s5m"]
    dataset.OPX_client = kwargs["OPX_OPX"]

    OPX_client = dataset.OPX_client
    NI_card_1 = dataset.NI_card_1
    NI_card_2 = dataset.NI_card_2
    NI_card_3 = dataset.NI_card_3
    camera_client = dataset.camera_client

    ensure_plot_structure(dataset)

    imaging_AOM_start = int(dataset.get_input_parameter("imaging_AOM_start"))
    imaging_AOM_end = int(dataset.get_input_parameter("imaging_AOM_end"))
    frame_1 = int(dataset.get_input_parameter("frame_1"))
    frame_2 = int(dataset.get_input_parameter("frame_2"))
    wait_time = float(dataset.get_input_parameter("wait_time"))
    exposure_time = float(dataset.get_input_parameter("exposure_time"))
    camera_gain = float(dataset.get_input_parameter("camera_gain"))

    opx_trigger_do = int(dataset.get_input_parameter("opx_trigger_do"))
    camera_trigger_do = "dio" + str(int(dataset.get_input_parameter("camera_trigger_do")))
    imaging_AOM_do = "dio" + str(int(dataset.get_input_parameter("imaging_AOM_do")))
    imaging_AOM_ao = "ao" + str(int(dataset.get_input_parameter("imaging_AOM_ao")))

    ni_sample_rate = 1000
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"

    camera_ttl_up = 1

    down_time = frame_2 - frame_1 - camera_ttl_up
    camera_trigger_pulse = [0] * frame_1 + [1] * camera_ttl_up + [0] * down_time + [1] * camera_ttl_up
    imaging_AOM_pulse = [0] * imaging_AOM_start + [1] * (imaging_AOM_end - imaging_AOM_start) + [0] * 100

    experiment_lenght_ms = max(int(len(camera_trigger_pulse) * 1.01), int(len(imaging_AOM_pulse) * 1.01))
    buffer = 0

    while thread.running:
        camera_client.set_hardware_trigger(
            line=trigger_line,
            activation=trigger_edge,
            selector="FrameStart",
            overlap="ReadOut",
            acquisition_mode="Continuous",
        )
        camera_client.set_exposure(exposure_time)
        camera_client.try_set_float("Gain", camera_gain)
        logger.info("Starting acquisition")
        dataset.camera_client.start_acquisition()

        NI_card_1.arm_clock(length=experiment_lenght_ms + buffer, sample_rate=ni_sample_rate)
        logger.info("Clock configured")

        NI_card_2.build_stack()
        NI_card_2.set_do_voltage(
            do_channel=camera_trigger_do,
            value=camera_trigger_pulse,
            sample_rate=ni_sample_rate
        )
        NI_card_2.set_do_voltage(
            do_channel=imaging_AOM_do,
            value=imaging_AOM_pulse,
            sample_rate=ni_sample_rate
        )

        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(
            ao_channel=imaging_AOM_ao,
            voltages=imaging_AOM_pulse,
            sample_rate=ni_sample_rate
        )

        OPX_client.build_stack()

        clock_elem = OPX_client.create_new_do_elem(
            do_channel=opx_trigger_do,
            length=500
        )
        N = experiment_lenght_ms
        with OPX_client.for_("i", 0, N, 1):
            OPX_client.set_digital_voltage(element=clock_elem)
            OPX_client.delay(999500)

        h1 = NI_card_2.arm()
        h2 = NI_card_3.arm()
        OPX_client.execute()
        NI_card_2.finalize(h1, timeout=120.0)
        NI_card_3.finalize(h2, timeout=120.0)
        NI_card_1.finalize_clock()

        def get_frame(timeout_ms=1000):
            b, shape, dtype = dataset.camera_client.get_frame_bytes(timeout_ms)
            logger.info(f"{shape}")
            return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

        try:
            logger.info("Requesting one frame")
            frame1 = get_frame(timeout_ms=10000)
            frame2 = get_frame(timeout_ms=100000)
            logger.info(
                f"Got frame1: shape={frame1.shape}, dtype={frame1.dtype}, "
                f"min={frame1.min()}, max={frame1.max()}"
            )
            logger.info(
                f"Got frame2: shape={frame2.shape}, dtype={frame2.dtype}, "
                f"min={frame2.min()}, max={frame2.max()}"
            )
        finally:
            logger.info("Stopping acquisition")
            dataset.camera_client.stop_acquisition()

        diff = frame1.astype(np.int32) - frame2.astype(np.int32)

        # Image plots
        frame1_ds = dataset.children["Frame 1"]
        frame2_ds = dataset.children["Frame 2"]
        diff_ds = dataset.children["Image Difference"]

        frame1_ds.set_data(frame1)
        frame1_ds.update()

        frame2_ds.set_data(frame2)
        frame2_ds.update()

        diff_ds.set_data(diff)
        diff_ds.update()

        diff_ds.graph.setLevels(*DIFF_LEVELS)
        frame1_ds.graph.setLevels(*FRAME1_LEVELS)
        frame2_ds.graph.setLevels(*FRAME2_LEVELS)

        avg = diff_ds.children["Image Differencecurrentavg"]
        avg.graph.setLevels(*AVG_DIFF_LEVELS)
        avg.graph.ui.histogram.autoHistogramRange = False

        # Gaussian fits from projections
        fit1 = project_and_fit(frame1)
        fit2 = project_and_fit(frame2)
        fit1_height = estimate_2d_gaussian_height(fit1)
        fit2_height = estimate_2d_gaussian_height(fit2)
        logger.info(
            f'Frame 1 fit: sigma_x={fit1["x_fit"]["sigma"]:.2f}, '
            f'sigma_y={fit1["y_fit"]["sigma"]:.2f}, '
            f'height={fit1_height:.2f}'
        )
        logger.info(
            f'Frame 2 fit: sigma_x={fit2["x_fit"]["sigma"]:.2f}, '
            f'sigma_y={fit2["y_fit"]["sigma"]:.2f}, '
            f'height={fit2_height:.2f}'
        )
        logger.info(f'datasets.children {list(dataset.children.keys())}')
        f1x_ds = dataset.children["Frame 1 X Fit"]
        f1y_ds = dataset.children["Frame 1 Y Fit"]
        f2x_ds = dataset.children["Frame 2 X Fit"]
        f2y_ds = dataset.children["Frame 2 Y Fit"]

        # plot measured projections with Gaussian fit overlaid in same window
        # measured data
        f1x_ds.set_data(data=fit1["x_proj"], x=fit1["x_axis"])
        f1x_ds.update()

        f1x_fit_overlay = f1x_ds.children.get("fit_curve", None)
        if f1x_fit_overlay is None:
            f1x_ds.add_child(
                name="fit_curve",
                data_type=Dataset,
                new_plot=False,
                color_index=1
            )
        f1x_ds.children["fit_curve"].set_data(data=fit1["x_fit"]["fit_y"], x=fit1["x_axis"])
        f1x_ds.children["fit_curve"].update()

        f1y_ds.set_data(data=fit1["y_proj"], x=fit1["y_axis"])
        f1y_ds.update()
        if "fit_curve" not in f1y_ds.children:
            f1y_ds.add_child(
                name="fit_curve",
                data_type=Dataset,
                new_plot=False,
                color_index=1
            )
        f1y_ds.children["fit_curve"].set_data(data=fit1["y_fit"]["fit_y"], x=fit1["y_axis"])
        f1y_ds.children["fit_curve"].update()

        f2x_ds.set_data(data=fit2["x_proj"], x=fit2["x_axis"])
        f2x_ds.update()
        if "fit_curve" not in f2x_ds.children:
            f2x_ds.add_child(
                name="fit_curve",
                data_type=Dataset,
                new_plot=False,
                color_index=1
            )
        f2x_ds.children["fit_curve"].set_data(data=fit2["x_fit"]["fit_y"], x=fit2["x_axis"])
        f2x_ds.children["fit_curve"].update()

        f2y_ds.set_data(data=fit2["y_proj"], x=fit2["y_axis"])
        f2y_ds.update()
        if "fit_curve" not in f2y_ds.children:
            f2y_ds.add_child(
                name="fit_curve",
                data_type=Dataset,
                new_plot=False,
                color_index=1
            )
        f2y_ds.children["fit_curve"].set_data(data=fit2["y_fit"]["fit_y"], x=fit2["y_axis"])
        f2y_ds.children["fit_curve"].update()

        # Waist history: keep each frame separate
        f1_wx = dataset.children[WAIST_HISTORY_ROOT]
        f1_wy = f1_wx.children["Frame 1 Waist Y"]
        f2_wx = f1_wx.children["Frame 2 Waist X"]
        f2_wy = f1_wx.children["Frame 2 Waist Y"]

        f1_wx.set_data(fit1["x_fit"]["sigma"])
        f1_wx.update()

        f1_wy.set_data(fit1["y_fit"]["sigma"])
        f1_wy.update()

        f2_wx.set_data(fit2["x_fit"]["sigma"])
        f2_wx.update()

        f2_wy.set_data(fit2["y_fit"]["sigma"])
        f2_wy.update()

        time.sleep(wait_time)
