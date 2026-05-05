import numpy as np
from itertools import product

from pylabnet.scripts.data_center.datasets import (
    Dataset,
    TriangleScan1D,
    Plot2D,
)
from qt_plotting import QtMatplotlibFrameViewer


# NumPy compatibility for older pylabnet codepaths
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


# ============================================================
# USER SETTINGS
# ============================================================
# Number of scan dimensions. Intended range: 1 to 4.
SCAN_DIM = 2

# Per-axis scan definitions. Only first SCAN_DIM entries are used.
# Triangular means the fastest axis runs forward on one line, backward on the next,
# then forward again, etc., so motion is continuous instead of sawtooth reset.
SCAN_AXES = [
    {"name": "AOM 1", "min": 0.0, "max": 1.0, "pts": 21},
    {"name": "AOM 2", "min": 0.0, "max": 1.0, "pts": 15},
    {"name": "AOM 3", "min": 0.0, "max": 1.0, "pts": 9},
    {"name": "AOM 4", "min": 0.0, "max": 1.0, "pts": 7},
]

# Which NI device/client and channels to use for the scan parameters.
# This assumes the scan parameters are set as analog outputs on NI_card_1.
# Adjust this mapping to your real hardware.
SCAN_OUTPUT_DEVICE = "nidaqmx_ni_daq_2"
SCAN_OUTPUT_CHANNELS = ["ao0", "ao1", "ao2", "ao3"]
SCAN_OUTPUT_SAMPLE_RATE = 1000000

# If True, use sum(diff) inside ROI. Otherwise sum whole image.
USE_ROI = False
ROI_X0 = 1000
ROI_X1 = 1400
ROI_Y0 = 800
ROI_Y1 = 1200


# ============================================================
# DATATAKER INPUTS
# ============================================================
INIT_DICT = {
    'imaging_AOM_start': {'Imaging AOM Start Time (us)': '1000'},
    'imaging_AOM_end': {'Imaging AOM End Time (us)': '2000'},
    'frame_1': {'Camera Frame 1 Time (us)': '1000'},
    'frame_2': {'Camera Frame 2 Time (us)': '1000'},
    'wait_time': {'Wait Time Between Cycles (s)': '0.0'},
    'camera_trigger_do': {'Camera Trigger DO Channel': '0'},
    'imaging_AOM_do': {'Imaging AOM DO Channel': '0'},
    'opx_trigger_do': {'OPX Trigger DO Channel': '1'},
    'blank1': {'filler': '0'},
    'blank2': {'filler': '0'},
    'blank3': {'filler': '0'},
    'blank4': {'filler': '0'},
}


def define_dataset():
    return 'Dataset'


# ============================================================
# HELPERS
# ============================================================
def _axis_values(axis_cfg):
    return np.linspace(axis_cfg["min"], axis_cfg["max"], axis_cfg["pts"])


def _triangular_grid(axis_cfgs):
    """
    Generate an N-dimensional triangular raster path.

    Convention:
    - axis 0 = fastest axis
    - whenever any slower-axis index changes, the direction of axis 0 flips
    - for d=1 this reduces to a forward scan, then the attached TriangleScan1D
      backward child is fed on alternate completed lines/repetitions.

    Returns a list of tuples:
        [((i0, i1, ...), (v0, v1, ...)), ...]
    """
    arrays = [_axis_values(cfg) for cfg in axis_cfgs]
    shapes = [len(a) for a in arrays]

    if len(arrays) == 1:
        return [((i,), (arrays[0][i],)) for i in range(shapes[0])]

    out = []
    slower_ranges = [range(n) for n in shapes[1:]]

    for slower_idx in product(*slower_ranges):
        parity = sum(slower_idx) % 2
        fast_iter = range(shapes[0]) if parity == 0 else range(shapes[0] - 1, -1, -1)

        for i0 in fast_iter:
            full_idx = (i0,) + tuple(slower_idx)
            vals = tuple(arrays[ax][full_idx[ax]] for ax in range(len(arrays)))
            out.append((full_idx, vals))

    return out


def _set_scan_outputs(dataset, values):
    """
    Writes the current scan-point values to NI analog outputs.
    Assumes static/constant AO write via your custom nidaqmx client.
    """
    scan_client = dataset.scan_client
    scan_client.build_stack()

    for ch, val in zip(dataset.scan_output_channels, values):
        scan_client.set_ao_voltage(
            ao_channel=ch,
            value=[float(val)],
            sample_rate=dataset.scan_output_sample_rate,
        )

    h = scan_client.arm()
    scan_client.finalize(h, timeout=30.0)


def _run_atomic_imaging_trial(dataset):
    """
    One trial with the same structure as your atomic imaging script:
      AOM on / trigger once / AOM off / trigger again / diff = frame2 - frame1

    Returns:
      score, diff_image, frame_1, frame_2
    """
    logger = dataset.log
    camera_client = dataset.camera_client
    NI_card_2 = dataset.NI_card_2
    OPX_client = dataset.OPX_client

    imaging_AOM_start = int(dataset.get_input_parameter("imaging_AOM_start"))
    imaging_AOM_end = int(dataset.get_input_parameter("imaging_AOM_end"))
    frame_1_t = int(dataset.get_input_parameter("frame_1"))
    frame_2_t = int(dataset.get_input_parameter("frame_2"))

    opx_trigger_do = int(dataset.get_input_parameter("opx_trigger_do"))
    camera_trigger_do = "dio" + str(int(dataset.get_input_parameter("camera_trigger_do")))
    imaging_AOM_do = "dio" + str(int(dataset.get_input_parameter("imaging_AOM_do")))

    ni_sample_rate = 1000000
    trigger_line = "Line0"
    trigger_edge = "RisingEdge"
    experiment_length_us = 10000
    camera_ttl_up = 50

    camera_trigger_pulse_1 = [0] * frame_1_t + [1] * camera_ttl_up
    camera_trigger_pulse_2 = [0] * frame_2_t + [1] * camera_ttl_up
    imaging_AOM_pulse = [0] * imaging_AOM_start + [1] * (imaging_AOM_end - imaging_AOM_start)

    camera_client.set_hardware_trigger(
        line=trigger_line,
        activation=trigger_edge,
        selector="FrameStart",
        overlap="ReadOut",
        acquisition_mode="Continuous",
    )
    camera_client.start_acquisition()

    NI_card_2.build_stack()
    NI_card_2.set_do_voltage(
        do_channel=camera_trigger_do,
        value=camera_trigger_pulse_1,
        sample_rate=ni_sample_rate,
    )
    NI_card_2.set_do_voltage(
        do_channel=camera_trigger_do,
        value=camera_trigger_pulse_2,
        sample_rate=ni_sample_rate,
    )
    NI_card_2.set_do_voltage(
        do_channel=imaging_AOM_do,
        value=imaging_AOM_pulse,
        sample_rate=ni_sample_rate,
    )

    OPX_client.build_stack()
    clock_elem = OPX_client.create_new_do_elem(
        do_channel=opx_trigger_do,
        length=500,
    )
    N = experiment_length_us
    with OPX_client.for_("i", 0, N, 1):
        OPX_client.set_digital_voltage(element=clock_elem)
        OPX_client.delay(500)

    h1 = NI_card_2.arm()
    OPX_client.execute()
