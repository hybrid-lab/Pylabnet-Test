"""View repeated LF input waveforms in DataTaker; counting is disabled.

Each refresh is a separate, untriggered ADC capture, not a segment of a
continuous recording. The plot and saved dataset retain the latest trace.
External SNSPD bias is managed separately; this script queues no AO pulses.
"""

import math
import time
import numpy as np

from pylabnet.scripts.data_center.opx1000_test_helpers import (
    integer, prepare_plot_updates, report_error, request_plot_update, wait_for_job,
)

INIT_DICT = {
    "fem_slot": {"LF FEM slot (3 or 4)": "3"},
    "ai_channel": {"Analog input (1 or 2)": "1"},
    "view_s": {"Raw waveform viewing duration (s)": "10"},
    "trace_ns": {"Trace duration per capture (ns, max 16000)": "10000"},
    "pause_s": {"Pause between captures (s)": "0.2"},
}


def define_dataset():
    return "Dataset"


def configure(**kwargs):
    dataset = kwargs["dataset"]
    dataset.opx = kwargs["OPX1000_OPX1000"]
    prepare_plot_updates(dataset)
    dataset.graph.setLabel("bottom", "Time within capture", units="ns")
    dataset.graph.setLabel("left", "Raw ADC code")
    dataset.graph.setTitle("LF input: latest raw waveform")


def experiment(**kwargs):
    dataset, thread = kwargs["dataset"], kwargs["thread"]
    opx = dataset.opx
    started = time.monotonic()
    try:
        if not thread.running:
            return
        slot, channel = integer(dataset, "fem_slot"), integer(dataset, "ai_channel")
        trace_ns = integer(dataset, "trace_ns")
        view_s = float(dataset.get_input_parameter("view_s"))
        pause_s = float(dataset.get_input_parameter("pause_s"))
        if not math.isfinite(view_s) or view_s <= 0:
            raise ValueError("view_s must be a positive finite number")
        if not math.isfinite(pause_s) or pause_s < 0:
            raise ValueError("pause_s must be finite and >= 0")
        if trace_ns < 16 or trace_ns > 16000 or trace_ns % 4:
            raise ValueError("Trace duration must be 16..16000 ns, divisible by 4")

        captures = 0
        first_frame = deadline = None
        dataset.log.info(
            "Viewing LF slot {} input {} for about {} s using separate {} ns captures".format(
                slot, channel, view_s, trace_ns,
            )
        )
        # Start the viewing interval after the first completed capture, so the
        # initial connection/compilation does not consume all the viewing time.
        while thread.running and (deadline is None or time.monotonic() < deadline):
            opx.build_stack()
            opx.get_ai_voltage(channel, length=trace_ns, fem_slot=slot, label="trace")
            opx.execute(wait=False)
            if not wait_for_job(opx, thread, 60):
                break
            trace = opx.get_results()["trace"]
            raw = np.asarray(trace["raw_adc"])
            dataset.set_data(raw, x=np.arange(raw.size) * 1e9 / trace["sample_rate_hz"])
            captures += 1
            if first_frame is None:
                first_frame = time.monotonic()
                deadline = first_frame + view_s
            request_plot_update(dataset)

            # Actual refresh spacing also includes RPC/compilation/acquisition
            # time. An in-flight capture may finish after the viewing deadline.
            pause_until = min(deadline, time.monotonic() + pause_s)
            while thread.running:
                remaining = pause_until - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))

        viewed_s = 0.0 if first_frame is None else time.monotonic() - first_frame
        dataset.log.update_metadata(opx1000_raw_waveform={
            "fem_slot": slot, "ai_channel": channel,
            "requested_view_s": view_s, "viewed_s": viewed_s,
            "elapsed_s": time.monotonic() - started,
            "trace_ns": trace_ns, "pause_s": pause_s, "captures": captures,
            "captured_live_time_s": captures * trace_ns * 1e-9,
            "continuous": False, "retained_data": "latest capture only",
            "stopped_by_user": not thread.running,
        })
        dataset.log.info(
            "Displayed {} raw captures over {:.2f} s viewing; latest trace retained. "
            "There are unrecorded gaps between captures.".format(captures, viewed_s)
        )
    except Exception as error:
        opx.close()
        report_error(kwargs, error)
    finally:
        request_plot_update(dataset)
        thread.running = False
