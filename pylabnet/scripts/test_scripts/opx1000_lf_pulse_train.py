"""Finite LF analog pulse train for scope verification; hardware output by default."""

import numpy as np

from pylabnet.scripts.data_center.opx1000_test_helpers import (
    flag, integer, prepare_plot_updates, report_error, request_plot_update, wait_for_job,
)

INIT_DICT = {
    "fem_slot": {"LF FEM slot (3 or 4)": "3"},
    "ao_channel": {"Analog output (1-8)": "1"},
    "amplitude_v": {"High voltage (V into 50 ohms)": "0.1"},
    "width_ns": {"Pulse high time (ns)": "1000"},
    "period_ns": {"Pulse period (ns)": "10000"},
    "pulse_count": {"Number of pulses": "1000000"},
    "simulate": {"Simulate only: 1=yes, 0=hardware output": "0"},
    "preview_ns": {"Waveform preview length (ns)": "16000"},
}


def define_dataset():
    return "Dataset"


def configure(**kwargs):
    dataset = kwargs["dataset"]
    dataset.opx = kwargs["OPX1000_OPX1000"]
    prepare_plot_updates(dataset)
    dataset.graph.setLabel("bottom", "Time", units="ns")
    dataset.graph.setLabel("left", "Output voltage", units="V")
    dataset.graph.setTitle(
        "QOP simulation preview" if flag(dataset, "simulate") else "Commanded output preview (verify on scope)"
    )


def experiment(**kwargs):
    dataset, thread = kwargs["dataset"], kwargs["thread"]
    opx = dataset.opx
    try:
        if not thread.running:
            return
        slot, channel = integer(dataset, "fem_slot"), integer(dataset, "ao_channel")
        width, period = integer(dataset, "width_ns"), integer(dataset, "period_ns")
        count = integer(dataset, "pulse_count")
        amplitude = float(dataset.get_input_parameter("amplitude_v"))
        simulate = flag(dataset, "simulate")
        preview = integer(dataset, "preview_ns")
        if preview < 16 or preview % 4 or preview > 1000000:
            raise ValueError("Preview must be 16..1000000 ns, divisible by 4")
        preview = min(preview, count * period)
        opx.build_stack()
        opx.pulse_train(
            channel, count, period, amplitude=amplitude, length=width, fem_slot=slot,
        )
        info = opx.execute(wait=False, simulate=simulate, duration=preview)
        if not wait_for_job(opx, thread, count * period * 1e-9 + 60):
            return
        if simulate:
            result = opx.get_simulated_samples()["con1"]
            port = "{}-{}".format(slot, channel)
            voltage = np.asarray(result["analog"][port])
            rate = result["analog_sampling_rate"][port]
            dataset.set_data(voltage, x=np.arange(len(voltage)) * 1e9 / rate)
            dataset.log.info("Simulation preview; no physical pulse train was emitted")
        else:
            # Commanded waveform preview, not a measurement of the physical output.
            time_ns = np.arange(preview)
            voltage = amplitude * ((time_ns % period) < width)
            dataset.set_data(voltage, x=time_ns)
            dataset.log.info(
                "Physical train completed: {} pulses, period {} ns, high time {} ns. "
                "Plot is the commanded preview; verify the actual output on a scope.".format(
                    count, period, width,
                )
            )
        dataset.log.update_metadata(opx1000_pulse_train={
            "job": info, "fem_slot": slot, "ao_channel": channel,
            "amplitude_v": amplitude, "width_ns": width, "period_ns": period,
            "pulse_count": count, "requested_duration_s": count * period * 1e-9,
        })
    except Exception as error:
        opx.close()
        report_error(kwargs, error)
    finally:
        request_plot_update(dataset)
        thread.running = False
