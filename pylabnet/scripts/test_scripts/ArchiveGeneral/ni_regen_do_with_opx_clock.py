import time

import nidaqmx
import numpy as np
from nidaqmx.constants import (
    AcquisitionType,
    LineGrouping,
    RegenerationMode,
)

from pylabnet.hardware.quantum_machines.OPX import Driver as OPXDriver


if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]


CLOCK_DEVICE = "PXI1Slot2_1"
CLOCK_INPUT = f"/{CLOCK_DEVICE}/PFI0"
CLOCK_EXPORT = f"/{CLOCK_DEVICE}/PXI_Trig7"

PULSE_DEVICE = "PXI1Slot4_1"
PULSE_DO_CHANNEL = f"{PULSE_DEVICE}/port0/line5"
PULSE_CLOCK_SOURCE = f"/{PULSE_DEVICE}/PXI_Trig7"

OPX_DEVICE = "OPX"
OPX_CLOCK_DO = 1
OPX_TTL_PULSE_NS = 500

SAMPLE_RATE_HZ = 1000
LOW_SAMPLES_1 = 10
HIGH_SAMPLES = 20
LOW_SAMPLES_2 = 20
NUM_CYCLES = 10


def main():
    sample_period_ns = int(round(1e9 / SAMPLE_RATE_HZ))
    delay_ns = sample_period_ns - OPX_TTL_PULSE_NS
    if delay_ns < 0:
        raise ValueError("SAMPLE_RATE_HZ is too high for a 500 ns OPX digital pulse")

    waveform = (
        [False] * LOW_SAMPLES_1
        + [True] * HIGH_SAMPLES
        + [False] * LOW_SAMPLES_2
    )
    samples_per_cycle = len(waveform)
    total_clock_ticks = samples_per_cycle * NUM_CYCLES

    opx = OPXDriver(device_name=OPX_DEVICE)

    with nidaqmx.Task() as clock_task, nidaqmx.Task() as do_task:
        # Slot 2 task exists only to receive the OPX-fed clock on PFI0 and
        # export that sample clock onto the PXI backplane.
        clock_task.ao_channels.add_ao_voltage_chan(f"{CLOCK_DEVICE}/ao0")
        clock_task.timing.cfg_samp_clk_timing(
            rate=float(SAMPLE_RATE_HZ),
            source=CLOCK_INPUT,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=samples_per_cycle,
        )
        clock_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        clock_task.write([0.0] * samples_per_cycle, auto_start=False)
        clock_task.export_signals.samp_clk_output_term = CLOCK_EXPORT

        # Slot 4 plays the digital pulse pattern from the exported PXI clock.
        do_task.do_channels.add_do_chan(
            PULSE_DO_CHANNEL,
            line_grouping=LineGrouping.CHAN_PER_LINE,
        )
        do_task.timing.cfg_samp_clk_timing(
            rate=float(SAMPLE_RATE_HZ),
            source=PULSE_CLOCK_SOURCE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=samples_per_cycle,
        )
        do_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        do_task.write(waveform, auto_start=False)

        opx.build_stack()
        clock_elem = opx.create_new_do_elem(
            do_channel=OPX_CLOCK_DO,
            length=OPX_TTL_PULSE_NS,
            delay=0,
            buffer=0,
        )
        with opx.for_("i", 0, total_clock_ticks, 1):
            opx.set_digital_voltage(
                element=clock_elem,
                pulse="ON",
                do_channel=OPX_CLOCK_DO,
                length=OPX_TTL_PULSE_NS,
                delay=0,
                buffer=0,
                simulate=False,
            )
            opx.delay(delay_ns, elements=[clock_elem])

        print(
            f"Starting regenerative DO on {PULSE_DO_CHANNEL} using clock from {CLOCK_INPUT} "
            f"exported by {CLOCK_DEVICE} to {CLOCK_EXPORT} for {NUM_CYCLES} cycles."
        )
        clock_task.start()
        do_task.start()
        opx.execute(wait=True, timeout=30.0)

        time.sleep(0.05)
        do_task.stop()
        clock_task.stop()

    print("Finished running regenerative NI digital pulse train.")


if __name__ == "__main__":
    main()
