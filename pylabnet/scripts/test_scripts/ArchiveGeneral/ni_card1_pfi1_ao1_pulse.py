from pylabnet.hardware.ni_daqs.nidaqmx_card import Driver


DEVICE_NAME = "PXI1Slot2_1"
AO_CHANNEL = "ao1"
TRIGGER_LINE = "PFI1"

SAMPLE_RATE_HZ = 1000
LOW_VOLTAGE = 0.0
HIGH_VOLTAGE = 1.0

PRE_SAMPLES = 1000
HIGH_SAMPLES = 2000
POST_SAMPLES = 2000


def main():
    ni = Driver(device_name=DEVICE_NAME)

    waveform = (
        [LOW_VOLTAGE] * PRE_SAMPLES
        + [HIGH_VOLTAGE] * HIGH_SAMPLES
        + [LOW_VOLTAGE] * POST_SAMPLES
    )

    ni.build_stack()
    ni.not_use_OPX_clock()
    ni.set_trigger(target="ao", trig_line=TRIGGER_LINE)
    ni.set_ao_voltage(
        ao_channel=AO_CHANNEL,
        voltages=waveform,
        sample_rate=SAMPLE_RATE_HZ,
    )

    print(
        f"Arming {DEVICE_NAME} on {AO_CHANNEL} with start trigger on {TRIGGER_LINE}. "
        "Waiting for rising edge..."
    )
    handle = ni.arm()
    ni.finalize(handle, timeout=30.0)
    print("Pulse complete.")


if __name__ == "__main__":
    main()
