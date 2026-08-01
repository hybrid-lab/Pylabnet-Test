import time
from nidaqmx.constants import Edge

# Optional INIT_DICT if your framework expects it
INIT_DICT = {
    'blank1': {'filler': '0'},
}


def define_dataset():
    """Specifies the type of plot to use for the data."""
    return 'InfiniteRollingLine'


def experiment(**kwargs):
    """
    Triggered digital output experiment using the provided NI Driver.

    What it does:
      - Builds a DO waveform on a chosen DIO line
      - Arms a digital start trigger on PFI0 (rising edge)
      - Calls execute() -> task starts and waits for trigger
      - After trigger, outputs: LOW -> HIGH -> LOW with hardware timing

    Wiring:
      Trigger source TTL OUT  -> NI PFI0
      Trigger source GND      -> NI DGND

      NI DO (chosen DIO line) -> your device TTL input / scope
      NI DGND                 -> your device ground / scope ground
    """

    thread = kwargs["thread"]
    dataset = kwargs.get("dataset", None)

    # Use whichever your framework actually passes.
    # If you have a direct kwarg client, use that. Otherwise fall back to dataset.NI_client.
    ni_2 = kwargs["nidaqmx_ni_daq_2"]
    ni_1 = kwargs["nidaqmx_ni_daq_1"]
    dataset.OPX = kwargs["OPX_OPX"]
    if ni_2 is None:
        if dataset is None or getattr(dataset, "NI_client", None) is None:
            raise RuntimeError("NI client not found in kwargs['ni_daq_2_PXI1Slot4_1'] or dataset.NI_client")
        ni_2 = dataset.NI_client_2

    dataset.NI_client_1 = ni_1

    # ---------------------------
    # Choose DO line + timing
    # ---------------------------
    do_channel = "dio1"       # uses your dioX -> portY/lineZ mapping
    sample_rate = 1000     # 100 kHz => 10 us/sample

    # Pulse shape: LOW 50 ms, HIGH 10 ms, LOW 50 ms
    low1_s = 0.500
    high_s = 0.300
    low2_s = 0.050

    low1_n = int(low1_s * sample_rate)
    high_n = int(high_s * sample_rate)
    low2_n = int(low2_s * sample_rate)

    waveform = ([0] * low1_n) + ([1] * high_n) + ([0] * low2_n)

    # Main loop
    while thread.running:
        ni_1.arm_clock(sample_rate=1000, length=1000)
        ni_2.build_stack()
        # Queue DO waveform (note: your API uses `value=...`, not `voltages=...`)
        ni_2.set_do_voltage(
            do_channel=do_channel,
            value=waveform,
            sample_rate=sample_rate,
        )
        ni_2.not_use_OPX_clock()
        OPX_client = dataset.OPX
        OPX_client.build_stack()

        clock_elem = OPX_client.create_new_do_elem(
            do_channel=1,
            length=500
        )
        N = 1100
        with OPX_client.for_("i", 0, N, 1):
            OPX_client.set_digital_voltage(
                element=clock_elem
            )
            OPX_client.delay(999500)

        # Arm trigger on PFI0 for DO task
        # ni.set_trigger(
        #     target="do",
        #     trig_line="PFI0",
        # )

        h = ni_2.arm()
        OPX_client.execute()

        ni_2.finalize(h, 30)
        ni_1.finalize_clock()

        # Optional: print debug metadata
        # print(out.get("_meta", {}))

        # Prevent tight looping/rearming
        time.sleep(0.2)
