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
    ni = kwargs["nidaqmx_ni_daq_2"]
    if ni is None:
        if dataset is None or getattr(dataset, "NI_client", None) is None:
            raise RuntimeError("NI client not found in kwargs['ni_daq_2_PXI1Slot4_1'] or dataset.NI_client")
        ni = dataset.NI_client

    # ---------------------------
    # Choose DO line + timing
    # ---------------------------
    do_channel = "dio1"       # uses your dioX -> portY/lineZ mapping
    sample_rate = 100_000     # 100 kHz => 10 us/sample

    # Pulse shape: LOW 50 ms, HIGH 10 ms, LOW 50 ms
    low1_s = 0.050
    high_s = 0.010
    low2_s = 0.050

    low1_n = int(low1_s * sample_rate)
    high_n = int(high_s * sample_rate)
    low2_n = int(low2_s * sample_rate)

    waveform = ([0] * low1_n) + ([1] * high_n) + ([0] * low2_n)

    # Main loop
    while thread.running:
        ni.build_stack()

        # Queue DO waveform (note: your API uses `value=...`, not `voltages=...`)
        ni.set_do_voltage(
            do_channel=do_channel,
            value=waveform,
            sample_rate=sample_rate,
        )

        # Arm trigger on PFI0 for DO task
        ni.set_trigger(
            target="do",
            trig_line="PFI0",
        )

        # Arms task; it will WAIT until the trigger edge arrives on PFI0
        ni.execute()

        # Optional: print debug metadata
        # print(out.get("_meta", {}))

        # Prevent tight looping/rearming
        time.sleep(0.2)
