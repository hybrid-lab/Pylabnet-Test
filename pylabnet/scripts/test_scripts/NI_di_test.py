import time
import numpy as np

#INIT_DICT is copied from NI_test experiment and does nothing at all
INIT_DICT = {
    'readout_len': {'Readout Length (ns)': '1000'},
    'avg_count': {'Points to Average': '10'},
    'take_data_rate': {'Update plot every __ seconds': '0.2'},
    'output_voltage': {'Output Voltage': '0.5'},
    'input_channel': {'Input Channel': '1'},
    'blank1': {'filler': '0'},
    'blank2': {'filler': '0'},
    'blank3': {'filler': '0'},
    'blank4': {'filler': '0'},
}


def define_dataset():
    """Specifies the type of plot to use for the data."""
    return 'InfiniteRollingLine'


def configure(**kwargs):
    """Sets up the hardware and the plot before the experiment runs."""\

    try:
        dataset = kwargs['dataset']
        logger = dataset.log # Get the logger for printing messages
        logger.error(f"Kwargs{kwargs}")

        NI_client = kwargs['nidaqmx_ni_daq_2']
        dataset.NI_client = NI_client

        # Add a child dataset for the plot
        # dataset.add_child(
        #     name='Real-time ADC',
        #     data_type=InfiniteRollingLine, # Use a rolling plot
        #     x_label='Timestamp (a.u.)',
        #     y_label='ADC Reading (a.u.)'
        # )
        # # Give the child dataset a more accessible name
        # dataset.adc_plot = dataset.children['Real-time ADC']

    except Exception as e:
        # This will catch ANY error and print it to the log
        dataset.log.error(f"An error occurred in CONFIGURE: {e}")
        # Re-raise the exception to make sure the script stops
        raise


def experiment(**kwargs):
    """
    Uses our NI driver to output a *triggered* digital waveform (TTL pulse train).

    Wiring (typical):
      - Trigger source TTL OUT  -> NI PFI0
      - Trigger source GND      -> NI DGND
      - NI DO line (e.g. port0/line0) -> your device TTL input / scope
      - NI DGND -> your device ground / scope ground
    """

    thread = kwargs["thread"]
    dataset = kwargs["dataset"]

    # ---- Waveform parameters ----
    sample_rate = 100_000  # 100 kHz -> 10 us per sample
    di_channel = "dio1"  # change to whatever DO line you want

    # Build a single pulse: LOW 10 ms, HIGH 2 ms, LOW 10 ms
    # (You can change these durations easily.)
    low1_samps = int(0.200 * sample_rate)  # 200 ms
    high_samps = int(1 * sample_rate)  # 200 ms
    low2_samps = int(0.200 * sample_rate)  # 200 ms

    do_waveform = ([0] * low1_samps) + ([1] * high_samps) + ([0] * low2_samps)

    while thread.running:
        ni = dataset.NI_client
        ni.build_stack()
        ni.get_di_state(
            di_channel="line0",
            sample_rate=100000,
            num_samples=100,
            port="port0"
        )
        ni.set_trigger(
            target="di",
            trig_line="PFI0"
        )
        data_batch = ni.execute()
        data_batch = data_batch["di_1"]

        dataset.log.error(f"DATA FETCHED")

        # # If data was fetched, process and plot it
        if data_batch is not None and len(data_batch) > 0:

            dataset.log.error("DATA BATCH: " + repr(data_batch))

            # Extract the measurement values (first element of each tuple)
            measurements = [item for item in data_batch]

            dataset.log.error(f"measurements: {measurements}")

            # Average the batch of points and plot the result
            for point in measurements:
                dataset.set_data(point)
        #dataset.set_data(avg_value)
        # rolling_dataset = dataset.children['Real-time ADC']
        # rolling_dataset.set_children_data()

        # Optional: slow down loop so you don’t re-arm immediately
        time.sleep(1.0)
