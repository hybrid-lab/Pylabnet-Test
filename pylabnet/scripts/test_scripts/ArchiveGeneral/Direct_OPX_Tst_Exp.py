import numpy as np
import time
from PyQt5 import QtCore


from pylabnet.scripts.data_center.take_data import ExperimentThread
from pylabnet.scripts.data_center.datasets import SawtoothScan1D, ErrorBarGraph, InfiniteRollingLine, Dataset, SawtoothScan1D_array_update

from pylabnet.launchers.siv_py_functions import upload_sequence, load_config

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from pylabnet.hardware.quantum_machines.OPXdriverConfig import *

INIT_DICT = {
    'readout_len': {'Readout Length (ns)': '1000'},
    'avg_count': {'Points to Average': '10'}
}


def define_dataset():
    """Specifies the type of plot to use for the data."""
    return 'InfiniteRollingLine'


def configure(**kwargs):
    """Sets up the hardware and the plot before the experiment runs."""\

    try:
        dataset = kwargs['dataset']
        logger = dataset.log # Get the logger for printing messages

        # Assumes your OPX client is named 'opx_qm' in the data_taker config
        ##Ideally we should be running everything through the driver through the client which is at kwargs[OPX_OPX]
        logger.info("Connecting to QM Manager client...")

        #qm_manager = kwargs[OPX_OPX]
        logger.info("Successfully connected to QM Manager client.")

        # Define the QUA program for continuous measurement
        with program() as realtime_measurement:
            # Declare a stream to send data from the OPX to the computer
            adc_stream = declare_stream(adc_trace=True)
            readout_len_cycles = int(dataset.get_input_parameter('readout_len') / 4) # This is not used if length is in config

            play("const", "generic_ai_elem_ch1")

            # This loop runs forever on the OPX
            # Measure the input and stream the raw ADC data
            measure("readout", "generic_output_elem_ch1_to_ch1", adc_stream)

            with stream_processing():
                # Save the raw ADC data with timestamps
                adc_stream.input1().with_timestamps().save_all("adc_stream")

        # Store the program and connect to the Quantum Machine
        dataset.qua_program = realtime_measurement

        logger.info("Opening Quantum Machine with provided config...")
        # This is the line that is likely failing:
        logger.info("Successfully opened Quantum Machine.")

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
    """The main experiment loop that runs when you click 'Run'."""

    thread = kwargs['thread']
    dataset = kwargs['dataset']

    # Execute the QUA program. It will run in the background on the OPX.
    qm_manager = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

    qm = qm_manager.open_qm(config)

    # Main loop to fetch and plot data
    while thread.running:
        job = qm.execute(dataset.qua_program)

    # Get the handle to the data stream
        adc_handle = job.result_handles

        #dataset.log.error(f"RUNNING {job.is_paused()}")

        # Fetch the latest data available in the stream
        adc_handle.wait_for_all_values() # Wait until there's at least 1 new data point
        #dataset.log.error(f"1 DATA VAL")

        data_batch = adc_handle.get('adc_stream').fetch_all()
        #dataset.log.error(f"DATA FETCHED")

        # If data was fetched, process and plot it
        if data_batch is not None and len(data_batch) > 0:

            dataset.log.error(f"DATA BATCH: {data_batch}")

            # Extract the measurement values (first element of each tuple)
            measurements = [item[0] for item in data_batch[0][0]]

            dataset.log.error(f"measurements: {measurements}")

            # Average the batch of points and plot the result
            avg_value = np.mean(measurements)
            dataset.set_data(avg_value)
            # rolling_dataset = dataset.children['Real-time ADC']
            # rolling_dataset.set_children_data()

        # A short pause to control the plot update rate
        time.sleep(0.02)
