import numpy as np
import time
from PyQt5 import QtCore
from rpyc.utils.classic import obtain

from pylabnet.scripts.data_center.take_data import ExperimentThread
from pylabnet.scripts.data_center.datasets import SawtoothScan1D, ErrorBarGraph, InfiniteRollingLine, Dataset, SawtoothScan1D_array_update

from pylabnet.launchers.siv_py_functions import upload_sequence, load_config

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel import *


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
        NI_client = kwargs['ni_daq_1_PXI1Slot2_1']
        dataset.NI_client = NI_client

    except Exception as e:
        # This will catch ANY error and print it to the log
        dataset.log.error(f"An error occurred in CONFIGURE: {e}")
        # Re-raise the exception to make sure the script stops
        raise


def experiment(**kwargs):
    """The main experiment loop that runs when you click 'Run'."""

    thread = kwargs['thread']
    dataset = kwargs['dataset']

    # Main loop to fetch and plot data
    while thread.running:

        data_batch = dataset.NI_client.get_ai_voltage(ai_channel="ai0", num_samples=1000)

        dataset.log.error(f"DATA FETCHED")

        # If data was fetched, process and plot it
        if data_batch is not None and len(data_batch) > 0:

            dataset.log.error("DATA BATCH: " + repr(data_batch))

            # Extract the measurement values (first element of each tuple)
            measurements = [item[1] for item in data_batch]

            # Average the batch of points and plot the result
            avg_value = np.mean(measurements)
            for point in measurements:
                dataset.set_data(point)

        # A short pause to control the plot update rate
        time.sleep(dataset.get_input_parameter('take_data_rate'))
