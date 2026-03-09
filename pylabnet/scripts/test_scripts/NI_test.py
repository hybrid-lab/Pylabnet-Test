import numpy as np
from PyQt5 import QtCore
from rpyc.utils.classic import obtain


from pylabnet.hardware.quantum_machines.OPX import Driver as OPX


from pylabnet.scripts.data_center.take_data import ExperimentThread
from pylabnet.scripts.data_center.datasets import SawtoothScan1D, ErrorBarGraph, InfiniteRollingLine, Dataset, SawtoothScan1D_array_update

from pylabnet.launchers.siv_py_functions import upload_sequence, load_config

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel import *
from pylabnet.network.client_server.nidaqmx_card import Client

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

        NI_client = kwargs['nidaqmx_ni_daq_1']
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
    """The main experiment loop that runs when you click 'Run'."""

    ramp_min = 0
    ramp_max = 3
    N = 10000
    ramp = np.linspace(ramp_min, ramp_max, N, dtype=np.float64)

    thread = kwargs['thread']
    dataset = kwargs['dataset']

    # Main loop to fetch and plot data
    while thread.running:
        ni = dataset.NI_client
        # dataset.NI_client.set_ao_voltage(ao_channel="ao1", voltages=1)
        ni.build_stack()
        ni.not_use_OPX_clock()
        ni.set_ao_voltage(ao_channel="ao0", voltages=ramp, sample_rate=100)
        h = ni.arm()
        ni.finalize(h, timeout=30)
