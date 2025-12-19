import numpy as np
import time
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
        OPX_client = kwargs['OPX_OPX']
        dataset.OPX_client = OPX_client

        logger.info("Connecting to QM Manager client...")

        #qm_manager = kwargs[OPX_OPX]
        logger.info("Successfully connected to QM Manager client.")

        # Add a child dataset for the plot
        dataset.add_child(
            name='Dependent Pulse',
            data_type=InfiniteRollingLine, # Use a rolling plot
        )
        dataset.add_child(
            name='I0 Value',
            data_type=InfiniteRollingLine, # Use a rolling plot

        )
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

    # Main loop to fetch and plot data
    while thread.running:
        c = dataset.OPX_client
        c.build_stack()
        with c.for_("i", 0, 10, 1):
            c.get_ai_voltage(length=500, ai_channel=1, ao_channel=1) # DO NOT MEASURE DIGITAL PULSE DIRECTLY
            c.set_ao_voltage(length=300, pulse="const", amplitude=0.4, ao_channel=3, frequency=50)

        # with c.if_gt("I0", -0.041248):
        #     c.get_ai_voltage(length=500, amplitude=0.4, ao_channel=2, ai_channel=2, frequency=0)
        #     c.set_ao_voltage(length=100, pulse="const", amplitude=0.4, ao_channel=3, frequency=50)
        #     with c.if_gt("I0", -0.041246):
        #         c.get_ai_voltage(length=500, amplitude=0.4, ao_channel=2, ai_channel=2, frequency=0)

        #         el1 = c.create_new_ao_elem(length=100, pulse="const", amplitude=0.4, ao_channel=3, frequency=10)
        #         c.delay(length=200, elements=[el1])
        #         c.set_ao_voltage(element=el1)

        # with c.elif_gt("I0", -0.041251):

        #     c.get_ai_voltage(length=500, amplitude=0.4, ao_channel=2, ai_channel=2, frequency=0)

        #     el = c.create_new_ao_elem(length=100, pulse="const", amplitude=0.4, ao_channel=3, frequency=0)
        #     c.delay(length=200, elements=[el])
        #     c.set_ao_voltage(element=el)

        # with c.else_():
        #     c.get_ai_voltage(length=500, amplitude=0.2, ao_channel=3, ai_channel=2, frequency=0)

        data_batch = c.execute()
        dataset.log.error(f"DATA FETCHED")

        # If data was fetched, process and plot it
        if data_batch is not None and len(data_batch) > 0:

            dataset.log.error("DATA BATCH: " + repr(data_batch))

            # Extract the measurement values (first element of each tuple)
            measurements = [item[1] for item in data_batch['raw_adc_1_in1']]
            i = 2
            measurements2 = None

            while True:
                key = f'raw_adc_{i}_in2'
                if key in data_batch and len(data_batch[key]) > 0:
                    measurements2 = [item[1] for item in data_batch[key]]
                    break
                i += 1

                # safety stop (optional)
                if i > 20:
                    raise KeyError("No raw_adc_*_in2 key found in data_batch")

            I_value = data_batch["I0"]

            # dataset.log.error(f"measurements: {measurements}")

            # Average the batch of points and plot the result
            avg_value = np.mean(measurements)

            dataset.set_data(np.asarray(measurements))
            dataset.children["Dependent Pulse"].set_data(np.asarray(measurements2))
            dataset.children["I0 Value"].set_data(I_value)
            # dataset.set_data(avg_value)
            # rolling_dataset = dataset.children['Real-time ADC']
            # rolling_dataset.set_children_data()

        # A short pause to control the plot update rate
        time.sleep(dataset.get_input_parameter('take_data_rate'))


def custom_body(q, cfg):
    I = q.declare(fixed)
    Q = q.declare(fixed)

    measure("readout_pulse", "readout_elem", None,
            dual_demod.full("cos", "sin", I, Q))

    with q.if_(I > 0.1):
        play("pi_pulse", "qubit")
    with q.else_():
        play("pi_half_pulse", "qubit")
