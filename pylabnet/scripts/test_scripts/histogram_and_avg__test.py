import numpy as np
import time
from PyQt5 import QtCore
from rpyc.utils.classic import obtain

from pylabnet.hardware.quantum_machines.OPX import Driver as OPX

from pylabnet.scripts.data_center.take_data import ExperimentThread
from pylabnet.scripts.data_center.datasets import *


# Optional helpers (kept for parity with the original template even if unused)
from pylabnet.launchers.siv_py_functions import upload_sequence, load_config

# QUA / QM imports are retained for parity; not directly used here
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from pylabnet.hardware.quantum_machines.OPXdriverConfig import *

# -------------------------
# UI inputs shown in DataTaker
# -------------------------
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
    """Tell DataTaker which base dataset class to instantiate."""
    return 'AveragedHistogram'


def configure(**kwargs):
    """Sets up the hardware/client handle on the dataset before the run."""
    try:
        dataset = kwargs['dataset']
        logger = dataset.log
        OPX_client = kwargs['OPX_OPX']
        dataset.OPX_client = OPX_client

        # Add a live rolling trace (second graph) just like the original stack script
        # It plots the last N samples while storing the full stream internally.
        # Choose a window size that matches the ADC batch (2000 points).
        if 'Rolling Trace' not in dataset.children:
            dataset.add_child(
                name='Rolling Trace',
                data_type=InfiniteRollingLine,
                data_length=2000,
                new_plot=True
            )
        if 'Running Average' not in dataset.children:
            dataset.add_child(
                name='Running Average',
                data_type=InfiniteRollingLine,
                data_length=1000,
                new_plot=True
            )

        dataset.graph.hide()

        logger.info("Configuring AveragedHistogram experiment; OPX client attached.")
    except Exception as e:
        dataset.log.error(f"CONFIGURE failed: {e}")
        raise


def experiment(**kwargs):
    """Main acquisition loop: fetch a batch from OPX and add it into the running histogram."""
    thread = kwargs['thread']
    dataset = kwargs['dataset']
    dataset.num_samples = 0

    gauss_amp = 0.3
    fuzzy_gauss_len = 400

    gauss_wf, gauss_der_wf = drag_gaussian_pulse_waveforms(
        gauss_amp, fuzzy_gauss_len, fuzzy_gauss_len / 5, alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
    )

    while thread.running:
        # client = dataset.OPX_client
        dataset.num_samples += 1

        # # Build an OPX command stack analogous to the original script
        # client.build_stack()
        # client.set_ao_voltage(
        #     "const",
        #     dataset.get_input_parameter('input_channel'),
        #     dataset.get_input_parameter('output_voltage')
        # )
        # # Fetch a batch of ADC samples on ai channel 2, length 600 (mirrors original)
        # client.get_ai_voltage(2, 600)

        # # Execute the stack and retrieve results
        # data_batch = client.execute()
        # dataset.log.info("OPX batch acquired for AveragedHistogram")
        rand_samples = np.random.normal(0, 0.05, fuzzy_gauss_len)
        rand_pulse = gauss_wf + rand_samples

        client = dataset.OPX_client

        client.build_stack()
        client.set_ao_voltage(pulse="arbitrary", ao_channel=dataset.get_input_parameter('input_channel'), wave_function=rand_pulse, length=fuzzy_gauss_len, frequency=0)
        client.get_ai_voltage(ai_channel=2, ao_channel=2, length=1000)
        data_batch = client.execute()

        if data_batch is not None and len(data_batch) > 0:
            # Original script indexes 'raw_adc_1_in2' and takes [1] from each tuple
            try:
                measurements = [item[1] for item in data_batch['raw_adc_1_in2']]
            except Exception as e:
                dataset.log.error(f"Unexpected data format from OPX: {type(data_batch)} keys={list(data_batch.keys()) if hasattr(data_batch,'keys') else 'n/a'} error={e}")
                measurements = []

            measure_array = np.asarray(measurements, dtype=float)

            if dataset.data is not None:
                total_data = (measure_array + dataset.data) / dataset.num_samples

            else:
                total_data = measure_array / dataset.num_samples

            if len(measurements) > 0:
                # AveragedHistogram expects a full histogram-like array;
                # it will add this batch into its running sum.
                dataset.set_data(measure_array)
                dataset.children['Rolling Trace'].set_data(measure_array)

                dataset.children['Running Average'].set_data(total_data)
                # Optionally also trigger child mappings if any
                dataset.set_children_data()

        # Throttle UI/plot updates
        time.sleep(dataset.get_input_parameter('take_data_rate'))
