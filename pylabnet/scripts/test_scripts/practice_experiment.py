import numpy as np
from PyQt5 import QtCore
from rpyc.utils.classic import obtain

from pylabnet.hardware.quantum_machines.OPX import Driver as OPX

from pylabnet.scripts.data_center.take_data import *


# Optional helpers (kept for parity with the original template even if unused)
from pylabnet.launchers.siv_py_functions import upload_sequence, load_config

# QUA / QM imports are retained for parity; not directly used here
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from pylabnet.hardware.quantum_machines.OPXdriverConfigmultelemsperchannel import *
from pylabnet.scripts.data_center.datasets import *

INIT_DICT = {
    'amp': {'Amplitude': '0.2'},
    'pulse_len': {'Pulse Length': '100'},
    'readout_len': {'Readout Length (ns)': '1000'},
    'output_channel': {'Pulse Output Channel': '1'},
    "measure_output_channel": {'Measure Pulse Output Channel': '2'},
    'measure_input_channel': {'Measure Input Channel': '2'},

    # Triangle scan parameters (optional; if not provided, Dataset shows a popup)
    # 'scan_min': {'Min (x)': '0.0'},
    # 'scan_max': {'Max (x)': '1.0'},
    # 'scan_pts': {'Points': '200'},
    'blank1': {'filler': '0'},
    'blank2': {'filler': '0'},
    'blank3': {'filler': '0'},
    'blank4': {'filler': '0'},
    'blank5': {'filler': '0'},
    'blank6': {'filler': '0'},
    'blank7': {'filler': '0'},
}


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.OPX_client = kwargs['OPX_OPX']
    measure_length = int(dataset.get_input_parameter('readout_len'))

    dataset.add_child(
        name='Measured Pulse',
        data_type=InfiniteRollingLine,
        data_length=measure_length,
        new_plot=True
    )


def experiment(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    client = dataset.OPX_client
    measure_length = int(dataset.get_input_parameter('readout_len'))
    amplitude = float(dataset.get_input_parameter('amp'))
    pulse_length = int(dataset.get_input_parameter('pulse_len'))
    output_channel = int(dataset.get_input_parameter('output_channel')) #output channel is channel where ao_output comes from for the pulse we're trying to measure
    measure_output_channel = int(dataset.get_input_parameter('measure_output_channel'))
    measure_input_channel = int(dataset.get_input_parameter('measure_input_channel'))

    while thread.running:

        client.build_stack()
        client.set_ao_voltage(pulse="const", length=pulse_length, amplitude=amplitude, ao_channel=output_channel)
        client.get_ai_voltage(ao_channel=measure_output_channel, ai_channel=measure_input_channel, length=measure_length)
        data_batch = client.execute()
        measurements = np.asarray([it[1] for it in data_batch['raw_adc_1_in2']])
        dataset.children['Measured Pulse'].set_data(measurements)
