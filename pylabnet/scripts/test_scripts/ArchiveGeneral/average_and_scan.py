import numpy as np
import time
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
from pylabnet.hardware.quantum_machines.OPXdriverConfig import *
from pylabnet.scripts.data_center.datasets import *


# Compatibility shims for NumPy 2.x with older libs
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "complex"):
    np.complex = complex


# -------------------------
# UI inputs shown in DataTaker
# -------------------------
INIT_DICT = {
    'readout_len': {'Readout Length (ns)': '1000'},
    'avg_count': {'Pulses to Average': '10'},
    'min_gauss': {'Min Gauss Amp': '0.2'},
    'max_gauss': {'Max Gauss Amp': '0.3'},
    'gauss_length': {'Gauss Length': '100'},
    'pulse_frequency': {'Pulse Frequency (Mhz)': '0'},
    'sample_num': {'Samples per Direction of Scan': '3'},
    'take_data_rate': {'Update plot every __ seconds': '0.2'},
    'input_channel': {'Input Channel': '2'},
    'output_channel': {'Output Channel': '1'},
    'noise_sd': {'Noise Standard Deviation': '0.05'},

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
    # Use a vanilla parent that is immediately fully initialized
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.OPX_client = kwargs['OPX_OPX']
    measure_length = int(dataset.get_input_parameter('readout_len'))

    dataset.add_child(
        name='Triangle',
        data_type=TriangleScan1D,
        min=0.0, max=1.0, pts=measure_length,   # use 600 if you want 600 points per sweep
        new_plot=True
    )

    dataset.add_child(
        name='Noisy Pulse',
        data_type=InfiniteRollingLine,
        data_length=measure_length,
        new_plot=True
    )

    dataset.add_child(
        name='AveragedData',
        data_type=AveragedHistogram,
        new_plot=False
    )

    dataset.children['AveragedData'].add_child(
        name='Running Average',
        data_type=InfiniteRollingLine,
        data_length=measure_length,
        new_plot=True
    )

    tri = dataset.children['Triangle']
    # (Optional but explicit) ensure x matches pts/min/max
    tri.x = np.linspace(tri.min, tri.max, tri.pts)

    # Hide the parent plot if you want
    dataset.graph.hide()


def experiment(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    client = dataset.OPX_client

    dataset.num_samples = 0

    min_gauss_amp = dataset.get_input_parameter('min_gauss')
    max_gauss_amp = dataset.get_input_parameter('max_gauss')
    pulse_num_to_average = int(dataset.get_input_parameter('avg_count'))
    fuzzy_gauss_len = int(dataset.get_input_parameter('gauss_length'))
    samples_per_direction = int(dataset.get_input_parameter('sample_num'))
    measure_length = int(dataset.get_input_parameter('readout_len'))
    noise_sd = dataset.get_input_parameter('noise_sd')

    #We're going to be sweeping the voltage magnitude:
    voltage_array = np.concatenate((np.linspace(min_gauss_amp, max_gauss_amp, samples_per_direction), np.linspace(max_gauss_amp, min_gauss_amp, samples_per_direction)))

    tri = dataset.children['Triangle']
    bwd = tri.children['Bwd trace']   # this child is created by TriangleScan1D

    while thread.running:
        for i in range(samples_per_direction * 2):
            gauss_amp = voltage_array[i]
            for _ in range(pulse_num_to_average):
                dataset.num_samples += 1

                gauss_wf, _ = drag_gaussian_pulse_waveforms(
                    gauss_amp, fuzzy_gauss_len, fuzzy_gauss_len / 5, alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
                )

                rand_samples = np.random.normal(0, noise_sd, fuzzy_gauss_len)
                rand_pulse = gauss_wf + rand_samples

                client.build_stack()
                client.set_ao_voltage(
                    pulse="arbitrary",
                    ao_channel=int(dataset.get_input_parameter('output_channel')),
                    wave_function=rand_pulse,
                    frequency=dataset.get_input_parameter('pulse_frequency')
                )
                client.get_ai_voltage(ai_channel=int(dataset.get_input_parameter('input_channel')), length=measure_length, ao_channel=int(dataset.get_input_parameter('input_channel')))
                data_batch = client.execute()

                if data_batch and 'raw_adc_1_in2' in data_batch:
                    measurements = np.asarray([it[1] for it in data_batch['raw_adc_1_in2']])
                    dataset.children['AveragedData'].set_data(measurements)
                    dataset.children['Noisy Pulse'].set_data(measurements)

                    if dataset.children['AveragedData'].data is not None:
                        total_data = (measurements + dataset.children['AveragedData'].data) / dataset.num_samples

                    else:
                        total_data = measurements / dataset.num_samples

                    dataset.children['AveragedData'].children['Running Average'].set_data(total_data)

            data = dataset.children['AveragedData'].data / dataset.num_samples
            dataset.log.error(f"data {data.size}")
            for point in data:
                if (i < samples_per_direction):
                    tri.set_data(point)     # forward
                else:
                    bwd.set_data(point)     # backward

            dataset.num_samples = 0

            dataset.children['AveragedData'].children['Running Average'].set_data([])
            dataset.children['AveragedData'].buffer = []
            dataset.children['Noisy Pulse'].buffer = []
            dataset.children['AveragedData'].data = np.zeros(measure_length)
            dataset.children['Noisy Pulse'].set_data([])

        time.sleep(dataset.get_input_parameter('take_data_rate'))
