from pylabnet.scripts.data_center.datasets import Dataset, TriangleScan1D
import numpy as np
import time
from PyQt5 import QtCore
from rpyc.utils.classic import obtain

from pylabnet.hardware.quantum_machines.OPX import Driver as OPX

from pylabnet.scripts.data_center.take_data import ExperimentThread
from pylabnet.scripts.data_center.datasets import TriangleScan1D

# Optional helpers (kept for parity with the original template even if unused)
from pylabnet.launchers.siv_py_functions import upload_sequence, load_config

# QUA / QM imports are retained for parity; not directly used here
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from pylabnet.hardware.quantum_machines.OPXdriverConfig import *


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
    'avg_count': {'Points to Average': '10'},
    'take_data_rate': {'Update plot every __ seconds': '0.2'},
    'output_voltage': {'Output Voltage': '0.5'},
    'input_channel': {'Input Channel': '1'},
    # Triangle scan parameters (optional; if not provided, Dataset shows a popup)
    # 'scan_min': {'Min (x)': '0.0'},
    # 'scan_max': {'Max (x)': '1.0'},
    # 'scan_pts': {'Points': '200'},
    'blank1': {'filler': '0'},
    'blank2': {'filler': '0'},
    'blank3': {'filler': '0'},
    'blank4': {'filler': '0'},
}

# In your experiment script


def define_dataset():
    # Use a vanilla parent that is immediately fully initialized
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.OPX_client = kwargs['OPX_OPX']

    dataset.add_child(
        name='Triangle',
        data_type=TriangleScan1D,
        min=0.0, max=1.0, pts=600,   # use 600 if you want 600 points per sweep
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

    samples_per_direction = 10
    #We're going to be sweeping the voltage magnitude:
    voltage_array = np.concatenate((np.linspace(1 / samples_per_direction, 0.5, samples_per_direction), np.linspace(0.5, 1 / samples_per_direction, samples_per_direction)))

    # a running counter of how many points we've sent this session
    k = getattr(dataset, "_scan_idx", 0)

    tri = dataset.children['Triangle']
    bwd = tri.children['Bwd trace']   # this child is created by TriangleScan1D

    while thread.running:
        voltage_val = voltage_array[(k // tri.pts) % (samples_per_direction * 2)]

        client.build_stack()
        client.set_ao_voltage(pulse="const",
                              ao_channel=dataset.get_input_parameter('input_channel'),
                              amplitude=voltage_val,
                              frequency=100)
        client.get_ai_voltage(ai_channel=2, ao_channel=2, length=600)   # returns 600 samples per loop
        data_batch = client.execute()

        if data_batch and 'raw_adc_1_in2' in data_batch:
            measurements = [it[1] for it in data_batch['raw_adc_1_in2']]

            for point in measurements:
                # decide direction based on how many full sweeps we've completed
                # even sweeps -> forward, odd sweeps -> backward
                if (k // (tri.pts * samples_per_direction)) % 2 == 0:
                    tri.set_data(point)     # forward
                else:
                    bwd.set_data(point)     # backward

                k += 1

        dataset._scan_idx = k
        time.sleep(dataset.get_input_parameter('take_data_rate'))
