from pathlib import Path
import numpy as np
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit

# These packages are imported here so that we don't have to import them in all the other files
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
import plotly.io as pio

pio.renderers.default = "browser"

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)
# Used to correct for IQ mixer imbalances


######################
# Network parameters #
######################
# IP address of the Quantum Orchestration Platform
qop_ip = "192.168.88.252"
cluster_name = "Cluster_1"


#############
# Save Path #
#############
# Path to save data
save_dir = Path(__file__).parent.resolve() / "Data"
save_dir.mkdir(exist_ok=True)

default_additional_files = {
    Path(__file__).name: Path(__file__).name,
    "optimal_weights.npz": "optimal_weights.npz",
}


scope_IF = 10 * u.MHz
const_len = 10000
dig_len = 1000000
const_amp = 0.5

config = {
    'version': 1,
    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0},
                3: {'offset': 0.0},
                4: {'offset': 0.0},
            },
            'digital_outputs': {
                1: {}
            },
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0}
            }
        }
    },
    'octaves': {},
    'elements': {
        'scope': {
            'singleInput': {'port': ('con1', 1)},
            'intermediate_frequency': scope_IF,
            'operations': {
                'const': 'const_pulse'
            }
        },
        "scope_dig": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
                #"activate2": {
                #    "port": ("con1", 10),
                #    "delay": 0,
                #    "buffer": 0,
                #},
            },
            "operations": {
                "ON": "ON_pulse",
            },
        }
    },
    'pulses': {
        'const_pulse': {
            'operation': 'control',
            'length': const_len,    # in nanoseconds, not clock cycles
            'waveforms': {
                'single': 'const'
            }
        },
        "ON_pulse": {
            "operation": "control",
            "length": dig_len,
            "digital_marker": "ON",
        }
    },
    'waveforms': {
        'const': {
            'type': 'constant',
            'sample': const_amp
        }
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    }
}
