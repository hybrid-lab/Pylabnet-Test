import numpy as np
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit

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


#####################
# OPX configuration #
#####################

# Frequencies
AOM_IF = 10 * u.MHz

# Gaussian pulse parameters
gauss_amp = 0.3  # The gaussian is used when calibrating pi and pi_half pulses
gauss_len = 100  # The gaussian is used when calibrating pi and pi_half pulses
gauss_wf, gauss_der_wf = drag_gaussian_pulse_waveforms(
    gauss_amp, gauss_len, gauss_len / 5, alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)

# Square pulse parameters
square_len = 100 # in ns
square_amp = 0.5 # in Volts

# Trigger pulse parameters
trigger_len = 320  # in ns

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
            "digital_outputs": {
                1: {},
                10: {},
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},
                2: {"offset": 0.0, "gain_db": 0},
            },
        },
    },
    "elements": {
        "AOM": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": AOM_IF,
            "operations": {
                "const": "square_pulse",
                "gauss": "gaussian_pulse",
            }
        },
        "AOM_2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": AOM_IF,
            "operations": {
                "const": "square_pulse",
                "gauss": "gaussian_pulse",
            }
        },
        "trigger": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "ON": "ON_pulse",
            },
        },
    },
    "pulses": {
        "ON_pulse": {
            "operation": "control",
            "length": trigger_len,
            "digital_marker": "ON",
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"single": "gaussian_wf"},
        },
        "square_pulse": {"operation": "control", "length": square_len, "waveforms": {"single": "square_wf"}},
    },
    "waveforms": {
        "square_wf": {"type": "constant", "sample": square_amp}
    },
}
