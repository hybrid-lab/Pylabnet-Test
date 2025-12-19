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
qop_ip = "192.168.88.251"
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

#####################
# OPX configuration #
#####################


#####################
# ARBITRARY PARAMS #
#####################

set_length_ch1 = 100
set_length_ch2 = 100
set_length_ch3 = 100
set_length_ch4 = 100
set_length_ch5 = 100
set_length_ch6 = 100
set_length_ch7 = 100
set_length_ch8 = 100
set_length_ch9 = 100
set_length_ch10 = 100

measure_length_ch1 = 100
measure_length_ch2 = 100


arbitrary_wf_ch1 = [0.3] * set_length_ch1
arbitrary_wf_ch2 = [0.3] * set_length_ch2
arbitrary_wf_ch3 = [0.3] * set_length_ch3
arbitrary_wf_ch4 = [0.3] * set_length_ch4
arbitrary_wf_ch5 = [0.3] * set_length_ch5
arbitrary_wf_ch6 = [0.3] * set_length_ch6
arbitrary_wf_ch7 = [0.3] * set_length_ch7
arbitrary_wf_ch8 = [0.3] * set_length_ch8
arbitrary_wf_ch9 = [0.3] * set_length_ch9
arbitrary_wf_ch10 = [0.3] * set_length_ch10

square_amp_ch1 = 0.2
square_amp_ch2 = 0.2
square_amp_ch3 = 0.3
square_amp_ch4 = 0.3
square_amp_ch5 = 0.3
square_amp_ch6 = 0.3
square_amp_ch7 = 0.3
square_amp_ch8 = 0.3
square_amp_ch9 = 0.3
square_amp_ch10 = 0.3

gauss_amp_ch1 = 0.3
gauss_amp_ch2 = 0.3
gauss_amp_ch3 = 0.3
gauss_amp_ch4 = 0.3
gauss_amp_ch5 = 0.3
gauss_amp_ch6 = 0.3
gauss_amp_ch7 = 0.3
gauss_amp_ch8 = 0.3
gauss_amp_ch9 = 0.3
gauss_amp_ch10 = 0.3

CHANNEL1_IF = 10 * u.MHz
CHANNEL2_IF = 10 * u.MHz
CHANNEL3_IF = 10 * u.MHz
CHANNEL4_IF = 10 * u.MHz
CHANNEL5_IF = 10 * u.MHz
CHANNEL6_IF = 10 * u.MHz
CHANNEL7_IF = 10 * u.MHz
CHANNEL8_IF = 10 * u.MHz
CHANNEL9_IF = 10 * u.MHz
CHANNEL10_IF = 10 * u.MHz

#Creates Gauss waveforms for each
gauss_wf_ch1, gauss_der_wf_ch1 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch1, set_length_ch1, set_length_ch1 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)
gauss_wf_ch2, gauss_der_wf_ch2 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch2, set_length_ch2, set_length_ch2 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)
gauss_wf_ch3, gauss_der_wf_ch3 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch3, set_length_ch3, set_length_ch3 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)
gauss_wf_ch4, gauss_der_wf_ch4 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch4, set_length_ch4, set_length_ch4 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)
gauss_wf_ch5, gauss_der_wf_ch5 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch5, set_length_ch5, set_length_ch5 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)
gauss_wf_ch6, gauss_der_wf_ch6 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch6, set_length_ch6, set_length_ch6 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)
gauss_wf_ch7, gauss_der_wf_ch7 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch7, set_length_ch7, set_length_ch7 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)
gauss_wf_ch8, gauss_der_wf_ch8 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch8, set_length_ch8, set_length_ch8 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)
gauss_wf_ch9, gauss_der_wf_ch9 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch9, set_length_ch9, set_length_ch9 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)
gauss_wf_ch10, gauss_der_wf_ch10 = drag_gaussian_pulse_waveforms(
    gauss_amp_ch10, set_length_ch10, set_length_ch10 / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)

#####################
# HARDCODED PARAMS #
#####################

# Frequencies
AOM_IF = 10 * u.MHz


measure_length = 200
# Gaussian pulse parameters
gauss_amp = 0.3  # The gaussian is used when calibrating pi and pi_half pulses
gauss_len = 100  # The gaussian is used when calibrating pi and pi_half pulses

#creates the list of constants needed to outline a gaussian wf. length of the list equals length of gauss_len
gauss_wf, gauss_der_wf = drag_gaussian_pulse_waveforms(
    gauss_amp, gauss_len, gauss_len / 5, alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)

# Square pulse parameters
square_len = 100 # in ns
square_amp = 0.1 # in Volts

set_length = 100
zero_pulse = [0.3] * set_length


on_len = 320  # in ns
IF = 0

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
                3: {"offset": +0.0},
                4: {"offset": +0.0},
                5: {"offset": +0.0},
                6: {"offset": +0.0},
                7: {"offset": +0.0},
                8: {"offset": +0.0},
                9: {"offset": +0.0},
                10: {"offset": +0.0},

            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
                6: {},
                7: {},
                8: {},
                9: {},
                10: {},
            },
            "analog_inputs": {
                1: {"offset": 0.0129, "gain_db": 0},
                2: {"offset": 0.0, "gain_db": 0},
            },
        },
    },
    "elements": {
        #####################################################
        # Arbitrary elements:
        "generic_ai_elem_ch1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": CHANNEL1_IF,
            "operations": {
                "const": "square_pulse_ch1",
                "gauss": "gaussian_pulse_ch1",
                "arbitrary": "arbitrary_pulse_ch1",
            }
        },
        "generic_ai_elem_ch2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": CHANNEL2_IF,
            "operations": {
                "const": "square_pulse_ch2",
                "gauss": "gaussian_pulse_ch2",
                "arbitrary": "arbitrary_pulse_ch2",
            }
        },
        "generic_ai_elem_ch3": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": CHANNEL3_IF,
            "operations": {
                "const": "square_pulse_ch3",
                "gauss": "gaussian_pulse_ch3",
                "arbitrary": "arbitrary_pulse_ch3",
            }
        },
        "generic_ai_elem_ch4": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": CHANNEL4_IF,
            "operations": {
                "const": "square_pulse_ch4",
                "gauss": "gaussian_pulse_ch4",
                "arbitrary": "arbitrary_pulse_ch4",
            }
        },
        "generic_ai_elem_ch5": {
            "singleInput": {"port": ("con1", 5)},
            "intermediate_frequency": CHANNEL5_IF,
            "operations": {
                "const": "square_pulse_ch5",
                "gauss": "gaussian_pulse_ch5",
                "arbitrary": "arbitrary_pulse_ch5",
            }
        },
        "generic_ai_elem_ch6": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": CHANNEL6_IF,
            "operations": {
                "const": "square_pulse_ch6",
                "gauss": "gaussian_pulse_ch6",
                "arbitrary": "arbitrary_pulse_ch6",
            }
        },
        "generic_ai_elem_ch7": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": CHANNEL7_IF,
            "operations": {
                "const": "square_pulse_ch7",
                "gauss": "gaussian_pulse_ch7",
                "arbitrary": "arbitrary_pulse_ch7",
            }
        },
        "generic_ai_elem_ch8": {
            "singleInput": {"port": ("con1", 8)},
            "intermediate_frequency": CHANNEL8_IF,
            "operations": {
                "const": "square_pulse_ch8",
                "gauss": "gaussian_pulse_ch8",
                "arbitrary": "arbitrary_pulse_ch8",
            }
        },
        "generic_ai_elem_ch9": {
            "singleInput": {"port": ("con1", 9)},
            "intermediate_frequency": CHANNEL9_IF,
            "operations": {
                "const": "square_pulse_ch9",
                "gauss": "gaussian_pulse_ch9",
                "arbitrary": "arbitrary_pulse_ch9",
            }
        },
        "generic_ai_elem_ch10": {
            "singleInput": {"port": ("con1", 10)},
            "intermediate_frequency": CHANNEL10_IF,
            "operations": {
                "const": "square_pulse_ch10",
                "gauss": "gaussian_pulse_ch10",
                "arbitrary": "arbitrary_pulse_ch10",
            }
        },
        "generic_output_elem_ch1_to_ch1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": CHANNEL1_IF,
            "operations": {
                "readout": "meas_pulse_in_ch1",
            },
            'time_of_flight': 100,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', 1)
            }
        },
        "generic_output_elem_ch2_to_ch2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": CHANNEL2_IF,
            "operations": {
                "readout": "meas_pulse_in_ch2",
            },
            'time_of_flight': 24,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', 2)
            }
        },
        "generic_di_elem_ch1": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },
        "generic_di_elem_ch2": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },
        "generic_di_elem_ch3": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },
        "generic_di_elem_ch4": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 4),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },
        "generic_di_elem_ch5": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 5),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },
        "generic_di_elem_ch6": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 6),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },
        "generic_di_elem_ch7": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 7),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },
        "generic_di_elem_ch8": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 8),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },
        "generic_di_elem_ch9": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 9),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },
        "generic_di_elem_ch10": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 10),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {"ON": "ON_pulse"},
        },

        #####################################################
        # Hardcoded elements:

        "AOM_2": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": AOM_IF,
            "operations": {
                "const": "square_pulse",
                "gauss": "gaussian_pulse",
            }
        },
        "generic_di_elem_ch": {
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
        },
        # "mixer_el": {
        #     "mixInputs": {
        #         'I': ('con1', 1),
        #         'Q': ('con1', 2),
        #         'mixer': 'mixer1',
        #         'lo_frequency': 5.1e9,
        #     },
        #     'intermediate_frequency': AOM_IF,
        #     "operations": {
        #         'gen_mixed_pulse': 'gen_mixed_pulse',
        #     },
        # },
    },
    "pulses": {
        #####################################################
        # Arbitrary pulses:
        "gaussian_pulse_ch1": {
            "operation": "control",
            "length": set_length_ch1,  # in ns
            "waveforms": {"single": "gaussian_wf_ch1"},
        },
        "square_pulse_ch1": {
            "operation": "control",
            "length": set_length_ch1,
            "waveforms": {"single": "square_wf_ch1"},
        },

        "gaussian_pulse_ch2": {
            "operation": "control",
            "length": set_length_ch2,  # in ns
            "waveforms": {"single": "gaussian_wf_ch2"},
        },
        "square_pulse_ch2": {
            "operation": "control",
            "length": set_length_ch2,
            "waveforms": {"single": "square_wf_ch2"},
        },

        "gaussian_pulse_ch3": {
            "operation": "control",
            "length": set_length_ch3,  # in ns
            "waveforms": {"single": "gaussian_wf_ch3"},
        },
        "square_pulse_ch3": {
            "operation": "control",
            "length": set_length_ch3,
            "waveforms": {"single": "square_wf_ch3"},
        },

        "gaussian_pulse_ch4": {
            "operation": "control",
            "length": set_length_ch4,  # in ns
            "waveforms": {"single": "gaussian_wf_ch4"},
        },
        "square_pulse_ch4": {
            "operation": "control",
            "length": set_length_ch4,
            "waveforms": {"single": "square_wf_ch4"},
        },

        "gaussian_pulse_ch5": {
            "operation": "control",
            "length": set_length_ch5,  # in ns
            "waveforms": {"single": "gaussian_wf_ch5"},
        },
        "square_pulse_ch5": {
            "operation": "control",
            "length": set_length_ch5,
            "waveforms": {"single": "square_wf_ch5"},
        },

        "gaussian_pulse_ch6": {
            "operation": "control",
            "length": set_length_ch6,  # in ns
            "waveforms": {"single": "gaussian_wf_ch6"},
        },
        "square_pulse_ch6": {
            "operation": "control",
            "length": set_length_ch6,
            "waveforms": {"single": "square_wf_ch6"},
        },

        "gaussian_pulse_ch7": {
            "operation": "control",
            "length": set_length_ch7,  # in ns
            "waveforms": {"single": "gaussian_wf_ch7"},
        },
        "square_pulse_ch7": {
            "operation": "control",
            "length": set_length_ch7,
            "waveforms": {"single": "square_wf_ch7"},
        },

        "gaussian_pulse_ch8": {
            "operation": "control",
            "length": set_length_ch8,  # in ns
            "waveforms": {"single": "gaussian_wf_ch8"},
        },
        "square_pulse_ch8": {
            "operation": "control",
            "length": set_length_ch8,
            "waveforms": {"single": "square_wf_ch8"},
        },

        "gaussian_pulse_ch9": {
            "operation": "control",
            "length": set_length_ch9,  # in ns
            "waveforms": {"single": "gaussian_wf_ch9"},
        },
        "square_pulse_ch9": {
            "operation": "control",
            "length": set_length_ch9,
            "waveforms": {"single": "square_wf_ch9"},
        },

        "gaussian_pulse_ch10": {
            "operation": "control",
            "length": set_length_ch10,  # in ns
            "waveforms": {"single": "gaussian_wf_ch10"},
        },
        "square_pulse_ch10": {
            "operation": "control",
            "length": set_length_ch10,
            "waveforms": {"single": "square_wf_ch10"},
        },
        "arbitrary_pulse_ch1": {
            "operation": "control",
            "length": set_length_ch1,  # in ns
            "waveforms": {"single": "arbitrary_ch1"},
        },
        "arbitrary_pulse_ch2": {
            "operation": "control",
            "length": set_length_ch2,  # in ns
            "waveforms": {"single": "arbitrary_ch2"},
        },
        "arbitrary_pulse_ch3": {
            "operation": "control",
            "length": set_length_ch3,  # in ns
            "waveforms": {"single": "arbitrary_ch3"},
        },
        "arbitrary_pulse_ch4": {
            "operation": "control",
            "length": set_length_ch4,  # in ns
            "waveforms": {"single": "arbitrary_ch4"},
        },
        "arbitrary_pulse_ch5": {
            "operation": "control",
            "length": set_length_ch5,  # in ns
            "waveforms": {"single": "arbitrary_ch5"},
        },
        "arbitrary_pulse_ch6": {
            "operation": "control",
            "length": set_length_ch6,  # in ns
            "waveforms": {"single": "arbitrary_ch6"},
        },
        "arbitrary_pulse_ch7": {
            "operation": "control",
            "length": set_length_ch7,  # in ns
            "waveforms": {"single": "arbitrary_ch7"},
        },
        "arbitrary_pulse_ch8": {
            "operation": "control",
            "length": set_length_ch8,  # in ns
            "waveforms": {"single": "arbitrary_ch8"},
        },
        "arbitrary_pulse_ch9": {
            "operation": "control",
            "length": set_length_ch9,  # in ns
            "waveforms": {"single": "arbitrary_ch9"},
        },
        "arbitrary_pulse_ch10": {
            "operation": "control",
            "length": set_length_ch10,  # in ns
            "waveforms": {"single": "arbitrary_ch10"},
        },

        'meas_pulse_in_ch1': {
            'operation': 'measurement',
            'length': measure_length_ch1,
            'waveforms': {
                'single': 'square_wf_ch1'
            },
            'integration_weights': { #doesn't get used if we're not demodulating
                'cos': 'cos',
                'sin': 'sin',
                'minus_sin': 'minus_sin',
            },
            'digital_marker': 'ON'  #why is this needed? Is it needed?
        },
        'meas_pulse_in_ch2': {
            'operation': 'measurement',
            'length': measure_length_ch2,
            'waveforms': {
                'single': 'square_wf_ch2'
            },
            'integration_weights': { #doesn't get used if we're not demodulating
                'cos': 'cos',
                'sin': 'sin',
                'minus_sin': 'minus_sin',
            },
            'digital_marker': 'ON'  #why is this needed? Is it needed?
        },

        #####################################################
        # Hardcoded pulses:


        "ON_pulse": {
            "operation": "control",
            "length": on_len,  # in ns
            "digital_marker": "ON",
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gauss_len,  # in ns
            "waveforms": {"single": "gaussian_wf"},
        },
        "square_pulse": {"operation": "control", "length": square_len, "waveforms": {"single": "square_wf"}},  # in ns
        #'pulse1': {
        #     'operation': 'control',
        #     'length': 12,
        #     'waveforms': {
        #         'I': 'wf_I',
        #         'Q': 'wf_Q',
        #     },
        # },
    },
    "waveforms": {

        #####################################################
        # Arbitrary waveforms:

        "arbitrary_ch1": {"type": "arbitrary", "samples": arbitrary_wf_ch1},
        "arbitrary_ch2": {"type": "arbitrary", "samples": arbitrary_wf_ch2},
        "arbitrary_ch3": {"type": "arbitrary", "samples": arbitrary_wf_ch3},
        "arbitrary_ch4": {"type": "arbitrary", "samples": arbitrary_wf_ch4},
        "arbitrary_ch5": {"type": "arbitrary", "samples": arbitrary_wf_ch5},
        "arbitrary_ch6": {"type": "arbitrary", "samples": arbitrary_wf_ch6},
        "arbitrary_ch7": {"type": "arbitrary", "samples": arbitrary_wf_ch7},
        "arbitrary_ch8": {"type": "arbitrary", "samples": arbitrary_wf_ch8},
        "arbitrary_ch9": {"type": "arbitrary", "samples": arbitrary_wf_ch9},
        "arbitrary_ch10": {"type": "arbitrary", "samples": arbitrary_wf_ch10},

        "square_wf_ch1": {"type": "constant", "sample": square_amp_ch1},
        "square_wf_ch2": {"type": "constant", "sample": square_amp_ch2},
        "square_wf_ch3": {"type": "constant", "sample": square_amp_ch3},
        "square_wf_ch4": {"type": "constant", "sample": square_amp_ch4},
        "square_wf_ch5": {"type": "constant", "sample": square_amp_ch5},
        "square_wf_ch6": {"type": "constant", "sample": square_amp_ch6},
        "square_wf_ch7": {"type": "constant", "sample": square_amp_ch7},
        "square_wf_ch8": {"type": "constant", "sample": square_amp_ch8},
        "square_wf_ch9": {"type": "constant", "sample": square_amp_ch9},
        "square_wf_ch10": {"type": "constant", "sample": square_amp_ch10},

        "gaussian_wf_ch1": {"type": "arbitrary", "samples": gauss_wf_ch1},
        "gaussian_wf_ch2": {"type": "arbitrary", "samples": gauss_wf_ch2},
        "gaussian_wf_ch3": {"type": "arbitrary", "samples": gauss_wf_ch3},
        "gaussian_wf_ch4": {"type": "arbitrary", "samples": gauss_wf_ch4},
        "gaussian_wf_ch5": {"type": "arbitrary", "samples": gauss_wf_ch5},
        "gaussian_wf_ch6": {"type": "arbitrary", "samples": gauss_wf_ch6},
        "gaussian_wf_ch7": {"type": "arbitrary", "samples": gauss_wf_ch7},
        "gaussian_wf_ch8": {"type": "arbitrary", "samples": gauss_wf_ch8},
        "gaussian_wf_ch9": {"type": "arbitrary", "samples": gauss_wf_ch9},
        "gaussian_wf_ch10": {"type": "arbitrary", "samples": gauss_wf_ch10},


        #####################################################
        # Hardcoded waveforms:

        "square_wf": {"type": "constant", "sample": square_amp},
        "gaussian_wf": {"type": "arbitrary", "samples": gauss_wf},
        # 'wf_I': {
        #     'type': 'arbitrary',
        #     'samples': [0.49, 0.47, 0.44, ...]
        # },
        # 'wf_Q': {
        #     'type': 'arbitrary',
        #     'samples': [-0.02, -0.03, -0.03, ...]
        # },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    'integration_weights': {
        'cos': {
            'cosine': [(4.0, 20)],
            'sine': [(0.0, 20)]
        },
        'sin': {
            'cosine': [(0.0, 20)],
            'sine': [(4.0, 20)]
        },
        'minus_sin': {
            'cosine': [(0.0, 20)],
            'sine': [(-4.0, 20)]
        },
    },
    # 'mixers': {
    #     'mixer1': [
    #         {'intermediate_frequency': 70e6, 'lo_frequency': 5.1e9, 'correction': [0.9, 0.003, 0.0, 1.05]}
    #     ],
    # },
}#DON'T PUT COMMA HERE

AI_CHANNEL_TO_GEN_EL = { ##why is this called AI_CHANNEL and not AO_CHANNEL
    1: "generic_ai_elem_ch",
    2: "generic_ai_elem_ch2",
    3: "generic_ai_elem_ch3",
    4: "generic_ai_elem_ch4",
    5: "generic_ai_elem_ch5",
    6: "generic_ai_elem_ch6",
    7: "generic_ai_elem_ch7",
    8: "generic_ai_elem_ch8",
    9: "generic_ai_elem_ch9",
    10: "generic_ai_elem_ch10",
}

AO_CHANNEL_TO_GEN_EL = {
    1: "generic_output_elem"
}##generic_output_elem is for analog input channel 1

DO_CHANNEL_TO_GEN_EL = { #I (Ankit) used naming convention here that I think makes sense
    1: "generic_di_elem"
}
