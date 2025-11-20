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

######################
# Network parameters #
######################
qop_ip = "192.168.88.252"
cluster_name = "Cluster_1"

#############
# Save Path #
#############
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
# PARAMS (DICT FORM)#
#####################

default_length = 100
default_frequency = 0
default_amplitude = 0.3
default_measure_length = 100
default_gauss_wf, default = drag_gaussian_pulse_waveforms(
    default_amplitude, default_length, default_length / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)

#####################
# HARDCODED PARAMS  #
#####################
# (These are your single, non-per-channel items)
AOM_IF = 10 * u.MHz
measure_length_default = 200
N_CHANNELS = 10
CHS = range(1, N_CHANNELS + 1)

# Per-channel lengths (was set_length_chX)
set_length = {ch: 100 for ch in CHS}

# Per-channel measurement lengths (expand as needed)
measure_length = {1: 100, 2: 100}

# Per-channel amplitudes
square_amp = {ch: 0.3 for ch in CHS}
gauss_amp = {ch: 0.3 for ch in CHS}

# Per-channel IFs (was CHANNELX_IF)
CHANNEL_IF = {ch: 10 * u.MHz for ch in CHS}

# Per-channel arbitrary waveforms (was arbitrary_wf_chX)
arbitrary_wf = {ch: [0.3] * set_length[ch] for ch in CHS}

# Per-channel Gaussian waveforms (was gauss_wf_chX / gauss_der_wf_chX)
gauss_wf, gauss_der_wf = {}, {}
for ch in CHS:
    wf, der = drag_gaussian_pulse_waveforms(
        gauss_amp[ch], set_length[ch], set_length[ch] / 5,
        alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
    )
    gauss_wf[ch] = wf
    gauss_der_wf[ch] = der

#####################
# HARDCODED PARAMS  #
#####################
# (These are your single, non-per-channel items)
AOM_IF = 10 * u.MHz
measure_length_default = 200
gauss_amp_single = 0.3
gauss_len_single = 100

gauss_wf_single, gauss_der_wf_single = drag_gaussian_pulse_waveforms(
    gauss_amp_single, gauss_len_single, gauss_len_single / 5,
    alpha=0, delta=1, anharmonicity=0, detuning=0, subtracted=True
)

square_len_single = 100   # ns
square_amp_single = 0.1   # Volts

set_length_single = 100
zero_pulse_single = [0.3] * set_length_single

on_len = 320  # ns
IF = 0

################
# CONFIG BUILD #
################

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0}, 2: {"offset": +0.0}, 3: {"offset": +0.0}, 4: {"offset": +0.0}, 5: {"offset": +0.0},
                6: {"offset": +0.0}, 7: {"offset": +0.0}, 8: {"offset": +0.0}, 9: {"offset": +0.0}, 10: {"offset": +0.0},
            },
            "digital_outputs": {k: {} for k in CHS},
            "analog_inputs": {
                1: {"offset": 0.0129, "gain_db": 0},
                2: {"offset": 0.0, "gain_db": 0},
            },
        },
    },

    "elements": {
        # ---------- Per-channel analog output elements (generic_ai_elem_chX)
        **{
            f"generic_ai_elem_ch{ch}": {
                "singleInput": {"port": ("con1", ch)},
                "intermediate_frequency": CHANNEL_IF[ch],
                "operations": {
                    "const": f"square_pulse_ch{ch}",
                    "gauss": f"gaussian_pulse_ch{ch}",
                    "arbitrary": f"arbitrary_pulse_ch{ch}",
                }
            } for ch in CHS
        },

        # ---------- Two per-channel "output -> input" elements you had hardcoded
        "generic_output_elem_ch1_to_ch1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": CHANNEL_IF[1],
            "operations": {"readout": "meas_pulse_in_ch1"},
            "time_of_flight": 24,
            "smearing": 0,
            "outputs": {"out1": ("con1", 1)},
        },
        "generic_output_elem_ch2_to_ch2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": CHANNEL_IF[2],
            "operations": {"readout": "meas_pulse_in_ch2"},
            "time_of_flight": 24,
            "smearing": 0,
            "outputs": {"out1": ("con1", 2)},
        },

        # ---------- Per-channel digital input elements (generic_di_elem_chX)
        **{
            f"generic_di_elem_ch{ch}": {
                "digitalInputs": {
                    "activate": {"port": ("con1", ch), "delay": 0, "buffer": 0},
                },
                "operations": {"ON": "ON_pulse"},
            } for ch in CHS
        },

        # ---------- Your hardcoded example element
        "AOM_2": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": AOM_IF,
            "operations": {"const": "square_pulse", "gauss": "gaussian_pulse"}
        },
    },

    "pulses": {
        # ---------- Per-channel gaussian/square/arbitrary pulses using set_length[ch]
        **{
            f"gaussian_pulse_ch{ch}": {
                "operation": "control",
                "length": set_length[ch],
                "waveforms": {"single": f"gaussian_wf_ch{ch}"},
            } for ch in CHS
        },
        **{
            f"square_pulse_ch{ch}": {
                "operation": "control",
                "length": set_length[ch],
                "waveforms": {"single": f"square_wf_ch{ch}"},
            } for ch in CHS
        },
        **{
            f"arbitrary_pulse_ch{ch}": {
                "operation": "control",
                "length": set_length[ch],
                "waveforms": {"single": f"arbitrary_ch{ch}"},
            } for ch in CHS
        },

        # ---------- Per-channel measurement pulses (two defined in your file)
        "meas_pulse_in_ch1": {
            "operation": "measurement",
            "length": measure_length[1],
            "waveforms": {"single": "square_wf_ch1"},
            "integration_weights": {"cos": "cos", "sin": "sin", "minus_sin": "minus_sin"},
            "digital_marker": "ON",
        },
        "meas_pulse_in_ch2": {
            "operation": "measurement",
            "length": measure_length[2],
            "waveforms": {"single": "square_wf_ch2"},
            "integration_weights": {"cos": "cos", "sin": "sin", "minus_sin": "minus_sin"},
            "digital_marker": "ON",
        },

        # ---------- Your hardcoded pulses
        "ON_pulse": {"operation": "control", "length": on_len, "digital_marker": "ON"},
        "gaussian_pulse": {"operation": "control", "length": gauss_len_single, "waveforms": {"single": "gaussian_wf"}},
        "square_pulse": {"operation": "control", "length": square_len_single, "waveforms": {"single": "square_wf"}},
    },

    "waveforms": {
        # ---------- Per-channel arbitrary/square/gaussian waveforms
        **{
            f"arbitrary_ch{ch}": {"type": "arbitrary", "samples": arbitrary_wf[ch]}
            for ch in CHS
        },
        **{
            f"square_wf_ch{ch}": {"type": "constant", "sample": square_amp[ch]}
            for ch in CHS
        },
        **{
            f"gaussian_wf_ch{ch}": {"type": "arbitrary", "samples": gauss_wf[ch]}
            for ch in CHS
        },

        # ---------- Your hardcoded waveforms
        "square_wf": {"type": "constant", "sample": square_amp_single},
        "gaussian_wf": {"type": "arbitrary", "samples": gauss_wf_single},
    },

    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },

    "integration_weights": {
        "cos": {"cosine": [(4.0, 20)], "sine": [(0.0, 20)]},
        "sin": {"cosine": [(0.0, 20)], "sine": [(4.0, 20)]},
        "minus_sin": {"cosine": [(0.0, 20)], "sine": [(-4.0, 20)]},
    },
}

# Channel → element name helpers
AI_CHANNEL_TO_GEN_EL = {ch: f"generic_ai_elem_ch{ch}" for ch in CHS}

# You only defined these two output->input elements explicitly:
AO_CHANNEL_TO_GEN_EL = {
    1: "generic_output_elem_ch1_to_ch1",
    2: "generic_output_elem_ch2_to_ch2",
}

DO_CHANNEL_TO_GEN_EL = {ch: f"generic_di_elem_ch{ch}" for ch in CHS}
