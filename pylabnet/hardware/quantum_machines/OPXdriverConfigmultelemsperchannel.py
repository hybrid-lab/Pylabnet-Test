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
qop_ip = "192.168.88.251"
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
default_frequency = 10 #in MHz
default_amplitude = 0.3
default_measure_length = 100

default_measure_pulse = "const"
default_digital_pulse = "ON"
defaut_time_of_flight = 24
default_smearing = 0
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
gauss_sd = 1
set_length_single = 100
zero_pulse_single = [0.3] * set_length_single

on_len = 320  # ns
IF = 0

CHS = range(1, 11)

################
# CONFIG BUILD #
################
hard_coded_config = {
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

    },

    "pulses": {

    },

    "waveforms": {
        # ---------- Your hardcoded waveforms
        "square_wf": {"type": "constant", "sample": square_amp_single},
        "gaussian_wf": {"type": "arbitrary", "samples": gauss_wf_single},
    },

    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },

    "integration_weights": {
        "cos": {"cosine": [(4.0, 500)], "sine": [(0.0, 500)]},
        "sin": {"cosine": [(0.0, 500)], "sine": [(4.0, 500)]},
        "minus_sin": {"cosine": [(0.0, 20)], "sine": [(-4.0, 20)]},
    },
}

config = hard_coded_config.copy()

# Channel → element name helpers
AI_CHANNEL_TO_GEN_EL = {ch: f"generic_ai_elem_ch{ch}" for ch in CHS}

# You only defined these two output->input elements explicitly:
AO_CHANNEL_TO_GEN_EL = {
    1: "generic_output_elem_ch1_to_ch1",
    2: "generic_output_elem_ch2_to_ch2",
}

DO_CHANNEL_TO_GEN_EL = {ch: f"generic_di_elem_ch{ch}" for ch in CHS}
