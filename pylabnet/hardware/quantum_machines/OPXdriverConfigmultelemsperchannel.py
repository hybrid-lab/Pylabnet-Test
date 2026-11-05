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
# B167 OPX1000. No cluster name: the gateway rejects "Cluster_1" and connects
# fine without one. The OPX server overrides these from configs/devices/OPX/OPX.json.
qop_ip = "192.168.0.188"
cluster_name = None

# OPX1000 only (the OPX+ is not supported). Every port is (controller, FEM slot,
# port). B167 chassis: slots 1-2 are MW-FEMs, slots 3-4 are LF-FEMs. All AO/DO/AI
# of this driver go through one LF-FEM, LF_FEM_SLOT; the OPX server sets it from
# lf_fem_slot in configs/devices/OPX/OPX.json via set_lf_fem_slot().
CONTROLLER = "con1"
LF_FEM_SLOT = 3

# Ports on an LF-FEM, numbered from 1
LF_FEM_PORTS = {"ao": 8, "do": 8, "ai": 2}
_PORT_NAMES = {"ao": "analog output", "do": "digital output", "ai": "analog input"}


def opx_port(ch, kind):
    """Config port tuple for channel `ch` of `kind` ('ao', 'do' or 'ai') on the driver's LF-FEM."""
    ch = int(float(ch))
    n = LF_FEM_PORTS[kind]
    if not 1 <= ch <= n:
        raise ValueError(f"{_PORT_NAMES[kind]} {ch} does not exist: the OPX1000 LF-FEM has "
                         f"{_PORT_NAMES[kind]}s 1-{n}")
    return (CONTROLLER, LF_FEM_SLOT, ch)

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


def _lf_fem_config():
    return {
        "type": "LF",
        "analog_outputs": {k: {"offset": 0.0} for k in range(1, LF_FEM_PORTS["ao"] + 1)},
        "digital_outputs": {k: {} for k in range(1, LF_FEM_PORTS["do"] + 1)},
        "analog_inputs": {k: {"offset": 0.0, "gain_db": 0} for k in range(1, LF_FEM_PORTS["ai"] + 1)},
    }


################
# CONFIG BUILD #
################
hard_coded_config = {
    "version": 1,
    "controllers": {
        CONTROLLER: {
            "type": "opx1000",
            "fems": {LF_FEM_SLOT: _lf_fem_config()},
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


def set_lf_fem_slot(slot):
    """Point the driver at the LF-FEM in `slot`. Call before any element is created."""
    global LF_FEM_SLOT
    slot = int(slot)
    # config is a shallow copy, so it shares this dict: edit it in place
    fems = hard_coded_config["controllers"][CONTROLLER]["fems"]
    fems.clear()
    fems[slot] = _lf_fem_config()
    LF_FEM_SLOT = slot
