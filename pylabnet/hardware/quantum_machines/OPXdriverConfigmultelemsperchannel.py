"""Connection settings, port helpers, defaults and the QUA config for the OPX1000 driver (OPX.py).

`config` starts with only the hardware and the shared pieces (digital "ON" marker,
integration weights). OPX.py adds an element, pulse and waveform to it for every
channel it uses, and opens the quantum machine with it.
"""
from pathlib import Path

import plotly.io as pio
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms    # used by OPX.py (import *)
from qualang_tools.units import unit

pio.renderers.default = "browser"    # simulation plots open in the browser

u = unit(coerce_to_integer=True)

##############
# Connection #
##############
# B167 OPX1000. No cluster name: the gateway rejects "Cluster_1" and connects
# fine without one. The OPX server overrides these from configs/devices/OPX/OPX.json.
qop_ip = "192.168.0.188"
cluster_name = None

#########
# Ports #
#########
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


# Simulation waveform reports (git-ignored)
save_dir = Path(__file__).parent.resolve() / "Data"
save_dir.mkdir(exist_ok=True)

######################################
# Defaults for the driver and client #
######################################
default_length = 100            # ns
default_frequency = 10          # MHz, intermediate frequency of analog elements
default_amplitude = 0.3         # V
default_measure_pulse = "const"
default_digital_pulse = "ON"
default_time_of_flight = 28      # ns; the OPX1000 needs a multiple of 4, >= 28
default_smearing = 0            # ns

##############
# QUA config #
##############


def _lf_fem_config():
    return {
        "type": "LF",
        "analog_outputs": {k: {"offset": 0.0} for k in range(1, LF_FEM_PORTS["ao"] + 1)},
        "digital_outputs": {k: {} for k in range(1, LF_FEM_PORTS["do"] + 1)},
        "analog_inputs": {k: {"offset": 0.0, "gain_db": 0} for k in range(1, LF_FEM_PORTS["ai"] + 1)},
    }


config = {
    "controllers": {
        CONTROLLER: {
            "type": "opx1000",
            "fems": {LF_FEM_SLOT: _lf_fem_config()},
        },
    },
    "elements": {},     # filled in by OPX.py
    "pulses": {},       # filled in by OPX.py
    "waveforms": {},    # filled in by OPX.py
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},    # high for the whole pulse; used by every digital pulse
    },
    # Referenced by every measurement pulse; measure_demod demodulates with "cos"
    "integration_weights": {
        "cos": {"cosine": [(4.0, 500)], "sine": [(0.0, 500)]},
        "sin": {"cosine": [(0.0, 500)], "sine": [(4.0, 500)]},
        "minus_sin": {"cosine": [(0.0, 20)], "sine": [(-4.0, 20)]},
    },
}


def set_lf_fem_slot(slot):
    """Point the driver at the LF-FEM in `slot`. Call before any element is created."""
    global LF_FEM_SLOT
    slot = int(slot)
    fems = config["controllers"][CONTROLLER]["fems"]
    fems.clear()    # in place: OPX.py holds a reference to this same dict
    fems[slot] = _lf_fem_config()
    LF_FEM_SLOT = slot
