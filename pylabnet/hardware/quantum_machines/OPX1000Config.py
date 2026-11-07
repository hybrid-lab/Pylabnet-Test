"""Minimal OPX1000 QUA hardware config: MW slots 1/2, LF slots 3/4.

No SDK imports, connections, directory creation, or experiment-specific elements.
See OPX1000_README.md before connecting to hardware.
"""

# Quantum Machines endpoint, separate from Pylabnet RPC addressing.
qop_ip = "192.168.0.188"  # Set the QM router IP/hostname before execution or simulation.
cluster_name = "OPX1000_cluster"  # Set the actual cluster name if required by your setup.

default_controller = "con1"
default_fem_slot = 3
default_length = 100  # ns
default_frequency = 0.0  # MHz; zero produces an unmodulated/DC envelope.
default_amplitude = 0.1  # V into a 50-ohm load


def make_config():
    """Return a fresh, independent full QUA config for the installed modules."""
    fems = {
        # Installed MW-FEMs are declared but have no configured/claimed ports.
        # Future MW output needs band, full_scale_power_dbm, upconverters,
        # and an element with MWInput, rather than LF singleInput.
        1: {"type": "MW", "analog_outputs": {}},
        2: {"type": "MW", "analog_outputs": {}},
    }
    for slot in (3, 4):
        fems[slot] = {
            "type": "LF",
            "analog_inputs": {
                port: {"offset": 0.0, "gain_db": 0, "sampling_rate": int(1e9)}
                for port in (1, 2)
            },
            "analog_outputs": {
                port: {
                    "offset": 0.0,
                    "sampling_rate": int(1e9),
                    "output_mode": "amplified",
                    "upsampling_mode": "pulse",
                }
                for port in range(1, 9)
            },
        }

    return {
        "version": 1,
        "controllers": {
            default_controller: {"type": "opx1000", "fems": fems},
        },
        "elements": {},
        "pulses": {},
        "waveforms": {},
    }


# Convenient editable template. Drivers deep-copy it; never mutate this globally.
config = make_config()
