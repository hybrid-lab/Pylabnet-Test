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

# DC offset (V) for individual LF analog inputs, keyed by (slot, input port); inputs not
# listed use 0.0. Each input's offset should be recalibrated whenever its gain_db changes.
# Quantum Machines' recipe: new offset = old offset - (mean raw code at 0 V input) / 4096.
# Measure the 0 V reading with a 50 ohm terminator on the input (nothing driving it), not
# with a loopback: a loopback also includes the output's own DC error.
# (3, 1): NOT SET. The 2026-11-12 loopback (slot 3 output 1 -> input 1, gain_db 0) read 0 V
#   as about -117.4 codes (which would mean +0.0287 V here), but that loopback can't tell
#   whether the offset is input 1's or output 1's; check with a terminator first.
input_offsets_v = {
}


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
                port: {"offset": input_offsets_v.get((slot, port), 0.0), "gain_db": 0,
                       "sampling_rate": int(1e9)}
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
