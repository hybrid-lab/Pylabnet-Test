# OPX1000 AO skeleton

Supports direct-driver use and Pylabnet client/server integration for DataTaker.
MW-FEM slots 1/2 are declared without ports. LF-FEM slots 3/4 expose AO ports
1-8 at 1 GS/s, zero offset, amplified mode and pulse upsampling for DC pulses.

Values are volts into 50 ohms: amplified permits [-2.5, 2.5) V, direct permits
[-0.5, 0.5) V. frequency is MHz, default 0 for DC. For modulated output,
consider direct mode and mw upsampling. Amplified bandwidth is 330 MHz;
1 GS/s limits IF magnitude to <500 MHz. Envelope bandwidth also matters.
Port settings affect every element on that port.

length/gauss_sd are ns. Lengths/delays must be integers >=16, divisible by 4.
Gaussian endpoints are zero. Arbitrary samples are absolute volts, one/ns;
amplitude applies only to const/gauss. This skeleton supports 1 GS/s LF only.

```python
from pylabnet.hardware.quantum_machines.OPX1000 import Driver

opx = Driver(dummy=True)  # Config/stack building needs no QM SDK.
opx.build_stack()
element = opx.set_ao_voltage(
    "gauss", fem_slot=3, ao_channel=1, amplitude=0.2, length=100,
)
opx.delay(40, element)
opx.set_ao_voltage(element=element)
# opx.build_program() requires qm-qua but makes no connection.
```

build_stack() deep-resets generated config/counters. Reuse returned names for
successive pulses on one timeline; different elements can overlap. Use
align()/delay() between elements. Playback is finite and requires stack mode.

Set qop_ip/cluster_name in config or constructor for QOP access. Simulation:
execute(simulate=True, duration=4000), duration in ns, returns a simulator job.
It requires QOP and dummy=False. Hardware execute() waits for Done;
wait=False returns after submission. timeout is seconds and does not cancel
jobs. Successful submission consumes the stack. Another hardware submission
replaces this driver's QM and stops its job. Call close() in a finally block.
Other clients' QMs are not closed; trim the default 16 ports when sharing.

QOP runs on the controller; qm-qua is the PC package. Check compatibility first.
Current qm-qua 1.4.1 needs Python 3.10-3.14. The repo venv has 1.2.2/Python 3.8;
its dependencies are unchanged. LF acquisition and a dedicated finite pulse-train
operation are available. General loops, digital output and MW output are not
implemented. Validate QOP simulation before hardware execution.

Run offline checks from the pylabnet package directory:
`python -B -m unittest discover -s hardware/quantum_machines/tests -v`

References:
- [FEM settings](https://docs.quantum-machines.co/latest/docs/Guides/opx1000_fems/)
- [SDK releases](https://docs.quantum-machines.co/latest/docs/Releases/qm_qua_releases/)
## Pylabnet client/server and DataTaker

The integration files are network/client_server/OPX1000.py and
launchers/servers/OPX1000.py. Launcher already selects these dynamically by
type; no Launcher modification is needed. The current DataTaker config selects:

```json
{"type": "OPX1000", "config": "OPX1000"}
```

This loads configs/devices/OPX1000/OPX1000.json. device_id identifies the server.
With no ip/port pair, Launcher can discover a registered matching server or
start one locally. An ip/port pair addresses an already running Pylabnet server.
QOP connection settings still come from OPX1000Config.py; the device JSON's
ip_address field is unused. Optional dummy must be a JSON boolean.

The server uses Pylabnet's existing GenericServer/ClientBase SSL defaults
(pylabnet.pem). The server environment needs compatible qm-qua for execution;
client calls and server startup do not establish a QOP connection.

A DataTaker experiment receives the client as kwargs["OPX1000_OPX1000"].
Retrieve/store it in configure(); build and execute sequences in experiment():

```python
def configure(**kwargs):
    kwargs["dataset"].opx = kwargs["OPX1000_OPX1000"]

def experiment(**kwargs):
    opx = kwargs["dataset"].opx
    opx.build_stack()
    element = opx.set_ao_voltage(
        "const", fem_slot=3, ao_channel=1, amplitude=0.1, length=1000,
    )
    info = opx.execute(simulate=True, duration=4000)
    samples = opx.get_simulated_samples()
    # samples["con1"]["analog"]["3-1"] is an ordinary list of values.
    # samples["con1"]["analog_sampling_rate"]["3-1"] gives samples/second.
    kwargs["dataset"].log.info("Simulation job: " + info["job_id"])
    kwargs["thread"].running = False  # Finish this one-shot test.
```

This illustrates client access; a complete DataTaker script also needs
INIT_DICT and define_dataset(), and may plot the returned simulated waveform.
Existing spectroscopy scripts still require general loops/demodulated readout,
which are not implemented by these verification methods.

Client methods: build_stack(), create_new_ao_elem(), set_ao_voltage(), delay(),
align(), execute(), get_config(), get_status(), get_simulated_samples(), close().
Units match the driver. execute() returns {"job_id": ..., "simulated": ...};
QM jobs/programs remain on the server. For wait=False, poll get_status().
get_config() is an independent snapshot. build_program() is a local-driver API.

The driver/stack is shared by all clients: use one experiment at a time per
device server. Closing a client connection leaves the driver available.
client.close() releases the shared driver's QM resources while keeping the
Pylabnet server running. close_server() releases resources and shuts down the
device server process. A hardware wait timeout leaves the submitted job active;
get_status() remains available afterward.

## LF verification experiments

Select either new file from the existing DataTaker experiment list:

- scripts/test_scripts/opx1000_snspd_dark_counts.py
- scripts/test_scripts/opx1000_lf_pulse_train.py

Restart the OPX1000 device server after updating its driver/RPC code, so the
running process loads the new methods. Use the repository virtual environment.
Each script is one-shot: Configure, then Run. There is no SNSPD bias control.

The input experiment now displays RAW WAVEFORMS ONLY. The existing filename
opx1000_snspd_dark_counts.py is retained, but it performs no threshold counting
or bias control. Its main plot is replaced with the latest ADC trace on each
refresh. There are no count-rate, accumulated-count, or threshold plots.

Defaults: LF slot 3, input 1, 10 seconds of viewing, a 10000 ns trace per
capture, and a 0.2 s pause between captures. The viewing interval begins after
the first completed capture; initial connection/compilation can add startup
time. Each capture is a separate short 1 GS/s acquisition, maximum 16000 ns.
There are unrecorded gaps between captures. This is NOT a continuous 10-second
waveform recording. RPC/compilation/fetch time adds to the pause, and an
in-flight capture can finish after the viewing deadline.

The x-axis is time within the latest capture in ns; the y-axis is raw ADC code,
not calibrated connector voltage. The plot refreshes while the run is active.
Stop prevents additional captures and cancels an active job when it is polled.
The saved dataset contains only the latest capture, not every trace seen during
the viewing interval. Metadata records capture count, requested/actual viewing
time and summed capture live time, explicitly marking the run non-continuous.

Inputs 1/2 on LF slots 3/4 are configured at 1 GS/s, zero offset, 0 dB gain.
Input-only measurement elements have no AO or physical digital-output mapping.
An internal pulse digital_marker gates the ADC. The base hardware config still
declares LF AO ports with their configured offsets. Default input range is
+/-0.5 V into 50 ohms; keep the conditioned SNSPD signal within its specification.
A short untriggered capture may contain no detector pulses at a low dark rate.

The driver's count_ai_pulses() API remains available for future counting work.
It is not called by this waveform-viewing experiment.

The pulse-train experiment defaults to simulate=1. Set simulate=0 to emit the
finite train on an LF AO, with the displayed voltage into a 50-ohm load. Length
and period are ns; both must be divisible by 4 and at least 16. The low interval
must be zero or at least 16 ns. pulse_count repetitions, including the final
low interval, are emitted in a QUA strict-timing loop. Requested duration is
pulse_count * period_ns. QOP compilation will reject unsupported strict timing.
Simulation is truncated to the selected preview duration (max 1e6 ns).
For hardware runs, the plot is an ideal commanded preview, not an acquired
output measurement. Use an oscilloscope to check the physical waveform.

Additional client methods:
- get_ai_voltage(ai_channel, length=..., fem_slot=..., label=...): queue raw ADC.
- count_ai_pulses(ai_channel, num_windows, window_ns=..., threshold_adc=...,
  polarity="Above"/"Below", max_tags=..., fem_slot=...): queue threshold counts.
- get_results(): fetch completed ADC/count streams as dictionaries/lists.
- pulse_train(ao_channel, pulse_count, period_ns, amplitude=..., length=...,
  fem_slot=...): queue a hardware-timed constant-pulse train.

SDK construction/config checks are not QOP compilation or hardware validation.
Time-tagging/gap behavior and polarity must be checked on the target system.
References:
- [ADC streaming](https://docs.quantum-machines.co/latest/docs/Guides/stream_proc/#streaming-raw-adc-results)
- [Time tagging](https://docs.quantum-machines.co/latest/docs/Guides/features/#time-tagging)
- [Operation timestamps](https://docs.quantum-machines.co/latest/docs/Guides/features/#timestamp-stream)
- [LF input specifications](https://docs.quantum-machines.co/latest/docs/Hardware/OPX1000_hardware/#lf-fem)
