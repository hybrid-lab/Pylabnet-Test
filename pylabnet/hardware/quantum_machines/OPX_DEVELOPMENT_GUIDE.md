# OPX Driver and Experiment Development Guide

This note summarizes the current Pylabnet OPX path and gives a starting point for
experiment or OPX1000 work. It describes the code as it exists in this repository;
it is not a claim that the current driver has been tested with an OPX1000.

## The different configuration files

There are three different configuration layers. They answer different questions:

1. **Pylabnet device config**  
   Example: [`configs/devices/OPX/OPX.json`](../../configs/devices/OPX/OPX.json)

   This is connection/identity information for Pylabnet's device server. Its
   `device_id` identifies the device entry, while `ip` and `port`, when present,
   tell a script Launcher where to connect to the Pylabnet server. These are not
   the QUA controller/port definitions.

2. **Pylabnet script config**  
   Example: [`configs/scripts/staticline/staticline_with_OPX.json`](../../configs/scripts/staticline/staticline_with_OPX.json)

   This tells the Launcher which device servers a run requires and which Python
   script to start. A server entry such as
   `{"type": "OPX", "config": "OPX"}` means: use the `OPX` server module and
   load the `OPX` device config.

3. **Quantum Machines / QUA hardware config**  
   Current module:
   [`OPXdriverConfigmultelemsperchannel.py`](OPXdriverConfigmultelemsperchannel.py)

   This Python dictionary defines the Quantum Machines configuration: controller
   ports, logical elements, operations, pulses, waveforms, mixers, and measurement
   settings. The OPX driver supplies this config when opening a Quantum Machine.
   It is separate from the Pylabnet JSON files.

## What the main files do

| File | Responsibility |
|---|---|
| [`launchers/launch_control.py`](../../launchers/launch_control.py) | GUI for choosing script/device configs, displaying processes/logging, and starting a script or device server. |
| [`launchers/launcher.py`](../../launchers/launcher.py) | Coordinates one script run: loads its script config, resolves required device servers, creates client objects, then invokes the script's `launch(...)`. |
| [`launchers/pylabnet_server.py`](../../launchers/pylabnet_server.py) | Entry point for a device-server process. Parses command-line values and calls the selected server module's `launch(...)`. |
| [`launchers/servers/OPX.py`](../../launchers/servers/OPX.py) | Creates the OPX hardware driver, wraps it in an RPC service, and starts the Pylabnet network server. |
| [`hardware/quantum_machines/OPX.py`](OPX.py) | Implements OPX operations, builds QUA programs, and communicates with the Quantum Machines API. |
| [`network/client_server/OPX.py`](../../network/client_server/OPX.py) | Defines the service methods exposed over RPC and the client methods an experiment calls. |
| QUA config module above | Defines the physical/logical Quantum Machines configuration used by the driver. |
| Experiment script | Contains experiment-specific logic and calls the OPX client to schedule operations. |

`OPX_new.py` is not the normal OPX server driver path: the current server module
imports `OPX` and instantiates `OPX.Driver`.

## Run flow: from GUI to hardware

For a script config that lists an OPX server, the usual flow is:

1. Launch Control starts a `Launcher` process for the selected script config.
2. The Launcher reads the script config and loads the named OPX device config.
3. It either connects to an already-running server or starts one, depending on
   the device config and runtime conditions.
   - If the device config has both `ip` and `port`, the current Launcher code
     connects to that address directly.
   - In the launch path, `launch_device_server(...)` passes the device ID,
     config name, and server port to `pylabnet_server.py`.
4. `pylabnet_server.py` imports `launchers/servers/OPX.py` and calls its
   `launch(...)` function with keyword arguments (`logger`, `port`, `device_id`,
   `config`). Python collects these in that function's `kwargs` dictionary
   because its signature is `launch(**kwargs)`.
5. The OPX server module creates `hardware.quantum_machines.OPX.Driver`, wraps
   the driver in the `OPX.Service`, and starts a `GenericServer`.
6. The Launcher creates an `OPX.Client` for the server and passes its client
   collection to the experiment script's `launch(...)` function.
7. Experiment calls go through the client/service to the driver; the driver
   builds or submits QUA programs through the Quantum Machines API.

There are two network endpoints to distinguish:

- The Pylabnet device JSON's `ip`/`port` are for the Pylabnet server/client RPC.
- `OPX.py` currently sets `qop_ip` and `cluster_name` in the driver; these are
  for the driver's Quantum Machines connection.

Changing one address does not automatically change the other.

## How the sequence stack works

The driver supports a build-then-execute pattern:

```python
opx.build_stack()

with opx.for_("i", 0, 100, 1):
    opx.set_ao_voltage(
        pulse="const",
        ao_channel=3,
        amplitude=0.4,
        length=100,
        frequency=50,
    )

job = opx.execute()
```

This is illustrative of the current API; obtain `opx` from the client object
passed to the experiment. Check the client wrapper for the exact callable
surface.

During `build_stack()`, the driver enters stack-building mode. Sequence methods
append operation descriptions to a Python list instead of running a separate
QUA program immediately. Context managers such as `for_`, `while_`, and
conditional helpers add nested blocks to that representation. The stack is an
intermediate description of a sequence, not a physical hardware queue.

During `execute()`, the driver:

1. Opens a QUA `program()` context.
2. Declares QUA variables needed by queued measurements/loops/conditionals.
3. Walks the operation stack in `_emit_stack()` and emits QUA statements such
   as `play`, `wait`, `measure`, loops, and conditionals.
4. Adds stream-processing statements when results are requested.
5. Opens the Quantum Machine using the QUA config and submits the completed
   program. Measurement paths fetch results and return them; other paths return
   the job.

Calls outside stack-building mode may execute immediately, and some methods
create infinite-loop programs in that mode. Prefer the explicit stack flow for
finite experimental sequences unless the immediate behavior is intentional.

## Elements, pulses, and automatic generation

QUA programs operate on **named elements**. The QUA config maps each element to
hardware ports and maps operation names to pulse definitions. Pulses in turn
refer to waveforms and other definitions. For example:

```text
element -> controller/FEM/port mapping
        -> operation name -> pulse -> waveform
```

This layering lets a QUA sequence say “play this operation on this element”
without spelling out the hardware mapping at every play statement.

The current driver can dynamically add definitions:

- `create_new_ao_elem(...)` generates an AO element name and adds element,
  pulse, and waveform entries to the QUA config.
- `create_new_do_elem(...)` generates a digital element and ON pulse.
- `create_new_ai_elem(...)` generates an element and measurement pulse for an
  AO/AI pair.

The generated element names include channel/counter information. The creator
returns the generated name. In stack mode, an operation such as
`set_ao_voltage(ao_channel=..., element=None, ...)` normally creates an element
and then queues the operation using it.

If you explicitly pass `element="my_element"`, the driver does not create that
specific name automatically. That element and the requested operation must
already exist in the active QUA config; otherwise QUA compilation or execution
can fail. To use an automatically generated element explicitly, call the
creator, save its returned name, then pass that name to later operations.

## Current design limitations to account for

These are useful investigation targets before relying on dynamic config
generation or using the driver as a base for another model:

1. **Config ownership/reset:** `OPX.py` imports `config` directly from the config
   module. `build_stack()` later assigns a new value to `opx_cfg.config`, but
   that does not rebind the driver's separately imported `config` name.
   Additionally, `.copy()` is shallow, so nested dictionaries may still be
   shared. Test and fix config ownership/reset behavior before assuming each
   sequence starts from a clean independent config.
2. **Coupled responsibilities:** the driver handles connection setup, dynamic
   QUA config mutation, sequence representation, QUA code generation, execution,
   and result fetching. The RPC client/service also mirrors many operation
   signatures. This makes changes cross-cutting and contributes to the
   perceived clunkiness.
3. **Implicit modes:** methods can queue operations in stack mode but execute
   immediately when stack mode is off. Method names do not always make the
   difference clear.
4. **Physical assumptions:** the driver/config use a hardcoded QOP host/cluster,
   a `con1` controller, and particular channel mappings.
5. **Timing/synchronization:** operations on different elements can be scheduled
   in parallel. Use explicit `align`/`wait` behavior where the experiment needs
   synchronization, and validate timing against the real setup.

## Suggested experiment-facing direction

The experiment goal is to describe **which pulse goes out of which logical
channel and when**, while keeping physical port details in the hardware config.
A cleaner future interface could look conceptually like:

```python
with opx.sequence() as seq:
    seq.play(channel="cooling", pulse="gaussian", amplitude=0.3, duration_ns=200)
    seq.wait(500, on="cooling")
    seq.play(channel="repump", pulse="square", amplitude=0.2, duration_ns=100)
    seq.measure(channel="probe", into="signal")

result = opx.run(seq)
```

This is a design sketch, **not an API currently implemented in this repository**.
The conceptual separation would be:

1. Hardware config maps stable logical channels to physical controller/FEM ports.
2. A pulse library defines reusable pulse shapes and defaults.
3. The experiment sequence specifies operations and timing.
4. A compiler translates that sequence into QUA config/program structures.

For a safe refactor, preserve the current client API initially, write tests for
representative sequences, and compare the emitted/compiled QUA behavior before
migrating experiment scripts.

## OPX1000: likely adaptation points

QUA may be shared across Quantum Machines hardware, but QUA alone does not
guarantee this driver/config work unchanged. The current code has concrete
OPX-specific assumptions that need to be checked:

- **Controller hardware config:** current config entries describe ports directly
  under `con1`. OPX1000 configurations describe controller FEMs and define
  hardware ports under the relevant FEM.
- **Port addresses:** OPX1000 FEM port references include controller, FEM slot,
  and port number, for example `("con1", 1, 2)`. The current dynamic builders
  use two-part addresses such as `("con1", ao_channel)`.
- **FEM type and setup:** configuration depends on the actual LF-FEM/MW-FEM
  hardware, slots, connected instruments, and QOP version.
- **Connection settings/API compatibility:** verify the installed QOP/QM SDK
  version and the OPX1000 connection details; the driver currently hardcodes
  `qop_ip` and `cluster_name`.
- **Measurement and timing:** validate input routing, acquisition settings,
  time-of-flight, sampling, output modes, and pulse constraints on the target
  hardware.

Recommended first step: keep the current OPX path unchanged, create an
OPX1000-specific hardware config, and run a minimal supported program on the
target system. Use a separate driver or adapter if connection setup, config
generation, or hardware operations differ. Avoid copying the whole driver as a
long-term design until the common and model-specific parts are understood. The
Pylabnet client/service wrapper can be shared if its exposed operations remain
compatible; server selection must still route the OPX1000 config to the correct
driver.

See the official Quantum Machines documentation for the current [QUA
configuration model](https://docs.quantum-machines.co/latest/docs/Introduction/config/)
and [QOP concepts](https://docs.quantum-machines.co/latest/docs/Introduction/qop_overview/).

## Before making changes

- Check `git status` and preserve unrelated working-tree changes.
- Confirm which server module and config the script actually selects.
- Identify whether the requested change concerns Pylabnet RPC addressing or
  the Quantum Machines/QOP connection.
- Add focused tests for generated elements, repeated `build_stack()` calls,
  nested sequence blocks, and config isolation before changing these behaviors.
- Validate on a simulator or supported test setup before hardware execution.
