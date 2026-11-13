# OPX1000 setup handoff

## Latest saved state: end-of-session update (2026-11-11)

This section supersedes older pending-test and recommended-action sections.

- User confirmed a successful DataTaker-to-OPX1000 connection with Python
  3.10.11 / .venv-opx310 and RPyC 6.0.2, with the launcher join() call disabled.
  Keep that interpreter selection and leave join() disabled unless requested.
- Python 3.12 .venv-opx is preserved. Regular .venv remains Python 3.8 with
  RPyC 6.0.2. requirements.txt still pins RPyC 5.0.1; setup.py permits >=4.1.2.
  Dependency manifests have not been changed; fresh installs need review.
- Fixed simulation completion handling: RPC get_status() returns the actual
  QOP job status; DataTaker waits for Done rather than treating Simulated as
  completion; Driver.execute(wait=True) now waits for simulated jobs too.
  Updated existing tests; all 40 OPX1000 tests passed on Python 3.10.11.
- opx1000_lf_pulse_train.py defaults now request HARDWARE OUTPUT (simulate=0)
  and 1,000,000 pulses. At the default 10,000 ns period, nominal output duration
  is 10 seconds, with 1,000 ns high time per pulse. Reload the experiment in
  DataTaker to refresh defaults. No measurements are acquired; the hardware-mode
  plot is a commanded waveform preview, not measured output. Physical output
  has not been independently verified by this agent.
- Pulse trains use a QUA hardware loop, not a million copied waveform entries.
  Current driver requires widths >=16 ns and nonzero gaps >=16 ns, in 4 ns
  increments. Existing config uses 1 GS/s and amplified LF outputs (330 MHz
  analog bandwidth); these differ from QUA instruction timing constraints.
- Actual user runs reached QOP: simulation sample retrieval failed before the
  completion fix; physical attempts encountered existing QM resource locks.
  Full template declares LF input/output ports on slots 3 and 4, so its resource
  claim is broader than this single-output experiment needs. Driver retains
  close_other_machines=False and does not reclaim other clients' resources.
- At user request, all open QMs were closed at approximately 15:55 and again
  15:58 local time; the final check at 15:58:45 returned []. This is historical
  state only: subsequent scripts/notebooks can reopen sessions. Closing all
  QMs stops their jobs. Do not assume the cluster is still empty.
- Read-only status check uses QuantumMachinesManager(host='192.168.0.188',
  cluster_name='OPX1000_cluster').list_open_qms(); close the manager channel
  afterward. Only close QM sessions when the user authorizes it.
- Upstream Lukin local launches use sys.executable. This fork added local
  python_path support in ada5d91a and custom-interpreter PYTHONPATH support in
  f82ba9ae. Other device configs without python_path retain their interpreter.
- All source/config edits remain local, uncommitted, and unpushed. Existing
  unrelated data_taker_xband.json is preserved. A local filesystem snapshot
  of changed/untracked files is stored under temp/opx1000-saved-*; it includes
  the pre-removal join() launcher backup when available. Venvs are not copied;
  the Python 3.10 package snapshot remains in its requirements-installed.txt.


## Latest update: Python 3.10 comparison (2026-11-11)

This section supersedes the older environment/current-worktree status below.

- Installed Python 3.10.11 through pyenv at
  `C:\Users\hybri\.pyenv\pyenv-win\versions\3.10.11\python.exe`.
- Created `.venv-opx310` in the Documents checkout; `.venv-opx` (Python 3.12)
  is preserved. The project default and regular `.venv` remain Python 3.8.
- OPX1000.json now selects `.venv-opx310\Scripts\python.exe` via python_path.
  The existing helper_methods.py PYTHONPATH support is sufficient; no new
  interpreter-launch logic was needed. Added `.venv-opx310/` to .gitignore.
- Installed qm-qua 1.5.0, RPyC 6.0.2, NumPy 2.2.6, Paramiko 5.1.0,
  python-decouple 3.8, pyqtgraph 0.14.0, PyQt5 5.15.11, slack-sdk 3.46.0,
  and resolved transitive dependencies. NumPy differs from the Python 3.12
  environment because Python 3.10 needs a compatible release (<2.3).
- The launcher join() call remains REMOVED for the comparison. Its unused
  GenericServer helper/import and launch-return assignment remain in source.
  The exact pre-removal launcher backup is
  `C:\Users\hybri\AppData\Local\Temp\pylabnet_server.before_join_test.738bcf1d455940f5b06a51397d40f914.py`.
- User reported repeated connection failures without join() under Python 3.12;
  one attempt advertised port 10907, disconnected from logging shortly after
  config load, and DataTaker received connection refusals. The supplied log
  did not contain a server-thread traceback establishing the cause.
- CPython issue https://github.com/python/cpython/issues/115533 documents a
  Python 3.12 behavior change preventing creation of additional threads after
  the main thread exits. This is a plausible mechanism for the earlier thread
  shutdown error; it has not been established as the exact production failure.
- Python 3.10 verification passed: pip check; server/QUA imports; temporary
  loopback TLS/RPC using the real OPX1000 launcher and Python 3.8 Client wrapper.
  The smoke server main thread returned without join(); the client retrieved
  initial status and controller config successfully. Only the temporary smoke
  process was terminated; existing lab/server processes were untouched.
- No QOP connection, simulation, or hardware job was performed. The user's
  next fresh DataTaker/server launch still needs to verify the production path.
- Installed package snapshot: `.venv-opx310/requirements-installed.txt`.


This note records the local OPX1000/Pylabnet setup and troubleshooting context
for future work. It describes this machine's state, not a portable installation.

## Which checkout is in use

- The Pylabnet checkout the user normally launches is
  `C:\Users\hybri\Documents\pylabnet\pylabnet`, now on branch
  `atoms-circuits`.
- A second clone exists at `C:\Users\hybri\Pylabnet-Test`, branch
  `atoms-circuits`. It is not the checkout used by the user's normal launcher.
- Both clones point to `https://github.com/hybrid-lab/Pylabnet-Test.git`.
- Changes in one checkout do not appear in the other unless deliberately
  transferred or committed/pulled.
- The OPX1000 interpreter/PYTHONPATH setup is committed in both branches:
  commit `90c292b2` on `gerlach_yajur` and commit `f82ba9ae` on
  `atoms-circuits`. The `atoms-circuits` commit was pushed to `origin`, and
  both local `atoms-circuits` checkouts were updated to it.
- The local `pylabnet/configs/network_configuration.json` selects
  `192.168.0.106` for Pylabnet's RPC server address. This is the machine's
  Ethernet address used by Windows to route to the OPX; the file is a local,
  machine-specific setting.

## OPX1000 Python environment

- The Documents checkout has an ignored venv at
  `C:\Users\hybri\Documents\pylabnet\pylabnet\.venv-opx`.
- Its interpreter is
  `C:\Users\hybri\Documents\pylabnet\pylabnet\.venv-opx\Scripts\python.exe`,
  Python 3.12.0.
- `qm-qua` 1.5.0 and the candidate server dependencies were installed there:
  `rpyc`, `numpy`, `paramiko`, `python-decouple`, `pyqtgraph`, `PyQt5`, and
  `slack-sdk` (plus their transitive dependencies).
- RPyC 6.0.2 is installed in both `.venv-opx` and the regular Pylabnet
  `.venv`. The regular `.venv` previously had RPyC 5.0.1.
- The server module import and `from qm import qua` were verified in this venv
  with the Documents checkout on `PYTHONPATH`. A live server startup and QOP
  hardware connection have not been verified.
- The OPX1000 device config
  `pylabnet/configs/devices/OPX1000/OPX1000.json` has a machine-specific
  `python_path` pointing to the venv above.
- `pylabnet/utils/helper_methods.py` was changed so that a locally launched
  device server with `python_path` receives the checkout root in its child
  process `PYTHONPATH`. This lets the venv import the existing Pylabnet source
  without installing Pylabnet into the venv.
- `.gitignore` now ignores `.venv-opx/`.
- The interpreter and launch changes are committed on both branches. The
  machine-specific interpreter path in the device config points to the
  Documents checkout's `.venv-opx`.

## Setup changes previously made in the Pylabnet-Test clone

The useful local setup changes were applied to both checkouts. Do not copy
back the stale interpreter path from the Pylabnet-Test clone:

1. `.gitignore`: add `.venv-opx/` under the environment ignore rules. This is
   committed on both branches.
2. `pylabnet/utils/helper_methods.py`: for local device-server launches, use
   `config_dict.get('python_path', sys.executable)`. When the config supplies
   `python_path`, copy `os.environ`, prepend the checkout root to the child
   process's `PYTHONPATH` (without duplicating it), and pass that environment
   to `subprocess.Popen`. Preserve the existing `Popen(cmd, shell=True)` path
   when no custom interpreter is configured. This is committed on both
   branches.
3. `pylabnet/configs/devices/OPX1000/OPX1000.json`: the Pylabnet-Test clone's
   local path was
   `C:\Users\hybri\Pylabnet-Test\pylabnet\.venv-opx\Scripts\python.exe`.
   That venv was deleted. Both configs now point to the Documents checkout's
   working venv instead.

The Pylabnet-Test clone also contains
`pylabnet/configs/scripts/data_taker/data_taker_opx_only.json`; it selects
`OPX` (not `OPX1000`) and has placeholder `C:\Users\User\...` script and data
paths. It is not a usable OPX1000 config. No unique working DataTaker config
from that clone needs to be preserved.

The previous attempt to install the checkout's full, old `requirements.txt`
failed while building its pinned `matplotlib==3.4.3` under Python 3.12.
Matplotlib is not imported on the OPX1000 server's direct startup path. Do not
assume the entire project requirements set is installed in `.venv-opx`.

## Earlier QOP error and current status

- A prior execute/simulate traceback came from the Documents checkout's regular
  `.venv` (Python 3.8.10), which had `qm-qua==1.2.2`.
- The reported controller version was QOP 3.8.1. The error was
  `QMConnectionError: ... V1BETA is not supported`.
- `qm-qua` 1.5.0 was installed in `.venv-opx`; Quantum Machines release notes
  report it tested against QOP 3.8.x. Using the newer venv may address that
  version mismatch, but it has not yet been confirmed against hardware.
- A later DataTaker launch produced an RPyC
  `ValueError: not enough values to unpack (expected 3, got 0)` with
  `client connection terminated abruptly`. This indicates a connection closed
  before a complete RPyC message. A loopback TLS/RPC test reproduced that
  symptom when the Python 3.8 client used RPyC 5.0.1 and the Python 3.12 server
  used RPyC 6.0.2.
- RPyC 5.0.1 was briefly tried in `.venv-opx`, but it cannot serve under Python
  3.12 because it calls the removed `ssl.wrap_socket`. It was restored to
  6.0.2. The regular Pylabnet `.venv` was upgraded to RPyC 6.0.2; imports
  succeeded, and a loopback TLS/RPC call between Python 3.8/RPyC 6.0.2 and
  Python 3.12/RPyC 6.0.2 returned `pong`.
- A loopback smoke test using the real OPX1000 server launcher and its
  `Client` wrapper passed TLS/RPC calls from the regular `.venv` to
  `.venv-opx`, returning the controller config and initial status. This did not
  contact QOP hardware.
- A `RuntimeError: can't create new thread at interpreter shutdown` was
  observed during an earlier production-launch attempt. A `join()` change was
  made in response, and a Python 3.8 client subsequently read status/config
  from the server at `192.168.0.106:9967`. However, the RPyC upgrade and
  launcher change were both in place by then, so that success does not prove
  the `join()` change caused the connection to work.
- The user has since reported that DataTaker connected on a later retry
  (2026-11-11, approximately 14:33 local time). This is the first reported
  successful DataTaker connection, but it still does not isolate which change
  mattered. No QOP simulation or hardware execution was run.
- **Do not describe the `join()` change as the established root cause.**
  Existing Pylabnet server launchers use the same `GenericServer` background
  thread pattern. Its thread is created without `daemon=True`, so it would
  normally keep the Python process alive; the necessity of the added `join()`
  is therefore uncertain. The earlier thread-shutdown exception was real, but
  its causal relationship to the original connection failure is unproven.
- RPyC compatibility is a separately confirmed issue: mismatched RPyC 5.0.1
  and 6.0.2 reproduced an abrupt TLS/RPC disconnect, and RPyC 5.0.1 could not
  serve under Python 3.12 because it uses the removed `ssl.wrap_socket`.
  RPyC 6.0.2 is now installed in both local environments. Since the working
  DataTaker retry happened after both the RPyC upgrade and launcher change, it
  is not yet proven that the RPyC change alone fixed the user's connection.
- During a later retry, DataTaker reported the OPX1000 server at
  `169.254.44.104`. Windows identified that address as the Npcap Loopback
  Adapter, and no listener was present on the reported port. Pylabnet's
  `get_ip()` returned that address because no `network_configuration.json`
  existed. The local network config has since been added and verified to make
  `get_ip()` return `192.168.0.106` from both the regular venv and `.venv-opx`.
  A subsequent DataTaker attempt correctly targeted the OPX RPC address, but
  no listener remained. The OPX server's log client connected to the log server
  and loaded `OPX1000.json`, then disconnected; this did not prove that the
  separate device RPC server remained usable. The user later reported a
  successful DataTaker connection after both RPyC and launcher changes, but
  neither change has been isolated as the cause.

## DataTaker config follow-up

- The Documents checkout's current DataTaker config is
  `pylabnet/configs/scripts/data_taker/data_taker.json`; on `atoms-circuits`
  it contains placeholders and is not the OPX1000-only config.
- The OPX1000-only config is now
  `pylabnet/configs/scripts/data_taker/data_taker_opx1000_only.json`. It
  selects only OPX1000 and retains the Documents checkout's DataTaker script,
  experiment, and save paths.
- Its `auto_save` value is JSON `false` (not the string `"False"`), because
  DataTaker tests that setting for truthiness.
- A tagged `data_taker_opx_only.json` exists in the Pylabnet-Test clone, but it
  is for `OPX`, not `OPX1000`, and contains placeholder paths. Do not treat it
  as the requested working config.

## Current worktree and change inventory

The Documents checkout is on `atoms-circuits`, tracking `origin/atoms-circuits`.
The following changes are local and uncommitted; none of them has been pushed:

- `pylabnet/launchers/pylabnet_server.py`: retains the `launch()` return value
  and calls `join()` when it is a `GenericServer`.
- `pylabnet/network/core/generic_server.py`: adds `GenericServer.join()`.
- `pylabnet/configs/network_configuration.json`: local machine IP selection
  (`192.168.0.106`).
- `pylabnet/configs/scripts/data_taker/data_taker_opx1000_only.json`: local
  DataTaker config selecting OPX1000 only.
- `OPX1000_SETUP_HANDOFF.md`: this handoff.
- `pylabnet/configs/scripts/data_taker/data_taker_xband.json` is also
  untracked in the worktree and was not created or modified as part of this
  troubleshooting; inspect before touching it.

RPyC 6.0.2 was installed directly into both `.venv` (Python 3.8.10) and
`.venv-opx` (Python 3.12.0). These are environment changes, not tracked source
changes or a dependency-manifest update. Preserve both installations during
the proposed test unless the user specifically asks to test RPyC versions.

`git diff --check` passed and Pylance reported no errors in the two modified
Python files. A targeted test invocation found no tests at the specified path;
no full test suite was run. The server was last confirmed listening on
`192.168.0.106:9967` at about 14:26 on 2026-11-11, before the user's later
successful DataTaker retry. Check with the user before assuming it remains
running or stopping/restarting it.

## Codex handoff: recommended next action

The user wants to determine whether the RPyC upgrade alone fixed the
DataTaker-to-OPX connection, or whether the `join()` code change was also
necessary. The user has asked that all changes be preserved so they can be
quickly restored. Do not remove or rewrite the `join()` change without
performing the controlled comparison below and do not revert the RPyC upgrade.

1. Coordinate a test window with the user. Restarting/stopping the OPX server
   can interrupt their live connection. Do not start a second server on port
   `9967` or terminate the currently running server without confirming it is
   safe.
2. Keep RPyC 6.0.2 installed in both environments. Temporarily disable only
   the `server_instance.join()` call in `pylabnet/launchers/pylabnet_server.py`
   (preserve the current diff so it can be restored exactly).
3. Restart the OPX1000 server through the same Pylabnet/DataTaker path and
   repeat the same DataTaker connection check. Record the server process,
   listener address/port, and DataTaker result/log. Do not run a QOP job as
   part of this connection-only test.
4. If DataTaker connects without `join()`, that suggests the call is not
   required for this run. If it fails, restore `join()` and repeat once to
   check whether the result is reproducible. In either case, restore and keep
   the original `join()` change in the worktree after the comparison unless
   the user explicitly asks to remove it; the user wants it preserved for
   quick reimplementation.
5. Keep conclusions limited to what the comparison shows. A successful/failed
   retry can still be affected by process state, port selection, or other
   configuration, so record the actual endpoint and ensure the test changes
   only one variable.

Separate follow-ups, only after the user asks or confirms:

- If the OPX1000-only DataTaker config is wanted in the remote repository,
  review it and then commit/push it from `atoms-circuits`.
- Investigate the unrelated `static_proxy.json` logger path and stale
  `master_ip` only after confirming intended local values.
- Test QOP simulation or hardware only with explicit lab readiness and user
  direction. Neither has been tested by the RPC connection checks.
