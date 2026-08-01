# -*- coding: utf-8 -*-
"""
NI DAQmx Driver with OPX-style stack infrastructure, extended for NI "Task" parallelism.

Key ideas:
- NI can run multiple tasks in parallel (e.g., AO and DO simultaneously), but each *subsystem*
  (AO/AI/DO/DI) should be managed by its own Task.
- This driver provides separate command stacks stored in a dict with keys:
    "ao", "ai", "do", "di"
- You can fill stacks (build_stack mode), then call execute() to:
    1) Compile each stack into a single DAQmx Task per subsystem
    2) Configure timing (sample clock) and optional digital start trigger per subsystem
    3) Start tasks in a safe order (inputs first, then outputs) and collect readbacks
    4) Return a dict of results for AI/DI operations (and optional debug metadata)

NEW (for simultaneous multi-device execution):
- Added arm() and finalize() so you can:
    arm(dev1); arm(dev2); start OPX; finalize(dev1); finalize(dev2)
  This prevents execute() from blocking/closing tasks before other devices are armed.

Backwards compatibility:
- execute() is still available; it now uses arm()+finalize() internally.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import re

import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode
import numpy as np

# Compatibility for older numpy code
if not hasattr(np, "bool"):
    np.bool = np.bool_

from pylabnet.utils.logging.logger import LogHandler
from pylabnet.utils.decorators.dummy_wrapper import dummy_wrap


# -----------------------------
# Internal operation descriptors
# -----------------------------

@dataclass
class _TriggerCfg:
    trig_line: str = "PFI0"          # e.g. "PFI0", "PXI_Trig0"
    edge: Edge = Edge.RISING        # nidaqmx.constants.Edge


@dataclass
class _ArmedHandle:
    """
    Returned by Driver.arm(). Holds open DAQmx Task objects that must be finalized/closed.
    """
    tasks: Dict[str, nidaqmx.Task]
    compiled: Dict[str, Dict[str, Any]]
    meta: Dict[str, Any]


# -----------------------------
# Driver
# -----------------------------

class Driver:
    """
    Driver for NI DAQmx card. Supports:
    - Analog Output (AO)
    - Analog Input (AI)
    - Digital Output (DO)
    - Digital Input (DI)
    - Optional per-subsystem digital start trigger

    NEW:
    - arm(): compile + create tasks + configure + write buffers + start (arm) tasks, but DO NOT wait/close.
    - finalize(handle): wait/read/close tasks from an arm() call.

    Stack usage example:

        dev = Driver("Dev1")
        dev.build_stack()
        dev.set_ao_voltage("ao0", [0, 1, 0], sample_rate=10000)
        dev.set_trigger(target="ao", trig_line="PFI0", edge=Edge.RISING)
        out = dev.execute()
    """

    STACK_KEYS = ("ao", "ai", "do", "di")

    def __init__(self, device_name: str, logger=None, dummy: bool = False):
        self.dev = device_name
        self.log = LogHandler(logger=logger)
        self.dummy = dummy

        # Defaults used inside _compile_stacks and _cfg_timing_if_needed
        self.default_clock_source = "/PXI1Slot2_1/PFI0"
        # NOTE: APFI0 is not a digital trigger terminal. Kept for compatibility with your existing file,
        # but NOT used by _cfg_trigger_if_needed (which uses set_trigger()).
        self.default_start_trigger = "/PXI1Slot2_1/APFI0"
        self.not_OPX_clock = False

        # Try to get info of DAQ device to verify connection
        try:
            ni_daq_device = nidaqmx.system.device.Device(name=device_name)
            self.log.info(
                "Successfully connected to NI DAQ '{device_name}' (type: {product_type})\n"
                "".format(
                    device_name=ni_daq_device.name,
                    product_type=ni_daq_device.product_type
                )
            )
        except (nidaqmx.DaqError, OSError):
            try:
                ni_daqs_names = nidaqmx.system._collections.device_collection.\
                    DeviceCollection().device_names
                self.log.error(
                    "NI DAQ card {} not found. \n"
                    "There are {} NI DAQs available: \n "
                    "    {}"
                    "".format(
                        device_name,
                        len(ni_daqs_names),
                        ni_daqs_names
                    )
                )
            except Exception:
                self.log.error("No NI modules found")
            if self.dummy:
                self.log.info("Entering dummy mode instead")

        # Timed counters (unchanged)
        self.counters: Dict[str, TimedCounter] = {}

        # Stack mode control
        self.adding_to_stack: bool = False
        self.stacks: Dict[str, List[list]] = {k: [] for k in self.STACK_KEYS}

        # Optional trigger config per subsystem
        self._triggers: Dict[str, Optional[_TriggerCfg]] = {k: None for k in self.STACK_KEYS}

        # Auto-labeling for read ops
        self._result_counter: int = 0

    # ---------------------------------------------------------------------
    # Stack control
    # ---------------------------------------------------------------------
    def not_use_OPX_clock(self):
        self.not_OPX_clock = True

    def build_stack(self) -> None:
        """Enable stack mode and clear existing stacks + triggers."""
        self.adding_to_stack = True
        self.stacks = {k: [] for k in self.STACK_KEYS}
        self._triggers = {k: None for k in self.STACK_KEYS}
        self._result_counter = 0

    def clear_stack(self) -> None:
        """Clear stacks (does not change stack mode on/off)."""
        self.stacks = {k: [] for k in self.STACK_KEYS}
        self._triggers = {k: None for k in self.STACK_KEYS}
        self._result_counter = 0

    def _append(self, stack_key: str, op: list) -> None:
        """Append an operation to the correct subsystem stack."""
        if stack_key not in self.stacks:
            raise ValueError(f"Unknown stack key {stack_key!r}. Must be one of {self.STACK_KEYS}.")
        self.stacks[stack_key].append(op)
        self.log.info(f"[NI] stack[{stack_key}] append: {op}")

    # ---------------------------------------------------------------------
    # Trigger API
    # ---------------------------------------------------------------------

    def set_trigger(self, target: str, trig_line: str = "PFI0", edge: Edge = Edge.RISING) -> None:
        """
        Define a digital edge start trigger for a given subsystem task.

        :param target: one of "ao", "ai", "do", "di"
        :param trig_line: e.g. "PFI0", "PXI_Trig0"
        :param edge: nidaqmx.constants.Edge.RISING / FALLING
        """
        if target not in self._triggers:
            raise ValueError(f"target must be one of {self.STACK_KEYS}, got {target!r}")
        cfg = _TriggerCfg(trig_line=trig_line, edge=edge)

        if self.adding_to_stack:
            # Record as metadata op so "stack replay" captures intent
            self._append(target, ["trigger", trig_line, edge])

        # Always set effective trigger config
        self._triggers[target] = cfg

    # ---------------------------------------------------------------------
    # AO / DO / AI / DI public methods (stack or immediate)
    # ---------------------------------------------------------------------

    @dummy_wrap
    def set_ao_voltage(
        self,
        ao_channel: str,
        voltages: Union[float, int, Sequence[float]],
        *,
        sample_rate: Optional[float] = None,
        max_range: float = 10.0,
    ) -> None:
        """
        Queue or immediately output analog voltage(s).
        """
        if self.adding_to_stack:
            self._append("ao", ["ao_write", ao_channel, _to_1d_array(voltages), sample_rate, max_range])
            return
        self._ao_write_now(ao_channel, voltages, sample_rate=sample_rate, max_range=max_range)

    def set_do_voltage(
        self,
        do_channel: str,
        value: Union[bool, int, Sequence[Union[bool, int]]],
        *,
        sample_rate: Optional[float] = None,
    ) -> None:
        """
        Queue or immediately output digital value(s).
        """
        # Allow user-friendly "dioX" naming (e.g., "dio0", "DIO18")
        # Maps DIO index -> port{index//8}/line{index%8}
        if isinstance(do_channel, str):
            s = do_channel.strip().lower()
            m = re.fullmatch(r"dio(\d+)", s)
            if m:
                dio = int(m.group(1))
                port = dio // 8
                line = dio % 8
                do_channel = f"port{port}/line{line}"

        if self.adding_to_stack:
            self._append("do", ["do_write", do_channel, _to_bool_array(value), sample_rate])
            return

        self._do_write_now(do_channel, value, sample_rate=sample_rate)

    def get_ai_voltage(
        self,
        ai_channel: str,
        *,
        num_samples: int = 1,
        max_range: float = 10.0,
        sample_rate: float = 100000.0,
        variable_name: Optional[str] = None,
    ) -> Union[List[float], str]:
        """
        Queue or immediately measure analog input.
        """
        if self.adding_to_stack:
            if variable_name is None:
                self._result_counter += 1
                variable_name = f"ai_{self._result_counter}"
            self._append("ai", ["ai_read", variable_name, ai_channel, int(num_samples), float(max_range), float(sample_rate)])
            return variable_name

        return self._ai_read_now(ai_channel, num_samples=num_samples, max_range=max_range, sample_rate=sample_rate)

    def get_di_state(
        self,
        port: str,
        di_channel: str,
        *,
        num_samples: int = 1,
        sample_rate: float = 100000.0,
        variable_name: Optional[str] = None,
    ) -> Union[bool, List[bool], str]:
        """
        Queue or immediately measure digital input line(s).
        """
        if self.adding_to_stack:
            if variable_name is None:
                self._result_counter += 1
                variable_name = f"di_{self._result_counter}"
            self._append("di", ["di_read", variable_name, port, di_channel, int(num_samples), float(sample_rate)])
            return variable_name

        return self._di_read_now(port, di_channel, num_samples=num_samples, sample_rate=sample_rate)

    # ---------------------------------------------------------------------
    # NEW: Build tasks helper + arm/finalize API
    # ---------------------------------------------------------------------

    def _build_tasks_from_compiled(
        self,
        compiled: Dict[str, Dict[str, Any]],
        regeneration: bool = False,
    ):
        """
        Create DAQmx tasks for each non-empty subsystem stack, add channels,
        configure timing + trigger, and write output buffers (AO/DO) with auto_start=False.
        Does NOT start tasks and does NOT close tasks.
        """
        tasks: Dict[str, nidaqmx.Task] = {}
        meta: Dict[str, Any] = {"compiled": {}}

        for key in self.STACK_KEYS:
            info = compiled[key]
            if not info["ops"]:
                continue

            task = nidaqmx.Task()
            tasks[key] = task

            if key == "ao":
                for ch in info["channels"]:
                    task.ao_channels.add_ao_voltage_chan(self._gen_ch_path(ch))
                self._cfg_timing_if_needed(task, info, key=key, regeneration=regeneration)
                _cfg_trigger_if_needed(task, info)
                if regeneration:
                    task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
                if info["write_buffer"] is not None:
                    task.write(info["write_buffer"], auto_start=False)

            elif key == "do":
                for ch in info["channels"]:
                    task.do_channels.add_do_chan(
                        self._gen_ch_path(ch),
                        line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE,
                    )
                self._cfg_timing_if_needed(task, info, key=key, regeneration=regeneration)
                _cfg_trigger_if_needed(task, info)
                if regeneration:
                    task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
                if info["write_buffer"] is not None:
                    task.write(info["write_buffer"], auto_start=False)

            elif key == "ai":
                for ch in info["channels"]:
                    ai = task.ai_channels.add_ai_voltage_chan(self._gen_ch_path(ch))
                    ai.ai_rng_high = float(info["max_range"])
                    ai.ai_rng_low = -float(info["max_range"])
                self._cfg_timing_if_needed(task, info, key=key, regeneration=regeneration)
                _cfg_trigger_if_needed(task, info)

            elif key == "di":
                for ch in info["channels"]:
                    task.di_channels.add_di_chan(
                        self._gen_ch_path(ch),
                        line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE,
                    )
                self._cfg_timing_if_needed(task, info, key=key, regeneration=regeneration)
                _cfg_trigger_if_needed(task, info)

            meta["compiled"][key] = {
                "channels": info["channels"],
                "total_samples": info["total_samples"],
                "sample_rate": info["sample_rate"],
                "trigger": info["trigger_source"],
                "clock_source": info.get("clock_source", None),
            }

        return tasks, meta

    def arm(self, regeneration: bool = False) -> _ArmedHandle:
        """
        Compile stacks -> create tasks -> configure -> write buffers -> START tasks (arm them).
        Does NOT wait for completion and does NOT close tasks.
        """
        self.adding_to_stack = False

        compiled = self._compile_stacks(self.stacks, self._triggers, dev=self.dev, log=self.log)

        if not any(compiled[k]["ops"] for k in self.STACK_KEYS):
            self.clear_stack()
            return _ArmedHandle(tasks={}, compiled=compiled, meta={"compiled": {}})

        tasks, meta = self._build_tasks_from_compiled(compiled, regeneration=regeneration)
        start_order = [k for k in ("ai", "di", "ao", "do") if k in tasks]
        for k in start_order:
            tasks[k].start()

        return _ArmedHandle(tasks=tasks, compiled=compiled, meta=meta)

    def finalize(
        self,
        handle: _ArmedHandle,
        *,
        timeout: float = 30.0,
        close: bool = True,
        force_finish: bool = False,
    ) -> Dict[str, Any]:
        """
        Wait for tasks to complete, read AI/DI, then stop/close tasks (default).
        Returns outputs + _meta.
        """
        tasks = handle.tasks
        compiled = handle.compiled
        outputs: Dict[str, Any] = {}

        try:
            # Read inputs (blocks until total_samples are available)
            if "ai" in tasks:
                info = compiled["ai"]
                raw = tasks["ai"].read(number_of_samples_per_channel=info["total_samples"])
                outputs.update(_split_ai_results(raw, info))

            if "di" in tasks:
                info = compiled["di"]
                raw = tasks["di"].read(number_of_samples_per_channel=info["total_samples"])
                outputs.update(_split_di_results(raw, info))

            # Wait for finite outputs unless caller explicitly wants to end
            # regenerative / continuous tasks by force at finalize time.
            if not force_finish:
                for k in ("ao", "do"):
                    if k in tasks:
                        tasks[k].wait_until_done(timeout=timeout)

        finally:
            if close:
                for t in tasks.values():
                    try:
                        t.stop()
                    except Exception:
                        pass
                    try:
                        t.close()
                    except Exception:
                        pass

                self.clear_stack()

        outputs["_meta"] = handle.meta
        return outputs

    # ---------------------------------------------------------------------
    # Execute: compile stacks -> tasks -> run -> return results
    # (kept for backwards compatibility; now implemented via arm()+finalize())
    # ---------------------------------------------------------------------

    def execute(self) -> Dict[str, Any]:
        """
        Backwards-compatible synchronous execution:
          arm() -> finalize()
        """
        h = self.arm()
        if not h.tasks:
            return {}
        return self.finalize(h, timeout=10.0, close=True)

    # ---------------------------------------------------------------------
    # Immediate-mode helpers (single task each call)
    # ---------------------------------------------------------------------

    def _ao_write_now(
        self,
        ao_channel: str,
        voltages: Union[float, int, Sequence[float]],
        *,
        sample_rate: Optional[float],
        max_range: float,
    ) -> None:
        channel = self._gen_ch_path(ao_channel)
        buf = _to_1d_array(voltages)
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(channel)
            if len(buf) > 1:
                if sample_rate is None:
                    raise ValueError("sample_rate is required for multi-sample AO waveforms.")
                task.timing.cfg_samp_clk_timing(
                    rate=float(sample_rate),
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=int(len(buf)),
                )
                task.write(buf.tolist(), auto_start=True)
            else:
                task.write(float(buf[0]), auto_start=True)

    def _do_write_now(
        self,
        do_channel: str,
        value: Union[bool, int, Sequence[Union[bool, int]]],
        *,
        sample_rate: Optional[float],
    ) -> None:
        channel = self._gen_ch_path(do_channel)
        buf = _to_bool_array(value)
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(
                channel,
                line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE,
            )
            if len(buf) > 1:
                if sample_rate is None:
                    raise ValueError("sample_rate is required for multi-sample DO waveforms.")
                task.timing.cfg_samp_clk_timing(
                    rate=float(sample_rate),
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=int(len(buf)),
                )
                task.write(buf.tolist(), auto_start=True)
            else:
                task.write(bool(buf[0]), auto_start=True)

    def _ai_read_now(
        self,
        ai_channel: str,
        *,
        num_samples: int,
        max_range: float,
        sample_rate: float,
    ) -> List[float]:
        channel = self._gen_ch_path(ai_channel)
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(channel)
            task.ai_channels[0].ai_rng_high = float(max_range)
            task.ai_channels[0].ai_rng_low = -float(max_range)
            task.timing.cfg_samp_clk_timing(
                rate=float(sample_rate),
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=int(num_samples),
            )
            return task.read(number_of_samples_per_channel=int(num_samples))

    def _di_read_now(
        self,
        port: str,
        di_channel: str,
        *,
        num_samples: int,
        sample_rate: float,
    ) -> Union[bool, List[bool]]:
        channel = self._gen_di_ch_path(port, di_channel)
        with nidaqmx.Task() as task:
            task.di_channels.add_di_chan(
                channel,
                line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE,
            )
            if num_samples > 1:
                task.timing.cfg_samp_clk_timing(
                    rate=float(sample_rate),
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=int(num_samples),
                )
                data = task.read(number_of_samples_per_channel=int(num_samples))
                return [bool(x) for x in data]
            return bool(task.read(number_of_samples_per_channel=1))

    # ---------------------------------------------------------------------
    # Channel path helpers
    # ---------------------------------------------------------------------

    def _gen_ch_path(self, channel: str) -> str:
        """Build DAQmx channel path string. Example: 'Dev1/ao0' or 'Dev1/port0/line0'."""
        return f"{self.dev}/{channel}"

    def _gen_di_ch_path(self, port: str, di_channel: str) -> str:
        """Build DI channel path string. Example: 'Dev1/port0/line0'."""
        return f"{self.dev}/{port}/{di_channel}"

    # ---------------------------------------------------------------------
    # TimedCounter API (unchanged)
    # ---------------------------------------------------------------------

    def create_timed_counter(self, counter_channel, physical_channel, duration=0.1, name=None):
        if name is None:
            name = f"counter_{len(self.counters)}"

        ctr_path = self._gen_ch_path(counter_channel) if "/" not in counter_channel else counter_channel
        phys = physical_channel
        if not phys.startswith("/"):
            if "/" in phys:
                phys = "/" + phys
            else:
                phys = f"/{self.dev}/{phys}"

        self.counters[name] = TimedCounter(
            logger=self.log,
            counter_channel=ctr_path,
            physical_channel=phys,
        )
        self.counters[name].set_parameters(duration)
        return name

    def start_timed_counter(self, name): self.counters[name].start()
    def stop_timed_counter(self, name): self.counters[name].terminate_counting()
    def close_timed_counter(self, name): self.counters[name].close()
    def get_count(self, name): return int(self.counters[name].count)

    # ---------------------------------------------------------------------
    # Compiler + timing config (unchanged except kept as in your original)
    # ---------------------------------------------------------------------

    def _compile_stacks(self, stacks: Dict[str, List[list]], triggers: Dict[str, Optional[_TriggerCfg]], *, dev: str, log: LogHandler) -> Dict[str, Dict[str, Any]]:
        """
        Turn stacks into compiled per-subsystem plan.
        """
        compiled: Dict[str, Dict[str, Any]] = {}

        for key in Driver.STACK_KEYS:
            ops = stacks.get(key, [])
            trig_cfg = triggers.get(key, None)
            for op in ops:
                if op and op[0] == "trigger":
                    trig_cfg = _TriggerCfg(trig_line=op[1], edge=op[2])

            info: Dict[str, Any] = {
                "ops": ops,
                "channels": [],
                "sample_rate": None,
                "total_samples": 0,
                "write_buffer": None,
                "read_plan": [],
                "max_range": 10.0,
                "trigger_cfg": trig_cfg,
                "trigger_source": None,
                "clock_source": self.default_clock_source,
                "start_trigger": self.default_start_trigger,
            }

            if trig_cfg is not None:
                info["trigger_source"] = f"/{dev}/{trig_cfg.trig_line}"

            if key == "ao":
                channels: List[str] = []
                per_chan_buffers: Dict[str, List[float]] = {}
                sample_rate: Optional[float] = None
                total_samples: int = 0

                for op in ops:
                    if not op:
                        continue
                    if op[0] == "trigger":
                        continue
                    if op[0] != "ao_write":
                        raise ValueError(f"Unsupported AO op {op[0]!r}")
                    _, ch, buf, sr, _max_range = op
                    buf = np.asarray(buf, dtype=float).reshape(-1)
                    if ch not in channels:
                        channels.append(ch)
                        per_chan_buffers[ch] = []
                    if buf.size > 1:
                        if sr is None:
                            raise ValueError("AO multi-sample waveform requires sample_rate.")
                        if sample_rate is None:
                            sample_rate = float(sr)
                        elif abs(sample_rate - float(sr)) > 1e-9:
                            raise ValueError("All AO waveform ops must share the same sample_rate in a single execute().")
                    per_chan_buffers[ch].extend(buf.tolist())
                    total_samples = max(total_samples, len(per_chan_buffers[ch]))

                for ch in channels:
                    buf_list = per_chan_buffers[ch]
                    if len(buf_list) == 0:
                        buf_list = [0.0]
                    if len(buf_list) < total_samples:
                        pad_val = buf_list[-1]
                        buf_list.extend([pad_val] * (total_samples - len(buf_list)))
                    per_chan_buffers[ch] = buf_list

                write_buffer = None
                if channels:
                    write_buffer = [per_chan_buffers[ch] for ch in channels]
                    if len(channels) == 1:
                        write_buffer = write_buffer[0]

                info.update({
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "total_samples": total_samples if total_samples > 0 else 1,
                    "write_buffer": write_buffer,
                })

            elif key == "do":
                channels: List[str] = []
                per_chan_buffers: Dict[str, List[bool]] = {}
                sample_rate: Optional[float] = None
                total_samples: int = 0

                for op in ops:
                    if not op:
                        continue
                    if op[0] == "trigger":
                        continue
                    if op[0] != "do_write":
                        raise ValueError(f"Unsupported DO op {op[0]!r}")
                    _, ch, buf, sr = op
                    buf = np.asarray(buf, dtype=np.bool_).reshape(-1)
                    if ch not in channels:
                        channels.append(ch)
                        per_chan_buffers[ch] = []
                    if buf.size > 1:
                        if sr is None:
                            raise ValueError("DO multi-sample waveform requires sample_rate.")
                        if sample_rate is None:
                            sample_rate = float(sr)
                        elif abs(sample_rate - float(sr)) > 1e-9:
                            raise ValueError("All DO waveform ops must share the same sample_rate in a single execute().")
                    per_chan_buffers[ch].extend([bool(x) for x in buf.tolist()])
                    total_samples = max(total_samples, len(per_chan_buffers[ch]))

                for ch in channels:
                    buf_list = per_chan_buffers[ch]
                    if len(buf_list) == 0:
                        buf_list = [False]
                    if len(buf_list) < total_samples:
                        pad_val = buf_list[-1]
                        buf_list.extend([pad_val] * (total_samples - len(buf_list)))
                    per_chan_buffers[ch] = buf_list

                write_buffer = None
                if channels:
                    write_buffer = [per_chan_buffers[ch] for ch in channels]
                    if len(channels) == 1:
                        write_buffer = write_buffer[0]

                info.update({
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "total_samples": total_samples if total_samples > 0 else 1,
                    "write_buffer": write_buffer,
                })

            elif key == "ai":
                channels: List[str] = []
                read_plan: List[dict] = []
                sample_rate: Optional[float] = None
                total_samples: int = 0
                max_range: float = 10.0

                for op in ops:
                    if not op:
                        continue
                    if op[0] == "trigger":
                        continue
                    if op[0] != "ai_read":
                        raise ValueError(f"Unsupported AI op {op[0]!r}")
                    _, var, ch, ns, mr, sr = op
                    ns = int(ns)
                    mr = float(mr)
                    sr = float(sr)

                    if ch not in channels:
                        channels.append(ch)

                    if sample_rate is None:
                        sample_rate = sr
                    elif abs(sample_rate - sr) > 1e-9:
                        raise ValueError("All AI ops must share the same sample_rate in a single execute().")

                    max_range = max(max_range, mr)
                    read_plan.append({"var": var, "ns": ns})
                    total_samples += ns

                info.update({
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "total_samples": total_samples,
                    "read_plan": read_plan,
                    "max_range": max_range,
                })

            elif key == "di":
                channels: List[str] = []
                read_plan: List[dict] = []
                sample_rate: Optional[float] = None
                total_samples: int = 0

                for op in ops:
                    if not op:
                        continue
                    if op[0] == "trigger":
                        continue
                    if op[0] != "di_read":
                        raise ValueError(f"Unsupported DI op {op[0]!r}")
                    _, var, port, line, ns, sr = op
                    ns = int(ns)
                    sr = float(sr)
                    ch = f"{port}/{line}"
                    if ch not in channels:
                        channels.append(ch)

                    if sample_rate is None:
                        sample_rate = sr
                    elif abs(sample_rate - sr) > 1e-9:
                        raise ValueError("All DI ops must share the same sample_rate in a single execute().")

                    read_plan.append({"var": var, "ns": ns})
                    total_samples += ns

                info.update({
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "total_samples": total_samples,
                    "read_plan": read_plan,
                })

            else:
                raise ValueError(f"Unexpected stack key {key!r}")

            compiled[key] = info

        return compiled

    def _cfg_timing_if_needed(self, task, info, key=None, regeneration: bool = False):
        N = int(info.get("total_samples", 0))
        fs = info.get("sample_rate", None)
        if N <= 1:
            return
        if fs is None:
            raise ValueError("sample_rate required for buffered timing")

        sample_mode = AcquisitionType.FINITE
        if regeneration and key in {"ao", "do"}:
            sample_mode = AcquisitionType.CONTINUOUS

        kwargs = dict(
            rate=float(fs),
            sample_mode=sample_mode,
            samps_per_chan=N,
        )

        if not self.not_OPX_clock:

            if self.dev == "PXI1Slot2_1":
                clk = info.get("clock_source", None)   # e.g. "/PXI1Slot2_1/PFI0"
                kwargs["source"] = clk
                task.timing.cfg_samp_clk_timing(**kwargs)
                # Export sample clock onto chassis backplane
                task.export_signals.samp_clk_output_term = "/PXI1Slot2_1/PXI_Trig7"
            else:
                # Consume backplane clock
                kwargs["source"] = f"/{self.dev}/PXI_Trig7"
                task.timing.cfg_samp_clk_timing(**kwargs)

        else:
            task.timing.cfg_samp_clk_timing(**kwargs)

    def _cfg_start_trigger(task, info):
        trig = info.get("start_trigger", None)
        if trig:
            task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source=trig,
                trigger_edge=Edge.RISING
            )

    def arm_clock(self, N, sample_rate, ao_channel="ao0"):
        """
        Create a dummy AO task that exports the master sample clock.

        Parameters
        ----------
        N : int
            Number of samples
        sample_rate : float
            Sample clock rate

        """
        self.build_stack()

        # Dummy waveform (all zeros)
        waveform = [0] * int(N)

        self.set_ao_voltage(
            ao_channel=ao_channel,
            voltages=waveform,
            sample_rate=sample_rate
        )

        self.clock_handle = self.arm(regeneration=True)

    def finalize_clock(self):
        self.finalize(handle=self.clock_handle, force_finish=True)


# -----------------------------
# Compilation helpers
# -----------------------------

def _to_1d_array(x: Union[float, int, Sequence[float]]) -> np.ndarray:
    if isinstance(x, (float, int, np.floating, np.integer)):
        return np.array([float(x)], dtype=float)
    arr = np.asarray(list(x), dtype=float).reshape(-1)
    return arr


def _to_bool_array(x: Union[bool, int, Sequence[Union[bool, int]]]) -> np.ndarray:
    if isinstance(x, (bool, int, np.bool_, np.integer)):
        return np.array([bool(x)], dtype=np.bool_)
    arr = np.asarray(list(x), dtype=np.bool_).reshape(-1)
    return arr


def _cfg_trigger_if_needed(
    task: nidaqmx.Task,
    info: Dict[str, Any],
) -> None:
    cfg: Optional[_TriggerCfg] = info.get("trigger_cfg", None)
    if cfg is None:
        return

    task.triggers.start_trigger.cfg_dig_edge_start_trig(
        trigger_source=info["trigger_source"],
        trigger_edge=cfg.edge,
    )


def _split_ai_results(raw: Any, info: Dict[str, Any]) -> Dict[str, Any]:
    plan = info["read_plan"]
    if not plan:
        return {}

    channels = info["channels"]
    if len(channels) == 1:
        arr = np.asarray(raw, dtype=float).reshape(1, -1)
    else:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

    out: Dict[str, Any] = {}
    cursor = 0
    for seg in plan:
        var = seg["var"]
        ns = int(seg["ns"])
        block = arr[:, cursor:cursor + ns]
        cursor += ns

        if arr.shape[0] == 1:
            data = block.reshape(-1).tolist()
            out[var] = data[0] if ns == 1 else data
        else:
            per_ch = {}
            for i, ch in enumerate(channels):
                data = block[i, :].tolist()
                per_ch[ch] = data[0] if ns == 1 else data
            out[var] = per_ch

    return out


def _split_di_results(raw: Any, info: Dict[str, Any]) -> Dict[str, Any]:
    plan = info["read_plan"]
    if not plan:
        return {}

    channels = info["channels"]

    if len(channels) == 1:
        if isinstance(raw, (bool, np.bool_)):
            arr = np.asarray([raw], dtype=np.bool_).reshape(1, 1)
        else:
            arr = np.asarray(raw, dtype=np.bool_).reshape(1, -1)
    else:
        arr = np.asarray(raw, dtype=np.bool_)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

    out: Dict[str, Any] = {}
    cursor = 0
    for seg in plan:
        var = seg["var"]
        ns = int(seg["ns"])
        block = arr[:, cursor:cursor + ns]
        cursor += ns

        if arr.shape[0] == 1:
            data = [bool(x) for x in block.reshape(-1).tolist()]
            out[var] = data[0] if ns == 1 else data
        else:
            per_ch = {}
            for i, ch in enumerate(channels):
                data = [bool(x) for x in block[i, :].tolist()]
                per_ch[ch] = data[0] if ns == 1 else data
            out[var] = per_ch

    return out


# -----------------------------
# TimedCounter (unchanged)
# -----------------------------

class TimedCounter:
    """Hardware class for NI gated counter (software-timed interval)."""

    def __init__(self, logger=None, counter_channel="Dev1/ctr0", physical_channel="/Dev1/20MHzTimebase"):
        self.log = logger
        self.ci_channel = None
        self.task = None
        self.duration = 0.1
        self._status = "Inactive"
        self.count = 0
        self.activate_task(counter_channel, physical_channel=physical_channel)

    def activate_task(self, counter_channel="Dev1/ctr0", physical_channel="/Dev1/20MHzTimebase"):
        self.task = nidaqmx.Task()
        try:
            self.ci_channel = self.task.ci_channels.add_ci_count_edges_chan(counter_channel)
            self.ci_channel.ci_count_edges_term = physical_channel
            self._status = "Active, but not counting"
            if self.log:
                self.log.info(f"Created counter {counter_channel} on physical channel {physical_channel}")
        except nidaqmx.DaqError:
            self.task.close()
            self._status = "Inactive"
            msg_str = f"Failed to activate counter {counter_channel} with physical channel {physical_channel}"
            if self.log:
                self.log.error(msg_str)

    def set_parameters(self, duration=0.1):
        self.duration = duration

    def close(self):
        try:
            self.task.stop()
        except nidaqmx.DaqError:
            if self.log:
                self.log.warn(f"Failed to stop NI DAQmx task {self.task.name}")
        self._status = "Inactive"
        self.task.close()

    def start(self):
        self._status = "Counting"
        try:
            current_time = time.time()
            self.task.start()
            while time.time() - current_time < self.duration:
                pass
            self.task.stop()
            self.count = self.task.read()
        except nidaqmx.DaqError:
            if self.log:
                self.log.warn(f"Failed to count on {self.task.name}")

    def terminate_counting(self):
        try:
            self.task.stop()
        except nidaqmx.DaqError:
            if self.log:
                self.log.warn(f"Failed to stop {self.task.name}")
        self._status = "Active, but not counting"

    def get_status(self):
        return self._status


if __name__ == "__main__":
    # Small smoke test (requires real hardware):
    dev = Driver("Dev1")
    dev.build_stack()
    dev.set_ao_voltage("ao0", [0.0, 1.0, 0.0], sample_rate=1000)
    tag = dev.get_ai_voltage("ai0", num_samples=100, sample_rate=1000, variable_name="trace")
    dev.set_trigger("ao", "PFI0")
    dev.set_trigger("ai", "PFI0")
    out = dev.execute()
    print(out.get(tag))
