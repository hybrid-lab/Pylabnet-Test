import pickle
import uuid
from typing import Any, Dict, Optional
from rpyc.utils.classic import obtain


from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


# -------------------------
# Helpers for trigger "edge"
# -------------------------
#
# We avoid importing nidaqmx.constants.Edge in the client/server layer because:
# - the server already has the module and can interpret the edge
# - Edge enums don't serialize cleanly across RPC boundaries
#
# Convention:
#   edge="rising" | "falling"  (default "rising")
#
_EDGE_STRINGS = {"rising", "falling"}


def _normalize_edge(edge: Any) -> str:
    if edge is None:
        return "rising"
    if isinstance(edge, str):
        e = edge.strip().lower()
        if e not in _EDGE_STRINGS:
            raise ValueError(f"edge must be one of {_EDGE_STRINGS}, got {edge!r}")
        return e
    # Allow bool/int as fallback: 1 -> rising, 0 -> falling
    try:
        return "rising" if int(edge) == 1 else "falling"
    except Exception as exc:
        raise ValueError(f"Unsupported edge type: {type(edge)}") from exc


# =========================
# Service (server-side RPC)
# =========================

class Service(ServiceBase):
    """
    RPC wrapper for the NI DAQmx driver.

    Supports the stacked driver interface:
      - build_stack()
      - clear_stack()
      - set_trigger(target="ao"/"ai"/"do"/"di", trig_line="PFI0", edge="rising"/"falling")
      - arm() -> returns {"handle_id": str|None, "meta": dict}
      - finalize(handle_id, timeout=..., close=...) -> dict of results
      - set_ao_voltage(..., sample_rate=..., max_range=...)
      - set_do_voltage(..., sample_rate=...)
      - get_ai_voltage(..., variable_name=...)  (returns label in stack mode, data in immediate mode)
      - get_di_state(..., num_samples=..., sample_rate=..., variable_name=...) (label or data)

    Notes:
      - We intentionally DO NOT expose execute() anymore; the correct multi-device flow is:
            build_stack()
            ... queue ops ...
            handle = arm()
            ... coordinate with OPX / other devices ...
            out = finalize(handle)

      - arm() returns a server-side handle token. The actual DAQmx Task objects remain on the server.
    """

   # ---- internal handle registry ----
    def _handles(self) -> Dict[str, Any]:
        # ServiceBase may or may not call __init__; store lazily.
        if not hasattr(self, "_armed_handles"):
            self._armed_handles = {}
        return self._armed_handles

    def exposed_not_use_OPX_clock(self):
        return self._module.not_use_OPX_clock()

   # ---- Stack control ----

    def exposed_build_stack(self):
        return self._module.build_stack()

    def exposed_clear_stack(self):
        return self._module.clear_stack()

    # ---- Trigger ----
    def exposed_set_trigger(self, target: str, trig_line: str = "PFI0", edge: str = "rising"):
        edge = _normalize_edge(edge)

        # Interpret edge string on the server using nidaqmx constants
        try:
            from nidaqmx.constants import Edge  # local import (server-side only)
            edge_enum = Edge.RISING if edge == "rising" else Edge.FALLING
        except Exception:
            # If nidaqmx is not available, just pass string (dummy mode / tests)
            edge_enum = edge

        return self._module.set_trigger(target=target, trig_line=trig_line, edge=edge_enum)

    # ---- Arm / Finalize ----
    def exposed_arm(self):
        """
        Arms the currently built stack and returns a pickled dict:
          {"handle_id": <uuid str or None>, "meta": <dict>}
        """
        handle = self._module.arm()
        if not getattr(handle, "tasks", None):
            payload = {"handle_id": None, "meta": getattr(handle, "meta", {})}
            return pickle.dumps(payload)

        hid = str(uuid.uuid4())
        self._handles()[hid] = handle
        payload = {"handle_id": hid, "meta": getattr(handle, "meta", {})}
        return pickle.dumps(payload)

    def exposed_finalize(self, handle_id, timeout=60.0, close=True):
        # Convert RPyC netrefs to local python objects when possible
        try:
            handle_id_local = obtain(handle_id)
        except Exception:
            handle_id_local = handle_id

        # Accept either:
        #   - handle_id string
        #   - the dict returned by arm(): {"handle_id": ..., "meta": ...}
        if isinstance(handle_id_local, dict):
            handle_id_local = handle_id_local.get("handle_id", None)

        # If arm() returned None (nothing was armed), just return empty output
        if handle_id_local is None:
            return pickle.dumps({})

        handle = self._handles().get(handle_id_local, None)
        if handle is None:
            raise KeyError(f"Unknown handle_id: {handle_id_local!r}")

        out = self._module.finalize(handle, timeout=timeout, close=close)

        # If tasks were closed, drop the stored handle
        if close:
            self._handles().pop(handle_id_local, None)

        # IMPORTANT: always return pickled bytes to match Client.finalize()
        return pickle.dumps(out)

        # Normal single-handle path (must be hashable)
        handle = self._handles().get(handle_id_local, None)
        if handle is None:
            raise KeyError(f"Unknown handle_id: {handle_id_local!r}")

        return self._module.finalize(handle, timeout=timeout, close=close)

    # ---- AO / DO ----
    def exposed_set_ao_voltage(
        self,
        ao_channel: str,
        voltage_pickle: bytes,
        sample_rate: Optional[float] = None,
        max_range: float = 10.0,
    ):
        voltages = pickle.loads(voltage_pickle)
        return self._module.set_ao_voltage(
            ao_channel=ao_channel,
            voltages=voltages,
            sample_rate=sample_rate,
            max_range=max_range,
        )

    def exposed_set_do_voltage(self, do_channel: str, value_pickle: bytes, sample_rate: Optional[float] = None):
        value = pickle.loads(value_pickle)
        return self._module.set_do_voltage(
            do_channel=do_channel,
            value=value,
            sample_rate=sample_rate,
        )

    # ---- AI / DI ----
    def exposed_get_ai_voltage(
        self,
        ai_channel: str,
        num_samples: int = 1,
        max_range: float = 10.0,
        sample_rate: float = 100000.0,
        variable_name: Optional[str] = None,
    ):
        result = self._module.get_ai_voltage(
            ai_channel=ai_channel,
            num_samples=num_samples,
            max_range=max_range,
            sample_rate=sample_rate,
            variable_name=variable_name,
        )
        return pickle.dumps(result)

    def exposed_get_di_state(
        self,
        port: str,
        di_channel: str,
        num_samples: int = 1,
        sample_rate: float = 100000.0,
        variable_name: Optional[str] = None,
    ):
        result = self._module.get_di_state(
            port=port,
            di_channel=di_channel,
            num_samples=num_samples,
            sample_rate=sample_rate,
            variable_name=variable_name,
        )
        return pickle.dumps(result)

    # ---- Backward-compat convenience ----
    def exposed_get_ai_voltage_triggered(
        self,
        ai_channel: str,
        trig_line: str = "PFI0",
        num_samples: int = 1000,
        sample_rate: float = 100000.0,
        max_range: float = 10.0,
        edge: str = "rising",
        timeout: float = 30.0,
    ):
        """
        Backward-compatible wrapper using the new stacked API:
          build_stack -> get_ai_voltage(label) -> set_trigger(target="ai") -> arm -> finalize -> return label data
        """
        edge = _normalize_edge(edge)

        # server-side conversion
        try:
            from nidaqmx.constants import Edge
            edge_enum = Edge.RISING if edge == "rising" else Edge.FALLING
        except Exception:
            edge_enum = edge

        self._module.build_stack()
        label = self._module.get_ai_voltage(
            ai_channel=ai_channel,
            num_samples=num_samples,
            max_range=max_range,
            sample_rate=sample_rate,
            variable_name="ai_triggered",
        )
        self._module.set_trigger(target="ai", trig_line=trig_line, edge=edge_enum)

        handle = self._module.arm()
        out = self._module.finalize(handle, timeout=timeout, close=True)
        return pickle.dumps(out[label])

    # ---- Timed counter passthrough (unchanged) ----
    def exposed_create_timed_counter(self, counter_channel, physical_channel, duration=0.1, name=None):
        return self._module.create_timed_counter(
            counter_channel=counter_channel,
            physical_channel=physical_channel,
            duration=duration,
            name=name
        )

    def exposed_start_timed_counter(self, name):
        return self._module.start_timed_counter(name)

    def exposed_close_timed_counter(self, name):
        return self._module.close_timed_counter(name)

    def exposed_get_count(self, name):
        return self._module.get_count(name)


# =======================
# Client (client-side API)
# =======================

class Client(ClientBase):

    # ---- Stack control ----
    def build_stack(self):
        return self._service.exposed_build_stack()

    def clear_stack(self):
        return self._service.exposed_clear_stack()

    # ---- Trigger ----
    def set_trigger(self, target: str, trig_line: str = "PFI0", edge: str = "rising"):
        edge = _normalize_edge(edge)
        return self._service.exposed_set_trigger(target=target, trig_line=trig_line, edge=edge)

    # ---- Arm / Finalize ----
    def arm(self) -> Dict[str, Any]:
        """
        Arms the server-side stack. Returns:
          {"handle_id": str|None, "meta": dict}
        """
        payload_pickle = self._service.exposed_arm()
        return pickle.loads(payload_pickle)

    def finalize(self, handle_id: str, timeout: float = 30.0, close: bool = True) -> Dict[str, Any]:
        """
        Finalizes a previously armed handle_id.
        You may pass either:
        - handle_id (str), or
        - the dict returned by arm(): {"handle_id": ..., "meta": ...}
        """
        if isinstance(handle_id, dict):
            handle_id = handle_id.get("handle_id", None)

        out_pickle = self._service.exposed_finalize(handle_id, timeout=timeout, close=close)
        return pickle.loads(out_pickle)

    # ---- AO / DO ----
    def set_ao_voltage(self, ao_channel: str, voltages, sample_rate: Optional[float] = None, max_range: float = 10.0):
        voltage_pickle = pickle.dumps(voltages)
        return self._service.exposed_set_ao_voltage(
            ao_channel=ao_channel,
            voltage_pickle=voltage_pickle,
            sample_rate=sample_rate,
            max_range=max_range,
        )

    def set_do_voltage(self, do_channel: str, value, sample_rate: Optional[float] = None):
        value_pickle = pickle.dumps(value)
        return self._service.exposed_set_do_voltage(
            do_channel=do_channel,
            value_pickle=value_pickle,
            sample_rate=sample_rate,
        )

    # ---- AI / DI ----
    def get_ai_voltage(
        self,
        ai_channel: str,
        num_samples: int = 1,
        max_range: float = 10.0,
        sample_rate: float = 100000.0,
        variable_name: Optional[str] = None,
    ):
        """
        Immediate mode: returns list[float] (or float if 1 sample) from server.
        Stack mode (after build_stack): returns a label (str) you can later use in finalize() output.
        """
        result_pickle = self._service.exposed_get_ai_voltage(
            ai_channel=ai_channel,
            num_samples=num_samples,
            max_range=max_range,
            sample_rate=sample_rate,
            variable_name=variable_name,
        )
        return pickle.loads(result_pickle)

    def get_di_state(
        self,
        di_channel: str,
        port: str,
        num_samples: int = 1,
        sample_rate: float = 100000.0,
        variable_name: Optional[str] = None,
    ):
        """
        Immediate mode: returns bool or list[bool].
        Stack mode: returns label (str) to use in finalize() output.
        """
        result_pickle = self._service.exposed_get_di_state(
            port=port,
            di_channel=di_channel,
            num_samples=num_samples,
            sample_rate=sample_rate,
            variable_name=variable_name,
        )
        return pickle.loads(result_pickle)

    # ---- Backward-compat convenience ----
    def get_ai_voltage_triggered(
        self,
        ai_channel: str,
        trig_line: str = "PFI0",
        num_samples: int = 1000,
        sample_rate: float = 100000.0,
        max_range: float = 10.0,
        edge: str = "rising",
        timeout: float = 30.0,
    ):
        """
        Legacy helper retained for old code paths.
        """
        edge = _normalize_edge(edge)
        voltages_pickle = self._service.exposed_get_ai_voltage_triggered(
            ai_channel=ai_channel,
            trig_line=trig_line,
            num_samples=num_samples,
            sample_rate=sample_rate,
            max_range=max_range,
            edge=edge,
            timeout=timeout,
        )
        return pickle.loads(voltages_pickle)

    # ---- Timed counter passthrough (unchanged) ----
    def create_timed_counter(self, counter_channel, physical_channel, duration=0.1, name=None):
        return self._service.exposed_create_timed_counter(
            counter_channel=counter_channel,
            physical_channel=physical_channel,
            duration=duration,
            name=name
        )

    def start_timed_counter(self, name):
        return self._service.exposed_start_timed_counter(name)

    def close_timed_counter(self, name):
        return self._service.exposed_close_timed_counter(name)

    def get_count(self, name):
        return self._service.exposed_get_count(name)

    def not_use_OPX_clock(self):
        return self._service.exposed_not_use_OPX_clock()
