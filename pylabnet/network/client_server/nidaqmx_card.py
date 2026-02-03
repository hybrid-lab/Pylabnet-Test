import pickle
from typing import Any, Dict, Optional

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

    Updated to support the new stacked driver interface:
      - build_stack()
      - clear_stack()
      - set_trigger(target="ao"/"ai"/"do"/"di", trig_line="PFI0", edge="rising"/"falling")
      - execute() -> dict of results
      - set_ao_voltage(..., sample_rate=..., max_range=...)
      - set_do_voltage(..., sample_rate=...)
      - get_ai_voltage(..., variable_name=...)  (returns label in stack mode, data in immediate mode)
      - get_di_state(..., num_samples=..., sample_rate=..., variable_name=...) (label or data)

    Backward compatibility:
      - exposed_get_ai_voltage_triggered is kept and implemented using the new stack+trigger flow.
    """

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

    # ---- Execute ----
    def exposed_execute(self):
        out = self._module.execute()
        return pickle.dumps(out)

    # ---- AO / DO ----
    def exposed_set_ao_voltage(self, ao_channel: str, voltage_pickle: bytes, sample_rate: Optional[float] = None, max_range: float = 10.0):
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
    ):
        """
        Backward-compatible wrapper using the new stacked API:
          build_stack -> get_ai_voltage(label) -> set_trigger(target="ai") -> execute -> return label data
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
        out = self._module.execute()
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

    # ---- Execute ----
    def execute(self) -> Dict[str, Any]:
        out_pickle = self._service.exposed_execute()
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
        Stack mode (after build_stack): returns a label (str) you can later use in execute() output.
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
        Stack mode: returns label (str) to use in execute() output.
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
