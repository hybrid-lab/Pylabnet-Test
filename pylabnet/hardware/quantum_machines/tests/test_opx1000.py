"""Focused offline regressions; run with unittest discover, no hardware access."""

from pylabnet.hardware.quantum_machines import OPX1000, OPX1000Config
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.driver = OPX1000.Driver()
        self.program = object()
        self.qua = SimpleNamespace(
            program=lambda: nullcontext(self.program),
            play=Mock(), wait=Mock(), align=Mock(),
        )

    def waveform(self, name):
        pulse = next(iter(self.driver.config["elements"][name]["operations"].values()))
        waveform = self.driver.config["pulses"][pulse]["waveforms"]["single"]
        return self.driver.config["waveforms"][waveform]

    def test_layout_and_independent_configs(self):
        original = deepcopy(OPX1000Config.config)
        other = OPX1000.Driver()
        custom = OPX1000Config.make_config()
        third = OPX1000.Driver(hardware_config=custom)
        third.config["controllers"]["con1"]["fems"][3]["analog_outputs"][1]["offset"] = 0.1
        self.assertEqual(custom, original)
        fems = other.config["controllers"]["con1"]["fems"]
        self.assertEqual([fems[i]["type"] for i in (1, 2, 3, 4)], ["MW", "MW", "LF", "LF"])
        self.assertEqual(len(fems[3]["analog_outputs"]), 8)
        self.driver.create_new_ao_elem(ao_channel=1)
        self.assertEqual(other.config, original)
        self.assertEqual(OPX1000Config.config, original)
        self.assertIsNone(self.driver.qmm)

    def test_routes_slots_and_converts_mhz_to_hz(self):
        names = [
            self.driver.create_new_ao_elem(ao_channel=1, fem_slot=slot, frequency=freq)
            for slot, freq in ((3, 0), (4, 50), (3, 0))
        ]
        self.assertEqual(len(set(names)), 3)
        for name, slot, freq in zip(names, (3, 4, 3), (0, 50000000, 0)):
            element = self.driver.config["elements"][name]
            self.assertEqual(element["singleInput"]["port"], ("con1", slot, 1))
            self.assertEqual(element["intermediate_frequency"], freq)

    def test_waveforms_have_correct_samples_and_no_aliasing(self):
        name = self.driver.create_new_ao_elem("gauss", ao_channel=1, length=100, amplitude=0.2)
        values = self.waveform(name)["samples"]
        self.assertEqual(self.waveform(name)["type"], "arbitrary")
        self.assertEqual(len(values), 100)
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 0)
        self.assertAlmostEqual(max(values), 0.2)
        self.assertEqual(values, list(reversed(values)))
        samples = [0.1] * 100
        arb = self.driver.create_new_ao_elem("arbitrary", ao_channel=2, wave_function=samples)
        samples[0] = 0.4
        self.assertEqual(self.waveform(arb)["samples"][0], 0.1)

    def test_rebuild_resets_generated_config_and_counters(self):
        self.driver.build_stack()
        first = self.driver.set_ao_voltage(ao_channel=1)
        self.driver.delay(40, first)
        self.driver.config["controllers"]["con1"]["fems"][3]["analog_outputs"][1]["offset"] = 0.2
        self.driver.build_stack()
        self.assertEqual(self.driver.config, OPX1000Config.make_config())
        self.assertEqual(self.driver.elems, [])
        self.assertEqual(self.driver.stack, [])
        self.assertEqual(self.driver.set_ao_voltage(ao_channel=1), first)

    def test_invalid_creation_does_not_mutate_config(self):
        invalid = [
            {"fem_slot": 1}, {"fem_slot": 2}, {"fem_slot": 5},
            {"controller": "absent"}, {"ao_channel": 9}, {"ao_channel": True},
            {"ao_channel": 1.5}, {"length": 12}, {"length": 18},
            {"length": 100.5}, {"amplitude": float("nan")},
            {"amplitude": 2.5}, {"amplitude": -2.6},
            {"pulse": "unknown"}, {"pulse": "gauss", "gauss_sd": 0},
            {"pulse": "arbitrary"}, {"pulse": "arbitrary", "wave_function": [0.1]},
            {"frequency": float("inf")}, {"frequency": 500},
            {"frequency": 331},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                params = {"ao_channel": 1}
                params.update(kwargs)
                with self.assertRaises(ValueError):
                    self.driver.create_new_ao_elem(**params)
                self.assertEqual(self.driver.config, OPX1000Config.make_config())
                self.assertEqual(self.driver.elems, [])

    def test_voltage_limits_include_offset_and_modulation(self):
        port = self.driver.config["controllers"]["con1"]["fems"][3]["analog_outputs"][1]
        port["output_mode"] = "direct"
        with self.assertRaises(ValueError):
            self.driver.create_new_ao_elem(ao_channel=1, amplitude=0.5)
        port["offset"] = 0.3
        with self.assertRaises(ValueError):
            self.driver.create_new_ao_elem(ao_channel=1, amplitude=0.25)
        self.driver.create_new_ao_elem(ao_channel=1, amplitude=-0.3)
        with self.assertRaises(ValueError):
            self.driver.create_new_ao_elem(ao_channel=1, amplitude=-0.3, frequency=10)
        port["sampling_rate"] = int(2e9)
        with self.assertRaisesRegex(ValueError, "1 GS/s"):
            self.driver.create_new_ao_elem(ao_channel=1)

    def test_stack_required_and_references_validated(self):
        with self.assertRaises(RuntimeError):
            self.driver.set_ao_voltage(ao_channel=1)
        self.assertFalse(self.driver.config["elements"])
        self.driver.build_stack()
        for call in (
            lambda: self.driver.set_ao_voltage(element="missing"),
            lambda: self.driver.delay(40, "missing"),
            lambda: self.driver.align("missing"),
            lambda: self.driver.delay(6),
            lambda: self.driver.build_program(),
        ):
            with self.assertRaises((ValueError, RuntimeError)):
                call()
        self.assertEqual(self.driver.stack, [])

    def test_emit_sequence_reuses_element_and_converts_wait(self):
        self.driver.build_stack()
        name = self.driver.set_ao_voltage(ao_channel=1)
        other = self.driver.create_new_ao_elem("gauss", fem_slot=4, ao_channel=1)
        self.driver.align(name, other)
        self.driver.delay(40, other)
        self.driver.set_ao_voltage(element=other)
        self.driver.delay(16)
        with patch.object(OPX1000, "_qua", return_value=self.qua):
            self.assertIs(self.driver.build_program(), self.program)
        self.assertEqual(len(self.driver.stack), 5)
        self.assertEqual(self.qua.play.call_args_list[0].args, ("const", name))
        self.assertEqual(self.qua.play.call_args_list[1].args, ("gauss", other))
        self.qua.align.assert_called_once_with(name, other)
        self.assertEqual(self.qua.wait.call_args_list[0].args, (10, other))
        self.assertEqual(self.qua.wait.call_args_list[1].args, (4,))

    def test_execute_consumes_only_after_submission_and_waits(self):
        self.driver.build_stack()
        self.driver.set_ao_voltage(ao_channel=1)
        manager = Mock()
        old_qm = self.driver.qm = Mock()
        with patch.object(OPX1000, "_qua", return_value=self.qua), patch.object(
            self.driver, "_get_qmm", return_value=manager
        ):
            job = self.driver.execute(timeout=2)
        old_qm.close.assert_called_once_with()
        manager.open_qm.assert_called_once_with(self.driver.config, close_other_machines=False)
        job.wait_until.assert_called_once_with("Done", timeout=2)
        self.assertIs(self.driver.job, job)
        self.assertEqual(self.driver.stack, [])
        self.assertFalse(self.driver.adding_to_stack)

    def test_failed_submission_preserves_stack(self):
        self.driver.build_stack()
        self.driver.set_ao_voltage(ao_channel=1)
        manager = Mock()
        manager.open_qm.return_value.execute.side_effect = RuntimeError("submission failed")
        with patch.object(OPX1000, "_qua", return_value=self.qua), patch.object(
            self.driver, "_get_qmm", return_value=manager
        ):
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                self.driver.execute()
        self.assertTrue(self.driver.adding_to_stack)
        self.assertEqual(len(self.driver.stack), 1)

    def test_simulation_converts_duration_without_opening_qm(self):
        fake_qm = SimpleNamespace(SimulationConfig=Mock())
        manager = Mock()
        with patch.dict(sys.modules, {"qm": fake_qm}), patch.object(
            self.driver, "_get_qmm", return_value=manager
        ):
            self.driver._simulate(self.program, duration=4000)
        fake_qm.SimulationConfig.assert_called_once_with(duration=1000)
        manager.open_qm.assert_not_called()
        manager.simulate.assert_called_once_with(
            self.driver.config, self.program, fake_qm.SimulationConfig.return_value
        )

    def test_simulation_submission_consumes_stack_and_keeps_hardware_qm(self):
        self.driver.build_stack()
        self.driver.set_ao_voltage(ao_channel=1)
        hardware_qm = self.driver.qm = Mock()
        with patch.object(OPX1000, "_qua", return_value=self.qua), patch.object(
            self.driver, "_simulate", return_value=Mock()
        ) as simulate:
            job = self.driver.execute(simulate=True, duration=2000)
        simulate.assert_called_once_with(self.program, duration=2000)
        hardware_qm.close.assert_not_called()
        job.wait_until.assert_not_called()
        self.assertFalse(self.driver.adding_to_stack)

    def test_dummy_and_missing_host_prevent_connection(self):
        with self.assertRaisesRegex(RuntimeError, "offline"):
            OPX1000.Driver(dummy=True).execute()
        self.driver.qop_ip = None  # Keep this test independent of lab connection settings.
        with self.assertRaisesRegex(ValueError, "qop_ip"):
            self.driver._get_qmm()

    def test_close_releases_only_owned_objects(self):
        manager, machine = Mock(), Mock()
        self.driver.qmm, self.driver.qm = manager, machine
        self.driver.close()
        machine.close.assert_called_once_with()
        manager.close.assert_called_once_with()
        self.assertIsNone(self.driver.qm)
        self.assertIsNone(self.driver.qmm)

    @unittest.skipUnless(importlib.util.find_spec("qm"), "qm-qua is not installed")
    def test_real_sdk_builds_qua_locally(self):
        self.driver.build_stack()
        for pulse in ("const", "gauss", "arbitrary"):
            self.driver.set_ao_voltage(
                pulse, ao_channel=1, wave_function=[0.1] * 100
            )
        with patch.object(self.driver, "_get_qmm", side_effect=AssertionError("Network access")):
            self.assertIsNotNone(self.driver.build_program())
        self.assertEqual(len(self.driver.stack), 3)


if __name__ == "__main__":
    unittest.main()
