"""Acquisition/counting regressions and headless DataTaker script checks."""

from pylabnet.scripts.data_center.opx1000_test_helpers import wait_for_job
from pylabnet.network.client_server.OPX1000 import Client, Service
from pylabnet.hardware.quantum_machines.OPX1000 import Driver
from rpyc.utils.factory import connect_thread
import numpy as np
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
if any(importlib.util.find_spec(name) is None for name in ("numpy", "rpyc")):
    raise unittest.SkipTest("Acquisition tests require the repository virtual environment")


def load_experiment(name):
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[3] / "scripts" / "test_scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.driver = Driver(dummy=True)
        self.driver.build_stack()

    def results(self, specs, streams):
        driver = self.driver
        job = Mock(id="test-acquisition")
        job.get_status.return_value = "Done"
        job.result_handles.wait_for_all_values.return_value = True
        handles = {}
        for label, data in streams.items():
            handle = Mock()
            handle.fetch_all.return_value = data
            handles[label] = handle
        job.result_handles.get.side_effect = handles.get
        driver.job = driver._result_job = job
        driver._last_result_specs = specs
        return driver.get_results()

    def test_input_only_routing_on_both_lf_modules(self):
        for slot in (3, 4):
            for channel in (1, 2):
                label = "trace_{}_{}".format(slot, channel)
                element = self.driver.get_ai_voltage(channel, fem_slot=slot, label=label)
                config = self.driver.config["elements"][element]
                self.assertEqual(config["outputs"], {
                    "out{}".format(channel): ("con1", slot, channel),
                })
                self.assertNotIn("singleInput", config)
                self.assertNotIn("digitalInputs", config)
                pulse = self.driver.config["pulses"][config["operations"]["readout"]]
                self.assertEqual(pulse["operation"], "measurement")
                self.assertIn("digital_marker", pulse)

    def test_acquisition_validation_precedes_config_mutation(self):
        for call in (
            lambda: self.driver.get_ai_voltage(3),
            lambda: self.driver.get_ai_voltage(1, fem_slot=1),
            lambda: self.driver.get_ai_voltage(1, length=16004),
            lambda: self.driver.get_ai_voltage(1, label="bad__label"),
            lambda: self.driver.count_ai_pulses(1, 0),
            lambda: self.driver.count_ai_pulses(1, 2, threshold_adc=2048),
            lambda: self.driver.count_ai_pulses(1, 2, max_tags=65537),
        ):
            with self.assertRaises(ValueError):
                call()
            self.assertEqual(self.driver.stack, [])
            self.assertEqual(self.driver.config["elements"], {})
        self.driver.get_ai_voltage(1, label="same")
        with self.assertRaises(ValueError):
            self.driver.get_ai_voltage(2, label="same")

    def test_train_is_a_single_hardware_operation_and_rejects_bad_gap(self):
        for period, length in ((1000, 1004), (1004, 1000)):
            with self.assertRaises(ValueError):
                self.driver.pulse_train(1, 1000, period, length=length)
        element = self.driver.pulse_train(1, 1000000, 2000, length=1000)
        self.assertEqual(len(self.driver.stack), 1)
        self.assertEqual(self.driver.stack[0], ("pulse_train", element, 1000000, 250))
        self.assertEqual(self.driver.config["elements"][element]["intermediate_frequency"], 0)

    def test_counts_use_live_time_and_four_ns_timestamp_cycles(self):
        specs = {"counts": {
            "kind": "counts", "num_windows": 2, "window_ns": 1000, "max_tags": 100,
        }}
        result = self.results(specs, {
            "counts": np.array([2, 3]),
            "counts__starts": np.array([100, 360]),
        })["counts"]
        self.assertEqual(result["total_counts"], 5)
        self.assertEqual(result["window_start_ns"], [0, 1040])
        self.assertAlmostEqual(result["live_time_s"], 2e-6)
        self.assertAlmostEqual(result["measurement_span_s"], 2.04e-6)
        self.assertAlmostEqual(result["inter_window_gap_s"], 40e-9)
        self.assertAlmostEqual(result["count_rate_hz"], 2.5e6)
        self.assertTrue(result["valid"])

    def test_saturation_and_partial_data_never_produce_a_rate(self):
        specs = {"counts": {
            "kind": "counts", "num_windows": 2, "window_ns": 1000, "max_tags": 4,
        }}
        result = self.results(specs, {
            "counts": np.array([4, 3]), "counts__starts": np.array([0, 260]),
        })["counts"]
        self.assertFalse(result["valid"])
        self.assertIsNone(result["count_rate_hz"])
        self.assertIsNone(result["live_time_s"])
        with self.assertRaisesRegex(RuntimeError, "partial"):
            self.results(specs, {
                "counts": np.array([1]), "counts__starts": np.array([0]),
            })
        with self.assertRaisesRegex(RuntimeError, "non-monotonic"):
            self.results(specs, {
                "counts": np.array([1, 1]), "counts__starts": np.array([260, 0]),
            })

    def test_overlap_invalidates_rate_and_raw_shape_is_checked(self):
        specs = {"counts": {
            "kind": "counts", "num_windows": 2, "window_ns": 1000, "max_tags": 10,
        }}
        result = self.results(specs, {
            "counts": np.array([1, 1]), "counts__starts": np.array([0, 200]),
        })["counts"]
        self.assertFalse(result["valid"])
        self.assertTrue(result["timing_overlap"])
        self.assertIsNone(result["count_rate_hz"])
        raw_spec = {"trace": {"kind": "adc", "length_ns": 16, "sample_rate_hz": 1e9}}
        raw = np.zeros(16, dtype=[("value", float)])
        self.assertEqual(len(self.results(raw_spec, {"trace": raw})["trace"]["raw_adc"]), 16)
        with self.assertRaisesRegex(RuntimeError, "sample count"):
            self.results(raw_spec, {"trace": np.zeros(15)})

    @unittest.skipUnless(importlib.util.find_spec("qm"), "qm-qua is not installed")
    def test_real_qua_programs_build_without_connecting(self):
        for kind in ("adc", "counts", "train"):
            self.driver.build_stack()
            if kind == "adc":
                self.driver.get_ai_voltage(2, fem_slot=4)
            elif kind == "counts":
                self.driver.count_ai_pulses(2, 1000, fem_slot=4)
            else:
                self.driver.pulse_train(1, 1000, 2000, length=1000)
            with patch.object(self.driver, "_get_qmm", side_effect=AssertionError("QOP access")):
                self.assertIsNotNone(self.driver.build_program())


class NewRpcTests(unittest.TestCase):
    def test_new_methods_cross_real_rpc_boundary(self):
        driver = Driver(dummy=True)
        service = Service()
        service.assign_module(driver)
        connection = connect_thread(
            remote_service=service, remote_config={"allow_public_attrs": True},
        )
        client = Client.__new__(Client)
        client._connection, client._service = connection, connection.root
        try:
            client.build_stack()
            client.get_ai_voltage(2, fem_slot=4)
            client.count_ai_pulses(2, 100, fem_slot=4, threshold_adc=-200, polarity="Above")
            client.pulse_train(1, 100, 2000, length=1000)
            self.assertEqual([op[0] for op in driver.stack], ["measure_adc", "count_adc", "pulse_train"])
            config = client.get_config()
            measurement = config["elements"][driver.stack[1][1]]
            self.assertEqual(measurement["timeTaggingParameters"]["signalThreshold"], -200)
            with self.assertRaises(RuntimeError):
                client.get_results()
            with patch.object(driver, "get_results", return_value={"trace": {"raw_adc": [1, 2]}}):
                self.assertIs(type(client.get_results()), dict)
                self.assertIs(type(client.get_results()["trace"]["raw_adc"]), list)
        finally:
            connection.close()
            driver.close()


class ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtCore import QCoreApplication
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setup_experiment(self, name, overrides=None):
        module = load_experiment(name)
        parameters = {
            key: float(next(iter(value.values()))) for key, value in module.INIT_DICT.items()
        }
        parameters.update(overrides or {})
        dataset = SimpleNamespace(graph=Mock(), log=Mock(), children={}, set_data=Mock(), update=Mock())
        dataset.get_input_parameter = parameters.__getitem__

        def add_child(name):
            dataset.children[name] = SimpleNamespace(graph=Mock(), set_data=Mock())

        dataset.add_child = add_child
        opx = Mock()
        opx.get_status.return_value = {"status": "Done"}
        opx.execute.return_value = {"job_id": "test", "simulated": False}
        thread = SimpleNamespace(running=True)
        kwargs = {"dataset": dataset, "thread": thread, "OPX1000_OPX1000": opx, "status_flag": Mock()}
        module.configure(**kwargs)
        return module, dataset, opx, thread, kwargs

    def test_raw_view_repeats_captures_until_view_deadline_without_counting(self):
        module, dataset, opx, thread, kwargs = self.setup_experiment(
            "opx1000_snspd_dark_counts", {"view_s": 0.0625, "pause_s": 0.03125, "trace_ns": 16},
        )
        clock = SimpleNamespace(now=0.0, monotonic=lambda: clock.now)
        clock.sleep = lambda seconds: setattr(clock, "now", clock.now + seconds)
        traces = []

        def fetch():
            clock.now += 0.015625
            traces.append([len(traces)] * 16)
            return {"trace": {"raw_adc": traces[-1], "sample_rate_hz": 1e9}}

        opx.get_results.side_effect = fetch
        with patch.object(module, "time", clock):
            module.experiment(**kwargs)
        self.assertEqual(opx.execute.call_count, 2)
        self.assertEqual(opx.get_ai_voltage.call_count, 2)
        opx.get_ai_voltage.assert_called_with(1, length=16, fem_slot=3, label="trace")
        opx.count_ai_pulses.assert_not_called()
        opx.set_ao_voltage.assert_not_called()
        opx.pulse_train.assert_not_called()
        self.assertEqual(dataset.children, {})
        np.testing.assert_array_equal(dataset.set_data.call_args.args[0], traces[-1])
        np.testing.assert_array_equal(dataset.set_data.call_args.kwargs["x"], np.arange(16))
        metadata = dataset.log.update_metadata.call_args.kwargs["opx1000_raw_waveform"]
        self.assertEqual(metadata["captures"], 2)
        self.assertFalse(metadata["continuous"])
        self.assertAlmostEqual(metadata["captured_live_time_s"], 32e-9)
        self.assertFalse(metadata["stopped_by_user"])
        kwargs["status_flag"].emit.assert_not_called()
        self.assertFalse(thread.running)
        self.app.processEvents()
        self.assertGreaterEqual(dataset.update.call_count, 2)

    def test_invalid_raw_view_settings_are_rejected_without_submission(self):
        for overrides in (
            {"view_s": 0}, {"view_s": float("nan")},
            {"pause_s": -1}, {"trace_ns": 16004},
        ):
            with self.subTest(overrides=overrides):
                module, dataset, opx, thread, kwargs = self.setup_experiment(
                    "opx1000_snspd_dark_counts", overrides,
                )
                module.experiment(**kwargs)
                opx.execute.assert_not_called()
                kwargs["status_flag"].emit.assert_called_once_with("BAD")
                self.assertFalse(thread.running)

    def test_raw_view_stop_prevents_further_captures(self):
        module, dataset, opx, thread, kwargs = self.setup_experiment(
            "opx1000_snspd_dark_counts", {"pause_s": 0},
        )

        def fetch():
            thread.running = False
            return {"trace": {"raw_adc": [0] * 16, "sample_rate_hz": 1e9}}

        opx.get_results.side_effect = fetch
        module.experiment(**kwargs)
        opx.execute.assert_called_once_with(wait=False)
        opx.count_ai_pulses.assert_not_called()
        metadata = dataset.log.update_metadata.call_args.kwargs["opx1000_raw_waveform"]
        self.assertTrue(metadata["stopped_by_user"])
        self.assertEqual(metadata["captures"], 1)
        self.assertFalse(thread.running)

    def test_physical_train_is_submitted_once_and_plot_is_only_a_preview(self):
        module, dataset, opx, thread, kwargs = self.setup_experiment(
            "opx1000_lf_pulse_train", {"simulate": 0, "pulse_count": 100},
        )
        module.experiment(**kwargs)
        opx.pulse_train.assert_called_once_with(
            1, 100, 10000, amplitude=0.1, length=1000, fem_slot=3,
        )
        opx.execute.assert_called_once_with(wait=False, simulate=False, duration=16000)
        opx.get_simulated_samples.assert_not_called()
        voltage = dataset.set_data.call_args.args[0]
        self.assertEqual(voltage[0], 0.1)
        self.assertEqual(voltage[1000], 0)
        self.assertEqual(voltage[10000], 0.1)
        self.assertFalse(thread.running)
        dataset.update.assert_not_called()
        self.app.processEvents()
        dataset.update.assert_called_once_with()

    def test_simulated_train_uses_selected_fem_and_sample_rate(self):
        module, dataset, opx, thread, kwargs = self.setup_experiment(
            "opx1000_lf_pulse_train", {"fem_slot": 4, "ao_channel": 2},
        )
        opx.get_simulated_samples.return_value = {
            "con1": {"analog": {"4-2": [0, 0.1, 0]}, "analog_sampling_rate": {"4-2": 1e9}},
        }
        module.experiment(**kwargs)
        np.testing.assert_array_equal(dataset.set_data.call_args.args[0], [0, 0.1, 0])
        np.testing.assert_array_equal(dataset.set_data.call_args.kwargs["x"], [0, 1, 2])
        self.assertFalse(thread.running)

    def test_stop_flag_closes_active_job_without_polling_again(self):
        opx = Mock()
        self.assertFalse(wait_for_job(opx, SimpleNamespace(running=False), 10))
        opx.close.assert_called_once_with()
        opx.get_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
