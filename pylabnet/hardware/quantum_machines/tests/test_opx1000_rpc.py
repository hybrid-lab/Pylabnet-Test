"""OPX1000 RPC and server wiring checks; loopback only, no QOP/hardware calls."""

from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.client_server.OPX1000 import Client, Service
from pylabnet.launchers.servers import OPX1000 as server_module
from pylabnet.hardware.quantum_machines.OPX1000 import Driver
from rpyc.utils.factory import connect_thread
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
if importlib.util.find_spec("rpyc") is None:
    raise unittest.SkipTest("RPC tests require the repository virtual environment")


class RpcTests(unittest.TestCase):
    def setUp(self):
        self.driver = Driver(dummy=True)
        self.service = Service()
        self.service.assign_module(self.driver)
        self.connection = connect_thread(
            remote_service=self.service,
            remote_config={"allow_public_attrs": True},
        )
        # Use the real Client methods over a real RPyC connection. ClientBase's
        # unchanged constructor/SSL setup is not part of these loopback tests.
        self.client = Client.__new__(Client)
        self.client._connection = self.connection
        self.client._service = self.connection.root

    def tearDown(self):
        self.connection.close()
        self.driver.close()

    def test_round_trip_pulses_slots_and_independent_config(self):
        self.client.build_stack()
        constant = self.client.set_ao_voltage(ao_channel=1, fem_slot=3)
        gaussian = self.client.create_new_ao_elem("gauss", ao_channel=1, fem_slot=4)
        self.client.set_ao_voltage(element=gaussian)
        samples = [0.15] * 100
        arbitrary = self.client.set_ao_voltage(
            "arbitrary", ao_channel=2, fem_slot=4, wave_function=samples,
        )
        samples[0] = 0.4
        config = self.client.get_config()
        self.assertIs(type(config), dict)
        self.assertEqual(config["elements"][constant]["singleInput"]["port"], ("con1", 3, 1))
        self.assertEqual(config["elements"][gaussian]["singleInput"]["port"], ("con1", 4, 1))
        pulse = config["elements"][arbitrary]["operations"]["arbitrary"]
        waveform = config["pulses"][pulse]["waveforms"]["single"]
        self.assertEqual(config["waveforms"][waveform]["samples"][0], 0.15)
        config["elements"].clear()
        self.assertEqual(len(self.client.get_config()["elements"]), 3)

    def test_timing_and_driver_errors_cross_rpc(self):
        with self.assertRaises(RuntimeError):
            self.client.set_ao_voltage(ao_channel=1)
        self.client.build_stack()
        element = self.client.set_ao_voltage(ao_channel=1)
        self.client.delay(40, [element])
        self.client.align(element)
        self.assertEqual(self.driver.stack[-2], ("wait", 10, (element,)))
        self.assertEqual(self.driver.stack[-1], ("align", (element,)))
        with self.assertRaises(ValueError):
            self.client.create_new_ao_elem(ao_channel=1, fem_slot=1)
        with self.assertRaises(ValueError):
            self.client.delay(19, element)

    def test_execute_returns_local_metadata_and_polls_status(self):
        job = Mock(id="hardware-job")
        job.get_status.return_value = "Running"
        self.driver.job = job
        self.driver.execute = Mock(return_value=job)
        result = self.client.execute(wait=False, timeout=2)
        self.assertIs(type(result), dict)
        self.assertEqual(result, {"job_id": "hardware-job", "simulated": False})
        self.driver.execute.assert_called_once_with(
            wait=False, timeout=2, simulate=False, duration=4000
        )
        self.assertEqual(self.client.get_status(), {
            "job_id": "hardware-job", "simulated": False, "status": "Running",
        })

    def test_simulation_samples_are_values_and_have_sampling_rates(self):
        job = Mock(id="simulation-job")
        job.get_status.side_effect = ["Running", "Done"]
        array = Mock()
        array.tolist.return_value = [0.0, 0.1, 0.0]
        job.get_simulated_samples.return_value = {
            "con1": SimpleNamespace(
                analog={"3-1": array}, digital={"3-1": [False, True, False]},
                analog_sampling_rate={"3-1": 1e9},
            ),
        }
        self.driver.execute = Mock(return_value=job)
        self.driver.job = job
        result = self.client.execute(simulate=True, duration=1000)
        self.assertEqual(result, {"job_id": "simulation-job", "simulated": True})
        self.assertEqual(self.client.get_status(), {
            "job_id": "simulation-job", "simulated": True, "status": "Running",
        })
        self.assertEqual(self.client.get_status()["status"], "Done")
        data = self.client.get_simulated_samples()
        self.assertIs(type(data), dict)
        self.assertIs(type(data["con1"]["analog"]["3-1"]), list)
        self.assertEqual(data["con1"]["analog"]["3-1"], [0.0, 0.1, 0.0])
        self.assertEqual(data["con1"]["analog_sampling_rate"]["3-1"], 1e9)
        self.assertEqual(job.get_status.call_count, 2)
        self.driver.execute.assert_called_once_with(
            wait=True, timeout=None, simulate=True, duration=1000
        )

    def test_no_samples_before_simulation_and_close_keeps_rpc_alive(self):
        self.assertIsNone(self.client.get_status()["job_id"])
        with self.assertRaises(RuntimeError):
            self.client.get_simulated_samples()
        self.client.close()
        self.client.build_stack()
        self.client.set_ao_voltage(ao_channel=1)
        self.assertEqual(len(self.driver.stack), 1)

    def test_wait_failure_retains_access_to_hardware_job(self):
        old_simulation = Mock(id="old-simulation")
        self.service._simulated_job = old_simulation
        job = Mock(id="timed-out-job")
        job.get_status.return_value = "Running"

        def submit(**kwargs):
            self.driver.job = job
            raise TimeoutError("completion wait timed out")

        self.driver.execute = submit
        with self.assertRaises(TimeoutError):
            self.client.execute(timeout=0.01)
        self.assertEqual(self.client.get_status(), {
            "job_id": "timed-out-job", "simulated": False, "status": "Running",
        })
        with self.assertRaises(RuntimeError):
            self.client.get_simulated_samples()

    def test_client_disconnect_preserves_driver_state(self):
        self.client.build_stack()
        element = self.client.set_ao_voltage(ao_channel=1)
        self.connection.close()
        self.assertIn(element, self.driver.config["elements"])

    def test_server_shutdown_attempts_cleanup_before_base_shutdown(self):
        with patch.object(self.driver, "close") as close, patch.object(
            ServiceBase, "close_server"
        ) as shutdown:
            self.service.close_server()
        close.assert_called_once_with()
        shutdown.assert_called_once_with()


class ServerLaunchTests(unittest.TestCase):
    def test_launch_reads_correct_config_and_builds_service(self):
        config = {"device_id": "OPX1000", "ip_address": "ignored"}
        logger = Mock()
        with patch.object(server_module, "load_device_config", return_value=config) as load, patch.object(
            server_module.OPX1000, "Driver"
        ) as driver_class, patch.object(
            server_module, "GenericServer"
        ) as generic_server, patch.object(server_module, "get_ip", return_value="127.0.0.1"):
            server = server_module.launch(config="OPX1000", port=15202, logger=logger)
        load.assert_called_once_with("OPX1000", "OPX1000", logger)
        driver_class.assert_called_once_with(
            device_name="OPX1000", logger=logger, dummy=False
        )
        self.assertIs(server, generic_server.return_value)
        options = generic_server.call_args.kwargs
        self.assertEqual(options["host"], "127.0.0.1")
        self.assertEqual(options["port"], 15202)
        self.assertIs(options["service"]._module, driver_class.return_value)
        server.start.assert_called_once_with()
        driver_class.return_value.close.assert_not_called()

    def test_start_failure_releases_driver(self):
        with patch.object(
            server_module, "load_device_config", return_value={"device_id": "OPX1000"}
        ), patch.object(server_module.OPX1000, "Driver") as driver_class, patch.object(
            server_module, "GenericServer"
        ) as generic_server, patch.object(server_module, "get_ip", return_value="127.0.0.1"):
            generic_server.return_value.start.side_effect = OSError("port busy")
            with self.assertRaises(OSError):
                server_module.launch(config="OPX1000", port=15202)
        driver_class.return_value.close.assert_called_once_with()

    def test_missing_or_invalid_config_rejected_before_driver_creation(self):
        for config in (None, {}, {"device_id": "OPX1000", "dummy": "False"}):
            with self.subTest(config=config), patch.object(
                server_module, "load_device_config", return_value=config
            ), patch.object(server_module.OPX1000, "Driver") as driver_class:
                with self.assertRaises((FileNotFoundError, ValueError)):
                    server_module.launch(config="OPX1000", port=15202)
                driver_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
