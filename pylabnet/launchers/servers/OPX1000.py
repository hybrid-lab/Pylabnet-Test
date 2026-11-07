"""Launch the OPX1000 driver behind Pylabnet's usual RPC server.

Selected by {"type": "OPX1000", "config": "<device JSON basename>"}.
QOP host/cluster and hardware defaults come from OPX1000Config.py. The device
JSON supplies device_id and optionally a JSON boolean dummy; its ip_address
field is unused. Launcher ip/port fields refer to the Pylabnet RPC endpoint.
"""

from pylabnet.hardware.quantum_machines import OPX1000
from pylabnet.network.client_server.OPX1000 import Client, Service
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, load_device_config


def launch(**kwargs):
    """Create Driver -> Service -> GenericServer, then start listening.

    kwargs: config (device config basename), port (RPC listening port),
    logger (optional Pylabnet logger). Construction does not connect to QOP;
    the first simulation/hardware execution establishes that connection.
    """
    logger = kwargs.get("logger")
    config_name = kwargs["config"]
    config = load_device_config("OPX1000", config_name, logger)
    if config is None:
        raise FileNotFoundError(
            "Missing device config: configs/devices/OPX1000/{}.json".format(config_name)
        )
    device_id = config.get("device_id")
    if not isinstance(device_id, str) or not device_id.strip():
        raise ValueError("OPX1000 device config needs a nonempty string device_id")
    dummy = config.get("dummy", False)
    if not isinstance(dummy, bool):
        raise ValueError("dummy must be a JSON boolean (true or false)")

    driver = OPX1000.Driver(device_name=device_id, logger=logger, dummy=dummy)
    try:
        service = Service()
        service.assign_module(module=driver)
        service.assign_logger(logger=logger)
        server = GenericServer(service=service, host=get_ip(), port=kwargs["port"])
        server.start()
    except Exception:
        driver.close()
        raise
    return server
