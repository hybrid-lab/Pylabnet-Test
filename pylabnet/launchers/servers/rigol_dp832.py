"""Connects to a Rigol DP832 power supply and launches its server."""

from pylabnet.hardware.power_supply.rigol_dp832 import Driver
from pylabnet.network.client_server.rigol_dp832 import Service, Client
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, load_device_config


def launch(**kwargs):
    """Connects to the supply and launches the server.

    :param kwargs: (dict) containing relevant kwargs
        :logger: instance of LogClient for logging purposes
        :port: (int) port number for the server
        :config: (str) name of the config in configs/devices/rigol_dp832
    """
    logger = kwargs['logger']
    config = load_device_config('rigol_dp832', kwargs['config'], logger)
    driver = Driver(device_id=config['device_id'], logger=logger)

    service = Service()
    service.assign_module(module=driver)
    service.assign_logger(logger=logger)
    server = GenericServer(service=service, host=get_ip(), port=kwargs['port'])
    server.start()
