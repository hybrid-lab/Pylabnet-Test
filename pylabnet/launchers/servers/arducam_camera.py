"""Connects to the Arducam OV9281 USB camera and launches its server."""

from pylabnet.hardware.cameras.arducam_camera import Driver, DEFAULT_NAME
from pylabnet.network.client_server.arducam_camera import Service, Client
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, load_device_config


def launch(**kwargs):
    """Connects to the camera and launches the server.

    :param kwargs: (dict) containing relevant kwargs
        :logger: instance of LogClient for logging purposes
        :port: (int) port number for the server
        :config: (str) name of the config in configs/devices/arducam_camera. The config
            has optionally "name" (the camera's name in Windows), "exposure_us" and "gain"
            (0-100)
    """
    logger = kwargs['logger']
    config = load_device_config('arducam_camera', kwargs['config'], logger)
    driver = Driver(
        name=config.get('name', DEFAULT_NAME),
        logger=logger,
        exposure_us=config.get('exposure_us', 1000),
        gain=config.get('gain', 0),
    )

    service = Service()
    service.assign_module(module=driver)
    service.assign_logger(logger=logger)
    server = GenericServer(service=service, host=get_ip(), port=kwargs['port'])
    server.start()
