"""Connects to a FLIR camera and launches its server."""

from pylabnet.hardware.cameras.flir_camera import Driver
from pylabnet.network.client_server.flir_camera import Service, Client
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, load_device_config


def launch(**kwargs):
    """Connects to the camera and launches the server.

    :param kwargs: (dict) containing relevant kwargs
        :logger: instance of LogClient for logging purposes
        :port: (int) port number for the server
        :config: (str) name of the config in configs/devices/flir_camera. The config has
            "serial", and optionally "exposure_us", "gain_db" and "pixel_format"
    """
    logger = kwargs['logger']
    config = load_device_config('flir_camera', kwargs['config'], logger)
    driver = Driver(
        serial=config.get('serial'),
        logger=logger,
        exposure_us=config.get('exposure_us', 5000),
        gain_db=config.get('gain_db', 0),
        pixel_format=config.get('pixel_format', 'Mono16'),
    )

    service = Service()
    service.assign_module(module=driver)
    service.assign_logger(logger=logger)
    server = GenericServer(service=service, host=get_ip(), port=kwargs['port'])
    server.start()
