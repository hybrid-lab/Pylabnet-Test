"""Connects to a camera read through OpenCV (USB or RTSP network camera) and launches its server."""

from pylabnet.hardware.cameras.opencv_camera import Driver
from pylabnet.network.client_server.opencv_camera import Service, Client
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, load_device_config


def launch(**kwargs):
    """Connects to the camera and launches the server.

    :param kwargs: (dict) containing relevant kwargs
        :logger: instance of LogClient for logging purposes
        :port: (int) port number for the server
        :config: (str) name of the config in configs/devices/opencv_camera. The config has
            "source" (USB device index, or RTSP URL) and optionally "exposure_us" (USB only)
    """
    logger = kwargs['logger']
    config = load_device_config('opencv_camera', kwargs['config'], logger)
    driver = Driver(
        source=config['source'],
        logger=logger,
        usb_exposure_us=config.get('exposure_us'),
    )

    service = Service()
    service.assign_module(module=driver)
    service.assign_logger(logger=logger)
    server = GenericServer(service=service, host=get_ip(), port=kwargs['port'])
    server.start()
