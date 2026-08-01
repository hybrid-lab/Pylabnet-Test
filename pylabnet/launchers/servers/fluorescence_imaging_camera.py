from pylabnet.hardware.fluorescence_imaging_camera import bfs_u3_51s5m
from pylabnet.network.client_server.fluorescence_imaging_camera import Service, Client
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, hide_console, load_config, load_device_config


def launch(**kwargs):
    """Connects to the fluorescence imaging camera and launches server.

    :param kwargs: (dict) containing relevant kwargs
        :logger: instance of LogClient for logging purposes
        :port: (int) port number for the fluorescence imaging camera server
        :device_id: (str) device identifier (optional if config provided)
        :serial: (str) camera serial number for selecting a specific camera
        :config: (str) path/name of config file (optional if device_id provided)
    """

    logger = kwargs.get("logger", None)
    config = None
    serial = kwargs.get("serial", None)

    if "config" in kwargs:
        try:
            config = load_device_config("fluorescence_imaging_camera", kwargs["config"], logger)
        except Exception:
            config = None

    if serial is None and config is not None:
        serial = config.get("serial", config.get("device_serial", None))

    device_name = kwargs.get("device_id", None)
    if device_name is None and config is not None:
        device_name = config.get("device", config.get("device_id", None))

    if device_name is None:
        if logger:
            logger.error("Please provide a valid config file or device_id")
        raise KeyError("device_id")

    try:
        fluorescence_imaging_camera_driver = bfs_u3_51s5m.Driver(
            device_name=device_name,
            logger=logger,
            serial=serial
        )
    except OSError:
        if logger:
            logger.error(f"Did not find camera device {device_name}")
        raise

    # Instantiate server
    fluorescence_imaging_camera_service = Service()
    fluorescence_imaging_camera_service.assign_module(module=fluorescence_imaging_camera_driver)
    fluorescence_imaging_camera_service.assign_logger(logger=logger)

    fluorescence_imaging_camera_server = GenericServer(
        service=fluorescence_imaging_camera_service,
        host=get_ip(),
        port=kwargs["port"]
    )
    fluorescence_imaging_camera_server.start()
