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
        :config: (str) path/name of config file (optional if device_id provided)
    """

    ######### DEBUG CODE
    debug_message = f"DEBUG: Port: '{kwargs.get('port')}'\n"
    with open("/home/porkpie/pylabnet/debug_log.txt", "a") as log_file:
        log_file.write(debug_message)
    ###############

    logger = kwargs.get("logger", None)

    # Instantiate driver
    try:
        fluorescence_imaging_camera_driver = bfs_u3_51s5m.Driver(
            device_name=kwargs["device_id"],
            logger=logger
        )
    except (KeyError, AttributeError):
        try:
            config = load_config(kwargs["config"])
            fluorescence_imaging_camera_driver = bfs_u3_51s5m.Driver(
                device_name=config["device"],
                logger=logger
            )
        except (KeyError, AttributeError):
            if logger:
                logger.error("Please provide a valid config file or device_id")
            raise
        except OSError:
            if logger:
                logger.error(f'Did not find camera device {config.get("device")}')
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
