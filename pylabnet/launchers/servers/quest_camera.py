from pylabnet.hardware.quest_camera import c15550
from pylabnet.network.client_server.c15550 import Service, Client
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, load_config


def launch(**kwargs):
    """Connects to the Hamamatsu ORCA-Quest camera and launches server.

    :param kwargs: (dict) containing relevant kwargs
        :logger: instance of LogClient for logging purposes
        :port: (int) port number for the Quest camera server
        :device_id: (str) device identifier / camera_id substring (optional)
        :serial: (str) serial / camera_id substring hint (optional)
        :index: (int) camera index (optional, default=0)
        :config: (str) path/name of config file (optional)
    """

    logger = kwargs.get("logger", None)

    quest_camera_driver = None

    # Try explicit kwargs first
    try:
        quest_camera_driver = c15550.Driver(
            device_name=kwargs.get("device_id", None),
            serial=kwargs.get("serial", None),
            index=kwargs.get("index", 0),
            logger=logger
        )

    except Exception as exc_explicit:

        # Try config file next
        try:
            config = load_config(kwargs["config"])

            quest_camera_driver = c15550.Driver(
                device_name=config.get("device", config.get("device_id", None)),
                serial=config.get("serial", None),
                index=config.get("index", 0),
                logger=logger
            )

        except KeyError:
            if logger:
                logger.error("Please provide a valid config file or device_id/serial/index")
            raise exc_explicit

        except OSError:
            if logger:
                logger.error("Could not load config file")
            raise

        except Exception as exc_config:
            if logger:
                logger.error(
                    f"Failed to initialize Quest camera from config: {exc_config}"
                )
            raise

    # Instantiate service
    quest_camera_service = Service()
    quest_camera_service.assign_module(module=quest_camera_driver)
    quest_camera_service.assign_logger(logger=logger)

    # Instantiate and start server
    quest_camera_server = GenericServer(
        service=quest_camera_service,
        host=get_ip(),
        port=kwargs["port"]
    )

    quest_camera_server.start()
