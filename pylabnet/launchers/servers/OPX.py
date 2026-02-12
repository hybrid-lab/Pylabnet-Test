from pylabnet.hardware.quantum_machines import OPX
from pylabnet.network.client_server.OPX import Service, Client
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, hide_console, load_config, load_device_config


def launch(**kwargs):
    """ Connects to OPX card and launches server
    :param kwargs: (dict) containing relevant kwargs
        :logger: instance of LogClient for logging purposes
        :port: (int) port number for the Cnt Monitor server
    """

#########DEBUG CODE
    debug_message = f"DEBUG: Port: '{kwargs['port']}'\n "
    with open("c:/users/hybri/pylabnet/debug_log.txt", "a") as log_file:
        log_file.write(debug_message)
    ###############

    # Instantiate driver
    try:
        logger = kwargs['logger']
        config_dict = load_device_config('OPX', kwargs['config'], logger)
        OPX_driver = OPX.Driver(
            device_name=config_dict['device_id'],
            logger=logger
        )
    except AttributeError:
        try:
            config = load_config(kwargs['config'])
            OPX_driver = OPX.Driver(
                device_name=config['device'],
                logger=logger
            )
        except AttributeError:
            logger.error('Please provide valid config file')
            raise
        except OSError:
            logger.error(f'Did not find OPX name {config["device"]}')
            raise
        except KeyError:
            logger.error('No device name provided. '
                         'Please make sure proper config file is provided')
            raise

    # Instantiate server
    OPX_service = Service()
    OPX_service.assign_module(module=OPX_driver)
    OPX_service.assign_logger(logger=logger)
    OPX_server = GenericServer(
        service=OPX_service,
        host=get_ip(),
        port=kwargs['port']
    )
    OPX_server.start()
