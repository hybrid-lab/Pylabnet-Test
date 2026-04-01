from pylabnet.hardware.oscilloscopes.picoscope_2000a import Driver
from pylabnet.network.client_server.picoscope_2000a import Service, Client
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, load_device_config, load_config


def launch(**kwargs):
    """
    Connects to Picoscope 2000A and launches server

    :param kwargs: (dict) containing relevant kwargs
        :logger: instance of LogClient for logging purposes
        :port: (int) port number for the Cnt Monitor server
    """

    #Instantiate driver
    logger = kwargs['logger']
    config = load_device_config('picoscope', kwargs['config'], logger)
    driver = Driver(
        sensor_name=config['device_id'],
        logger=logger
    )

    #Instantiate server
    service = Service()
    service.assign_module(module=driver)
    service.assign_logger(logger=logger)
    server = GenericServer(
        service=service,
        host=get_ip(),
        port=kwargs['port']
    )
    server.start()