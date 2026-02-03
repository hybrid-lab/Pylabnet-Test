from pylabnet.hardware.lab_env_sensor.sensorpush import Driver
from pylabnet.network.client_server.sensorpush import Service, Client
from pylabnet.network.core.generic_server import GenericServer
from pylabnet.utils.helper_methods import get_ip, load_device_config


def launch(**kwargs):
    """
    Connects to SensorPush and launches server

    :param kwargs: (dict) containing relevant kwargs
        :logger: instance of LogClient for logging purposes
        :port: (int) port number for the Cnt Monitor server
    """

    #Instantiate driver
    logger = kwargs['logger']
    config_dict = load_device_config('sensorpush', kwargs['config'], logger)
    driver = Driver(
        device_name=config_dict['device_name'],
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
