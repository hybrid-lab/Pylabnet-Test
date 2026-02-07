from pylabnet.scripts.pid import PID
from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase
from pylabnet.gui.pyqt.external_gui import Window
from pylabnet.utils.helper_methods import (unpack_launcher, create_server,
                                           load_config, get_gui_widgets, get_legend_from_graphics_view, add_to_legend, find_client,
                                           load_script_config, get_ip)
from pylabnet.utils.logging.logger import LogClient, LogHandler

import numpy as np
import pickle
import pyqtgraph as pg

TEMP_INDEX = 0
HUMIDITY_INDEX = 1


class RackMonitor:
    """ A script class for monitoring laser rack conditions and locking lasers based on the wavemeter """

    def __init__(self, sensorpush_client, logger_client, gui='rack_monitor_v1', display_pts=500, port=None):
        """ Instantiates WlmMonitor script object for monitoring wavemeter

        :param sensorpush_client: (obj) instance of sensorpush client
        :param gui_client: (obj) instance of GUI client.
        :param logger_client: (obj) instance of logger client.
        :param display_pts: (int, optional) number of points to display on plot
        :param port: (int) port number for update server
        """

        self.sensorpush_client = sensorpush_client
        self.display_pts = display_pts
        self.log = LogHandler(logger_client)

        # Instantiate gui
        self.gui = Window(
            gui_template=gui,
            host=get_ip(),
            port=port,
            log=self.log
        )

        # Setup stylesheet.
        self.gui.apply_stylesheet()

        self.widgets = get_gui_widgets(
            gui=self.gui,
            temp=1, humidity=1, graph=2
        )

        # Configure plots

        self.widgets['curve'] = []

        self.sensor = Sensor(self.sensorpush_client, log=self.log)
        self._initialize_sensor()

    def run(self):
        """Runs the WlmMonitor

        Can be stopped using the pause() method
        """

        self._update_sensor()
        self.gui.force_update()

    # Technical methods

    def _initialize_sensor(self):
        """Initializes a channel and outputs to the GUI

        Should only be called in the beginning of channel use to assign physical GUI widgets
        """
        self.sensor.initialize(self.display_pts)

        # Create curves
        # temperature
        self.widgets['curve'].append(self.widgets['graph'][TEMP_INDEX].plot(
            pen=pg.mkPen(color=self.gui.COLOR_LIST[0])
        ))

        # humidity
        self.widgets['curve'].append(self.widgets['graph'][HUMIDITY_INDEX].plot(
            pen=pg.mkPen(color=self.gui.COLOR_LIST[0])
        ))

    def _update_sensor(self):
        """ Updates all channels + displays

        Called continuously inside run() method to refresh WLM data and output on GUI
        """
        # Update data with the new temperature and humidity
        time = self.sensorpush_client.get_time()
        temp = self.sensorpush_client.get_temperature()
        humidity = self.sensorpush_client.get_humidity()
        self.sensor.update(time, temp, humidity)

        #TODO: error sending datetime object as x-axis --> pull out hour and/or minute
        #completely cursed, but time will be graphed as 'hour.(minute*60/100)' --> days loop around
        times = [t.hour + (t.minute / 60) for t in self.sensor.time]

        # Update temperature
        self.widgets['curve'][TEMP_INDEX].setData(x=times, y=self.sensor.temp)
        self.widgets['temp'].setValue(self.sensor.temp[0])

        # Update humidity
        self.widgets['curve'][HUMIDITY_INDEX].setData(x=times, y=self.sensor.humidity)
        self.widgets['humidity'].setValue(self.sensor.humidity[0])

    def get_env_data(self, num_points=1):
        return self.sensorpush_client.get_data(num_points)

    def get_time(self):
        return self.sensorpush_client.get_time()

    def get_temperature(self):
        return self.sensorpush_client.get_temperature()

    def get_humidity(self):
        return self.sensorpush_client.get_humidity()


class Service(ServiceBase):
    """ A service to enable external updating of Monitor parameters """

    def exposed_update_parameters(self, params_pickle):

        params = pickle.loads(params_pickle)
        return self._module.update_parameters(params)

    def exposed_reconnect_gui(self):
        return self._module.reconnect_gui()

    def exposed_pause(self):

        if isinstance(self._module, list):
            for module in self._module:
                module.pause()
            return 0

        else:
            return self._module.pause()

    def exposed_resume(self):
        return self._module.resume()

    def exposed_get_env_data(self, num_points=1):
        return self._module.get_env_data(num_points)

    def exposed_get_time(self):
        return self._module.get_time()

    def exposed_get_temperature(self):
        return self._module.get_temperature()

    def exposed_get_humidity(self):
        return self._module.get_humidity()


class Client(ClientBase):

    def update_parameters(self, params):

        params_pickle = pickle.dumps(params)
        return self._service.exposed_update_parameters(params_pickle)

    def reconnect_gui(self):
        return self._service.exposed_reconnect_gui()

    def pause(self):
        return self._service.exposed_pause()

    def resume(self):
        return self._service.exposed_resume()

    def get_env_data(self, num_points=1):
        return self._service.exposed_get_env_data(num_points)

    def get_time(self):
        return self._service.exposed_get_time()

    def get_temperature(self):
        return self._service.exposed_get_temperature()

    def get_humidity(self):
        return self._service.exposed_get_humidity()


class Sensor:
    """Object containing all information regarding a single sensor"""

    def __init__(self, sensor_client=None, log: LogHandler = None):
        """
        Initializes all parameters given, sets others to default. Also sets up some defaults + placeholders for data

        :param sensor_client: Sensorpush client object
        :param log: (LogHandler) instance of LogHandler for logging metadata
        """

        # Set channel parameters to default values
        self.sensor_client = sensor_client
        self.log = log
        self.labels_updated = False  # Flag to check if we have updated all labels

        # Initialize relevant placeholders
        self.time = np.array([])
        self.temp = np.array([])
        self.humidity = np.array([])

        #for updating data
        self.count = 0

    def initialize(self, display_pts=720):
        """
        Initializes the channel based on the current value

        :param display_pts: number of points to display on the plot
        """
        if self.sensor_client != None:
            self.time = self.sensor_client.get_data(display_pts)['datetime']
            self.temp = self.sensor_client.get_data(display_pts)['temperature']
            self.humidity = self.sensor_client.get_data(display_pts)['humidity']

    def update(self, time, temp, humidity):
        """
        Updates the data

        :param value: (float) current value
        """
        self.count = self.count + 1
        if self.count == 10:
            self.time = np.append(time, self.time[:-1])
            self.temp = np.append(temp, self.temp[:-1])
            self.humidity = np.append(humidity, self.humidity[:-1])
            self.count = 0


def launch(**kwargs):
    """ Launches the WLM monitor + lock script """

    logger = kwargs['logger']
    config = load_script_config(
        script='rack_monitor',
        config=kwargs['config'],
        logger=logger
    )

    device_id = config['device_id']

    sensorpush_client = find_client(
        clients=kwargs['clients'],
        settings=config,
        client_type='sensorpush',
        logger=logger
    )

    # Instantiate Monitor script
    rack_monitor = RackMonitor(
        sensorpush_client=sensorpush_client,
        logger_client=logger
    )

    update_service = kwargs['service']
    update_service.assign_module(module=rack_monitor)
    logger.update_data(data=dict(device_id=device_id))
    rack_monitor.gui.set_network_info(port=kwargs['server_port'])

    # Run continuously
    # Note that the actual operation inside run() can be paused using the update server
    while True:

        rack_monitor.run()
