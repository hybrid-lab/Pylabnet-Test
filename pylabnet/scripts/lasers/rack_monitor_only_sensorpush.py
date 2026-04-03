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
import matplotlib.dates as mdates
import pytz

TEMP_INDEX = 0
HUMIDITY_INDEX = 1


class RackMonitor:
    """ A script class for monitoring laser rack conditions and locking lasers based on the wavemeter """

    def __init__(self, sensorpush_clients, logger_client, gui='sensorpush_only', display_pts=800, port=None):
        """ Instantiates WlmMonitor script object for monitoring wavemeter

        :param sensorpush_client: (obj) instance of sensorpush client
        :param gui_client: (obj) instance of GUI client.
        :param logger_client: (obj) instance of logger client.
        :param display_pts: (int, optional) number of points to display on plot
        :param port: (int) port number for update server
        """

        self.sensorpush_clients = sensorpush_clients
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
            temp=2, humidity=2, graph=4, reset=4
        )

        # Configure plots

        self.widgets['curve'] = []
        self.sensors = []
        for client in self.sensorpush_clients:
            self.sensors.append(Sensor(client, log=self.log))
        for n in range(len(self.sensors)):
            self._initialize_sensor(n)

    def run(self):
        """Runs the WlmMonitor

        Can be stopped using the pause() method
        """

        self._update_sensors()
        self.gui.force_update()

    def reset(self, plot_index):
        """resets the plot back to initial framing"""

        self.widgets['graph'][plot_index].getPlotItem().enableAutoRange(axis='xy', enable=True)

    # Technical methods

    def _initialize_sensor(self, index):
        """Initializes a channel and outputs to the GUI

        Should only be called in the beginning of channel use to assign physical GUI widgets
        """
        self.sensors[index].initialize(self.display_pts)

        # Create curves
        # temperature
        axis1 = pg.DateAxisItem(orientation='bottom')
        self.widgets['graph'][2 * index + TEMP_INDEX].setAxisItems({'bottom': axis1})
        self.widgets['curve'].append(self.widgets['graph'][2 * index + TEMP_INDEX].plot(
            pen=pg.mkPen(color=self.gui.COLOR_LIST[0])
        ))
        self.widgets['reset'][2 * index + TEMP_INDEX].clicked.connect(
            lambda: self.reset(2 * index + TEMP_INDEX)
        )

        # humidity
        axis2 = pg.DateAxisItem(orientation='bottom')
        self.widgets['graph'][2 * index + HUMIDITY_INDEX].setAxisItems({'bottom': axis2})
        self.widgets['curve'].append(self.widgets['graph'][2 * index + HUMIDITY_INDEX].plot(
            pen=pg.mkPen(color=self.gui.COLOR_LIST[0])
        ))
        self.widgets['reset'][2 * index + HUMIDITY_INDEX].clicked.connect(
            lambda: self.reset(2 * index + HUMIDITY_INDEX)
        )

    def _update_sensors(self):
        """ Updates all channels + displays

        Called continuously inside run() method to refresh WLM data and output on GUI
        """
        # Update data with the new temperature and humidity
        for index in range(len(self.sensorpush_clients)):
            time = self.sensorpush_clients[index].get_time()
            temp = self.sensorpush_clients[index].get_temperature()
            humidity = self.sensorpush_clients[index].get_humidity()
            self.sensors[index].update(time, temp, humidity)

            times = [t.timestamp() for t in self.sensors[index].time]

            # Update temperature
            self.widgets['curve'][2 * index + TEMP_INDEX].setData(x=times, y=self.sensors[index].temp)
            self.widgets['temp'][index].setValue(self.sensors[index].temp[0])

            # Update humidity
            self.widgets['curve'][2 * index + HUMIDITY_INDEX].setData(x=times, y=self.sensors[index].humidity)
            self.widgets['humidity'][index].setValue(self.sensors[index].humidity[0])

    def get_env_data(self, index, num_points=1):
        return self.sensorpush_clients[index].get_data(num_points)

    def get_time(self, index):
        return self.sensorpush_clients[index].get_time()

    def get_temperature(self, index):
        return self.sensorpush_clients[index].get_temperature()

    def get_humidity(self, index):
        return self.sensorpush_clients[index].get_humidity()


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
            data = self.sensor_client.get_data(display_pts)
            self.time = data['datetime']
            self.temp = data['temperature']
            self.humidity = data['humidity']

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
    client_configs = []
    for server in config['servers']:
        if server["type"] == 'sensorpush':
            client_configs.append(server['config'])

    sensorpush_clients = []
    for n in range(len(client_configs)):
        sensorpush_clients.append(
            find_client(
                clients=kwargs['clients'],
                settings=config,
                client_type='sensorpush',
                client_config=client_configs[n],
                logger=logger
            )
        )

    # Instantiate Monitor script
    rack_monitor = RackMonitor(
        sensorpush_clients=sensorpush_clients,
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
