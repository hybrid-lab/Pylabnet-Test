from pylabnet.scripts.pid import PID
from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase
from pylabnet.gui.pyqt.external_gui import Window
from pylabnet.utils.helper_methods import (unpack_launcher, create_server,
                                           load_config, get_gui_widgets, get_legend_from_graphics_view, add_to_legend, find_client,
                                           load_script_config, get_ip)
from pylabnet.utils.logging.logger import LogClient, LogHandler

from pylabnet.network.client_server.nidaqmx_card import Client as NI_Client

import numpy as np
import pickle
import pyqtgraph as pg
import matplotlib.dates as mdates
import pytz

TEMP_INDEX = 0
HUMIDITY_INDEX = 1


class RackMonitor:
    """ A script class for monitoring laser rack conditions and locking lasers based on the wavemeter """

    def __init__(self, sensorpush_client, ni_client, logger_client, channel_params, display_pts=1000, port=None):
        """ Instantiates WlmMonitor script object for monitoring wavemeter

        :param sensorpush_client: (obj) instance of sensorpush client
        :param gui_client: (obj) instance of GUI client.
        :param logger_client: (obj) instance of logger client.
        :param display_pts: (int, optional) number of points to display on plot
        :param port: (int) port number for update server
        """

        self.sensorpush_client = sensorpush_client
        self.ni_client = ni_client
        self.display_pts = display_pts
        self.log = LogHandler(logger_client)

        #initiate channel objects
        self.channels = []
        self.init_channels(channel_params)
        num_lasers = len(self.channels)

        gui = f'rack_monitor_{num_lasers}lasers'

        # Instantiate gui
        self.gui = Window(
            gui_template=gui,
            host=get_ip(),
            port=port,
            log=self.log
        )

        # Setup stylesheet.
        self.gui.apply_stylesheet()

        num_plots = 2 + num_lasers
        self.widgets = get_gui_widgets(
            gui=self.gui,
            temp=1, humidity=1, voltage=num_lasers,
            graph=num_plots, reset=num_plots
        )

        # Configure plots

        self.widgets['curve'] = []

        self.sensor = Sensor(self.sensorpush_client, log=self.log)
        self._initialize_sensor(TEMP_INDEX)
        self._initialize_sensor(HUMIDITY_INDEX)

        [self._initialize_channel(c) for c in self.channels]

    def init_channels(self, channel_params):
        """initiates NI_channel objects"""

        # Initialize each channel individually
        for num, params in channel_params.items():
            self.channels.append(NI_channel(params, self.ni_client, log=self.log))

    def run(self):
        """Runs the WlmMonitor

        Can be stopped using the pause() method
        """

        self._update_sensor()
        [self._update_channel(c) for c in self.channels]
        self.gui.force_update()

    def reset(self, plot_index):
        """resets the plot back to initial framing"""

        self.widgets['graph'][plot_index].getPlotItem().enableAutoRange(axis='xy', enable=True)

    # Technical methods

    def _initialize_sensor(self, index):
        """Initializes a channel and outputs to the GUI

        Should only be called in the beginning of channel use to assign physical GUI widgets
        """
        #TODO: maybe put temp and humidity on same graph
        self.sensor.initialize(self.display_pts)

        # Create curves
        axis = pg.DateAxisItem(orientation='bottom')
        self.widgets['graph'][index].setAxisItems({'bottom': axis})
        self.widgets['curve'].append(self.widgets['graph'][index].plot(
            pen=pg.mkPen(color=self.gui.COLOR_LIST[0])
        ))
        self.widgets['reset'][index].clicked.connect(
            lambda: self.reset(index)
        )

    def _update_sensor(self):
        """ Updates all channels + displays

        Called continuously inside run() method to refresh WLM data and output on GUI
        """
        # Update data with the new temperature and humidity
        time = self.sensorpush_client.get_time()
        temp = self.sensorpush_client.get_temperature()
        humidity = self.sensorpush_client.get_humidity()
        self.sensor.update(time, temp, humidity)

        times = [t.timestamp() for t in self.sensor.time]

        # Update temperature
        self.widgets['curve'][TEMP_INDEX].setData(x=times, y=self.sensor.temp)
        self.widgets['temp'].setValue(self.sensor.temp[0])

        # Update humidity
        self.widgets['curve'][HUMIDITY_INDEX].setData(x=times, y=self.sensor.humidity)
        self.widgets['humidity'].setValue(self.sensor.humidity[0])

    def _initialize_channel(self, channel):
        """Initializes NI channel"""
        channel.initialize()
        index = channel.index + 2

        #Create curves
        self.widgets['curve'].append(self.widgets['graph'][index].plot(
            pen=pg.mkPen(color=self.gui.COLOR_LIST[0])
        ))
        self.widgets['reset'][index].clicked.connect(
            lambda: self.reset(index)
        )

    def _update_channel(self, channel):
        voltage = self.ni_client.get_ai_voltage(ai_channel=channel)
        channel.update(voltage)
        index = channel.index + 2

        self.widgets['curve'][index].setData(channel.data)
        self.widgets['voltage'][channel.index].setValue(channel.data[-1])

    def get_env_data(self, num_points=1):
        return self.sensorpush_client.get_data(num_points)

    def get_time(self):
        return self.sensorpush_client.get_time()

    def get_temperature(self):
        return self.sensorpush_client.get_temperature()

    def get_humidity(self):
        return self.sensorpush_client.get_humidity()

    def get_voltage(self, channel):
        return self.ni_client.get_ai_voltage(ai_channel=channel)


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

    def exposed_get_voltage(self, channel):
        return self._module.get_voltage(channel)


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

    def get_voltage(self, channel):
        return self._service.exposed_get_voltage(channel)


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


class NI_channel:
    """Object containing all information regarding a single NI Channel"""

    def __init__(self, params, ni_client=None, log: LogHandler = None):
        """
        Initializes all parameters given, sets others to default. Also sets up some defaults + placeholders for data

        :param params: list of channel parameters
        :param NI_client: NI_client object
        :param log: (LogHandler) instance of LogHandler for logging metadata
        """

        self.params = params

        self.channel = params['channel']
        self.index = params['index']
        self.ni_client = ni_client
        self.log = log
        self.labels_updated = False  # Flag to check if we have updated all labels

        # Initialize relevant placeholders
        self.data = np.array([])

    def initialize(self, display_pts=1000):
        """
        Initializes the channel based on the current value

        :param display_pts: number of points to display on the plot
        """
        self.log.info(f'NI_client is of type {type(self.ni_client)}')
        if self.ni_client != None:
            self.data = self.ni_client.get_ai_voltage(ai_channel=self.channel, num_samples=display_pts, sample_rate=1000)

    def update(self, new_data):
        """
        Updates the data

        :param value: (float) current value
        """
        self.data = np.append(self.data[1:], new_data)


def launch(**kwargs):
    """ Launches the rack monitor script """

    logger = kwargs['logger']

    config = load_script_config(
        script='rack_monitor',
        config=kwargs['config'],
        logger=logger
    )

    device_id = config['device_id']
    channel_params = config['channels']

    sensorpush_client = find_client(
        clients=kwargs['clients'],
        settings=config,
        client_type='sensorpush',
        logger=logger
    )

    ni_client = find_client(
        clients=kwargs['clients'],
        settings=config,
        client_type='nidaqmx',
        logger=logger
    )

    # Instantiate Monitor script
    rack_monitor = RackMonitor(
        sensorpush_client=sensorpush_client,
        ni_client=ni_client,
        logger_client=logger,
        channel_params=channel_params
    )

    update_service = kwargs['service']
    update_service.assign_module(module=rack_monitor)
    logger.update_data(data=dict(device_id=device_id))
    rack_monitor.gui.set_network_info(port=kwargs['server_port'])

    # Run continuously
    # Note that the actual operation inside run() can be paused using the update server
    while True:

        rack_monitor.run()
