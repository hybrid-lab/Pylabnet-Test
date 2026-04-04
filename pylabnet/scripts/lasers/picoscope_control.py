from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase
from pylabnet.gui.pyqt.external_gui import Window
from pylabnet.utils.helper_methods import (unpack_launcher, create_server,
                                           load_config, get_gui_widgets, get_legend_from_graphics_view, add_to_legend, find_client,
                                           load_script_config, get_ip)
from pylabnet.utils.logging.logger import LogClient, LogHandler

from pylabnet.network.client_server.picoscope_2000a import Client as Pico_Client

import numpy as np
import pickle
import pyqtgraph as pg

import tracemalloc


class Pico_Control:
    """A script class for controlling and displaying picoscope information. Includes functions
    that can be directly accessed through interface"""

    def __init__(self, pico_clients, logger_client, params, gui='picoscope_4', port=None):
        self.pico_clients = pico_clients
        self.num_picos = len(pico_clients)
        self.log = LogHandler(logger_client)

        #setup GUI
        self.gui = Window(
            gui_template=gui,
            host=get_ip(),
            port=port,
            log=self.log
        )

        self.gui.apply_stylesheet()

        self.widgets = get_gui_widgets(
            gui=self.gui,
            graph=self.num_picos
        )
        self.widgets['curve'] = []

        #Pico stuff
        self.channel_params = []
        self.trigger_params = []

        self.blockModeOn = []
        self.rapidBlockModeOn = []
        self.ETSModeOn = []
        self.streamingModeOn = []

        self._initiatePicos(params)

    def openUnit(self, n):
        """n is the index of the specific pico"""
        self.pico_clients[n].openUnit()

    #channels
    def setChannels(self, n):
        self.pico_clients[n].setChannel(self.channel_params[n])

    def setChannelCoupling(self, n, channel, coupling_type):
        """AC or DC"""
        self.pico_clients[n].setChannelCoupling(channel, coupling_type)

    #Block Mode
    def setupBlock(self, n):
        trigger_params = self.trigger_params[n]
        self.pico_clients[n].setupBlock(trigger_params)

    def runBlock(self, n):
        self.blockModeOn[n] = True

    def stopBlock(self, n):
        self.pico_clients[n].stop()
        self.blockModeOn[n] = False

    #closing unit
    def closeUnit(self, n):
        self.pico_clients[n].closeUnit()
        self.blockModeOn[n] = False
        self.rapidBlockModeOn[n] = False
        self.ETSModeOn[n] = False
        self.streamingModeOn[n] = False

    def run(self):
        """Runs the Pico Control. Can be paused with pause()"""
        for n in range(self.num_picos):
            if self.blockModeOn[n] is True:
                self._runBlock(n)

    #Technical methods
    def _initiatePicos(self, params):
        for n in range(self.num_picos):
            #set channels
            channel_params = params[f"{n}"]["channels"]
            self.channel_params = channel_params

            #assign trigger_params
            trigger_params = params[f"{n}"]['trigger_params']
            self.trigger_params.append(trigger_params)

            #initiate modes
            self.blockModeOn.append(False)
            self.rapidBlockModeOn.append(False)
            self.ETSModeOn.append(False)
            self.streamingModeOn.append(False)

            self.pico_clients[n].setTime()

            # Create curves
            # Ch A
            self.widgets['curve'].append(self.widgets['graph'][n].plot(
                pen=pg.mkPen(color=self.gui.COLOR_LIST[0])
            ))
            add_to_legend(
                legend=self.widgets['legend'][n],
                curve=self.widgets['curve'][2 * n],
                curve_name=f'Picoscope {n} Channel A'
            )

            # Ch B
            self.widgets['curve'].append(self.widgets['graph'][n].plot(
                pen=pg.mkPen(color=self.gui.COLOR_LIST[1])
            ))
            add_to_legend(
                legend=self.widgets['legend'][n],
                curve=self.widgets['curve'][2 * n + 1],
                curve_name=f'Picoscope {n} Channel B'
            )

    def _runBlock(self, n):
        pico = self.pico_clients[n]
        time, data = pico.runBlock()

        for d in range(len(data)):
            self.widgets['curve'][2 * n + d].setData(x=time, y=data[d])


class Service(ServiceBase):
    """ A service to enable external updating of Control parameters.
     TODO: add all channel and trigger parameter functions here """

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


def launch(**kwargs):
    """ Launches the rack monitor script """
    tracemalloc.start()

    logger = kwargs['logger']

    config = load_script_config(
        script='pico_control',
        config=kwargs['config'],
        logger=logger
    )

    device_id = config['device_id']
    params = config['params']

    client_configs = []
    for server in config['servers']:
        if server['type'] == 'picoscope':
            client_configs.append(server['config'])

    pico_clients = []
    for client_config in client_configs:
        pico_clients.append(
            find_client(
                clients=kwargs['clients'],
                settings=config,
                client_type='picoscope',
                client_config=client_config,
                logger=logger
            )
        )

    # Instantiate Monitor script
    pico_control = Pico_Control(
        pico_clients=pico_clients,
        logger_client=logger,
        params=params
    )

    update_service = kwargs['service']
    update_service.assign_module(module=pico_control)
    logger.update_data(data=dict(device_id=device_id))
    pico_control.gui.set_network_info(port=kwargs['server_port'])

    # Run continuously
    # Note that the actual operation inside run() can be paused using the update server
    while True:

        pico_control.run()
