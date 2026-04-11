import tracemalloc
from time import sleep
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
pg.setConfigOptions(useOpenGL=False)


class Pico_Control:
    """A script class for controlling and displaying picoscope information. Includes functions
    that can be directly accessed through interface"""

    def __init__(self, pico_clients, logger_client, params, gui='picoscope_4', port=None):
        self.pico_clients = pico_clients
        self.num_picos = len(pico_clients)
        self.log = LogHandler(logger_client)
        self.log.debug('initiating gui')

        #setup GUI
        self.gui = Window(
            gui_template=gui,
            host=get_ip(),
            port=port,
            log=self.log,
            run=True
        )
        self.log.debug('gui initiated')

        self.gui.apply_stylesheet()
        self.log.debug('ready to get widgets')
        self.widgets = get_gui_widgets(
            gui=self.gui,
            graph=self.num_picos, legend=self.num_picos,
        )
        self.log.debug(f'collected widgets: {self.widgets}')
        self.widgets['curve'] = []

        #Pico stuff
        self.channel_params = []
        self.trigger_params = []

        self.blockModeOn = []
        self.rapidBlockModeOn = []
        self.ETSModeOn = []
        self.streamingModeOn = []

        self.log.debug('ready to initiate picos')
        for n in range(self.num_picos):
            self.openUnit(n)
        self.log.debug('pico is open')
        self._initiatePicos(params)

    def openUnit(self, n):
        """n is the index of the specific pico"""
        self.pico_clients[n].openUnit()

    #channels
    def setChannels(self, n):
        self.log.debug('entered setChannels')
        self.pico_clients[n].setChannel(self.channel_params[n])

    def setChannelCoupling(self, n, channel, coupling_type):
        """AC or DC"""
        self.pico_clients[n].setChannelCoupling(channel, coupling_type)

    #Block Mode
    def setupBlock(self, n):
        trigger_params = self.trigger_params[n]
        self.pico_clients[n].setupBlock(trigger_params)

    def runBlock(self, n):
        trigger_params = self.trigger_params[n]
        self.pico_clients[n].setupBlock(trigger_params)
        self.blockModeOn[n] = True

    def stopBlock(self, n):
        self.pico_clients[n].stop()
        self.blockModeOn[n] = False

    #Rapid Block Mode
    def setupRapidBlock(self, n, nSegments=10):
        """nSegments=nCaptures"""
        trigger_params = self.trigger_params[n]
        self.pico_clients[n].setupRapidBlock(trigger_params, nSegments, nSegments)

    def runRapidBlock(self, n):
        self.rapidBlockModeOn[n] = True

    def stopRapidBlock(self, n):
        self.pico_clients[n].stop()
        self.rapidBlockModeOn = False

    #closing unit
    def closeUnit(self, n):
        self.pico_clients[n].stop()
        self.pico_clients[n].closeUnit()
        self.blockModeOn[n] = False
        self.rapidBlockModeOn[n] = False
        self.ETSModeOn[n] = False
        self.streamingModeOn[n] = False

    def run(self):
        """Runs the Pico Control. Can be paused with pause()"""
        try:
            for n in range(self.num_picos):
                if self.blockModeOn[n]:
                    self.log.debug('running Block')
                    self._runBlock(n)
                if self.rapidBlockModeOn[n]:
                    self.log.info('running RapidBlock')
                    self._runRapidBlock(n)
        except:
            self.log.info('closing unit')
            self.closeUnit(n)

    #Technical methods
    def _initiatePicos(self, params):
        self.log.debug('entered _initiatePicos')
        for n in range(self.num_picos):
            #set channels
            channel_params = params[f"{n}"]["channels"]
            self.channel_params.append(channel_params)
            self.setChannels(n)
            self.log.debug('successfully set channels')

            #assign trigger_params
            trigger_params = params[f"{n}"]['trigger_params']
            self.trigger_params.append(trigger_params)

            self.pico_clients[n].setTime()
            self.log.debug('time as been set')

            #initiate modes
            self.blockModeOn.append(True)
            self.setupBlock(n)
            self.rapidBlockModeOn.append(False)
            self.ETSModeOn.append(False)

            self.log.debug('ready to create curves')
            # Create curves
            # Ch A
            if self.num_picos == 1:
                self.widgets['curve'].append(self.widgets['graph'].plot(
                    pen=pg.mkPen(color=self.gui.COLOR_LIST[0])
                ))
                self.log.debug('Ch A curve appended')
                self.widgets['legend'].addItem(self.widgets['curve'][0])
                self.log.debug('legend made for ChA')
            else:
                self.widgets['curve'].append(self.widgets['graph'][n].plot(
                    pen=pg.mkPen(color=self.gui.COLOR_LIST[0])
                ))
                self.log.debug('Ch A curve appended')
                self.widgets['legend'][n].addItem(self.widgets['curve'][2 * n])
                self.log.debug('legend made for ChA')

            # Ch B
            if self.num_picos == 1:
                self.widgets['curve'].append(self.widgets['graph'].plot(
                    pen=pg.mkPen(color=self.gui.COLOR_LIST[1])
                ))
                self.log.debug('Ch B curve appended')
                self.widgets['legend'].addItem(self.widgets['curve'][1])
                self.log.debug('legend made for ChB')
            else:
                self.widgets['curve'].append(self.widgets['graph'][n].plot(
                    pen=pg.mkPen(color=self.gui.COLOR_LIST[1])
                ))
                self.log.debug('Ch B curve appended')
                self.widgets['legend'][n].addItem(self.widgets['curve'][2 * n + 1])
                self.log.debug('legend made for ChB')

    def _runBlock(self, n):
        pico = self.pico_clients[n]
        self.log.debug('entered _runBlock')
        time, data = pico.runBlock()
        self.log.debug('data gathered')

        for d in range(len(data)):
            self.widgets['curve'][2 * n + d].setData(x=time, y=data[d])
            self.log.debug('data plotted')

    def _runRapidBlock(self, n):
        pico = self.pico_clients[n]
        time, offset, data = pico.runRapidBlock()
        dt_prev = 0
        for ind in range(len(offset)):
            for d in range(len(data)):
                graph_data = data[d][ind]
                self.widgets['curve'][2 * n + d].setData(x=time, y=graph_data)
            dt = offset[ind]
            sleep(dt - dt_prev)
            dt_prev = dt


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
    """ Launches the picoscope control script """
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
        if server['type'] == 'picoscope_2000a':
            client_configs.append(server['config'])

    pico_clients = []
    for client_config in client_configs:
        pico_clients.append(
            find_client(
                clients=kwargs['clients'],
                settings=config,
                client_type='picoscope_2000a',
                client_config=client_config,
                logger=logger
            )
        )

    logger.debug('clients hopefully found')
    # Instantiate Monitor script
    pico_control = Pico_Control(
        pico_clients=pico_clients,
        logger_client=logger,
        params=params
    )

    logger.debug('pico_control initiated')

    update_service = kwargs['service']
    update_service.assign_module(module=pico_control)
    logger.update_data(data=dict(device_id=device_id))
    pico_control.gui.set_network_info(port=kwargs['server_port'])

    # Run continuously
    # Note that the actual operation inside run() can be paused using the update server
    while True:

        pico_control.run()
