import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from pylabnet.utils.logging.logger import LogClient
from pylabnet.network.core.generic_server import GenericServer

from pylabnet.hardware.oscilloscopes.picoscope_2000a import Driver
from pylabnet.network.client_server.picoscope_2000a import Client
from pylabnet.network.client_server.picoscope_2000a import Service

"""This file is used to test and demonstrate the code for picoscope"""

#instantiate
logger = LogClient(
    host='192.168.50.101',
    port=38967,
    module_tag='Picoscope'
)

#open driver
scope_driver = Driver(serial='12459/0199', logger=logger)

#start service
scope_service = Service()
scope_service.assign_module(module=scope_driver)
scope_service.assign_logger(logger=None)
scope_service_server = GenericServer(service=scope_service, host='localhost', port=60496)

scope_service_server.start()

#start client
scope = Client(host='localhost', port=60496)

### RUN BLOCK TEST/DEMO
channel_params = {
    'A' : {
        'range' : '2V'
    },
    'B' : {
        'range' : '2V',
        'offset' : 0.1,
        'coupling' : 'AC'
    }
}

scope.setChannel(channel_params) #set a default list. Implement other settings with gui options

#time axis stuff --> sommehow make more concise
scope.setNoSamples() #need to adjust time axis in gui
scope.getTimebase(2) #add some type of setting to adjust this in gui

#setup block
trigger_params = {
    'channel' : 'A'
}
scope.setupBlock(trigger_params)

#run block
fig, ax = plt.subplost()
def update(frame):
    time, data = scope.runBlock()
    for channel in data:
        ax.plot(time, data[channel])
ani = FuncAnimation(fig, update, interval=10)
plt.show()
