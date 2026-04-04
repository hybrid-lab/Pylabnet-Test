from pylabnet.hardware.oscilloscopes.picoscope_2000a import Driver
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
sys.path.insert(0, '/home/porkpie/pylabnet')


#open driver
scope = Driver(serial='12459/0199', logger=None)

### RUN BLOCK TEST/DEMO
"""channel_params = {
    'A' : {
        'range' : '2V'
    },
    'B' : {
        'range' : '2V',
        'offset' : 0.1,
        'coupling' : 'AC'
    }
}"""

channel_params = {
    'A': {
        'range': '2V'
    }
}

scope.setChannel(channel_params) #set a default list. Implement other settings with gui options

#time axis stuff --> sommehow make more concise
scope.setNoSamples() #need to adjust time axis in gui
scope.getTimebase(2) #add some type of setting to adjust this in gui

#setup block
trigger_params = {
    'channel': 'A'
}
scope.setupBlock(trigger_params)

#run block
fig, ax = plt.subplots()
num = 0


def update(frame):
    global num
    plt.cla()
    time, data = scope.runBlock(0)
    for channel_data in data:
        ax.plot(time, channel_data)
    print(f'cycle{num}')
    num = num + 1


ani = FuncAnimation(fig, update, interval=0.0001)
plt.show()

scope.stop()
scope.closeUnit()
