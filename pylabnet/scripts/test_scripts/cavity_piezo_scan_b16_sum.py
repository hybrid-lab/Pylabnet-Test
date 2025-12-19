import numpy as np
from pylabnet.utils.helper_methods import load_config

config_dict = load_config('global_daqports')


def configure(self):

    ctr = self.clients[('si_tt', 'sitt_b16')]
    ctr.create_combined_channel(
        channel_name='1+2',
        channel_list=[1, 2]
    )
    ctr.start_rate_monitor(name='sweeper', ch_list=['1+2'])


def experiment(piezo_voltage, self, **kwargs):

    ctr, nidaq = self.clients[('si_tt', 'sitt_b16')], self.clients[('nidaqmx', 'PXI1Slot4_2')]

    piezo_ao = config_dict["piezo_ao"]
    nidaq.set_ao_voltage(piezo_ao, piezo_voltage)

    return ctr.get_count_rate(
        name='sweeper',
        integration=0.1
    )[0]
