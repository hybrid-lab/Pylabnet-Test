import matplotlib.pyplot as plt
import numpy as np

from pylabnet.utils.logging.logger import LogClient
from pylabnet.network.core.generic_server import GenericServer

from pylabnet.hardware.oscilloscopes.picoscope_2000a import Driver
from pylabnet.network.client_server.picoscope_2000a import Client
from pylabnet.network.client_server.picoscope_2000a import Server

"""This file is used to test and demonstrate the code for picoscope"""

