import ctypes
import numpy as np
from picosdk.ps2000a import ps2000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

from pylabnet.utils.logging.logger import LogHandler

class Driver:
    