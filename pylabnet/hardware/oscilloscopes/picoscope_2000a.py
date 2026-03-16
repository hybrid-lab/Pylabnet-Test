"""
    This houses all the functions that can be run on the picoscope
    Functions come from picosdk-python-wrappers github repository
    https://github.com/picotech/picosdk-python-wrappers/tree/master
"""

import ctypes
import numpy as np
from picosdk.ps2000a import ps2000a as ps
from picosdk.functions import adc2mV, assert_pico_ok
import time

ENABLED = 1
DISABLED = 0

class Driver:
    def __init__(self, serial=None, analog_offset=0.0):
        """"""

        #Create chandle and status ready for use
        self.chandle = ctypes.c_int16()
        self.status = {}
        self.serial = serial
        self.analog_offset = analog_offset

        self.openUnit()

    def openUnit(self):
        """
        Opens the picoscope
        """
        self.status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(self.chandle), self.serial.encode('utf-8'))
        assert_pico_ok(self.status["openunit"])
    
    def pingUnit(self):
        """
        Checks that already opened device is still connected to the USB port and communication is successful
        """
        self.status["pingunit"] = ps.ps2000aPingUnit(self.chandle)
        assert_pico_ok(self.status["pingunit"])

    def getUnitInfo(self, req_info, stringLength=10):
        """
        Retrieve information about the specified oscilloscope

        :req_info: (str) requrested information. The options are: driver_version, usb_version, hardware_version, 
            variant_info, batch_and_serial, cal_date (calibration date), kernel_version, digital_hardware_version, 
            analogue_hardware_version, firmware_version_1, firmware_version_2
        :stringLength: maximum number of chars that may be written to output string (info_out) --- c stuff
        """
        info_out = ctype.c_int8()
        required_size = ctype.c_int(16)

        req_info = req_info.upper()
        info = ps.PICO_INFO[f'PICOC_{req_info}']
        self.status["getUnitInfo"] = ps.ps2000aGetUnitInfo(self.chandle, ctypes.byref(info_out), stringLength, ctypes.byref(required_size), info)
        assert_pico_ok(self.status["getUnitInfo"])

        #print or something, idk yet

    def flashLED(self, num_flashed):
        """
        Flashes the LED on the front of the scope without blocking the calling thread. runStreaming() and runBlock() 
        cancels any flashing started by this function. Not possible to set LED to be constantly illuminated (indicates not initiated)

        :num_flashed: 
            < 0: flash LED indefinitely
            0  : stop flashing
            > 0: flash the LED given number of times. If already flashing at start of function, flash count reset to num_flashed
        """
        self.status["flashLED"] = ps.ps2000aFlashLed(self.chandle, num_flashed)
        assert_pico_ok(self.status["flashLED"])
    
    def closeUnit(self)
        """Closes the unit"""

        self.status["closeUnit"] = ps.ps2000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["closeUnit"])

    def setChannel(self, channel_name, coupling_type='AC', input_range='20V'):
        """
        Sets/initiates a channel on the picoscope

        :channel: (str) The name of the channel being set (A or B)
        :coupling_type: (str) The type of current being passed through the channel (AC or DC)
        :range: (str) Input range (voltage) of input +/-(10mV, 20mV, 50mV, 100mV, 200 mV, 500 mV, 1V, 2V, 5V, 10V, 20V)

        TODO: add error messages for wrong paramter inputs
        """

        if channel_name != 'A' and channel_name != 'B':
            return
        
        if coupling_type != 'AC' and coupling_type != 'DC':
            return
        
        input_range = input_range.upper()
        ALLOWED_RANGES = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V', '20V']
        if !(input_range in ALLOWED_RANGES):
            return
        
        channel = ps.PS2000A_CHANNEL[f'PS2000A_CHANNEL_{channel_name}']
        coupling = ps.PS2000A_COUPLING[f'PS2000A_{coupling_type}']
        channel_range = ps.PS2000A_RANGE[f'PS2000A_{input_range}']
        self.status[f"setCh{name}"] = ps.ps2000aSetChannel(self.chandle, channel, ENABLED, 
                                                            coupling, channel_range, self.analog_offset)
        assert_pico_ok(self.status[f"setCh{name}"])

    def simpleTrigger(self, channel, threshold=1024, threshold_direction='RISING', delay=0, auto_trigger=1000):
        """
        Sets trigger on given channel

        :channel: (str) The name of the channel being used as trigger source (A or B)
        :threshold: (int) ADC counts (lowkey don't know what this is)
        :threshold_direction: (str) above, below, rising, falling, rising_or_falling, above_lower, below_lower, rising_lower, falling_lower
        :delay: (int, s) 
        :auto_trigger: (int, ms)
        """

        source = ps.PS2000A_CHANNEL[f'PS2000A_CHANNEL_{channel}']
        direction = ps.PS2000A_THRESHOLD_DIRECTION[f'PS2000A_{threshold_direction}']

        self.status[f"trigger"] = ps.ps2000aSetSimpleTrigger(self.chandle, ENABLED, source, threshold, direction, delay, auto_trigger)