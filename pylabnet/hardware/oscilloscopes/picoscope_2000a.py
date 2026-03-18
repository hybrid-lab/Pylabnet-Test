"""
    This houses all the functions that can be run on the picoscope
    Functions come from picosdk-python-wrappers github repository
    https://github.com/picotech/picosdk-python-wrappers/tree/master

    Refer to PicoScope 2000 Series (A API) Programmer's Guide:
    https://www.picotech.com/download/manuals/picoscope-2000-series-a-api-programmers-guide.pdf
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
        self.isOpen = False

        self.openUnit()

    def openUnit(self):
        """
        Opens the picoscope
        """
        self.status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(self.chandle), self.serial.encode('utf-8'))
        assert_pico_ok(self.status["openunit"])
        self.isOpen = True
    
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
        info_out = ctypes.c_int8()
        required_size = ctypes.c_int(16)

        req_info = req_info.upper()
        info = ps.PICO_INFO[f'PICOC_{req_info}']
        self.status["getUnitInfo"] = ps.ps2000aGetUnitInfo(self.chandle, ctypes.byref(info_out), stringLength, ctypes.byref(required_size), info)
        assert_pico_ok(self.status["getUnitInfo"])

        return info_out.value, required_size.value

    def flashLED(self, num_flashed):
        """
        Flashes the LED on the front of the scope without blocking the calling thread. runStreaming() and runBlock() 
        cancels any flashing started by this function. Not possible to set LED to be constantly illuminated (indicates not initiated)

        :num_flashed: (int)
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
        self.isOpen = False

    def memorySegments(self, nSegments=1):
        """
        Sets the number of memory segments (divides memory into a number of segments so scope can store several waveforms sequentially)
        The max number of waveforms PicoScopee 2206B can handle is 32 MS (shared between channels). Returns the number of samples 
        available in each segment (total number over the 2 channels)

        :nSegments: (int) number of segments required
        """
        nMaxSamples = ctypes.c_int32()
        self.status["memorySegments"] = ps.ps2000aMemorySegments(self.chandle, nSegments, ctypes.byref(nMaxSamples))
        assert_pico_ok(self.status["memorySegments"])
        return nMaxSamples.value

    def getMaxSegments(self):
        """
        returns the maximum number of segments allowed for the opened variant
        """
        maxsegments = ctypes.c_uint32()
        self.status["getMaxSegments"] = ps.ps2000aGetMaxSegments(self.chandle, ctypes.byref(maxsegments))
        assert_pico_ok(self.status["getMaxSegments"])
        return maxsegments.value

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
        self.status[f"setCh{channel_name}"] = ps.ps2000aSetChannel(self.chandle, channel, ENABLED, 
                                                            coupling, channel_range, self.analog_offset)
        assert_pico_ok(self.status[f"setCh{channel_name}"])
    
    def getChannelInformation(self, req_info, channel_name):
        """
        Queries which ranges are available on a scope device
        
        :req_info: (str) requested information: currently, on CI_RANGES is supported
        :channel_name: (str) The name of the channel being queried (A or B)
        """ 
        info = ps.PS2000A_CHANNEL_INFO["PS2000A_CI_RANGES"]
        channel = ps.PS2000A_CHANNEL[f"PS2000A_CHANNEL_{channel_name}"]

        ranges = ctypes.c_int32()
        length = ctypes.c_int32()

        self.status[f"getCh{channel_name}Info"] = ps.ps2000aGetChannelInformation(self.chandle, info, 0, ctypes.byref(ranges), 
                                                                                ctypes.byref(length), channel)
        assert_pico_ok(self.status[f"getCh{channel_name}Info"])
        return ranges.value, length.value

    def setNoOfCaptures(self, nCaptures):
        """
        Sets the number of captures to be collected in one run of rapid block mode. Must be called before a run, or else
        driver will capture only one waveform. Value remains constant unless changed

        :nCaptures: (int) the number of waveforms to capture in one run
        """
        self.status["setNoOfCaptures"] = ps.ps2000aSetNoOfCaptures(self.chandle, nCaptures)
        assert_pico_ok(self.status["setNoOfCaptures"])

    def getTimebase(self, timebase, noSamples, segmentIndex):
        """
        Calculates the sampling rate and maximum number of samples for a given timebase under specified conditions.
        Depends on the number of channels enabled by the last call to setChannels().
        Before using, estimate the timebase number that you require using timebase guide in Programmer's Guide
        (linked above at start of this file)

        :timebase: (int) 
        :noSamples: (int) the number of samples required
        :segmentIndex: (int) the idnex of the memory segment to use

        returns timeIntervalNanoseconds (float, time interval between readins at selected timebase), 
                maxSamples (int, maximum number of samples available)
        """
        timeIntervalNanoseconds = ctypes.c_float()
        maxSamples = ctypes.c_int32()
        oversample = 0
        self.status["getTimebase"] = ps.ps2000aGetTimebase2(self.chandle, timebase, noSamples, 
                                                            ctypes.byref(timeIntervalNanoseconds), oversample,
                                                            ctypes.byref(maxSamples), segmentIndex)
        assert_pico_ok(self.status["getTimebase"])
        return timeIntervalNanoseconds.value, maxSamples.value
    
    def isReady(self):
        """
        May be used instead of a callback function to receive data from runBlock(). To use, pass NULL pointer as lpReady
        argument to runBlock(). Then, poll driver to see if it has finisehd collected the requested samples.

        If returns 0, device is still collecting. If non-zero, device has finished collecting and getValues() can be used to retrieve data
        """
        ready = ctypes.c_int16()
        self.status["isReady"] = ps.ps2000aIsReady(self.chandle, ctypes.byref(ready))
        assert_pico_ok(self.status["isReady"])
        return ready.value
    
    def stop(self):
        """
        Stops the scope device while it is waiting for a trigger or capturing data
        Block mode: terminates current capture. any data in buffer is invalid
        Rapid block mode: terminates the sequence of captures. Any completed capptures will contain valid data
        Streaming mode: terminates data capture. If called before trigger event, oscilloscope may not contain valid data.
                        If capture has already started, buffer will contain valid data
        """
        self.status["stop"] = ps.ps2000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])
    
    def holdOff(): #reserved for future use
        return
    
    def enumerateUnits(self, serialLth): 
        """
        counts the number of unopened units connected to the computer and returns a list of all serial numbers as a string. 
        Does not detect units that already have a handle assigned to them by the driver.

        :serialLth: length of the char buffer pointed to by serials
        """
        count = ctypes.c_int16()
        serials = ctypes.c_int8()
        serialLth_out = ctypes.c_int16() #lol idk how this works
    
    def getAnalogOffset(self, input_range, coupling_type):
        """
        Returns maximum and minimum allowable analog offset for specific voltage range

        :volt_range: (str) voltage range to be used when gathering min and max information
            10mV, 20mV, 50mV, 100mV, 200 mV, 500 mV, 1V, 2V, 5V, 10V, 20V
        :coupling_type: (str) AC or DC
        """
        input_range = input_range.upper()
        ALLOWED_RANGES = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V', '20V']
        if input_range not in ALLOWED_RANGES:
            return
        
        coupling = ps.PS2000A_COUPLING[f'PS2000A_{coupling_type}']
        volt_range = ps.PS2000A_RANGE[f'PS2000A_{input_range}']
        max_volt = ctypes.c_float()
        min_volt = ctypes.c_float()
        self.status["getAnalogOffset"] = ps.ps2000aGetAnalogueOffset(self.chandle, volt_range, coupling, ctypes.byref(max_volt), ctypes.byref(min_volt))
        assert_pico_ok(self.status["getAnalogOffset"])
        return max_volt.value, min_volt.value


    
    ### SAMPLING MODES FUNCTIONS

    #ETS (Equivalent-time sampling)
    def ETSOff(self):
        """
        Disables ETS
        """
        sampleTimePicoseconds = ctypes.c_int32()
        mode = ps.PS2000A_ETS_MODE['PS2000A_ETS_OFF']
        self.status["ETSOff"] = ps.ps2000aSetEts(self.chandle, mode, 1, 1, ctypes.byref(sampleTimePicoseconds))
        assert_pico_ok(self.status["ETSOff"])

    def setETSFast(self, etsCycles=500, etsInterleave=40):
        """
        Enables Fast ETS. This mode provides etsCycles of data, which may contain data from previously returned cycles.
        Returns the effective sampling interval of ETS data (captured sample time / etsInterleave)

        :etsCycles: (int) number of cycles to store, maximum value is 500
        :etsInterleave: (int) number of waveforms to combine into a single ETS capture. Maximum value is 40
        """
        sampleTimePicoseconds = ctypes.c_int32()
        mode = ps.PS2000A_ETS_MODE['PS2000A_ETS_FAST']
        self.status["ETSFast"] = ps.ps2000aSetEts(self.chandle, mode, etsCycles, etsInterleave, ctypes.byref(sampleTimePicoseconds))
        assert_pico_ok(self.status["ETSFast"])
        return sampleTimePicoseconds.value

    def setETSSlow(self, etsCycles=500, etsInterleave=40):
        """
        Enables Slow ETS. This mode provides fresh data every etsCycles. Takes longer than fast mode, but the data sets
        are more stable and are guaranteed to contain only new data.
        Returns the effective sampling interval of ETS data (captured sample time / etsInterleave)

        :etsCycles: (int) number of cycles to store, maximum value is 500
        :etsInterleave: (int) number of waveforms to combine into a single ETS capture. Maximum value is 40
        """
        sampleTimePicoseconds = ctypes.c_int32()
        mode = ps.PS2000A_ETS_MODE['PS2000A_ETS_SLOW']
        self.status["ETSSlow"] = ps.ps2000aSetEts(self.chandle, mode, etsCycles, etsInterleave, ctypes.byref(sampleTimePicoseconds))
        assert_pico_ok(self.status["ETSSlow"])
        return sampleTimePicoseconds.value
    
    def setEtsTimeBuffer(self, buffers):
        """
        Tells the driver where to find application's ETS time buffers. Contains the 64-bit timing information 
        for each ETS sample after running block-mode ETS capture

        :buffers: (array) array of 64-bit words, each representing the time in femtoseconds (10^-15 s) 
                at which the sample was captured
        """
        self.status["setEtsTimeBuffer"] = ps.ps2000aSetEtsTimeBuffer(self.chandle, buffers, len(buffers))
        assert_pico_ok(self.status["setEtsTimeBuffer"])
    
    #Block
    def runBlock():
        return
    
    def blockReady():  #different in ps2000a.py in picosdk-python-wrappers, idk why
        return
    
    #Streaming
    def runStreaming():
        return
    
    def getStreamingLatestValues():
        return
    
    def streamingReady(): #different in ps2000a.py in picosdk-python-wrappers, idk why
        return
    
    def noOfStreamingValues():
        return
    
    
    ### GATHERING DATA
    
    def getMaxDownSampleRatio():
        return
    
    def getValues():
        return
    
    def getValuesBulk():
        return
        
    def getValuesAsync():
        return
    
    def getValuesOverlapped():
        return
    
    def getValuesOverlappedBulk():
        return

    def maximumValue(self):
        """
        returns the maximum ADC count returned by calls to the getValues functions
        """
        maxVal = ctypes.c_int16()
        self.status["maxValue"] = ps.ps2000aMaximumValue(self.chandle, ctypes.byref(maxVal))
        assert_pico_ok(self.status["maxValue"])
        return maxVal.value
    
    def minimumValue(self):
        """
        returns the minimum ADC count returned by calls to the getValues functions
        """
        minVal = ctypes.c_int16()
        self.status["minValue"] = ps.ps2000aMinimumValue(self.chandle, ctypes.byref(minVal))
        assert_pico_ok(self.status["minValue"])
        return minVal.value


    ### TRIGGER FUNCTIONS

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
        assert_pico_ok(self.status["trigger"])
    
    def triggerChannelConditions():
        return
    
    def triggerChannelDirections():
        return

    def triggerChannelProperties():
        return
    
    def triggerDelay():
        return

    def setPulseWidthQualifier():
        return
    
    def isTriggerOrPulseWidthQualifierEnabled():
        return
    
    def getTriggerTimeOffset64():
        return
    
    def getValuesTriggerTimeOffsetBulk64():
        return
    

    ### CAPTURES FUNCTIONS

    def getNoOfCaptures():
        return
    
    def getNoOfProcessedCaptures():
        return

    def setDataBuffer():
        return
    
    def setDataBuffers():
        return

    
    ### SIGNAL GENERATOR FUNCTIONS

    def setSigGenArbitrary():
        return

    def setSigGenBuiltIn():
        return
    
    def setSigGenPropertiesArbitrary():
        return
    
    def setSigGenPropertiesBuiltIn():
        return

    def sigGenFrequencyToPhase():
        return

    def sigGenArbitraryMinMaxValues():
        return

    def sigGenSoftwareControl():
        return
