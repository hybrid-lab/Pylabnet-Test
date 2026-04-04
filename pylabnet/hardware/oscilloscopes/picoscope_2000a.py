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

ENABLED = 1
DISABLED = 0


class Driver:
    ### INITATION / GENERAL SETUP
    def __init__(self, serial=None, logger=None):
        """
        Opens picoscope unit (specified by serial number) and sets up channels with given input_ranges and analog_offsets.

        :serial: (str) serial number of picoscope to be opened (if None, opens first picoscope)
        :timebase: (int) An estimated timebase. See Programmer's guide p.28
        """
        self.logger = logger
        self.serial = serial

        #Create chandle and status ready for use
        self.chandle = ctypes.c_int16()
        self.status = {}

        self.isOpen = False
        self.oversample = ctypes.c_int16(0)

        self.channels = None
        self.timebase = 2
        self.trigger_parms = None
        self.preTriggerSamples = 0
        self.postTriggerSamples = 0
        self.totalSamples = 0
        self.time_interval_ns = 0

    def openUnit(self):
        self.status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(self.chandle), self.serial.encode('utf-8'))
        assert_pico_ok(self.status["openunit"])
        self.isOpen = True

    #set Channel stuff
    def setChannel(self, params):
        """
        :params: (dict) 2D dictionary of initial channel setup parameters
            - 'range': (str) Viewed voltage range of input. Options: 10mV, 20mV, 50mV, 100mV, 200 mV, 500 mV, 1V, 2V, 5V, 10V, 20V
            - 'offset': (float, V) Voltage added to input channel before digitization. If None, offset set to 0.0
            - 'coupling': (str) Coupling type, 'AC' or 'DC'. If None, coupling is set to 'AC'
        """
        self.channels = params
        for name, param in self.channels.items():
            if 'coupling' in param:
                coupling = param['coupling']
            else:
                coupling = 'AC'
                param['coupling'] = coupling

            volt_range = param['range']

            if 'offset' in param:
                offset = param['offset']
            else:
                offset = 0.0
                param['offset'] = 0.0

            maximum, minimum = self._getAnalogOffset(volt_range, coupling)
            if offset > maximum or offset < minimum:
                raise ValueError(f"{offset} offset is outside allowed range of {minimum} to {maximum}")

            self._setChannel(name, input_range=volt_range, analog_offset=offset)
            param['channelOn'] = True

    def setChannelCoupling(self, channel_name, coupling_type):
        channel = self.channels[channel_name]
        channel['coupling'] = coupling_type
        self._setChannel(channel_name, coupling_type, channel['range'], channel['offset'])
        channel['channelOn'] = True

    def setChannelRange(self, channel_name, volt_range):
        channel = self.channels[channel_name]
        channel['range'] = volt_range
        self._setChannel(channel_name, channel['coupling'], volt_range, channel['offset'])
        channel['channelOn'] = True

    def setChannelOffset(self, channel_name, offset):
        channel = self.channels[channel_name]
        maximum, minimum = self._getAnalogOffset(channel['range'], channel['coupling'])
        if offset > maximum or offset < minimum:
            raise ValueError(f"{offset} offset is outside allowed range of {minimum} to {maximum}")
        else:
            channel['offset'] = offset
            self._setChannel(channel_name, channel['coupling'], channel['range'], offset)
            channel['channelOn'] = True

    def closeChannel(self, channel_name):
        channel_name = channel_name.upper()
        if channel_name != 'A' and channel_name != 'B':
            raise ValueError(f"Channel {channel_name} does not exist")
        channel = ps.PS2000A_CHANNEL[f'PS2000A_CHANNEL_{channel_name}']
        self.status[f"closeCh{channel_name}"] = ps.ps2000aSetChannel(self.chandle, channel, DISABLED, 0, 0, 0.0)
        assert_pico_ok(self.status[f"closeCh{channel_name}"])
        self.channels[channel_name]['channelOn'] = False

    #timebase
    def setTime(self, preTriggerTime, postTriggerTime, totalSamples):
        totalTime = preTriggerTime + postTriggerTime
        ideal_ns = int(totalTime / totalSamples) * 1e9
        preTriggerSamples = int(preTriggerTime / ideal_ns)
        postTriggerSamples = int(postTriggerTime / ideal_ns)

        if ideal_ns < 9e-9:
            timebase = int(np.log2(5e8 * ideal_ns))
            if self.channels['A']['channelOn'] is True and self.channels['B']['channelOn'] is True and timebase is 1:
                timebase = 2
        else:
            timebase = int(6.25e7 * ideal_ns)
        ns, maxSamples = self._getTimebase(timebase)
        if totalSamples < maxSamples:
            self.totalSamples = preTriggerSamples + postTriggerSamples
            self.time_interval_ns = ns
            self.preTriggerSamples = preTriggerSamples
            self.postTriggerSamples = postTriggerSamples

    def _getTimebase(self, timebase, segmentIndex=0):
        """
        Calculates the sampling rate and maximum number of samples for a given timebase under specified conditions.
        Depends on the number of channels enabled by the last call to setChannels().
        Before using, estimate the timebase number that you require using timebase guide in Programmer's Guide
        (linked above at start of this file)

        :timebase: (int)
        :segmentIndex: (int) the index of the memory segment to use

        returns timeIntervalNanoseconds (float, time interval between readins at selected timebase),
                maxSamples (int, maximum number of samples available)
        """
        self.timebase = timebase
        timeIntervalns = ctypes.c_float()
        maxSamples = ctypes.c_int32()
        self.status["getTimebase"] = ps.ps2000aGetTimebase2(self.chandle, self.timebase, self.totalSamples,
                                                            ctypes.byref(timeIntervalns), self.oversample,
                                                            ctypes.byref(maxSamples), segmentIndex)
        assert_pico_ok(self.status["getTimebase"])
        self.time_interval_ns = timeIntervalns.value
        return timeIntervalns.value, maxSamples.value

    #trigger stuff
    def setTrigger(self, params):
        """
        :trigger_params: (dict) dictionary of trigger parameteres for simpleTrigger
            - 'channel': (str) channel to trigger on 'A' or 'B'
            - 'threshold': (int) ADC count at which trigger will fire (default = 1024)
            - 'direction': (str) direction in which signal must move to cause a trigger (default = 'rising')
                Options are 'Above', 'Below', Rising', 'Falling', and 'Rising_or_falling'
            - 'delay': (int, ms) time between trigger occuring and first sample being taken (default = 0)
            - 'autoTrigger_ms': (int, ms) number of ms the device will wait if no trigger occurs. If 0, scope will wait
                indefinitely for a trigger (default = 1000)
        """
        self.trigger_params = params
        triggerChannel = self.trigger_params['channel']

        if 'threshold' in self.trigger_params:
            triggerThreshold = self.trigger_params['threshold']
        else:
            triggerThreshold = 1024

        if 'direction' in self.trigger_params:
            triggerDirection = self.trigger_params['direction']
        else:
            triggerDirection = 'RISING'

        if 'delay' in self.trigger_params:
            triggerDelay = self.trigger_params['delay']
        else:
            triggerDelay = 0

        if 'autoTrigger_ms' in self.trigger_params:
            autoTrigger = self.trigger_params['autoTrigger_ms']
        else:
            autoTrigger = 1000

        self.simpleTrigger(triggerChannel, triggerThreshold, triggerDirection, triggerDelay, autoTrigger)
        self.totalSamples = self.preTriggerSamples + self.postTriggerSamples

    def closeUnit(self):
        """Closes the unit"""

        self.status["closeUnit"] = ps.ps2000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["closeUnit"])
        self.isOpen = False

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

    ### BLOCK MODE

    # Block Setup

    def setupBlock(self, trigger_params, downsampling_mode=None, nSegments=1):
        """
        Setup trigger with SimpleTrigger, and setup Data buffers.

        :trigger_params: see setTrigger()
        :preTriggerSamples: (int) number of samples to store before the trigger event
        :postTriggerSamples: (int) number of samples to store after the trigger event
        :downsampling_mode: (str) the downsampling mode used to store data
        :nSegments: (int) the number of memory segments used (number of buffers to setup)
        """
        self.setTrigger(trigger_params)

        mode = self._checkDownsampleMode(downsampling_mode)

        #Create and set buffers ready for assigning pointers for data collection
        for channel, param in self.channels.items():
            if nSegments == 1:
                param['buffersMax'] = (ctypes.c_int16 * self.totalSamples)()
                param['buffersMin'] = (ctypes.c_int16 * self.totalSamples)()
                self._setDataBuffers(channel, param['buffersMax'], param['buffersMin'], self.totalSamples, 0, mode)
            else:
                param['buffersMax'] = []
                param['buffersMin'] = []
                for index in range(nSegments):
                    param['buffersMax'].append((ctypes.c_int16 * self.totalSamples)())
                    param['buffersMin'].append((ctypes.c_int16 * self.totalSamples)())
                    self._setDataBuffers(channel, param['buffersMax'][-1], param['buffersMin'][-1], self.totalSamples, index, mode)

    #Run Block
    def runBlock(self, segmentIndex=0, downsample_ratio=1, downsample_mode=None):
        """
        Starts collecting data in block mode. Number of samples collected is determined by noOfPreTriggerSamples and
        noOfPostTriggerSamples. Total number of samples must not be more than the size of the segment referred to by
        segmentIndex.

        Loop this
        Returns the data to graph

        :segmentIndex: (int) which memory segment to use
        """
        cTotalSamples = ctypes.c_int32(self.totalSamples)
        overflow = ctypes.c_int16()

        self.status["runBlock"] = ps.ps2000aRunBlock(self.chandle, self.preTriggerSamples, self.postTriggerSamples, self.timebase,
                                                     self.oversample, None, segmentIndex, None, None)
        assert_pico_ok(self.status["runBlock"])
        self._isReady()

        #get values
        mode = self._checkDownsampleMode(downsample_mode)
        self._getValues(0, cTotalSamples, segmentIndex, overflow, mode, downsample_ratio)

        time, data = self._getGraphValues(cTotalSamples)
        return time, data

    ### RAPID BLOCK MODE

    def setupRapidBlock(self, nSegments=10, nCaptures=10, downsampling_mode=None):
        """
        Sets up additional rapid block requirements ontop of basic block set up stuff

        :preTriggerSamples: (int) number of samples to store before the trigger event
        :postTriggerSamples: (int) number of samples to store after the trigger event
        :nSegments: (int) number of segments required
        :nCaptures: (int) number of waveforms to capture in one run
        :downsampling_mode: (str) the downsampling mode used to store data
        """
        self._memorySegments(self.totalSamples, nSegments)
        self._setNoOfCaptures(nCaptures)
        self.setupBlock(self.preTriggerSamples, self.postTriggerSamples, downsampling_mode, nSegments)

    ## Run Rapid Block

    def runRapidBlock(self, segmentIndex=0, downsample_ratio=1, downsample_mode=None):
        """
        Starts collecting data in block mode. Number of samples collected is determined by noOfPreTriggerSamples and
        noOfPostTriggerSamples. Total number of samples must not be more than the size of the segment referred to by
        segmentIndex.

        Loop this

        :preTriggerSamples: (int) number of samples to store before the trigger event
        :postTriggerSamples: (int) number of samples to store after the trigger event
        :segmentIndex: (int) which memory segment to use
        """
        cTotalSamples = ctypes.c_int32(self.totalSamples)
        overflow = ctypes.c_int16()

        self.status["runBlock"] = ps.ps2000aRunBlock(self.chandle, self.preTriggerSamples, self.postTriggerSamples, self.timebase,
                                                     self.oversample, None, segmentIndex, None, None)
        assert_pico_ok(self.status["runBlock"])
        self._isReady()

        self._getValues(0, cTotalSamples, segmentIndex, overflow, downsample_ratio, downsample_mode)

    ### ETS MODE

    ### STREAMING MODE

    ### AWG (Arbitrary Wave Generator)

    #helper functions

    def _setChannel(self, channel_name, coupling_type='AC', input_range='20V', analog_offset=0.0):
        """
        Sets/initiates a channel on the picoscope

        :channel: (str) The name of the channel being set (A or B)
        :coupling_type: (str) The type of current being passed through the channel (AC or DC)
        :range: (str) Input range (voltage) of input +/-(10mV, 20mV, 50mV, 100mV, 200 mV, 500 mV, 1V, 2V, 5V, 10V, 20V)

        """
        channel_name = channel_name.upper()
        if channel_name != 'A' and channel_name != 'B':
            raise ValueError(f"Channel {channel_name} does not exist")

        coupling_type = coupling_type.upper()
        if coupling_type != 'AC' and coupling_type != 'DC':
            raise ValueError(f"The given coupling type is {coupling_type}. Coupling type must be AC or DC")

        input_range = input_range.upper()
        ALLOWED_RANGES = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V', '20V']
        if input_range not in ALLOWED_RANGES:
            raise ValueError(f"The given voltage range {input_range} is not within the allowed ranges. See p. 76 on the Picoscope Programmer's Guide")

        channel = ps.PS2000A_CHANNEL[f'PS2000A_CHANNEL_{channel_name}']
        coupling = ps.PS2000A_COUPLING[f'PS2000A_{coupling_type}']
        channel_range = ps.PS2000A_RANGE[f'PS2000A_{input_range}']
        self.status[f"setCh{channel_name}"] = ps.ps2000aSetChannel(self.chandle, channel, ENABLED,
                                                                   coupling, channel_range, analog_offset)
        assert_pico_ok(self.status[f"setCh{channel_name}"])

    def _getAnalogOffset(self, input_range, coupling_type):
        """
        Returns maximum and minimum allowable analog offset for specific voltage range

        :volt_range: (str) voltage range to be used when gathering min and max information
            10mV, 20mV, 50mV, 100mV, 200 mV, 500 mV, 1V, 2V, 5V, 10V, 20V
        :coupling_type: (str) AC or DC
        """
        input_range = input_range.upper()
        ALLOWED_RANGES = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V', '20V']
        if input_range not in ALLOWED_RANGES:
            raise ValueError(f"The given voltage range {input_range} is not within the allowed ranges. See p. 76 on the Picoscope Programmer's Guide")

        coupling = ps.PS2000A_COUPLING[f'PS2000A_{coupling_type}']
        volt_range = ps.PS2000A_RANGE[f'PS2000A_{input_range}']
        max_volt = ctypes.c_float()
        min_volt = ctypes.c_float()
        self.status["getAnalogOffset"] = ps.ps2000aGetAnalogueOffset(self.chandle, volt_range, coupling, ctypes.byref(max_volt), ctypes.byref(min_volt))
        assert_pico_ok(self.status["getAnalogOffset"])
        return max_volt.value, min_volt.value

    def _setDataBuffer():
        return

    def _setDataBuffers(self, channel, bufferMax, bufferMin, totalSamples, segmentIndex=0, mode=None):
        """
        Tells the driver the location of one or two buffers for receiving data.

        :channel: (str) A or B
        :bufferMax: user-allocated buffer to receive the maximum data values in aggregation mode,
                or the non-aggregated values otherwise. each value is a  16-bit ADC count scaled according to
                the selected voltage range.
        :bufferMin: user-allocated buffer to receive the minimum data values in aggregation mode. Not normally
                used in other modes, but you can direct the driver to write non-aggregated values to this buffer by
                setting bufferMax to NULL. To enable aggregation, the downsampling ratio and mode must be set
                appropriately when calling one of the ps2000aGetValues() functions.
        :segmentIndex: (int) the number of the memory segment to be used
        :mode: downsampling mode (none, aggregate, average, decimate)
        """
        source = ps.PS2000A_CHANNEL[f'PS2000A_CHANNEL_{channel.upper()}']
        self.status[f"setDataBuffers{channel}"] = ps.ps2000aSetDataBuffers(self.chandle, source, ctypes.byref(bufferMax),
                                                                           ctypes.byref(bufferMin), totalSamples, segmentIndex,
                                                                           mode)
        assert_pico_ok(self.status[f"setDataBuffers{channel}"])

    def _checkDownsampleMode(self, mode):
        if mode is None:
            mode = 'NONE'
        mode = mode.upper()
        DOWNSAMPLING_MODES = ['NONE', 'AGGREGATE', 'AVERAGE', 'DECIMATE']
        if mode not in DOWNSAMPLING_MODES:
            raise ValueError(f"{mode} is not a valid downsampling mode")
        return ps.PS2000A_RATIO_MODE[f'PS2000A_RATIO_MODE_{mode}']

    def _runBlock(self, preTriggerSamples, postTriggerSamples, segmentIndex=0):
        self.status["runBlock"] = ps.ps2000aRunBlock(self.chandle, preTriggerSamples, postTriggerSamples, self.timebase,
                                                     self.oversample, None, segmentIndex, None, None)
        assert_pico_ok(self.status["runBlock"])

    def _isReady(self):
        """
        Checks for data collection to finish using ps2000aIsReady
        """
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.status["isReady"] = ps.ps2000aIsReady(self.chandle, ctypes.byref(ready))

    def _getValues(self, start_index, cTotalSamples, segmentIndex, overflow, mode, downsample_ratio=1):
        """
        Retrieves block-mode data, with or without downsampling, starting at the specified sample number. Used to get stored
        data after data collection has stopped and store it in a user buffer previously passed to _setDataBuffer().
        Blocks the calling function while retrieving data.
        Sufficient to retrieve data for all channels at same time

        :start_index:
        :cTotalSamples:
        :segmentIndex:
        :overflow:
        :downsample_ratio:
        :downsample_mode:
        """
        self.status["getValues"] = ps.ps2000aGetValues(self.chandle, start_index, ctypes.byref(cTotalSamples),
                                                       downsample_ratio, mode, 0, segmentIndex, ctypes.byref(overflow))
        assert_pico_ok(self.status["getValues"])

    """TODO"""

    def _getValuesBulk(self, cTotalSamples, firstSegmentIndex, lastSegmentIndex, overflow, downsammple_ratio, downsample_mode=None):
        """
        Retrieves waveforms captured using RAPID BLOCK MODE. Waveforms must have been collected sequentially and in the same run.
        Sufficient to retrieve data for all channels at the same time.
        """
        if downsample_mode is None:
            downsample_mode = 'NONE'
        downsample_mode = downsample_mode.upper()
        DOWNSAMPLING_MODES = ['NONE', 'AGGREGATE', 'AVERAGE', 'DECIMATE']
        if downsample_mode not in DOWNSAMPLING_MODES:
            raise ValueError(f"{downsample_mode} is not a valid downsampling mode")
        mode = ps.PS2000A_RATIO_MODE[f'PS2000A_RATIO_MODE_{downsample_mode}']

    def _getGraphValues(self):
        """Returns time axis in microseconds"""
        start = self.preTriggerSamples * self.time_interval_ns * 0.001
        end = self.postTriggerSamples * self.time_interval_ns * 0.001
        time = np.linspace(start, end, self.totalSamples.value)

        maxADC = self._maximumValue()
        data = []
        for channel, param in self.channels.items():
            maxBuffer = param['buffersMax']
            chRange = ps.PS2000A_RANGE[f'PS2000A_{param['range']}']
            data.append(adc2mV(maxBuffer, chRange, maxADC))

        return time, data

    def _memorySegments(self, maxSamples, nSegments=10):
        """
        Sets the number of memory segments (divides memory into a number of segments so scope can store several waveforms sequentially)
        The max number of waveforms PicoScopee 2206B can handle is 32 MS (shared between channels). Returns the number of samples
        available in each segment (total number over the 2 channels)

        :nSegments: (int) number of segments required
        """
        cMaxSamples = ctypes.c_int32(maxSamples)
        self.status["memorySegments"] = ps.ps2000aMemorySegments(self.chandle, nSegments, ctypes.byref(cMaxSamples))
        assert_pico_ok(self.status["memorySegments"])
        return cMaxSamples.value

    def _setNoOfCaptures(self, nCaptures=10):
        """
        Sets the number of captures to be collected in one run of rapid block mode. Must be called before a run, or else
        driver will capture only one waveform. Value remains constant unless changed

        :nCaptures: (int) the number of waveforms to capture in one run
        """
        self.status["setNoOfCaptures"] = ps.ps2000aSetNoOfCaptures(self.chandle, nCaptures)
        assert_pico_ok(self.status["setNoOfCaptures"])

    ### TRIGGER FUNCTIONS

    def _samplePeriod(self, msTime):
        """Calculates number of sample periods from time input, using set timebase. msTime is in milliseconds"""
        if self.timebase < 3:
            interval = np.power(2, self.timebase) / 500000000
        else:
            interval = (self.timebase - 2) / 62500000

        #seconds to ms
        interval = 1e6 * interval

        return int(msTime / interval)

    def simpleTrigger(self, channel, threshold=1024, threshold_direction='RISING', delay=0, auto_trigger=1000):
        """
        Sets trigger on given channel, given trigger params
        """
        channel = channel.upper()
        if channel != 'A' and channel != 'B':
            raise ValueError(f"Channel {channel} does not exist")

        threshold_direction = threshold_direction.upper()
        DIRECTIONS = ['ABOVE', 'BELOW', 'RISING', 'FALLING', 'RISING_OR_FALLING']
        if threshold_direction not in DIRECTIONS:
            raise ValueError(f"Threshold Direction {threshold_direction} is not supported.")

        source = ps.PS2000A_CHANNEL[f'PS2000A_CHANNEL_{channel}']
        direction = ps.PS2000A_THRESHOLD_DIRECTION[f'PS2000A_{threshold_direction}']
        delay = self._samplePeriod(delay)

        self.status[f"trigger"] = ps.ps2000aSetSimpleTrigger(self.chandle, ENABLED, source, threshold, direction, delay, auto_trigger)
        assert_pico_ok(self.status["trigger"])

    def triggerChannelConditions(): #idk how to implement this
        return

    def triggerChannelDirections(self, channelA, channelB):
        """
        Sets the direction of the trigger for each channel.

        :channelA: (str) direction for channel A
        :channelB: (str) direction for channel B

        See p. 104 on Programmer's Guide for allowed directions
        """
        DIRECTIONS = ['ABOVE', 'ABOVE_LOWER', 'BELOW', 'BELOW_LOWER', 'RISING', 'RISING_LOWER', 'FALLING', 'FALLING_LOWER',
                      'RISING_OR_FALLING', 'INSIDE', 'OUTSIDE', 'ENTER', 'EXIT', 'ENTER_OR_EXIT', 'NONE']

        channelA = channelA.upper()
        channelB = channelB.upper()
        if channelA not in DIRECTIONS:
            raise ValueError(f"Threshold Direction {channelA} is not supported.")
        if channelB not in DIRECTIONS:
            raise ValueError(f"Threshold Direction {channelB} is not supported.")

        A = ps.PS2000A_THRESHOLD_DIRECTION[f'PS2000A_{channelA}']
        B = ps.PS2000A_THRESHOLD_DIRECTION[f'PS2000A_{channelB}']
        self.status["triggerDirections"] = ps.ps2000aSetTriggerChannelDirections(self.chandle, A, B, None, None, None, None)
        assert_pico_ok(self.status["triggerDirections"])

    def triggerChannelProperties():
        return

    def triggerDelay(self, delay):
        """
        Sets the post-trigger delay. Causes capture to start a defined time after the trigger event

        :delay: (int) time between the trigger occuring and the first sample, milliseconds
        """
        delay = self._samplePeriod(delay)
        self.status["TriggerDelay"] = ps.ps2000aSetTriggerDelay(self.chandle, delay)
        assert_pico_ok(self.status["TriggerDelay"])

    def setPulseWidthQualifier():
        return

    def isTriggerOrPulseWidthQualifierEnabled():
        return

    def getTriggerTimeOffset64():
        return

    def getValuesTriggerTimeOffsetBulk64():
        return

    ### EXTRA FUNCTIONS THAT COME IN DRIVER

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

    def getMaxSegments(self):
        """
        returns the maximum number of segments allowed for the opened variant
        """
        maxsegments = ctypes.c_uint32()
        self.status["getMaxSegments"] = ps.ps2000aGetMaxSegments(self.chandle, ctypes.byref(maxsegments))
        assert_pico_ok(self.status["getMaxSegments"])
        return maxsegments.value

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

    def getValuesAsync():
        return

    def getValuesOverlapped():
        return

    def getValuesOverlappedBulk():
        return

    def _maximumValue(self):
        """
        returns the maximum ADC count returned by calls to the getValues functions
        """
        maxVal = ctypes.c_int16()
        self.status["maxValue"] = ps.ps2000aMaximumValue(self.chandle, ctypes.byref(maxVal))
        assert_pico_ok(self.status["maxValue"])
        return maxVal

    def _minimumValue(self):
        """
        returns the minimum ADC count returned by calls to the getValues functions
        """
        minVal = ctypes.c_int16()
        self.status["minValue"] = ps.ps2000aMinimumValue(self.chandle, ctypes.byref(minVal))
        assert_pico_ok(self.status["minValue"])
        return minVal

    ### CAPTURES FUNCTIONS

    def getNoOfCaptures():
        return

    def getNoOfProcessedCaptures():
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
