from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


class Service(ServiceBase):
    def exposed_setChannel(self, params=None):
        return self._module.setChannel(params)

    def exposed_setChannelCoupling(self, channel_name, coupling_type):
        return self._module.setChannelCoupling(channel_name, coupling_type)

    def exposed_setChannelRange(self, channel_name, volt_range):
        return self._module.setChannelRange(channel_name, volt_range)
    
    def exposed_setChannelOffset(self, channel_name, offset):
        return self._module.setChannelOffset(channel_name, offset)
    

    def exposed_setTrigger(self, params):
        return self._module.setTrigger(params)
    

    def exposed_getTimebase(self, timebase, noSamples, segmentIndex):
        return self._module.getTimebase(timebase, noSamples, segmentIndex)


    #Block Mode
    def exposed_setupBlock(self, trigger_params, preTriggerSamples, postTriggerSamples, downsampling_mode=None, nSegments=1):
        return self._module.setupBlock(trigger_params, preTriggerSamples, postTriggerSamples, downsampling_mode, nSegments)
    
    def exposed_runBlock(self, segmentIndex, downsample_ratio=1, downsample_mode=None):
        return self._module.runBlock(segmentIndex, downsample_ratio, downsample_mode)
    
    #Rapid Block Mode
    def exposed_setupRapidBlock(self, preTriggerSamples, postTriggerSamples, nSegments=10, nCaptures=10, downsampling_mode=None):
        return self._module.setupRapidBlock(preTriggerSamples, postTriggerSamples, nSegments, nCaptures, downsampling_mode)
    
    def exposed_runRapidBlock(self, segmentIndex, downsample_ratio=1, downsample_mode=None):
        return self._module.runRapidBlock(segmentIndex, downsample_ratio, downsample_mode)

    
    #Closing unit stuff
    def exposed_closeUnit(self):
        return self._module.closeUnit()
    
    def exposed_stop(self):
        return self._module.stop()

class Client(ClientBase):
    def setChannel(self, params=None):
        return self._service.exposed_setChannel(params)

    def setChannelCoupling(self, channel_name, coupling_type):
        return self._module.exposed_setChannelCoupling(channel_name, coupling_type)

    def setChannelRange(self, channel_name, volt_range):
        return self._module.exposed_setChannelRange(channel_name, volt_range)
    
    def setChannelOffset(self, channel_name, offset):
        return self._module.exposed_setChannelOffset(channel_name, offset)
    

    def setTrigger(self, params):
        return self._module.exposed_setTrigger(params)
    

    def getTimebase(self, timebase, noSamples, segmentIndex):
        return self._module.exposed_getTimebase(timebase, noSamples, segmentIndex)


    #Block Mode
    def setupBlock(self, trigger_params, preTriggerSamples, postTriggerSamples, downsampling_mode=None, nSegments=1):
        return self._module.exposed_setupBlock(trigger_params, preTriggerSamples, postTriggerSamples, downsampling_mode, nSegments)
    
    def runBlock(self, segmentIndex, downsample_ratio=1, downsample_mode=None):
        return self._module.exposed_runBlock(segmentIndex, downsample_ratio, downsample_mode)
    
    #Rapid Block Mode
    def setupRapidBlock(self, preTriggerSamples, postTriggerSamples, nSegments=10, nCaptures=10, downsampling_mode=None):
        return self._module.exposed_setupRapidBlock(preTriggerSamples, postTriggerSamples, nSegments, nCaptures, downsampling_mode)
    
    def runRapidBlock(self, segmentIndex, downsample_ratio=1, downsample_mode=None):
        return self._module.exposed_runRapidBlock(segmentIndex, downsample_ratio, downsample_mode)
    
    #Closing unit stuff
    def closeUnit(self):
        return self._module.exposed_closeUnit()
    
    def stop(self):
        return self._module.exposed_stop()