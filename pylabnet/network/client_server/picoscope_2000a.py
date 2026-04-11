from pylabnet.network.core.service_base import ServiceBase
from pylabnet.network.core.client_base import ClientBase


class Service(ServiceBase):
    def exposed_openUnit(self):
        return self._module.openUnit()

    def exposed_setChannel(self, params=None):
        return self._module.setChannel(params)

    def exposed_setChannelCoupling(self, channel_name, coupling_type):
        return self._module.setChannelCoupling(channel_name, coupling_type)

    def exposed_setChannelRange(self, channel_name, volt_range):
        return self._module.setChannelRange(channel_name, volt_range)

    def exposed_setChannelOffset(self, channel_name, offset):
        return self._module.setChannelOffset(channel_name, offset)

    def exposed_closeChannel(self, channel_name):
        return self._module.closeChannel(channel_name)

    def exposed_setTime(self, preTriggerTime, postTriggerTime, totalSamples):
        return self._module.setTime(preTriggerTime, postTriggerTime, totalSamples)

    def exposed_setTrigger(self, params):
        return self._module.setTrigger(params)

    #Block Mode

    def exposed_setupBlock(self, trigger_params, downsampling_mode=None):
        return self._module.setupBlock(trigger_params, downsampling_mode, 1)

    def exposed_runBlock(self, segmentIndex=0, downsample_ratio=1, downsample_mode=None):
        return self._module.runBlock(segmentIndex, downsample_ratio, downsample_mode)

    #Rapid Block Mode
    def exposed_setupRapidBlock(self, trigger_params, nSegments=10, nCaptures=10, downsampling_mode=None):
        return self._module.setupRapidBlock(trigger_params, nSegments, nCaptures, downsampling_mode)

    def exposed_runRapidBlock(self, segmentIndex, downsample_ratio=1, downsample_mode=None):
        return self._module.runRapidBlock(segmentIndex, downsample_ratio, downsample_mode)

    #Closing unit stuff

    def exposed_closeUnit(self):
        return self._module.closeUnit()

    def exposed_stop(self):
        return self._module.stop()


class Client(ClientBase):
    def openUnit(self):
        return self._service.exposed_openUnit()

    def setChannel(self, params=None):
        return self._service.exposed_setChannel(params)

    def setChannelCoupling(self, channel_name, coupling_type):
        return self._service.exposed_setChannelCoupling(channel_name, coupling_type)

    def setChannelRange(self, channel_name, volt_range):
        return self._service.exposed_setChannelRange(channel_name, volt_range)

    def setChannelOffset(self, channel_name, offset):
        return self._service.exposed_setChannelOffset(channel_name, offset)

    def closeChannel(self, channel_name):
        return self._service.exposed_closeChannel(channel_name)

    def setTime(self, preTriggerTime=5000, postTriggerTime=5000, totalSamples=500):
        return self._service.exposed_setTime(preTriggerTime, postTriggerTime, totalSamples)

    def setTrigger(self, params):
        return self._service.exposed_setTrigger(params)

    #Block Mode

    def setupBlock(self, trigger_params, downsampling_mode=None):
        return self._service.exposed_setupBlock(trigger_params, downsampling_mode)

    def runBlock(self, segmentIndex=0, downsample_ratio=1, downsample_mode=None):
        return self._service.exposed_runBlock(segmentIndex, downsample_ratio, downsample_mode)

    #Rapid Block Mode
    def setupRapidBlock(self, trigger_params, nSegments=10, nCaptures=10, downsampling_mode=None):
        return self._service.exposed_setupRapidBlock(trigger_params, nSegments, nCaptures, downsampling_mode)

    def runRapidBlock(self, segmentIndex=0, downsample_ratio=1, downsample_mode=None):
        return self._service.exposed_runRapidBlock(segmentIndex, downsample_ratio, downsample_mode)

    #Closing unit stuff
    def closeUnit(self):
        return self._service.exposed_closeUnit()

    def stop(self):
        return self._service.exposed_stop()
