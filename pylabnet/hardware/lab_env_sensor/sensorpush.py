import os
import numpy as np
from pysensorpush import PySensorPush #if running first time, run 'pip3 install pysensorpush'

from datetime import datetime
import pytz #may need to run 'pip3 install pytz'

tz_pacific = pytz.timezone('US/Pacific')
date_format = '%Y-%m-%dT%H:%M:%S.%fZ'


class Driver:
    def __init__(self, sensor_name, logger=None):
        """
        Instantiate driver class

        :param sensor_name: Name of specific sensor (type: string)
        :param logger: An instance of a LogClient
        """
        user = os.getenv('SENSORPUSH_USER', 'hybridlab2025@gmail.com')
        password = os.getenv('SENSORPUSH_PASSWORD', 'mmWave40')

        if None in (user, password):
            print('ERROR! Must define env variables SENSORPUSH_USER and SENSORPUSH_PASSWORD')
            raise SystemExit

        self.logger = logger
        self.sensor_id = None
        self.sensorpush = PySensorPush(user, password)
        self.logger.info('sensorpush initiated')
        for s, info in self.sensorpush.sensors.items():
            if info['name'] == sensor_name:
                self.sensor_id = s
                self.logger.info('sensor_name initiated')

    def get_data(self, num_points=1):
        samples = self.sensorpush.samples(limit=num_points)
        data = {}
        data['datetime'] = []
        data['humidity'] = []
        data['temperature'] = []

        for sensor, reading in samples['sensors'].items():
            if sensor == self.sensor_id:
                for r in reading:
                    time = r['observed'] #GMT
                    dt = datetime.strptime(time, date_format)
                    dt = dt.replace(tzinfo=pytz.UTC)
                    data['datetime'].append(dt.astimezone(tz_pacific))

                    data['humidity'].append(r['calibrated_humidity'])
                    data['temperature'].append((r['calibrated_temperature'] - 32) * 5 / 9)  #data read in farenheit

        return data
