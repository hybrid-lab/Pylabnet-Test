import os
import numpy as np
from pysensorpush import PySensorPush #if running first time, run 'pip3 install pysensorpush'

from datetime import datetime
import pytz #may need to run 'pip3 install pytz'
tz_pacific = pytz.timezone('US/Pacific')


class Driver:
    def __init__(self, logger=None):
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

        self.sensorpush = PySensorPush(user, password)

    def get_data(self, num_points=1):
        sensors_config = self.sensorpush.sensors
        samples = self.sensorpush.samples(limit=num_points)
        sensors = []

        for sensor, reading in samples['sensors'].items():
            sensor_data = {}
            sensors.append(sensor_data)

            sensor_data['name'] = sensors_config[sensor]['name']
            sensor_data['datetime'] = []
            sensor_data['humidity'] = []
            sensor_data['temperature'] = []

            for r in reading:
                time = r['observed'] #GMT
                dt = datetime.fromisoformat(time)
                sensor_data['datetime'].append(dt.astimezone(tz_pacific))

                sensor_data['humidity'].append(r['calibrated_humidity'])
                sensor_data['temperature'].append((r['calibrated_temperature'] - 32) * 5 / 9)  #data read in farenheit

        return sensor_data
