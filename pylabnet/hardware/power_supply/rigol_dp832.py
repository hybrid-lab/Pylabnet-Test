"""Driver for the Rigol DP832 triple-output DC power supply over USB (VISA).

Every current setpoint is capped at MAX_CURRENT_A, a constant in this file and not a
setting, because the supply drives the MOT coils. The cap applies to every channel.
The driver refuses any request above the cap, and it sets the supply's own
over-current protection (OCP) to OCP_A. If the current goes above OCP_A anyway
(e.g. someone raises the setpoint on the front panel), the supply switches that
output off by itself.

SCPI commands are from the Rigol DP800 Programming Guide.
"""

from pyvisa import ResourceManager, VisaIOError

from pylabnet.utils.logging.logger import LogHandler

# Highest current setpoint the driver accepts, in A. Raise it only if the coils and
# wiring are rated for more.
MAX_CURRENT_A = 0.28
# Hardware over-current trip, in A. It sits a little above MAX_CURRENT_A so the trip
# doesn't fire on noise while the supply regulates at the cap.
OCP_A = 0.30

# Rated output of each channel: (V, A)
CHANNEL_RATINGS = {1: (30.0, 3.0), 2: (30.0, 3.0), 3: (5.0, 3.0)}

# USB vendor and product ID of the DP800 series, used to find the supply automatically
RIGOL_DP800_USB_ID = '0X1AB1::0X0E11'


class Driver:

    def __init__(self, device_id, logger=None):
        """Connect to the supply and make every channel safe.

        :param device_id: (str) VISA address, e.g. 'USB0::0x1AB1::0x0E11::DP8C123456789::INSTR',
            or 'auto' to use the only DP800 found on USB
        :param logger: instance of LogClient
        """
        self.log = LogHandler(logger=logger)
        self.rm = ResourceManager()

        if device_id == 'auto':
            device_id = self._find_supply()

        try:
            self.device = self.rm.open_resource(device_id)
        except VisaIOError:
            self.log.error(f'Could not open {device_id}. Is the supply on and plugged in over USB?')
            raise
        self.device.write_termination = '\n'
        self.device.read_termination = '\n'
        self.device.timeout = 5000  # ms
        self.log.info(f'Connected to {self.device.query("*IDN?").strip()}')

        for channel in CHANNEL_RATINGS:
            self._make_safe(channel)

    def _find_supply(self):
        """Return the VISA address of the only DP800 on USB."""
        found = [r for r in self.rm.list_resources() if RIGOL_DP800_USB_ID in r.upper()]
        if len(found) != 1:
            msg = f'Expected one Rigol DP800 on USB, found {found}. Put its VISA address in the device config.'
            self.log.error(msg)
            raise RuntimeError(msg)
        return found[0]

    def _make_safe(self, channel):
        """Arm OCP; if the setpoint is above the cap, switch the output off and lower it."""
        self.device.write(f':OUTP:OCP:VAL CH{channel},{OCP_A}')
        self.device.write(f':OUTP:OCP CH{channel},ON')
        setpoint = self.get_current_setpoint(channel)
        if setpoint > MAX_CURRENT_A:
            self.output_off(channel)
            self.device.write(f':SOUR{channel}:CURR {MAX_CURRENT_A}')
            self.log.warn(f'CH{channel} was set to {setpoint} A, above the {MAX_CURRENT_A} A cap: '
                          f'switched it off and lowered the setpoint to {MAX_CURRENT_A} A')

    def _check_channel(self, channel):
        channel = int(channel)
        if channel not in CHANNEL_RATINGS:
            raise ValueError(f'Channel must be 1, 2 or 3, not {channel}')
        return channel

    def set_current(self, channel, current):
        """Set the current setpoint of a channel, in A. Refuses anything above MAX_CURRENT_A."""
        channel = self._check_channel(channel)
        current = float(current)
        if not 0 <= current <= MAX_CURRENT_A:
            msg = f'Refusing CH{channel} current {current} A: must be between 0 and {MAX_CURRENT_A} A'
            self.log.error(msg)
            raise ValueError(msg)
        self.device.write(f':SOUR{channel}:CURR {current}')
        # Read back: the supply silently ignores commands it doesn't understand
        readback = self.get_current_setpoint(channel)
        if abs(readback - current) > 1e-3:
            raise RuntimeError(f'CH{channel} current setpoint reads {readback} A after setting {current} A')
        self.log.info(f'CH{channel} current setpoint {current} A')

    def set_voltage(self, channel, voltage):
        """Set the voltage setpoint of a channel, in V. With a coil load this is the compliance
        voltage: the supply runs in constant-current mode as long as it is high enough."""
        channel = self._check_channel(channel)
        voltage = float(voltage)
        v_max = CHANNEL_RATINGS[channel][0]
        if not 0 <= voltage <= v_max:
            raise ValueError(f'CH{channel} voltage must be between 0 and {v_max} V, not {voltage}')
        self.device.write(f':SOUR{channel}:VOLT {voltage}')
        self.log.info(f'CH{channel} voltage setpoint {voltage} V')

    def get_current_setpoint(self, channel):
        """Current setpoint of a channel, in A (not the measured current)."""
        channel = self._check_channel(channel)
        return float(self.device.query(f':SOUR{channel}:CURR?'))

    def get_voltage_setpoint(self, channel):
        """Voltage setpoint of a channel, in V (not the measured voltage)."""
        channel = self._check_channel(channel)
        return float(self.device.query(f':SOUR{channel}:VOLT?'))

    def output_on(self, channel):
        """Switch a channel's output on, after checking its setpoint is within the cap."""
        channel = self._check_channel(channel)
        setpoint = self.get_current_setpoint(channel)
        if setpoint > MAX_CURRENT_A:
            msg = f'Refusing to switch CH{channel} on: setpoint {setpoint} A is above the {MAX_CURRENT_A} A cap'
            self.log.error(msg)
            raise RuntimeError(msg)
        # Re-arm OCP in case it was changed or tripped from the front panel
        self.device.write(f':OUTP:OCP:CLEAR CH{channel}')
        self.device.write(f':OUTP:OCP:VAL CH{channel},{OCP_A}')
        self.device.write(f':OUTP:OCP CH{channel},ON')
        self.device.write(f':OUTP CH{channel},ON')
        self.log.info(f'CH{channel} output on ({setpoint} A setpoint)')

    def output_off(self, channel):
        """Switch a channel's output off."""
        channel = self._check_channel(channel)
        self.device.write(f':OUTP CH{channel},OFF')
        self.log.info(f'CH{channel} output off')

    def is_output_on(self, channel):
        channel = self._check_channel(channel)
        return self.device.query(f':OUTP? CH{channel}').strip().upper() in ('ON', '1')

    def is_ocp_tripped(self, channel):
        """True if the over-current protection has switched the channel off."""
        channel = self._check_channel(channel)
        return self.device.query(f':OUTP:OCP:QUES? CH{channel}').strip().upper() in ('YES', '1')

    def measure(self, channel):
        """Measured (voltage V, current A) at a channel's output."""
        channel = self._check_channel(channel)
        voltage = float(self.device.query(f':MEAS:VOLT? CH{channel}'))
        current = float(self.device.query(f':MEAS:CURR? CH{channel}'))
        return voltage, current

    def close(self):
        self.device.close()
