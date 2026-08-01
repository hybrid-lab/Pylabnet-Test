# Turn on RF output on both DIM-3000 devices.
# Does not change frequency, amplitude, or segment settings.

import serial
import time

for port in ['COM4', 'COM5']:
    try:
        ser = serial.Serial(port=port, baudrate=19200, bytesize=8,
                            parity='N', stopbits=1, timeout=1)
        ser.flushInput()
        ser.write(b'*IDN?\n')
        time.sleep(0.15)
        idn = ser.readline().decode().strip()
        ser.write(b'OUT_on\n')
        time.sleep(0.1)
        ser.write(b'AMP:340\n')
        time.sleep(0.1)
        print(f"{port}: {idn} -> RF ON, AMP +34.0 dBm")
        ser.close()
    except Exception as e:
        print(f"{port}: ERROR - {e}")
