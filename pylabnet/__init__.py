"""

pylabnet

Client-server, python-based laboratory software
"""

__version__ = '1.0.0'
__author__ = 'Lukin SiV Team'

# Load the NI-DAQmx library (nicaiu.dll) before anything imports scipy, pandas
# or PyQt5. With NI-DAQmx 26.x, loading it after those makes every
# DAQmxCreateTask fail with "access violation reading 0x0000000000000000".
# Every pylabnet entry point imports this package first, so doing it here
# covers servers, scripts and DataTaker alike.
try:
    from nidaqmx._lib import lib_importer as _ni_lib_importer
    _ni_lib_importer.windll
except Exception:   # no nidaqmx / no NI driver on this machine: nothing to protect
    pass
