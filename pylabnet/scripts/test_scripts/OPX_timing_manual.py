from pylabnet.hardware.quantum_machines.OPXdriverConfig import config  # your OPX config
from qm.qua import program, infinite_loop_, play, wait
from nidaqmx.system import System
from nidaqmx.constants import AcquisitionType
import nidaqmx
import numpy as np
import qm

from qm import SimulationConfig
from qm.qua import infinite_loop_, play, program, wait
from qm import LoopbackInterface
from qm import QuantumMachinesManager

from pylabnet.scripts.data_center.datasets import Dataset, InfiniteRollingLine
from pylabnet.hardware.ni_daqs.nidaqmx_card import Driver
from pylabnet.hardware.quantum_machines.OPXdriverConfig import config

# ni = Driver(device_name="PXI1Slot2_1")
# ni_2 = Driver(device_name="PXI1Slot6_1")

# ni_ao = "ao0"
# ni_2_ao = "ao15"
qop_ip = "192.168.88.253"
cluster_name = "Cluster_1"
# #OPX timing frequency is 10 kHz
# ramp_min = 0
# ramp_max = 3
# N = 10000
# #ramp = [1] * N
# ramp = np.linspace(ramp_min, ramp_max, N, dtype=np.float64)
# voltages = ramp
# fs = 100000


# ni.build_stack()
# ni.set_trigger("ao","PFI0")
# ni.set_ao_voltage(ao_channel=ni_ao, voltages=ramp, sample_rate=fs) #For future, make a set_timing function that will set the timing of a card to be an external clock
# ni.execute()

# ni_2.build_stack()
# ni_2.set_trigger("ao", "PXI_Trig0")
# ni_2.set_ao_voltage(ao_channel=ni_2_ao, voltages=ramp, sample_rate=fs)
# ni_2.execute()

# with program() as pulse_train:
#     with infinite_loop_():
#         play("ON", "generic_di_elem_ch1")
#         wait(22500)

# qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
# qm = qmm.open_qm(config)
# job = qm.execute(pulse_train)


# -------------------------
# OPX SETTINGS
# -------------------------
qop_ip = "192.168.88.253"
cluster_name = "Cluster_1"

# IMPORTANT: this must be an OUTPUT element in your OPX config that drives a real pin
# that is physically wired to NI card2 PFI0.
OPX_CLK_ELEM = "generic_do_elem_ch1"  # <-- CHANGE THIS to your real output element name

# The pulse you play must exist in config under pulses/waveforms/etc.
# It should be a SHORT pulse (or a constant level pulse) used to make a clock.
OPX_CLK_PULSE = "ON"  # <-- must exist in your config for OPX_CLK_ELEM

# 10 kHz clock: period = 100 us.
# In OPX, wait/play durations are in machine clock cycles (typically 4 ns).
# 100 us / 4 ns = 25,000 cycles.
OPX_CYCLES_PER_PERIOD = 25_000

# If your "ON" pulse length is short (e.g., a few cycles), you can just wait the remainder.
# If you want ~50% duty, define a pulse length ~12,500 cycles in config and wait 12,500 here.


# -------------------------
# NI SETTINGS
# -------------------------
MASTER_DEV = "PXI1Slot2_1"  # card 2
SLAVE_DEV = "PXI1Slot6_1"   # card 6
MASTER_AO_CH = "ao0"
SLAVE_AO_CH = "ao15"

# OPX clock comes into master PFI0
OPX_CLK_IN = f"/{MASTER_DEV}/PFI0"

# We route that clock onto the backplane PXI_Trig0
PXI_TRIG0_MASTER = f"/{MASTER_DEV}/PXI_Trig0"
PXI_TRIG0_SLAVE = f"/{SLAVE_DEV}/PXI_Trig0"

FS = 100_000  # expected sample clock rate (must match OPX clock frequency, e.g. 100 kHz if that's what you're outputting)
N = 10_000

v_min, v_max = 0.0, 3.0
wave_master = np.linspace(v_min, v_max, N, dtype=np.float64)
wave_slave = np.linspace(v_max, v_min, N, dtype=np.float64)


def configure_ao_task(dev: str, ao_ch: str, data: np.ndarray) -> nidaqmx.Task:
    t = nidaqmx.Task(new_task_name=f"AO_{dev}_{ao_ch}")
    t.ao_channels.add_ao_voltage_chan(f"{dev}/{ao_ch}")
    t.timing.cfg_samp_clk_timing(
        rate=float(FS),
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=int(len(data)),
    )
    t.write(data.tolist(), auto_start=False)
    return t


def build_opx_clock_program():
    """
    Infinite clock: repeatedly play a pulse on OPX_CLK_ELEM then wait to set repetition rate.
    NOTE: The actual high-time is determined by the pulse length of OPX_CLK_PULSE in your OPX config.
    """
    with program() as pulse_train:
        with infinite_loop_():
            play("ON", "generic_di_elem_ch1")
            wait(22500)

    qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
    qm = qmm.open_qm(config)
    job = qm.execute(pulse_train)


def main():

    # 1) Prepare NI routing: explicitly connect master PFI0 -> master PXI_Trig0
    sys = System.local()
    sys.connect_terms(OPX_CLK_IN, PXI_TRIG0_MASTER)

    # 2) Create NI tasks
    master = configure_ao_task(MASTER_DEV, MASTER_AO_CH, wave_master)
    slave = configure_ao_task(SLAVE_DEV, SLAVE_AO_CH, wave_slave)

    # 3) Configure sample clock sources
    # Master uses OPX clock directly on PFI0
    master.timing.cfg_samp_clk_timing(
        rate=float(FS),
        source=OPX_CLK_IN,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=int(N),
    )

    # Slave uses backplane PXI_Trig0 (which we routed from master PFI0)
    slave.timing.cfg_samp_clk_timing(
        rate=float(FS),
        source=PXI_TRIG0_SLAVE,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=int(N),
    )

    # 4) Build OPX program (but DO NOT start it yet)
    build_opx_clock_program()

    try:
        # 5) Arm NI first (so it is already waiting for clocks)
        slave.start()
        master.start()

        # 6) Now start OPX clock (this should immediately start feeding PFI0)

        # 7) Wait for NI to finish
        slave.wait_until_done(timeout=30.0)
        master.wait_until_done(timeout=30.0)

        print("SUCCESS: card2+card6 completed using OPX clock routed onto PXI_Trig0.")

    finally:
        # stop NI tasks
        try:
            slave.stop()
        except Exception:
            pass
        try:
            master.stop()
        except Exception:
            pass
        slave.close()
        master.close()

        try:
            sys.disconnect_terms(OPX_CLK_IN, PXI_TRIG0_MASTER)
        except Exception:
            pass


if __name__ == "__main__":
    main()
