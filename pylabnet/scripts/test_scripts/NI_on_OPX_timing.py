import numpy as np
import time

from pylabnet.scripts.data_center.datasets import Dataset, InfiniteRollingLine

# DataTaker parses every INIT_DICT value with float(...),
# so EVERYTHING here must be numeric strings.
INIT_DICT = {
    # OPX clock train
    "clock_rate_hz": {"OPX clock rate (Hz)": "100000"},     # 100 kHz
    "clock_seconds": {"Clock duration (s)": "0.500"},       # 500 ms

    # NI ramp
    "ni_ao_channel_num": {"NI AO channel number (0=ao0, 1=ao1)": "0"},
    "ramp_min_v": {"Ramp min (V)": "0.0"},
    "ramp_max_v": {"Ramp max (V)": "3.0"},

    # OPX digital outs (numeric)
    "opx_clock_do": {"OPX DO for clock": "7"},
    "opx_start_do": {"OPX DO for start trig": "8"},

    # Pulse widths (ns; must be multiple of 4 ns for your OPX delay/play timing)
    "clock_pulse_ns": {"Clock pulse width (ns)": "40"},
    "start_pulse_ns": {"Start pulse width (ns)": "200"},

    # NI trigger controls (numeric-coded)
    "use_start_trigger": {"Use NI start trigger? (1=yes,0=no)": "0"},
    "ni_trigger_target_code": {"NI trigger target (0=ao,1=ai,2=di,3=do)": "0"},
    "ni_trigger_pfi_num": {"NI trigger PFI number (0=PFI0,1=PFI1)": "0"},

    "take_data_rate": {"Update plot every __ seconds": "0.2"},
    "blank1": {"filler": "0"},
    "blank2": {"filler": "0"},
    "blank3": {"filler": "0"},
    "blank4": {"filler": "0"},
}


def define_dataset():
    return "Dataset"


def configure(**kwargs):
    dataset = kwargs["dataset"]
    dataset.NI_client = kwargs["nidaqmx_ni_daq_1"]
    dataset.NI_client2 = kwargs["nidaqmx_ni_daq_3"]
    dataset.OPX_client = kwargs["OPX_OPX"]

    dataset.add_child(
        name="Ramp monitor",
        data_type=InfiniteRollingLine,
        data_length=400,
        new_plot=True
    )
    dataset.graph.hide()


def experiment(**kwargs):
    dataset = kwargs["dataset"]
    thread = kwargs["thread"]

    ni = dataset.NI_client
    opx = dataset.OPX_client

    ni_2 = dataset.NI_client2

    ni_ao = "ao0"
    ni_2_ao = "ao15"
    qop_ip = "192.168.88.253"
    cluster_name = "Cluster_1"
    #OPX timing frequency is 10 kHz
    ramp_min = 0
    ramp_max = 3
    N = 100000
    #ramp = [1] * N
    ramp = np.linspace(ramp_min, ramp_max, N, dtype=np.float64)
    voltages = ramp
    fs = 100000

    while thread.running:

        # ni.build_stack()
        # ni.set_ao_voltage(ao_channel=ni_ao, voltages=ramp, sample_rate=fs) #For future, make a set_timing function that will set the timing of a card to be an external clock

        # ni_2.build_stack()
        # ni_2.set_ao_voltage(ao_channel=ni_2_ao, voltages=ramp, sample_rate=fs)

        opx.build_stack()
        clock_elem = opx.create_new_do_elem(
            do_channel=1,
            length=1000
        )
        opx.set_digital_voltage(
            element=clock_elem
        )
        # with opx.for_("i", 0, N, 1):
        #     opx.set_digital_voltage(
        #         element=clock_elem
        #     )
        #     opx.delay(99000, elements=[clock_elem])

        # ni.execute()
        # ni_2.execute()
        opx.execute()

        time.sleep(float(dataset.get_input_parameter("take_data_rate")))
        time.sleep(10)
