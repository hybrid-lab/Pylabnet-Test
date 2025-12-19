import numpy as np
import time
from PyQt5 import QtCore


from pylabnet.scripts.data_center.take_data import ExperimentThread
from pylabnet.scripts.data_center.datasets import SawtoothScan1D, ErrorBarGraph, InfiniteRollingLine, Dataset, SawtoothScan1D_array_update

from pylabnet.gui.pyqt.external_gui import Window, ParameterPopup

from scipy.stats import poisson
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import pylabnet.hardware.awg.zi_hdawg as zi_hdawg
from pylabnet.launchers.siv_py_functions import upload_sequence, load_config


# config_dict = load_config('datataker_test')
# dataset_folder = config_dict['dataset_path']

BIN_NUMS = 100


# # contrast check
# UCONT_TRSH = -1
# LCONT_TRSH = 1.4
# WAVELENGTH_RANGE = [406.635, 406.636] # If laser optimizer wants to step outside this range, reset laser to middle of range

# # CNOT_check
# CNOT_F_THRES = 0.9
# CNOT_CHECK_T = 10


pulse_var_dict_G12 = load_config('pulse_vars')
dio_dict_G12 = load_config('dio_assignment_global')


IQ_AMP_I = pulse_var_dict_G12["IQ_AMP_I"]
IQ_AMP_Q = pulse_var_dict_G12["IQ_AMP_Q"]
IQ_PHASE1 = pulse_var_dict_G12["IQ_PHASE1"]
IQ_PHASE2 = pulse_var_dict_G12["IQ_PHASE2"]

Selected_Osc_1 = pulse_var_dict_G12["Selected_Osc_1"]
Selected_Osc_2 = pulse_var_dict_G12["Selected_Osc_2"]
Selected_Osc_3 = pulse_var_dict_G12["Selected_Osc_3"]
XYPhaseDifference = pulse_var_dict_G12["XYPhaseDifference"]
CNOT_check_repeat = 500


def define_dataset():
    return 'Dataset'


def rebin_histogram(dataset, prev_dataset):
    """ Rebins a histogram """

    raw_data = dataset.data[:2 * dataset.repeat_per_time]
    prev_dataset.data, _ = np.histogram(raw_data, bins=BIN_NUMS, range=(0, BIN_NUMS - 1))


def double_poisson(x, a1, mu1, a2, mu2):
    return a1 * poisson.pmf(x, mu1) + a2 * poisson.pmf(x, mu2)


def fit_histogram(dataset, prev_dataset, peakindex):
    """ fits histogram """

    raw_data = dataset.data

    # bounds=([0, 0.1, 0, np.argmax(raw_data)],
    #     [400, np.argmax(raw_data)+5, 400, 40])

    bounds = ([0, 0.1, 0, np.argmax(raw_data)],
              [400, np.argmax(raw_data) + 5, 400, 400])

    try:
        fit = curve_fit(double_poisson, np.arange(BIN_NUMS), raw_data, p0=[100, 5, 100, 25], bounds=bounds)
        if fit[0][1] < fit[0][3]:
            peaks = [fit[0][1], fit[0][3], min((fit[0][3] / fit[0][1], 30))]
        else:
            peaks = [fit[0][3], fit[0][1], min((fit[0][1] / fit[0][3], 30))]

        if fit[0][2] < 0.1 or fit[0][3] == 40:
            peaks = [fit[0][1], fit[0][1], 1]
    except RuntimeError:
        peaks = [1, 1, 1]

    prev_dataset.set_data(peaks[peakindex])


def upper_line(dataset, prev_dataset):
    prev_dataset.set_data(dataset.line1)


def lower_line(dataset, prev_dataset):
    prev_dataset.set_data(dataset.line2)


def upper_rabi(dataset, prev_dataset):
    prev_dataset.set_data(dataset.RABI_amp1)


def lower_rabi(dataset, prev_dataset):
    prev_dataset.set_data(dataset.RABI_amp2)


def init_fidel(dataset, prev_dataset):
    prev_dataset.set_data(dataset.init_fidelity)


def pi_fidel(dataset, prev_dataset):
    prev_dataset.set_data(dataset.pi_fidelity)


def reading(dataset, prev_dataset):
    rawdata = dataset.data
    for i in range(dataset.pts):
        data_repeats = rawdata[(dataset.repeat_per_time * i): (dataset.repeat_per_time * (i + 1))]
        prev_dataset.set_data(np.mean(data_repeats))


def photons_low(dataset, prev_dataset):
    data_2Dscan = np.reshape(dataset.data, (2 * dataset.repeat_per_time, dataset.n_hist_bins))
    data_2D_scan_low = data_2Dscan[0:2 * dataset.repeat_per_time:2, :]
    dataset.log.info("data_2D_scan_low.shape" + str(data_2D_scan_low.shape))
    data_mean = np.mean(data_2D_scan_low, axis=0)
    dataset.log.info("data_mean.shape" + str(data_mean.shape))

    prev_dataset.set_data(data_mean)


def photons_high(dataset, prev_dataset):
    data_2Dscan = np.reshape(dataset.data, (2 * dataset.repeat_per_time, dataset.n_hist_bins))
    data_2D_scan_high = data_2Dscan[1:2 * dataset.repeat_per_time:2, :]
    dataset.log.info("data_2D_scan_high.shape" + str(data_2D_scan_high.shape))
    data_mean = np.mean(data_2D_scan_high, axis=0)
    dataset.log.info("data_mean.shape" + str(data_mean.shape))

    prev_dataset.set_data(data_mean)

    # raw_data = np.zeros(dataset.n_hist_bins)

    # for ii in range(dataset.repeat_per_time):
    #     raw_data += np.array(dataset.data[(dataset.n_hist_bins*ii):(dataset.n_hist_bins*(ii+1))] )# since data is flattened, separate histograms for each experiment

    # raw_data /= dataset.repeat_per_time

    # for idx, pt in enumerate(raw_data):
    #      prev_dataset.set_data(pt)


INIT_DICT = {
    "cnot1_freq": {"CNOT1 frequency (MHz) (first guess):": pulse_var_dict_G12["cnot_1_freq"]},
    "cnot2_freq": {"CNOT2 frequency (MHz) (first guess):": pulse_var_dict_G12["cnot_2_freq"]},
    "thres": {"treshold (counts):": pulse_var_dict_G12["readout_params"]["low_thres"]},
    "rpt_per_time": {"Repeat per time:": '100'},
    "cnot1_amp": {"CNOT1 amplitude (first guess):": pulse_var_dict_G12["cnot_pis"]["cnot1_pi_amp"]},
    "cnot2_amp": {"CNOT2 amplitude (first guess):": pulse_var_dict_G12["cnot_pis"]["cnot2_pi_amp"]},
    "h_cnot1_amp": {"Half CNOT1 amplitude (first guess):": '0.033'},
    "h_cnot2_amp": {"Half CNOT2 amplitude (first guess):": '0.033'},
    "cnot_len": {"CNOT pulse length:": pulse_var_dict_G12["cnot_pis"]["len"]},
    "rf_len": {"RF pulse length:": pulse_var_dict_G12["si_29_params"]["RF1_pulselength_0.8V"]},
    "upper_thres": {"Upper Threshold (counts)": pulse_var_dict_G12["readout_params"]["high_thres"]},
    "RF_freq": {"RF frequency (MHz)": pulse_var_dict_G12["si_29_params"]["RF1_freq"]},
    "tuneup_cnot": {"tuneup_cnot, 1:cnot1; 2:cnot2": "1"},
    "eomPeriods": {"EOM number of periods (phase read time, Sa):": '5000'},
    "eomAmp": {"EOM amplitude (in 5V settomg):": '0.25'},
    "eomFreq": {"EOM frequency:": '320'}

}


def push_sequence(dataset):

    f1 = dataset.get_input_parameter("cnot1_freq")
    f2 = dataset.get_input_parameter("cnot2_freq")
    eomFreq = dataset.get_input_parameter("eomFreq")
    eomAmp = dataset.get_input_parameter("eomAmp")
    eomPeriods = dataset.get_input_parameter("eomPeriods")
    trsh = dataset.get_input_parameter("thres")
    upper_trsh = dataset.get_input_parameter("upper_thres")
    rp = int(dataset.get_input_parameter("rpt_per_time"))
    amp1 = float(dataset.get_input_parameter("cnot1_amp"))
    amp2 = float(dataset.get_input_parameter("cnot2_amp"))
    mw_len = int(dataset.get_input_parameter("cnot_len"))
    RF_len = int(dataset.get_input_parameter("rf_len"))
    upper_trsh = int(dataset.get_input_parameter("upper_thres"))
    RF = float(dataset.get_input_parameter("RF_freq"))
    # Save values into metadata when saved
    # Update this when adding fields!
    dataset.log.update_metadata(pushed_dict={
        "f1": f1,
        "f2": f2,
        "amp1": amp1,
        "amp2": amp2,
        "trsh": trsh,
        "mw_len": mw_len,
        "rp": rp,
        "RF freq": RF,
        "RF_len": RF_len,
        "upper_trsh": upper_trsh
    })

    dataset.f1 = f1
    dataset.f2 = f2

    dataset.line1 = f1
    dataset.line2 = f2

    dataset.RABI_amp1 = amp1
    dataset.RABI_amp2 = amp2

    dataset.trsh = trsh
    dataset.upper_trsh = upper_trsh

    dataset.repeat_per_time = rp

    dataset.HDAWG.setd('oscs/4/freq', f1 * 1e6)
    dataset.HDAWG.setd('oscs/5/freq', f2 * 1e6)
    dataset.HDAWG.setd('oscs/7/freq', eomFreq * 1e6)
    #dataset.HDAWG.setd('oscs/3/freq', RF*1e6)

    #amps = np.linspace(0.5,1.0,21)
    #dataset.HDAWG.seti('awgs/0/userregs/2', int(np.argmin((amps-amp1)**2)))
    #dataset.HDAWG.seti('awgs/0/userregs/3', int(np.argmin((amps-amp2)**2)))

    plugin = f"""
    const side_read_AOM = 1 << {dio_dict_G12["side_read_AOM_TTL"]};
    const phase_read_aom = 1 << {dio_dict_G12["phase_read_AOM_TTL"]};
    const SNSPDS = (1 << 27) | (1 << 24);

    const OFF = SNSPDS;
    const G12TT = OFF | (1 << {dio_dict_G12["gate"]});
    const phase_read_on =  side_read_AOM | phase_read_aom | SNSPDS;
    const HIPO_bloCK = 1 << {dio_dict_G12["High_Power_Switch"]} | OFF;
    const GREEN_G12 = OFF | (1 << {dio_dict_G12["green"]});
    const OLD_TISA_READ = OFF | (1 << {dio_dict_G12["toptica"]}) | side_read_AOM;
    const TISATT = OLD_TISA_READ | G12TT;


    const DIO_delay = 30;

    const t_green = 50000;
    const t_TiSa = {pulse_var_dict_G12["readout_params"]["readout_len"]};


    const repeat_per_time = {rp};

    const trsh = {trsh};
    const hi_trsh = {upper_trsh};

    wave marker_w = join(marker({mw_len+24}, 1), marker(6, 1));
    wave marker_w_MF = join(marker(30, 0), marker({mw_len+26}, 1)); // Shift the marker to start later due to cable length

    wave pi1_MF = join( {amp1} * ones({mw_len//4}) , zeros(2));
    wave pi2_MF = join( {amp2} * ones({mw_len//4}) , zeros(2) );
    wave w_zeros = zeros({mw_len//4} + 2);
    wave pi1_x = interleave(pi1_MF, w_zeros, w_zeros, w_zeros) + marker_w_MF;
    wave pi1_y = interleave(w_zeros, pi1_MF, w_zeros, w_zeros) + marker_w_MF;
    wave pi2_x = interleave(w_zeros, w_zeros, pi2_MF, w_zeros) + marker_w_MF;
    wave pi2_y = interleave(w_zeros, w_zeros, w_zeros, pi2_MF) + marker_w_MF;
    wave pi_z = interleave(w_zeros, w_zeros, w_zeros, w_zeros);

    wave RF_pulse = interleave(ones({RF_len//4}), zeros({RF_len//4}), zeros({RF_len//4}), zeros({RF_len//4}));
    wave RF_zero = 0*RF_pulse;


    wave x_unit = join(ones({mw_len//4}), zeros(2));
    wave pi1_x_unit = interleave(x_unit, 0*x_unit, 0*x_unit, 0*x_unit) + marker_w_MF;
    wave pi2_x_unit = interleave(0*x_unit, 0*x_unit, x_unit, 0*x_unit) + marker_w_MF;
    const IQ_AMP_I = {IQ_AMP_I};
    const IQ_AMP_Q = {IQ_AMP_Q};
    const RF_len = {RF_len};

    ////////////////////////////////////////////////////////////////////////////////
    // EOM PULSE PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////

    const repeat_times = {rp};
    const amplitude = {eomAmp};
    const phaseOffset = 0;
    const CNOT_check_repeat = {CNOT_check_repeat};
    const phr_samples = {eomPeriods//4};

    const tt_buffer = 250;
    const gate_length = 10;


    // AOM opens, EOM creates sidebands, AOM closes
    wave SINE = join(ones(phr_samples),
                     zeros(tt_buffer + gate_length));
    wave SINE_MF = interleave(SINE, 0*SINE, 0*SINE, 0*SINE);
    wave marker_timetagger = join( marker(tt_buffer*4, 0), marker(gate_length*4, 1), marker(phr_samples*4, 0));

    """

    constant = """

    const ionized_trsh = 30; // after how many repeats do we send the green laser?

    wave marker_w_sweep = join(marker(2640,1), marker(16,0));

    void CNOT1_MFx () {
        playWave(1, pi_z, 2,pi_z, 3, IQ_AMP_I * pi1_x, 4, IQ_AMP_Q * pi1_x);
        waitWave();
    }

    void CNOT2_MFx () {
        playWave(1, pi_z, 2, pi_z, 3, IQ_AMP_I * pi2_x, 4, IQ_AMP_Q * pi2_x);
        waitWave();
    }

    void RF_MF () {
        playWave(1, RF_zero, 2, RF_pulse, 3, RF_zero, 4, RF_zero);
        waitWave();
    }


    void E_INIT_L(){
        var el_perro_verde = 0;

        setTrigger(0b0001);
        setDIO(OLD_TISA_READ);
        wait(t_TiSa);
        setDIO(OFF);
        setTrigger(0b0000);
        wait(1000);

        while (getCnt(0) > trsh) {

            if (el_perro_verde > ionized_trsh) {
                setDIO(GREEN_G12);
                wait(t_green);
                setDIO(OFF);
                wait(t_green);

                el_perro_verde = 0;
            }

            wait(100);

            CNOT1_MFx();
            CNOT2_MFx();

            wait(DIO_delay);

            setTrigger(0b0001);
            setDIO(OLD_TISA_READ);
            wait(t_TiSa);
            setDIO(OFF);
            setTrigger(0b0000);
            wait(1000);

            el_perro_verde += 1;
        }
        return;
    }


    void E_INIT_H(){
        var el_perro_verde = 0;

        setTrigger(0b0001);
        setDIO(OLD_TISA_READ);
        wait(t_TiSa);
        setDIO(OFF);
        setTrigger(0b0000);
        wait(1000);

        while (getCnt(0) < hi_trsh) {

            if (el_perro_verde > ionized_trsh) {
                setDIO(GREEN_G12);
                wait(t_green);
                setDIO(OFF);
                wait(t_green);

                el_perro_verde = 0;
            }

            wait(100);

            CNOT1_MFx();
            CNOT2_MFx();

            wait(DIO_delay);

            setTrigger(0b0001);
            setDIO(OLD_TISA_READ);
            wait(t_TiSa);
            setDIO(OFF);
            setTrigger(0b0000);
            wait(1000);

            el_perro_verde += 1;
        }
        return;
    }


    void SWAP_ON_SI29 () {
        wait(100);
        CNOT1_MFx();
        RF_MF();
        CNOT1_MFx();
        wait(100);
    }

    void E_N_INIT_extra(){
        var g12_not_initialized = 1;

        while (g12_not_initialized) {
            E_INIT_L();
            SWAP_ON_SI29();
            E_INIT_H();
            CNOT1_MFx();
            wait(100);

            setTrigger(0b0001);
            setDIO(OLD_TISA_READ);
            wait(t_TiSa);
            setDIO(OFF);
            setTrigger(0b0000);
            wait(1000);

            if (getCnt(0) <= trsh) {
                g12_not_initialized = 0; // SUCCESS! e and Si29 both initialized
                wait(1000);
            }
        }

    }

    void E_N_INIT(){
        E_INIT_L();
        SWAP_ON_SI29();
        E_INIT_L();

    }



    while(1) {
        var run_mode = getUserReg(0);
        var active = getUserReg(1);

        wait(1000);

        if (active) {
            switch (run_mode) {
            case 4: // Phase read

                repeat(repeat_times){
                    // Initialize low

                    E_INIT_L();
                    wait(100);
                    waitWave();

                    wait(100);

                    // Phase Read Pulse
                    resetOscPhase();
                    wait(100);
                    setDIO(phase_read_on);
                    wait(100);

                    playWave(1, amplitude*SINE_MF + marker_timetagger, 2, 0*SINE_MF, 3, 0*SINE_MF, 4, 0*SINE_MF);
                    waitWave();

                    wait(100);
                    setDIO(OFF);
                    wait(1000);

                    //Read
                    setDIO(TISATT);
                    wait(t_TiSa);
                    setDIO(OFF);
                    wait(200);

                    // Initialize high
                    E_INIT_L();
                    wait(100);

                    CNOT1_MFx(); //1st pi pulse
                    CNOT2_MFx(); //1st pi pulse
                    waitWave();

                    wait(100);


                    // Phase Read Pulse
                    resetOscPhase();
                    wait(100);
                    setDIO(phase_read_on);
                    wait(100);

                    playWave(1, amplitude*SINE_MF + marker_timetagger, 2, 0*SINE_MF, 3, 0*SINE_MF, 4, 0*SINE_MF);
                    waitWave();

                    wait(100);
                    setDIO(OFF);
                    wait(100);

                    //Read
                    setDIO(TISATT);
                    wait(t_TiSa);
                    setDIO(OFF);
                    wait(200);
                }
            }
            setUserReg(0,0);
            setUserReg(1,0);
        }
    }
    """

    dataset.awgModule = dataset.HDAWG.daq.awgModule()
    dataset.awgModule.set("device", dataset.HDAWG.device_id)
    dataset.awgModule.set("index", 0) #Hardcoded sequencer idndex
    dataset.awgModule.execute()

    upload_sequence(dataset, plugin + constant)

    text_file = open("C:\\Users\\Roger\\Dropbox (Lukin SiV)\\SiV Quick Sharing\\G12 experiments\\Datataker Experiments\\experiment_scripts\\hdawg_code.txt", "w")
    n = text_file.write(plugin + constant)
    text_file.close()

    tt_binsize = 0.1 # ns
    tt_meas_len = 0.2 # us
    tt_nbins = int(1000 * tt_meas_len / tt_binsize)

    dataset.n_hist_bins = tt_nbins

    # Get histogram of arrival times
    delay = 0.18 * 1e6 # start at ~0.14 us (in ps)
    dataset.ctr.create_delayed_channel('START', 6, delay)

    dataset.ctr.start_histogram(
        name='hist_5',
        start_ch='START',
        click_ch=8, # clicks come in on channel 8 bon tt in g12
        next_ch=6,
        binwidth=int(tt_binsize * 1000), # width of bins in ps
        n_bins=tt_nbins, # tt_meas_len is in us, tt_binsize in ns
        n_histograms=rp * 2
    )

    dataset.add_child(
        name='phase read low',
        data_type=SawtoothScan1D_array_update,
        min=0,
        max=tt_meas_len,
        pts=tt_nbins,
        mapping=photons_low
    )

    dataset.add_child(
        name='phase read high',
        data_type=SawtoothScan1D_array_update,
        min=0,
        max=tt_meas_len,
        pts=tt_nbins,
        mapping=photons_high
    )

    # dataset.add_child(
    #     name='phase read avg',
    #     data_type=Dataset,
    #     x=np.linspace(0, tt_meas_len, tt_nbins),
    #     pts=tt_nbins
    # )

    dataset.avg = np.zeros(tt_nbins)
    dataset.reps = 0

    # dataset.add_child(
    #     name='phase read ave',
    #     data_type= Dataset,
    #     min=0,
    #     max=tt_meas_len,
    #     pts=int(1000 * tt_meas_len / tt_binsize),
    #     mapping = photons
    # )

    # Ctr 5 gated
    dataset.ctr.start_gated_counter(
        name='gated_8',
        click_ch=8,
        gate_ch=3,
        bins=rp * 2
    )

    # # Used when we have 2 TT channels (e.g. TDI config)
    # # Ctr 8 gated
    # dataset.ctr.start_gated_counter(
    #     name='gated_8',
    #     click_ch=8,
    #     gate_ch=3,
    #     bins=10000
    # )


def configure(**kwargs):

    dataset: Dataset = kwargs['dataset']

    dataset.prev_RABI1_vec = None
    dataset.prev_RABI2_vec = None

    dataset.wm = kwargs['wlm_monitor_two_topticas']

    ## the time tagger
    dataset.ctr = kwargs["si_tt_standard_ctr"]
    #dataset.HDAWG = kwargs['zi_hdawg_b16_hdawg']

    dataset.HDAWG = zi_hdawg.Driver('dev8227', None)

    # set stepping frequency
    dataset.HDAWG.setd('oscs/2/freq', 500000)

    # set DIO in AWG-controlled  mode
    dataset.HDAWG.seti('dios/0/mode', 1)

    # set oscillators in non-AWG-controlled mode
    dataset.HDAWG.seti('system/awg/oscillatorcontrol', 0)

    # set right waveform modulation mode
    dataset.HDAWG.seti('awgs/0/outputs/0/modulation/mode', 5)
    dataset.HDAWG.seti('awgs/0/outputs/1/modulation/mode', 5)
    dataset.HDAWG.seti('awgs/1/outputs/0/modulation/mode', 5)
    dataset.HDAWG.seti('awgs/1/outputs/1/modulation/mode', 5)

    # set IQ params (amps)
    dataset.HDAWG.setd('awgs/1/outputs/0/gains/0', IQ_AMP_I)
    dataset.HDAWG.setd('awgs/1/outputs/0/gains/1', IQ_AMP_Q)
    dataset.HDAWG.setd('awgs/1/outputs/1/gains/0', IQ_AMP_I)
    dataset.HDAWG.setd('awgs/1/outputs/1/gains/1', IQ_AMP_Q)

    # Set MF settings for Output 3 - oscillators
    dataset.HDAWG.setd('awgs/1/outputs/0/modulation/carriers/0/oscselect', Selected_Osc_1)
    dataset.HDAWG.setd('awgs/1/outputs/0/modulation/carriers/1/oscselect', Selected_Osc_1)
    dataset.HDAWG.setd('awgs/1/outputs/0/modulation/carriers/2/oscselect', Selected_Osc_2)
    dataset.HDAWG.setd('awgs/1/outputs/0/modulation/carriers/3/oscselect', Selected_Osc_2)

    # Set MF settings for Output 1 - oscillators. Phase read
    dataset.HDAWG.setd('awgs/0/outputs/0/modulation/carriers/0/oscselect', 7)
    dataset.HDAWG.setd('awgs/0/outputs/0/modulation/carriers/0/phaseshift', 0)

    # Phase
    dataset.HDAWG.setd('awgs/1/outputs/0/modulation/carriers/0/phaseshift', IQ_PHASE1)
    dataset.HDAWG.setd('awgs/1/outputs/0/modulation/carriers/1/phaseshift', IQ_PHASE1 + XYPhaseDifference)
    dataset.HDAWG.setd('awgs/1/outputs/0/modulation/carriers/2/phaseshift', IQ_PHASE1)
    dataset.HDAWG.setd('awgs/1/outputs/0/modulation/carriers/3/phaseshift', IQ_PHASE1 + XYPhaseDifference)
    # Set MF settings for Output 4 - oscillators
    dataset.HDAWG.setd('awgs/1/outputs/1/modulation/carriers/0/oscselect', Selected_Osc_1)
    dataset.HDAWG.setd('awgs/1/outputs/1/modulation/carriers/1/oscselect', Selected_Osc_1)
    dataset.HDAWG.setd('awgs/1/outputs/1/modulation/carriers/2/oscselect', Selected_Osc_2)
    dataset.HDAWG.setd('awgs/1/outputs/1/modulation/carriers/3/oscselect', Selected_Osc_2)
    # Phase
    dataset.HDAWG.setd('awgs/1/outputs/1/modulation/carriers/0/phaseshift', IQ_PHASE2)
    dataset.HDAWG.setd('awgs/1/outputs/1/modulation/carriers/1/phaseshift', IQ_PHASE2 + XYPhaseDifference)
    dataset.HDAWG.setd('awgs/1/outputs/1/modulation/carriers/2/phaseshift', IQ_PHASE2)
    dataset.HDAWG.setd('awgs/1/outputs/1/modulation/carriers/3/phaseshift', IQ_PHASE2 + XYPhaseDifference)

    # RF oscillator
    dataset.HDAWG.setd('awgs/0/outputs/1/modulation/carriers/0/oscselect', Selected_Osc_3)
    # RF Phase
    dataset.HDAWG.setd('awgs/0/outputs/1/modulation/carriers/0/phaseshift', 0)

    # Logic counter settings
    dataset.HDAWG.seti('cnts/0/enable', 1)
    dataset.HDAWG.seti('cnts/0/mode', 3)
    dataset.HDAWG.seti('cnts/0/operation', 0)
    dataset.HDAWG.seti('cnts/0/inputselect', 35)
    dataset.HDAWG.seti('cnts/0/gateselect', 32)

    dataset.HDAWG.seti('cnts/2/enable', 1)
    dataset.HDAWG.seti('cnts/2/mode', 3)
    dataset.HDAWG.seti('cnts/2/operation', 0)
    dataset.HDAWG.seti('cnts/2/inputselect', 35)
    dataset.HDAWG.seti('cnts/2/gateselect', 32)

    dataset.HDAWG.seti('cnts/1/enable', 1)
    dataset.HDAWG.seti('cnts/1/mode', 3)
    dataset.HDAWG.seti('cnts/1/operation', 0)
    dataset.HDAWG.seti('cnts/1/inputselect', 33)
    dataset.HDAWG.seti('cnts/1/gateselect', 32)

    dataset.HDAWG.seti('cnts/3/enable', 1)
    dataset.HDAWG.seti('cnts/3/mode', 3)
    dataset.HDAWG.seti('cnts/3/operation', 0)
    dataset.HDAWG.seti('cnts/3/inputselect', 33)
    dataset.HDAWG.seti('cnts/3/gateselect', 32)

    dataset.add_child(
        name='Histogram',
        data_type=ErrorBarGraph,
        window='histogram_fits',
        window_title='Histogram and Fits',
        mapping=rebin_histogram
    )

    histogram_dataset = dataset.children['Histogram']
    histogram_dataset.x = np.arange(BIN_NUMS)
    histogram_dataset.graph.getPlotItem().setLabels(
        bottom='Counts', left='Occurences'
    )

    dataset.FLAG = True
    dataset.ON = False
    dataset.cnot_check_it = 0

    push_sequence(dataset)


def experiment(**kwargs):

    thread: ExperimentThread = kwargs['thread']
    dataset: Dataset = kwargs['dataset']
    # histogram_dataset = dataset.children['Histogram']

    # experiment
    dataset.HDAWG.seti('awgs/0/userregs/0', 4)
    time.sleep(0.2)
    dataset.HDAWG.seti('awgs/0/userregs/1', 1)
    time.sleep(0.1)
    dataset.ON = True

    while dataset.ON:
        time.sleep(0.1)
        if not thread.running:
            break
        dataset.ON = dataset.HDAWG.geti('awgs/0/userregs/1')

    raw_data = np.array(dataset.ctr.get_counts(name='hist_5')).flatten()
    dataset.set_data(raw_data)

    histogram_dataset = dataset.children['Histogram']
    read_data = dataset.ctr.get_counts(name='gated_8')
    dataset.log.info(str(read_data.shape))
    histogram_dataset.data, _ = np.histogram(read_data[:2 * dataset.repeat_per_time], bins=BIN_NUMS, range=(0, BIN_NUMS - 1))
    histogram_dataset.set_children_data()

    dataset.ctr.clear_ctr(name='hist_5')
    dataset.ctr.clear_ctr(name='gated_8')
    time.sleep(0.5)
