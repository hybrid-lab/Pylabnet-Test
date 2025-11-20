import numpy as np
import time
import textwrap
import os
import json

from scipy.stats import poisson
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

#config_dict = load_config('datataker_test')
#BIN_NUMS = config_dict['histogram_params']['num_bins']
BIN_NUMS = 60

config_file_folder = "C:\\Users\\Roger\\Lukin SiV Dropbox\\SiV Quick Sharing\\G12 experiments\\config_files\\"


def load_config(filename):
    filepath = os.path.join(config_file_folder, f'{filename}.json')
    f = open(filepath)
    data = json.load(f)
    return data


def rebin_histogram(dataset, prev_dataset, n_data):
    """ Rebins a histogram """

    # If n_data is a function, apply that function on dataset and get the value
    if callable(n_data):
        n_data = n_data(dataset)

    raw_data = dataset.data[:n_data]
    prev_dataset.data, _ = np.histogram(raw_data, bins=BIN_NUMS, range=(0, BIN_NUMS - 1))


def double_poisson(x, a1, mu1, a2, mu2):
    return a1 * poisson.pmf(x, mu1) + a2 * poisson.pmf(x, mu2)


def fit_histogram(dataset, prev_dataset, peakindex, bounds_fn, p0_fn):
    """ Fits histogram """

    # Bounds and p0 are functions from dataset to actual numerical bounds
    bounds = bounds_fn(dataset)
    p0 = p0_fn(dataset)

    # bounds = ([450,  0.2, 50, np.argmax(raw_data)], [1000, np.argmax(raw_data)+5, 600, 40])
    # p0 = [500, np.argmax(raw_data)+1, 500, 10],

    try:
        popt, pcov = curve_fit(double_poisson, np.arange(BIN_NUMS), dataset.data, p0=p0, bounds=bounds)
        a1, mu1, a2, mu2 = popt

        if mu1 < mu2:
            peaks = [mu1, mu2, min((mu2 / mu1, 30))]
        else:
            peaks = [mu2, mu1, min((mu1 / mu2, 30))]

        if a2 < 0.1 or mu2 == 40:
            peaks = [mu1, mu1, 1]

    except RuntimeError:
        peaks = [1, 1, 1]
    except ValueError as e: # When initial guesses don't match bounds
        dataset.log.warn(e)
        dataset.log.warn(f"Bounds = {bounds}, p0 = {p0}")
        raise e # Re-raise to stop experiment

    prev_dataset.set_data(peaks[peakindex])


def upper_line(dataset, prev_dataset):
    prev_dataset.set_data(dataset.line1)


def lower_line(dataset, prev_dataset):
    prev_dataset.set_data(dataset.line2)


def upper_rabi(dataset, prev_dataset):
    prev_dataset.set_data(dataset.RABI_amp1)


def lower_rabi(dataset, prev_dataset):
    prev_dataset.set_data(dataset.RABI_amp2)


def init_fidel(dataset, prev_dataset, n_data):
    rawdata = dataset.data
    if dataset.children['Histogram'].children['Lower peak averages'].data is not None:
        lower_cnt_avg = dataset.children['Histogram'].children['Lower peak averages'].data[-1]
        # P(measure low | initialize low). This is the theoretical readout fidelity from
        # the Poisson distribution, using the measured low-count peak location.
        readout_fidelity = sum(poisson.pmf(x, lower_cnt_avg) for x in range(0, READ_THRESH + 1))
    else:
        readout_fidelity = 1

    # P(measuring low) ~~ P(init low) x P(measuring low | init low), neglect cases where
    # we init high but measure low. Thus P(init low) ~~ P(measure low) / P(mesaure low | init low).
    init_fidelity = min(np.mean(rawdata[0:n_data:2] <= READ_THRESH) / readout_fidelity, 1)

    prev_dataset.set_data(init_fidelity)


def pi_fidel(dataset, prev_dataset, n_data):
    rawdata = dataset.data
    if dataset.children['Histogram'].children['Lower peak averages'].data is not None:
        lower_cnt_avg = dataset.children['Histogram'].children['Lower peak averages'].data[-1]
        # P(measure low | initialize low), readout fidelity in low state.
        readout_fidelity_low = sum(poisson.pmf(x, lower_cnt_avg) for x in range(0, READ_THRESH + 1))
    else:
        readout_fidelity_low = 1

    init_fidelity = min(np.mean(rawdata[0:n_data:2] <= READ_THRESH) / readout_fidelity_low, 1)

    if dataset.children['Histogram'].children['Upper peak averages'].data is not None:
        higher_cnt_avg = dataset.children['Histogram'].children['Upper peak averages'].data[-1]
        # P(measure high | initialize high), readout fidelity in high state.
        readout_fidelity_high = 1 - sum(poisson.pmf(x, higher_cnt_avg) for x in range(0, READ_THRESH + 1))

    else:
        readout_fidelity_high = 1

    pi_fidelity = min(np.mean(rawdata[1:n_data:2] > READ_THRESH) / init_fidelity / readout_fidelity_high, 1)

    if pi_fidelity > 0.97:
        dataset.tunedup = True

    if pi_fidelity < 0.93:
        dataset.tunedup = False

    prev_dataset.set_data(pi_fidelity)


def init_fidel_no_compute(dataset, prev_dataset):
    prev_dataset.set_data(dataset.init_fidelity)


def pi_fidel_no_compute(dataset, prev_dataset):
    prev_dataset.set_data(dataset.pi_fidelity)


def CNOT_fidelity_test(self, ucount_trsh, lcount_trsh, n_data):

    self.HDAWG.seti('awgs/0/userregs/0', 3)
    time.sleep(0.3)
    self.HDAWG.seti('awgs/0/userregs/1', 1)
    time.sleep(0.1)
    self.ON = True

    while self.ON:
        time.sleep(0.05)
        self.ON = self.HDAWG.geti('awgs/0/userregs/1')

    raw_data = self.ctr.get_counts(name='gated_5')
    self.ctr.clear_ctr(name='gated_5')

    histogram_dataset = self.children['Histogram']
    histogram_dataset.data, _ = np.histogram(raw_data[:n_data], bins=BIN_NUMS, range=(0, BIN_NUMS - 1))
    histogram_dataset.set_children_data()

    upper_counts = histogram_dataset.children['Upper peak averages'].data[-1]
    lower_counts = histogram_dataset.children['Lower peak averages'].data[-1]

    # P(measure low | initialize low)
    readout_fidelity_low = sum(poisson.pmf(x, lower_counts) for x in range(0, READ_THRESH + 1))
    # P(measure high | initialize high)
    readout_fidelity_high = 1 - sum(poisson.pmf(x, upper_counts) for x in range(0, READ_THRESH + 1))

    self.init_fidelity = min(np.mean(raw_data[0:n_data:2] <= READ_THRESH) / readout_fidelity_low, 1)
    self.pi_fidelity = min(np.mean(raw_data[1:n_data:2] > READ_THRESH) / self.init_fidelity / readout_fidelity_high, 1)
    self.log.info(f'pi fidelity is {self.pi_fidelity}')

    # ucount_trsh = 9
    # lcount_trsh = 1.5

    if upper_counts < ucount_trsh or lower_counts > lcount_trsh:
        self.log.info("Contrast bad, reposition laser")
        run_contrast_measurement_and_move_line(self, histogram_dataset, lower_counts, lcount_trsh, upper_counts, ucount_trsh, n_data)

    return self.pi_fidelity


def ramsey(dataset, prev_dataset):
    rawdata = dataset.data
    for i in range(dataset.pts):
        prev_dataset.set_data(np.mean(rawdata[(dataset.repeat_per_time * i):(dataset.repeat_per_time * i + dataset.repeat_per_time)]))


def first_reading(dataset, prev_dataset):
    rawdata = dataset.data
    prev_dataset.set_data(np.mean(rawdata[0:(2 * dataset.repeat_per_freq):2]))


def second_reading(dataset, prev_dataset):
    rawdata = dataset.data
    prev_dataset.set_data(np.mean(rawdata[1:(2 * dataset.repeat_per_freq):2]))


def plot_odmr(dataset, prev_dataset):
    rawdata = dataset.ODMR_vec

    for i in range(141):
        prev_dataset.set_data(rawdata[i])


def find_ODMR_lines(self, low, high, min_contrast):
    self.HDAWG.seti('awgs/0/userregs/0', 1)
    time.sleep(0.3)
    self.HDAWG.seti('awgs/0/userregs/1', 1)
    time.sleep(0.1)
    self.ON = True

    time.sleep(0.1)

    while self.ON:
        self.ON = self.HDAWG.geti('awgs/0/userregs/1')

    count_vec = self.ctr.get_counts(name='gated_5')
    self.ctr.clear_ctr(name='gated_5')
    freq_vec = np.linspace(low, high, 141)

    ODMR_vec = np.zeros(141)
    for ii in range(141):
        ODMR_vec[ii] = np.mean(count_vec[(10 * ii):(10 * ii + 10)])

    if max(ODMR_vec) < min_contrast * np.mean(ODMR_vec):
        return 0
    else:
        self.ODMR_vec = ODMR_vec
        if freq_vec[np.argmax(ODMR_vec)] > ((low + high) / 2):
            self.HDAWG.setd('oscs/0/freq', (freq_vec[np.argmax(ODMR_vec)] - 66) * 1e6)
            self.HDAWG.setd('oscs/1/freq', (freq_vec[np.argmax(ODMR_vec)]) * 1e6)
            self.line1 = freq_vec[np.argmax(ODMR_vec)] - 66
            self.line2 = freq_vec[np.argmax(ODMR_vec)]
            self.log.info(f'found ODMR frequencies at {self.line1}MHz and {self.line2}MHz')
        else:
            self.HDAWG.setd('oscs/0/freq', (freq_vec[np.argmax(ODMR_vec)]) * 1e6)
            self.HDAWG.setd('oscs/1/freq', (freq_vec[np.argmax(ODMR_vec)] + 66) * 1e6)
            self.line1 = freq_vec[np.argmax(ODMR_vec)]
            self.line2 = freq_vec[np.argmax(ODMR_vec)] + 66
            self.log.info(f'found ODMR frequencies at {self.line1}MHz and {self.line2}MHz')
        return 1


def find_Rabi_amps(self):

    # first RABI
    self.HDAWG.seti('awgs/0/userregs/0', 2)
    time.sleep(0.3)
    self.HDAWG.seti('awgs/0/userregs/1', 1)
    time.sleep(0.1)
    self.ON = True

    time.sleep(0.1)

    while self.ON:
        self.ON = self.HDAWG.geti('awgs/0/userregs/1')

    count_vec = self.ctr.get_counts(name='gated_5')
    self.ctr.clear_ctr(name='gated_5')
    amp_vec = np.linspace(0.5, 1.0, 21)

    RABI_vec = np.zeros(21)
    for ii in range(21):
        for jj in range(49):
            RABI_vec[ii] = RABI_vec[ii] + np.abs(count_vec[(50 * ii) + jj] - count_vec[(50 * ii) + jj + 1])
        #RABI_vec[ii] = np.mean(count_vec[(50*ii):(50*ii+50)])

    if self.prev_RABI1_vec is None:
        RABI_amp1_index = np.argmin(RABI_vec)
        self.prev_RABI1_vec = RABI_vec
        RABI_amp1 = amp_vec[RABI_amp1_index]
    else:
        RABI_amp1_index = np.argmin((RABI_vec + self.prev_RABI1_vec) / 2)
        self.prev_RABI1_vec = (self.prev_RABI1_vec * 2 + RABI_vec) / 3
        RABI_amp1 = amp_vec[RABI_amp1_index]

    self.log.info(f'first Rabi pi amplitude found at {RABI_amp1}')

    # second RABI
    self.HDAWG.setd('oscs/0/freq', self.line2 * 1e6)
    self.HDAWG.setd('oscs/1/freq', self.line1 * 1e6)

    self.HDAWG.seti('awgs/0/userregs/0', 2)
    time.sleep(0.3)
    self.HDAWG.seti('awgs/0/userregs/1', 1)
    time.sleep(0.1)
    self.ON = True

    time.sleep(0.1)

    while self.ON:
        self.ON = self.HDAWG.geti('awgs/0/userregs/1')

    count_vec = self.ctr.get_counts(name='gated_5')
    self.ctr.clear_ctr(name='gated_5')

    amp_vec = np.linspace(0.5, 1.0, 21)

    RABI_vec = np.zeros(21)
    for ii in range(21):
        for jj in range(49):
            RABI_vec[ii] = RABI_vec[ii] + np.abs(count_vec[(50 * ii) + jj] - count_vec[(50 * ii) + jj + 1])
        #RABI_vec[ii] = np.mean(count_vec[(50*ii):(50*ii+50)])

    if self.prev_RABI2_vec is None:
        RABI_amp2_index = np.argmin(RABI_vec)
        self.prev_RABI2_vec = RABI_vec
        RABI_amp2 = amp_vec[RABI_amp2_index]
    else:
        RABI_amp2_index = np.argmin((RABI_vec + self.prev_RABI2_vec) / 2)
        self.prev_RABI2_vec = (self.prev_RABI2_vec * 2 + RABI_vec) / 3
        RABI_amp2 = amp_vec[RABI_amp2_index]

    self.log.info(f'second Rabi pi amplitude found at {RABI_amp2}')

    self.HDAWG.seti('awgs/0/userregs/2', RABI_amp1_index)
    self.HDAWG.seti('awgs/0/userregs/3', RABI_amp2_index)
    self.HDAWG.seti('awgs/1/userregs/2', RABI_amp1_index)
    self.HDAWG.seti('awgs/1/userregs/3', RABI_amp2_index)
    self.RABI_amp1 = RABI_amp1
    self.RABI_amp2 = RABI_amp2
    self.HDAWG.setd('oscs/0/freq', self.line1 * 1e6)
    self.HDAWG.setd('oscs/1/freq', self.line2 * 1e6)


def run_contrast_measurement_and_move_line(self, histogram_dataset,
                                           lower_counts, lcount_trsh,
                                           upper_counts, ucount_trsh,
                                           n_data):

    if lower_counts > lcount_trsh:

        go_up(self)
        going_up = True

        while True:
            self.HDAWG.seti('awgs/0/userregs/0', 3)
            time.sleep(0.3)
            self.HDAWG.seti('awgs/0/userregs/1', 1)
            time.sleep(0.1)
            self.ON = True

            while self.ON:
                # if dataset.stop:
                #     thread.running = False
                # if not thread.running:
                #     break
                self.ON = self.HDAWG.geti('awgs/0/userregs/1')

            raw_data = self.ctr.get_counts(name='gated_5')
            self.ctr.clear_ctr(name='gated_5')
            histogram_dataset.data, _ = np.histogram(raw_data[:n_data], bins=BIN_NUMS, range=(0, BIN_NUMS - 1))
            histogram_dataset.set_children_data()

            #new_contrast = histogram_dataset.children['Contrast'].data[-1]
            new_lcounts = histogram_dataset.children['Lower peak averages'].data[-1]

            if new_lcounts < lcount_trsh:
                break

            if going_up:
                if new_lcounts < lower_counts:
                    go_up(self)
                else:
                    go_down(self)
                    going_up = False
            else:
                if new_lcounts < lower_counts:
                    go_down(self)
                else:
                    go_up(self)
                    going_up = True

            lower_counts = new_lcounts

    else:
        go_up(self)
        going_up = True

        while True:
            self.HDAWG.seti('awgs/0/userregs/0', 3)
            time.sleep(0.3)
            self.HDAWG.seti('awgs/0/userregs/1', 1)
            time.sleep(0.1)
            self.ON = True

            while self.ON:
                # if dataset.stop:
                #     thread.running = False
                # if not thread.running:
                #     break
                self.ON = self.HDAWG.geti('awgs/0/userregs/1')

            raw_data = self.ctr.get_counts(name='gated_5')
            self.ctr.clear_ctr(name='gated_5')
            histogram_dataset.data, _ = np.histogram(raw_data[:n_data], bins=BIN_NUMS, range=(0, BIN_NUMS - 1))
            histogram_dataset.set_children_data()

            #new_contrast = histogram_dataset.children['Contrast'].data[-1]
            new_ucounts = histogram_dataset.children['Upper peak averages'].data[-1]

            if new_ucounts > ucount_trsh:
                break

            if going_up:
                if new_ucounts > upper_counts:
                    go_up(self)
                else:
                    go_down(self)
                    going_up = False
            else:
                if new_ucounts > upper_counts:
                    go_down(self)
                else:
                    go_up(self)
                    going_up = True

            upper_counts = new_ucounts

    self.log.info("laser repositioned, found good contrast again")


def go_up(self):
    wl = self.wm.get_wavelength(1)
    self.wm.update_parameters([{
        'channel': 1, 'setpoint': wl + 0.00005
    }])


def go_down(self):
    wl = self.wm.get_wavelength(1)
    self.wm.update_parameters([{
        'channel': 1, 'setpoint': wl - 0.00005
    }])


def upload_sequence(dataset, program):
    dataset.log.info("Uploading to HDAWG...")
    dataset.awgModule.set("compiler/sourcestring", textwrap.dedent(program))

    # While uploading
    while dataset.awgModule.getInt('compiler/status') == -1:
        dataset.log.info("Waiting for HDAWG compiler...")
        time.sleep(1)

    status = dataset.awgModule.getInt('compiler/status')
    # Compilation failed
    if status == 1:
        dataset.log.warn(f"Compilation failed: {dataset.awgModule.getString('compiler/statusstring')}")
        return
    # No warnings
    elif status == 0:
        dataset.log.info("Compilation successful with no warnings, will upload the program to the instrument.")
    # Warnings
    elif status == 2:
        dataset.log.warn("Compilation successful with warnings, will upload the program to the instrument.")
        dataset.log.warn(f"Compiler warning: {dataset.awgModule.getString('compiler/statusstring')}")
    else:
        dataset.log.warn(f"Unknown status. Compiler warning: {dataset.awgModule.getString('compiler/statusstring')}")

    # Wait for the waveform upload to finish
    while (dataset.awgModule.getDouble('progress') < 1.0) and (dataset.awgModule.getInt('elf/status') != 1):
        dataset.log.info(f"Progress: {dataset.awgModule.getDouble('progress'):.2f}")
        time.sleep(0.2)

    elf_status = dataset.awgModule.getInt('elf/status')
    if elf_status == 0:
        dataset.log.info("Upload to the instrument successful.")
    elif elf_status == 1:
        dataset.log.warn("Upload to the instrument failed.")
