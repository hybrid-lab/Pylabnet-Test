import numpy as np
from pylabnet.scripts.data_center.datasets import Dataset

INIT_DICT = {
    'readout_if': {'Readout IF (MHz)': '50'},
    'qubit_if_center': {'Qubit IF Center (MHz)': '100'},
    'qubit_if_span': {'Qubit IF Span (MHz)': '40'},
    'qubit_if_points': {'Qubit IF Points': '201'},
    'n_avg': {'Averages': '2000'},
    'thermalization': {'Thermalization (us)': '200'},
}


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.OPX_client = kwargs['OPX_OPX']


def experiment(**kwargs):
    thread = kwargs['thread']
    dataset = kwargs['dataset']
    OPX = dataset.OPX_client
    log = dataset.log

    readout_if = int(float(dataset.get_input_parameter('readout_if')) * 1e6)
    qif_center = float(dataset.get_input_parameter('qubit_if_center')) * 1e6
    qif_span = float(dataset.get_input_parameter('qubit_if_span')) * 1e6
    qif_points = int(dataset.get_input_parameter('qubit_if_points'))
    n_avg = int(dataset.get_input_parameter('n_avg'))
    therm_us = int(dataset.get_input_parameter('thermalization'))

    fs = np.linspace(qif_center - qif_span / 2,
                     qif_center + qif_span / 2, qif_points).astype(int)
    f_start = int(fs[0])
    f_stop = int(fs[-1])
    f_step = int(fs[1] - fs[0])
    therm_ns = therm_us * 1000

    while thread.running:
        OPX.build_stack()

        with OPX.for_('n', 0, n_avg, 1):
            with OPX.for_('f', f_start, f_stop + 1, f_step):
                OPX.update_frequency('qubit', 'f')
                OPX.play_pulse('saturation', 'qubit')
                OPX.align('qubit', 'resonator')
                OPX.measure_demod_iq('resonator', 'I', 'Q')
                OPX.delay(therm_ns, ['qubit', 'resonator'])

        results = OPX.execute(
            average_streams={'I': qif_points, 'Q': qif_points},
            wait=True
        )

    # while thread.running:
    #     OPX.build_stack()

    #     # Pin the resonator IF to the value found in resonator spectroscopy.
    #     # We hardcode it as the QUA literal here via a one-shot for_ trick:
    #     # set the variable once, then use the standard update_frequency op.
    #     with OPX.for_('rif', readout_if, readout_if + 1, 1):
    #         OPX.update_frequency('resonator', 'rif')

    #         with OPX.for_('n', 0, n_avg, 1):                       # averaging
    #             with OPX.for_('f', f_start, f_stop + 1, f_step):   # qubit sweep
    #                 OPX.update_frequency('qubit', 'f')
    #                 OPX.play_pulse('saturation', 'qubit')
    #                 OPX.align('qubit', 'resonator')
    #                 OPX.measure_demod_iq('resonator', 'I', 'Q')
    #                 OPX.delay(therm_ns, ['qubit', 'resonator'])

    #     results = OPX.execute(
    #         average_streams={'I': qif_points, 'Q': qif_points},
    #         wait=True
    #     )

    #     I = np.asarray(results['I'])
    #     Q = np.asarray(results['Q'])
    #     S = I + 1j * Q
    #     mag   = np.abs(S)
    #     phase = np.unwrap(np.angle(S))

    #     # The qubit feature is usually most visible as a phase deviation
    #     # from the off-resonant baseline.
    #     phase_dev = phase - np.median(phase)
    #     f_qubit_if = fs[np.argmax(np.abs(phase_dev))]
    #     log.info(f"Qubit feature at IF = {f_qubit_if/1e6:.3f} MHz "
    #              f"(|phase dev| = {np.max(np.abs(phase_dev)):.3f} rad)")

    #     dataset.set_data(np.column_stack([fs, mag, phase]))
    #     dataset.update()

    #     thread.running = False
