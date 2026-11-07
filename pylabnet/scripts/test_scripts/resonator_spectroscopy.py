import numpy as np
from pylabnet.scripts.data_center.datasets import Dataset

INIT_DICT = {
    'if_center': {'IF Center (MHz)': '50'},
    'if_span': {'IF Span (MHz)': '20'},
    'if_points': {'IF Points': '201'},
    'n_avg': {'Averages': '1000'},
    'reset_time': {'Reset Time Between Shots (us)': '10'},
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

    if_center = float(dataset.get_input_parameter('if_center')) * 1e6
    if_span = float(dataset.get_input_parameter('if_span')) * 1e6
    if_points = int(dataset.get_input_parameter('if_points'))
    n_avg = int(dataset.get_input_parameter('n_avg'))
    reset_us = int(dataset.get_input_parameter('reset_time'))

    fs = np.linspace(if_center - if_span / 2,
                     if_center + if_span / 2, if_points).astype(int)
    f_start = int(fs[0])
    f_stop = int(fs[-1])
    f_step = int(fs[1] - fs[0])
    reset_ns = reset_us * 1000

    while thread.running:
        OPX.build_stack()

        with OPX.for_('n', 0, n_avg, 1):
            with OPX.for_('f', f_start, f_stop + 1, f_step):
                OPX.update_frequency('resonator', 'f')
                OPX.measure_demod_iq('resonator', 'I', 'Q')
                OPX.delay(reset_ns, ['resonator'])

        results = OPX.execute(
            average_streams={'I': if_points, 'Q': if_points},
            wait=True
        )

        I = np.asarray(results['I'])
        Q = np.asarray(results['Q'])
        S = I + 1j * Q
        mag = np.abs(S)
        phase = np.unwrap(np.angle(S))

        f_min_if = fs[np.argmin(mag)]
        log.info(f"Min |S| at IF = {f_min_if/1e6:.3f} MHz "
                 f"(|S|={mag.min():.3e})")

        dataset.set_data(np.column_stack([fs, mag, phase]))
        dataset.update()

        thread.running = False
