# Fast single image, TWO cameras: MOT -> CMOT -> release -> image
# Same sequence as jingfastMOT.py, but a second camera looks at the same MOT
# from a different position and is exposed at exactly the same moments.
#
# Both cameras get the identical trigger waveform, on separate DO lines of NI
# card 2 (dio0 -> camera A Line0, dio8 -> camera B Line0). Both are
# sample-locked to the same clock, so the two frame pairs are exposed
# simultaneously under the same imaging light pulse.
#
# dio8 is port1/line0 on the PXIe-6535, which is hardware-timed like port0.
# dio7 is deliberately NOT used: older scripts (lattice_bothCameras.py) put the
# second camera there, but it is now the COM5 lattice driver's Sync gate
# (latticemovedouble_sweep.py, fastLatticeMoveSweep2_dual.py). This script
# never writes dio7, so it cannot step that driver.
#
# Everything that does not depend on the second camera -- fits, atom number,
# the image panel class, safe-state values -- is imported from jingfastMOT.py
# rather than copied, so a fix there applies here too. The datataker only
# reloads THIS module on each run, so after editing jingfastMOT.py restart the
# datataker for the change to take effect.
#
# Clocking: NI only, no OPX. See jingfastMOT.py.

import numpy as np

import jingfastMOT as base
from jingfastMOT import (
    AutoLevelPlot2D, fit_gaussian_1d, sensor_full_scale, localized_signal,
    results_table_html, ROOT_HALF_PI, INTEGRATION_RADII,
    FRAME_W, FRAME_H, COLORMAP, PIXEL_FORMAT, CAMERA_GAIN_DB,
    SENSOR_MAX_MONO16, CARD2_DEV, CARD3_DEV, CLOCK_MASTER_DEV,
)

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
# One entry per camera. 'client' is the datataker client key; 'trigger_do' is
# the NI card 2 line wired to that camera's Line0.
#
# Each camera has its own imaging geometry, because a camera at a different
# position generally has a different magnification and looks through a
# different viewport. Those only rescale the reported widths, atom number and
# density -- the fits themselves are in pixels and need nothing.
#
# !! Camera B's magnification / window / distance below are PLACEHOLDERS copied
# !! from camera A. Measure them for camera B's actual position, or its atom
# !! number and widths in um are wrong (widths by M_A/M_B, density by its cube).
CAMERAS = [
    {
        'id': 'A',
        'label': 'Camera A (top chamber)',
        'client': 'fluorescence_imaging_camera_bfs_u3_51s5m_top_chamber',
        'trigger_do': 'dio0',
        'enable_param': 'enable_camera_a',
        'magnification': base.MAGNIFICATION,
        'window_mm': base.WINDOW_DIAMETER_MM,
        'distance_mm': base.APERTURE_DISTANCE_MM,
    },
    {
        'id': 'B',
        'label': 'Camera B (second view)',
        'client': 'fluorescence_imaging_camera_bfs_u3_51s5m_bottom_chamber',
        'trigger_do': 'dio8',
        'enable_param': 'enable_camera_b',
        'magnification': base.MAGNIFICATION,       # PLACEHOLDER -- measure
        'window_mm': base.WINDOW_DIAMETER_MM,      # PLACEHOLDER -- measure
        'distance_mm': base.APERTURE_DISTANCE_MM,  # PLACEHOLDER -- measure
    },
]


def image_name(cam):
    return f"{cam['label']}: MOT image (frame 1 - frame 2)"


def window_attr(cam):
    """Name of the GUI attribute holding this camera's popup window."""
    return f"camera_{cam['id']}_window"


# Initial size of each camera's popup (width, height in px). Tall enough for
# the image, the x profile and the results table underneath it.
WINDOW_SIZE = (1100, 1000)


# Once the first camera's frame has arrived the trigger has definitely fired,
# so the other camera's frame is already exposed and only needs reading out
# (~15 ms for 5 MP over USB3). A short timeout keeps a dead second camera from
# stalling every cycle by a whole frame timeout.
SECONDARY_TIMEOUT_MS = 1500

# A camera that misses this many grabs in a row is dropped for the rest of the
# run so the other one keeps going at full rate. Camera A is never dropped:
# it behaves exactly like jingfastMOT and keeps retrying.
MAX_CONSECUTIVE_MISSES = 5

# jingfastMOT's safe state plus the second trigger line.
SAFE_AO = base.SAFE_AO
SAFE_DO = dict(base.SAFE_DO, dio8=0)


def park_outputs(NI_do, NI_ao, logger):
    """jingfastMOT.park_outputs, extended to park dio8. Never raises."""
    try:
        NI_ao.write_static(ao=SAFE_AO)
        NI_do.write_static(do=SAFE_DO)
    except Exception as exc:
        logger.warn(f"write_static unavailable ({exc}); parking channel by channel")
        for channel, value in SAFE_AO.items():
            try:
                NI_ao.set_ao_voltage(ao_channel=channel, voltages=value)
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value} V: {e}")
        for channel, value in SAFE_DO.items():
            try:
                NI_do.set_do_voltage(do_channel=channel, value=bool(value))
            except Exception as e:
                logger.warn(f"Could not park {channel} at {value}: {e}")
    logger.info(
        "Outputs parked: VCOs 0 V, coils 0 V, bias 0 V, AOM drive 0.8 V, "
        "all light off, both camera triggers low."
    )


def cloud_metrics_for(cam, diff, fx, fy, dtype, exposure_us):
    """jingfastMOT.cloud_metrics, rescaled to this camera's geometry.

    cloud_metrics assumes camera A's pixel size and viewport. Neither enters
    the fit, only the conversion to physical units, so the result is simply
    rescaled: N goes as 1/eta, widths as the pixel size, and peak density as
    N / width^3.
    """
    met = base.cloud_metrics(diff, fx, fy, dtype, exposure_us,
                             gain_db=cam['gain_db'])
    if met is None:
        return None
    k = cam['pixel_um'] / base.PIXEL_UM
    eta = base.collection_efficiency(cam['window_mm'], cam['distance_mm'])
    r = met['eta'] / eta
    met['n_atoms'] *= r
    met['n_atoms_total'] *= r
    met['wx_um'] *= k
    met['wy_um'] *= k
    met['density'] *= r / k ** 3
    met['eta'] = eta
    return met


INIT_DICT = {
    # ===================== CAMERAS =====================
    'enable_camera_a': {'Enable Camera A / top (1=yes, 0=no)': '1'},
    'enable_camera_b': {'Enable Camera B / second view (1=yes, 0=no)': '1'},

    # ===================== PHASE 1: MOT =====================
    'mot_loading_time': {'MOT Loading Time (ms)': '800'},
    'mot_aom_power': {'MOT AOM Power (V)': '0.8'},
    'repump_aom_power': {'Repump AOM Power (V)': '0.8'},
    'mot_coils_voltage': {'MOT Coils Voltage (V)': '6'},
    'MOT_VCO_loading': {'MOT VCO Loading (V)': '0', },
    'Repump_VCO': {'Repump VCO (V)': '0'},

    # ===================== PHASE 2: CMOT =====================
    'mot_switch_time': {'MOT Switch Time (ms) [MOT->CMOT ramp]': '15'},
    'cmot_hold_time': {'CMOT Hold Time (ms)': '5'},
    'cmot_mot_power': {'CMOT MOT AOM Power (V)': '0.6'},
    'cmot_repump_power': {'CMOT Repump AOM Power (V)': '0.2'},
    'cmot_detuning_voltage': {'CMOT MOT VCO (V) [-40MHz]': '-0.6'},
    'cmot_coils_voltage': {'CMOT Coils Voltage (V)': '8'},
    'cmot_repump_vco': {'CMOT Repump VCO (V) [-3MHz]': '-0.2'},

    # ===================== RELEASE + IMAGING =====================
    'tof_us': {'TOF (us)': '0'},
    'coils_ttl_lead_ms': {'Coils TTL Lead (ms before release)': '2'},
    'coils_restore_delay_ms': {'Coils Restore (ms after frame 2)': '10'},
    'coils_rampback_ms': {'Coils Rampback Duration (ms)': '10'},
    'MOT_VCO_imaging': {'MOT VCO Imaging (V)': '0.3'},
    'vco_lead_us': {'VCO Lead Time Before AOM (us)': '100'},
    'imaging_repump_lead_us': {'Imaging Repump Lead Before MOT Light (us)': '200'},
    'imaging_repump_power': {'Imaging Repump AOM Power (V)': '0.2'},
    'imaging_repump_vco': {'Imaging Repump VCO (V)': '0'},
    # Shared by both cameras: the imaging light pulse is sized from it, so a
    # different exposure on one camera would only add background, not signal.
    'camera_exposure_us': {'Camera Exposure Time (us)': '200'},
    'atoms_clear_time': {'Atoms Clear Time (ms)': '100'},

    # ===================== BIAS COILS =====================
    'bias_x_v': {'Bias X (V)': '0.0'},
    'bias_y_v': {'Bias Y (V)': '0.0'},
    'bias_z_v': {'Bias Z (V)': '0.0'},

    # ===================== DISPLAY (both panels) =====================
    # See jingfastMOT.py for what these do.
    'display_autoscale': {'Autoscale Display? (1=yes, 0=manual)': '1'},
    'display_min': {'Display Min / Background (counts)': '-5'},
    'display_max': {'Display Max (counts)': '2000'},
}


def define_dataset():
    return 'Dataset'


def configure(**kwargs):
    dataset = kwargs['dataset']
    dataset.NI_card_2 = kwargs['nidaqmx_ni_daq_2']
    dataset.NI_card_3 = kwargs['nidaqmx_ni_daq_3']

    disp_min = float(dataset.get_input_parameter("display_min"))
    disp_max = float(dataset.get_input_parameter("display_max"))
    disp_min = max(disp_min, -float(SENSOR_MAX_MONO16))
    disp_max = min(disp_max, float(SENSOR_MAX_MONO16))
    dataset.diff_levels = (disp_min, disp_max)

    # One image panel per camera, each in its own popup window. Built for every
    # camera, enabled or not, so toggling one between runs never needs a
    # reconfigure.
    for cam in CAMERAS:
        name = image_name(cam)
        win_attr = window_attr(cam)
        # Plot2D keeps its popup as an attribute on the GUI, and the datataker
        # only clears the MAIN window on reconfigure -- so without this every
        # reconfigure would add another panel to the old popup, stacked under
        # the dead one from the previous run.
        old = getattr(dataset.gui, win_attr, None)
        if old is not None:
            try:
                old.close()
                old.deleteLater()
            except Exception:
                pass
            delattr(dataset.gui, win_attr)
        dataset.add_child(
            name=name,
            data_type=AutoLevelPlot2D,
            min_x=0, max_x=FRAME_W, pts_x=FRAME_W,
            min_y=0, max_y=FRAME_H, pts_y=FRAME_H,
            new_plot=True,
            window=win_attr,
            window_title=cam['label'],
        )
        getattr(dataset.gui, win_attr).resize(*WINDOW_SIZE)
        img = dataset.children[name]
        img.graph.view.setAspectLocked(True)
        img.graph.view.invertY(True)
        img.update_colormap(COLORMAP)
        img.graph.ui.histogram.hide()
        img.graph.ui.roiBtn.hide()
        img.graph.ui.menuBtn.hide()
        img.graph.setLevels(disp_min, disp_max)
        img.attach_colorbar(
            disp_min, disp_max,
            label=f"{cam['label']} (counts, frame 1 - frame 2)")
        img.attach_profiles()
    dataset.graph.hide()


def experiment(**kwargs):
    dataset = kwargs['dataset']
    thread = kwargs['thread']
    logger = dataset.log

    NI_card_2 = kwargs["nidaqmx_ni_daq_2"]
    NI_card_3 = kwargs["nidaqmx_ni_daq_3"]

    # =========================================================================
    # READ GUI PARAMETERS
    # =========================================================================
    mot_loading_ms = int(dataset.get_input_parameter("mot_loading_time"))
    mot_aom_power = float(dataset.get_input_parameter("mot_aom_power"))
    repump_aom_power = float(dataset.get_input_parameter("repump_aom_power"))
    mot_coils_v = float(dataset.get_input_parameter("mot_coils_voltage"))
    vco_loading = float(dataset.get_input_parameter("MOT_VCO_loading"))
    repump_vco = float(dataset.get_input_parameter("Repump_VCO"))

    mot_switch_ms = int(dataset.get_input_parameter("mot_switch_time"))
    cmot_hold_ms = int(dataset.get_input_parameter("cmot_hold_time"))
    cmot_mot_power = float(dataset.get_input_parameter("cmot_mot_power"))
    cmot_repump_power = float(dataset.get_input_parameter("cmot_repump_power"))
    cmot_detuning = float(dataset.get_input_parameter("cmot_detuning_voltage"))
    cmot_coils_v = float(dataset.get_input_parameter("cmot_coils_voltage"))
    cmot_repump_vco = float(dataset.get_input_parameter("cmot_repump_vco"))

    tof_us = int(dataset.get_input_parameter("tof_us"))
    coils_lead_ms = int(dataset.get_input_parameter("coils_ttl_lead_ms"))
    coils_restore_ms = int(dataset.get_input_parameter("coils_restore_delay_ms"))
    coils_rampback_ms = int(dataset.get_input_parameter("coils_rampback_ms"))
    vco_imaging = float(dataset.get_input_parameter("MOT_VCO_imaging"))
    vco_lead_us = int(dataset.get_input_parameter("vco_lead_us"))
    img_repump_lead_us = int(dataset.get_input_parameter("imaging_repump_lead_us"))
    img_repump_power = float(dataset.get_input_parameter("imaging_repump_power"))
    img_repump_vco = float(dataset.get_input_parameter("imaging_repump_vco"))
    camera_exposure_us = int(dataset.get_input_parameter("camera_exposure_us"))
    atoms_clear_ms = int(dataset.get_input_parameter("atoms_clear_time"))

    bias_x_v = float(dataset.get_input_parameter("bias_x_v"))
    bias_y_v = float(dataset.get_input_parameter("bias_y_v"))
    bias_z_v = float(dataset.get_input_parameter("bias_z_v"))

    autoscale = bool(int(dataset.get_input_parameter("display_autoscale")))
    disp_min_req = float(dataset.get_input_parameter("display_min"))
    disp_max_req = float(dataset.get_input_parameter("display_max"))

    # =========================================================================
    # ACTIVE CAMERAS
    # =========================================================================
    # Per-run state for each camera lives in a copy of its CAMERAS entry, so
    # the two never share a peak, a full-scale or a display range.
    cams = []
    for spec in CAMERAS:
        if int(dataset.get_input_parameter(spec['enable_param'])) == 0:
            logger.info(f"{spec['label']}: disabled")
            continue
        client = kwargs.get(spec['client'])
        if client is None:
            logger.error(
                f"{spec['label']}: client '{spec['client']}' is not connected "
                f"to this datataker -- running without it. Add its server to "
                f"the datataker config (see data_taker_config_with_everything)."
            )
            continue
        cam = dict(spec)
        cam.update(
            client=client,
            img_ds=dataset.children[image_name(spec)],
            pixel_um=base.PIXEL_PITCH_UM / spec['magnification'],
            gain_db=CAMERA_GAIN_DB,
            disp_min=disp_min_req, disp_max=disp_max_req,
            run_peak=None, limit=None,
            misses=0, pairs=0, alive=True, streaming=False,
        )
        cams.append(cam)
    if not cams:
        logger.error("No camera enabled and connected; nothing to do.")
        return
    logger.info("Cameras: " + ", ".join(
        f"{c['label']} on {c['trigger_do']} ({c['pixel_um']:.3f} um/px)"
        for c in cams))

    # =========================================================================
    # TIMING  (identical to jingfastMOT)
    # =========================================================================
    ni_rate = 20000
    SPM = ni_rate // 1000
    SPU = ni_rate / 1e6
    ttl_up = max(1, SPM // 20)
    def ms2s(t): return t * SPM
    def us2s(t): return int(round(t * SPU))

    mot_s = ms2s(mot_loading_ms)
    cmot_ramp_s = ms2s(mot_switch_ms)
    cmot_hold_s = ms2s(cmot_hold_ms)
    tof_samp = us2s(tof_us)
    clear_s = ms2s(atoms_clear_ms)
    vco_lead_s = us2s(vco_lead_us)
    img_repump_lead_s = us2s(img_repump_lead_us)
    coils_lead_s = ms2s(coils_lead_ms)
    coils_restore_s = ms2s(coils_restore_ms)
    coils_rampback_s = ms2s(coils_rampback_ms)
    aom_pulse_s = max(ttl_up, int(np.ceil(camera_exposure_us * SPU)))

    # =========================================================================
    # TIMELINE: MOT -> CMOT -> release -> TOF -> repump pre-pulse -> image
    # =========================================================================
    cmot_start = mot_s
    cmot_hold_start = cmot_start + cmot_ramp_s
    cmot_end = cmot_start + cmot_ramp_s + cmot_hold_s
    release = cmot_end
    coils_off = release - coils_lead_s
    f1 = release + tof_samp
    f2 = f1 + clear_s
    coils_restore_t = f2 + aom_pulse_s + coils_restore_s
    N = coils_restore_t + coils_rampback_s + 2

    logger.info(
        f"Fast MOT+CMOT (2 cameras): mot={mot_loading_ms}ms, "
        f"cmot={mot_switch_ms}+{cmot_hold_ms}ms, tof={tof_us}us, "
        f"release={release/SPM:.0f}ms, "
        f"repump lead={img_repump_lead_s / SPU:.0f}us, N={N}"
    )
    if img_repump_lead_s > tof_samp:
        logger.warn(
            f"Imaging repump lead ({img_repump_lead_us} us) is longer than the "
            f"TOF ({tof_us} us), so the repump pre-pulse starts before release "
            f"and overwrites the last {(img_repump_lead_s - tof_samp) / SPU:.0f} us "
            f"of the CMOT hold with the imaging repump power/VCO."
        )
    if img_repump_lead_us > 0 and img_repump_lead_s == 0:
        logger.warn(
            f"Imaging repump lead {img_repump_lead_us} us is below the "
            f"{1e6 / ni_rate:.0f} us sample period and rounds to zero; the "
            f"repump and MOT light will turn on together."
        )

    # =========================================================================
    # BUILD WAVEFORMS  (identical to jingfastMOT)
    # =========================================================================
    # One camera trigger waveform, played on every active camera's line.
    cam_trig = [0] * N
    for s in range(ttl_up):
        cam_trig[f1 + s] = 1
        cam_trig[f2 + s] = 1

    ado = [0] * N
    aao = [mot_aom_power] * N
    rdo = [0] * N
    rao = [repump_aom_power] * N

    # MOT
    for s in range(cmot_start):
        ado[s] = 1
        aao[s] = mot_aom_power
        rdo[s] = 1
        rao[s] = repump_aom_power
    # CMOT ramp
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        ado[t] = 1
        aao[t] = mot_aom_power + frac * (cmot_mot_power - mot_aom_power)
        rdo[t] = 1
        rao[t] = repump_aom_power + frac * (cmot_repump_power - repump_aom_power)
    # CMOT hold
    for s in range(cmot_hold_s):
        t = cmot_hold_start + s
        ado[t] = 1
        aao[t] = cmot_mot_power
        rdo[t] = 1
        rao[t] = cmot_repump_power
    # TOF: digital off, analog stays at full power (thermal stability)
    for s in range(release, min(f1, N)):
        ado[s] = 0
        rdo[s] = 0
    # Imaging pulses: repump leads, MOT light joins for the exposure.
    for frame_t in (f1, f2):
        pulse_end = min(frame_t + aom_pulse_s, N)
        for s in range(max(0, frame_t - img_repump_lead_s), pulse_end):
            rdo[s] = 1
            rao[s] = img_repump_power
        for s in range(frame_t, pulse_end):
            ado[s] = 1
            aao[s] = mot_aom_power

    # Coils: TTL off from coils_off through frame 2 + restore delay, then back
    # on with the analog ramped 0 -> MOT value.
    cdo = [0] * N
    for s in range(coils_off, min(coils_restore_t, N)):
        cdo[s] = 1
    cao = [mot_coils_v] * N
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        cao[t] = mot_coils_v + frac * (cmot_coils_v - mot_coils_v)
    for s in range(cmot_hold_start, N):
        cao[s] = cmot_coils_v
    for s in range(coils_rampback_s):
        t = coils_restore_t + s
        if t >= N:
            break
        frac = s / max(coils_rampback_s - 1, 1)
        cao[t] = frac * mot_coils_v
    for s in range(coils_restore_t + coils_rampback_s, N):
        cao[s] = mot_coils_v

    # MOT VCO
    mvco = [vco_loading] * N
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        mvco[t] = vco_loading + frac * (cmot_detuning - vco_loading)
    for s in range(cmot_hold_start, N):
        mvco[s] = cmot_detuning
    for s in range(max(0, f1 - vco_lead_s), min(N, f2 + aom_pulse_s)):
        mvco[s] = vco_imaging

    # Repump VCO
    rvco = [repump_vco] * N
    for s in range(cmot_ramp_s):
        t = cmot_start + s
        frac = s / max(cmot_ramp_s - 1, 1)
        rvco[t] = repump_vco + frac * (cmot_repump_vco - repump_vco)
    for s in range(cmot_hold_start, N):
        rvco[s] = cmot_repump_vco
    for s in range(max(0, f1 - img_repump_lead_s), min(N, f2 + aom_pulse_s)):
        rvco[s] = img_repump_vco

    # Lattice channels: all off
    lat_off_do = [1] * N
    lat_off_ao = [0.0] * N

    # =========================================================================
    # LOAD, ARM, EXECUTE
    # =========================================================================
    h1 = None
    h2 = None

    cycle_ms = N / SPM
    frame_timeout_ms = int(2 * cycle_ms + 1000)
    logger.info(
        f"Sequence cycle {cycle_ms:.0f} ms ({N} samples @ {ni_rate} S/s); "
        f"frame timeout {frame_timeout_ms} ms"
    )

    def get_frame(cam, timeout_ms):
        b, shape, dtype = cam['client'].get_frame_bytes(timeout_ms)
        return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)

    def setup_camera(cam):
        """Trigger, pixel format, exposure and gain for one camera.

        Same settings as jingfastMOT, and the same gain read-back: whatever
        the camera actually applied is what the atom number is computed from.
        """
        client, label = cam['client'], cam['label']
        client.set_hardware_trigger(
            line="Line0", activation="RisingEdge",
            selector="FrameStart", overlap="ReadOut",
            acquisition_mode="Continuous")
        if not client.try_set_enum("PixelFormat", PIXEL_FORMAT):
            logger.warn(
                f"{label}: could not set PixelFormat to {PIXEL_FORMAT}; the "
                f"camera keeps whatever it had."
            )
        client.set_exposure(camera_exposure_us)
        gain_set = client.try_set_float("Gain", CAMERA_GAIN_DB)
        gain_db = None
        try:
            gain_db = client.get_float("Gain")
        except Exception as exc:
            logger.warn(f"{label}: cannot read back gain ({exc}).")
        if gain_db is None:
            gain_db = CAMERA_GAIN_DB if gain_set else 0.0
            logger.warn(
                f"{label}: gain could not be read back; assuming "
                f"{gain_db:.2f} dB."
            )
        if abs(gain_db - CAMERA_GAIN_DB) > 0.01:
            logger.error(
                f"{label}: gain is {gain_db:.2f} dB, NOT the requested "
                f"{CAMERA_GAIN_DB:.2f} dB. Atom number uses the actual value."
            )
        else:
            logger.info(f"{label}: gain confirmed at {gain_db:.2f} dB")
        cam['gain_db'] = gain_db
        client.start_acquisition()
        cam['streaming'] = True

    def describe_arm(name, handle):
        hid = handle.get("handle_id") if isinstance(handle, dict) else None
        if hid is None:
            logger.error(
                f"{name}: arm() returned no handle -- nothing was queued, so "
                f"this card is NOT driving its outputs."
            )
            return
        compiled = (handle.get("meta") or {}).get("compiled", {})
        if not compiled:
            logger.warn(f"{name}: armed, but reported no compiled subsystems.")
        for key, c in compiled.items():
            logger.info(
                f"{name}[{key}]: {len(c.get('channels', []))} channels, "
                f"{c.get('total_samples')} samples @ {c.get('sample_rate')} S/s, "
                f"clock_source={c.get('clock_source')}"
            )

    def process_pair(cam, frame1, frame2):
        """Everything jingfastMOT does with one frame pair, for one camera."""
        label = cam['label']
        img_ds = cam['img_ds']
        diff = frame1.astype(np.int32) - frame2.astype(np.int32)

        frame_peak = int(max(frame1.max(), frame2.max()))
        if cam['run_peak'] is None or frame_peak > cam['run_peak']:
            cam['run_peak'] = frame_peak

        if cam['limit'] is None:
            limit = sensor_full_scale(frame1)
            cam['limit'] = limit
            cam['disp_min'] = max(cam['disp_min'], -float(limit))
            cam['disp_max'] = min(cam['disp_max'], float(limit))
            logger.info(f"{label}: full scale {limit} counts ({frame1.dtype}).")
            if frame1.dtype == np.uint8:
                logger.error(
                    f"{label}: sending 8-bit frames despite the request for "
                    f"{PIXEL_FORMAT}; every count is 16x coarser than Mono16."
                )
            img_ds.set_levels_hint(cam['disp_min'], cam['disp_max'])
        disp_min, disp_max = cam['disp_min'], cam['disp_max']

        if autoscale:
            frame_max = float(np.percentile(diff, 99.95))
            if frame_max <= disp_min + 1:
                frame_max = disp_min + 10
            lo, hi = disp_min, frame_max
        else:
            lo, hi = disp_min, disp_max
            if hi <= lo:
                hi = lo + 1

        if img_ds.fit_to_frame(diff.shape):
            logger.warn(
                f"{label}: frame is {diff.shape[1]}x{diff.shape[0]}, not "
                f"{FRAME_W}x{FRAME_H}; image plot resized to match."
            )

        x_prof = diff.sum(axis=0).astype(float)
        y_prof = diff.sum(axis=1).astype(float)
        fx = fit_gaussian_1d(x_prof)
        fy = fit_gaussian_1d(y_prof)

        table = results_table_html(None, None, None, None, False)
        if fx and fy:
            peak = fx['amplitude'] / (fy['w'] * ROOT_HALF_PI)
            good = fx['reliable'] and fy['reliable']
            flag = '' if good else '   [UNRELIABLE]'
            met = cloud_metrics_for(cam, diff, fx, fy, frame1.dtype,
                                    camera_exposure_us)
            if met:
                logger.info(
                    f"{label} cloud: brightness({INTEGRATION_RADII:.0f}w)="
                    f"{met['counts']:.3e} cnt | N={met['n_atoms']:.3e} atoms | "
                    f"n0={met['density']:.3e} cm^-3 | "
                    f"w=({met['wx_um']:.0f},{met['wy_um']:.0f}) um | "
                    f"gain={met['gain_db']:.1f}dB{flag}"
                )
            logger.info(
                f"{label} fit: 1/e2 wx={fx['w']:.1f}px wy={fy['w']:.1f}px | "
                f"centre=({fx['center']:.1f},{fy['center']:.1f}) | "
                f"peak={peak:.4f} cnt/px | "
                f"snr x={fx['snr']:.1f} y={fy['snr']:.1f}{flag}"
            )
            table = results_table_html(fx, fy, peak, met, good)

        img_ds.set_levels_hint(lo, hi)
        img_ds.set_profile_hint(
            x_prof, fx['curve'] if fx else None,
            y_prof, fy['curve'] if fy else None,
            table)
        img_ds.set_data(diff)

        # Condensed version of jingfastMOT's "why is it dark" diagnostics.
        area = float(diff.sum(dtype=np.float64))
        noise_area = float(diff.std()) * np.sqrt(diff.size)
        loc = localized_signal(diff)
        logger.info(
            f"{label} raw f1={float(frame1.mean()):.0f} "
            f"f2={float(frame2.mean()):.0f} "
            f"peak={cam['run_peak']}/{cam['limit']} | "
            f"area={area:.3e} ({area / max(noise_area, 1):+.0f}sig) | "
            + (f"cloud {loc['peak'] - loc['background']:+.3f}/px "
               f"({loc['excess_sigma']:+.1f}sig) at ({loc['x']},{loc['y']}) | "
               if loc else "")
            + f"levels=({lo:.0f},{hi:.0f})"
        )
        if cam['run_peak'] is not None and cam['limit'] \
                and cam['run_peak'] < 0.02 * cam['limit']:
            logger.warn(f"{label}: sensor is nearly dark -- no light reaching it.")
        elif area < -5.0 * noise_area:
            # Also what a frame pair that slipped by one looks like (frame 2
            # of one cycle paired with frame 1 of the next), e.g. after a
            # dropped frame on this camera.
            logger.warn(
                f"{label}: signal is NEGATIVE ({area / max(noise_area, 1):.0f} "
                f"sigma) -- frame 2 brighter than frame 1, the pair is swapped."
            )

    def grab(cam, timeout_ms, phase):
        """One frame from one camera, or None. Drops a camera that keeps missing."""
        try:
            frame = get_frame(cam, timeout_ms)
            cam['misses'] = 0
            return frame
        except Exception as exc:
            if not thread.running:
                return None
            cam['misses'] += 1
            if cam['misses'] == 1:
                logger.warn(f"{cam['label']}: no frame on {phase} ({exc}).")
            if cam is not cams[0] and cam['misses'] >= MAX_CONSECUTIVE_MISSES:
                cam['alive'] = False
                logger.error(
                    f"{cam['label']}: {cam['misses']} missed grabs in a row -- "
                    f"dropping it for the rest of this run. Check that "
                    f"{cam['trigger_do']} on NI card 2 is wired to its Line0."
                )
            elif cam['misses'] in (5, 20) or cam['misses'] % 50 == 0:
                logger.error(
                    f"{cam['label']}: {cam['misses']} missed grabs, "
                    f"{cam['pairs']} pairs so far. Check {cam['trigger_do']} "
                    f"on NI card 2 is wired to its Line0."
                )
            return None

    def grab_all(phase, first_timeout_ms):
        """Same frame from every live camera. The first one waits for the
        trigger; the rest were exposed by the same edge, so they get a short
        timeout."""
        frames = {}
        timeout = first_timeout_ms
        for cam in cams:
            if not cam['alive'] or not thread.running:
                continue
            frame = grab(cam, timeout, phase)
            if frame is not None:
                frames[cam['id']] = frame
                timeout = SECONDARY_TIMEOUT_MS
        return frames

    try:
        for cam in cams:
            setup_camera(cam)

        card2_is_master = (CARD2_DEV == CLOCK_MASTER_DEV)
        NI_card_2.set_clock(
            master_dev=CLOCK_MASTER_DEV,
            source="internal" if card2_is_master else None)
        NI_card_3.set_clock(
            master_dev=CLOCK_MASTER_DEV,
            source="internal" if not card2_is_master else None)
        logger.info(
            f"Clock master: {CLOCK_MASTER_DEV} "
            f"({'card 2 DO' if card2_is_master else 'card 3 AO'}), exporting "
            f"PXI_Trig7; the other card slaves to it."
        )

        NI_card_2.build_stack()
        # Every trigger line is written -- active cameras get the pattern,
        # the rest a flat zero -- so the lines never float.
        active_lines = {c['trigger_do'] for c in cams}
        for line in sorted({c['trigger_do'] for c in CAMERAS}):
            NI_card_2.set_do_voltage(
                do_channel=line,
                value=cam_trig if line in active_lines else [0] * N,
                sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio1", value=ado, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio2", value=cdo, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio3", value=rdo, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio4", value=lat_off_do, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio5", value=lat_off_do, sample_rate=ni_rate)
        NI_card_2.set_do_voltage(do_channel="dio6", value=[0] * N, sample_rate=ni_rate)

        NI_card_3.build_stack()
        NI_card_3.set_ao_voltage(ao_channel="ao1", voltages=aao, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao2", voltages=cao, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao3", voltages=mvco, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao4", voltages=rvco, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao5", voltages=rao, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao6", voltages=lat_off_ao, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao7", voltages=lat_off_ao, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao8", voltages=[bias_x_v] * N, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao9", voltages=[bias_y_v] * N, sample_rate=ni_rate)
        NI_card_3.set_ao_voltage(ao_channel="ao10", voltages=[bias_z_v] * N, sample_rate=ni_rate)

        # Slave first, master last (see jingfastMOT).
        def arm_card(card, name, is_master):
            try:
                return card.arm(regeneration=True)
            except Exception as exc:
                role = "MASTER" if is_master else "SLAVE"
                logger.error(
                    f"{name} failed to arm as clock {role}: {exc}. See "
                    f"jingfastMOT.arm_card for the usual causes."
                )
                raise

        if card2_is_master:
            h2 = arm_card(NI_card_3, "NI card 3 (AO)", False)
            h1 = arm_card(NI_card_2, "NI card 2 (DO)", True)
        else:
            h1 = arm_card(NI_card_2, "NI card 2 (DO)", False)
            h2 = arm_card(NI_card_3, "NI card 3 (AO)", True)
        describe_arm("NI card 2 (DO)", h1)
        describe_arm("NI card 3 (AO)", h2)

        # Discard the first pair on every camera, so they start in step.
        logger.info("Discarding first frame pair on each camera")
        for phase in ("first frame 1", "first frame 2"):
            got = grab_all(phase, frame_timeout_ms + 5000)
            for cam in cams:
                if cam['id'] not in got and cam['alive']:
                    logger.error(
                        f"{cam['label']}: no frame within "
                        f"{frame_timeout_ms + 5000} ms of arming. Nothing is "
                        f"triggering it -- check {cam['trigger_do']} -> Line0."
                    )
        cam_by_id = {c['id']: c for c in cams}

        while thread.running:
            first = grab_all("frame 1", frame_timeout_ms)
            if not thread.running:
                break
            if not first:
                continue
            second = grab_all("frame 2", frame_timeout_ms)
            if not thread.running:
                break
            for cid, frame1 in first.items():
                frame2 = second.get(cid)
                if frame2 is None:
                    continue
                cam = cam_by_id[cid]
                cam['pairs'] += 1
                process_pair(cam, frame1, frame2)
    finally:
        for cam in cams:
            if cam['streaming']:
                try:
                    cam['client'].stop_acquisition()
                except Exception:
                    pass
        if h1 is not None:
            try:
                NI_card_2.finalize(h1, timeout=120.0, force_finish=True)
            except Exception:
                pass
        if h2 is not None:
            try:
                NI_card_3.finalize(h2, timeout=120.0, force_finish=True)
            except Exception:
                pass
        park_outputs(NI_card_2, NI_card_3, logger)
        # Restore default clock routing so the next OPX-clocked script can
        # claim PXI_Trig7 (see jingfastMOT).
        for card, label in ((NI_card_2, 'card 2'), (NI_card_3, 'card 3')):
            try:
                card.reset_clock()
            except Exception as exc:
                logger.error(
                    f"Could not restore default clock routing on {label}: {exc}. "
                    f"It may still own PXI_Trig7 -- the next OPX-clocked script "
                    f"will fail to arm its clock until the NI servers restart."
                )
