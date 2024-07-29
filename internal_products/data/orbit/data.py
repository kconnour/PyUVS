import warnings

from astropy.io import fits
from h5py import File
import numpy as np

import pyuvs as pu


hdulist = fits.hdu.hdulist.HDUList


def add_leading_axis_if_necessary(data: np.ndarray, expected_axes: int) -> np.ndarray:
    """Add a leading axis to an array such that it has the expected number of axes.

    Parameters
    ----------
    data: np.ndarray
        Any array
    expected_axes
        The expected number of axes the array should have

    Returns
    -------
    np.ndarray
       The original data with an empty, leading axis added if necessary

    Notes
    -----
    I assume the IUVS data can only be smaller than the expected number of
    dimensions by up to one dimension.

    """
    return data if np.ndim(data) == expected_axes else data[None, :]


def get_integrations_per_file(hduls: list[hdulist]) -> list[int]:
    return [f['integration'].data['et'].shape[0] for f in hduls]


def make_spatial_bin_latitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_corner_lat'], 3) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_longitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_corner_lon'], 3) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_tangent_altitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_corner_mrh_alt'], 3) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_tangent_altitude_rate(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_corner_mrh_alt_rate'], 3) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_line_of_sight(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_corner_los'], 3) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_solar_zenith_angle(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_solar_zenith_angle'], 2) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_emission_angle(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_emission_angle'], 2) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_phase_angle(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_phase_angle'], 2) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_zenith_angle(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_zenith_angle'], 2) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_local_time(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_local_time'], 2) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_right_ascension(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_corner_ra'], 3) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_declination(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_corner_dec'], 3) for f in hduls]) if hduls else np.array([])


def make_spatial_bin_vector(hduls: list[hdulist]) -> np.ndarray:
    # original shape: (n_integrations, 3, spatial_bins, 5)
    # new shape: (n_integrations, n_spatial_bins, 5, 3)
    return np.moveaxis(np.concatenate([add_leading_axis_if_necessary(f['pixelgeometry'].data['pixel_vec'], 4) for f in hduls]), 1, -1) if hduls else np.array([])


def make_subsolar_latitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['sub_solar_lat'] for f in hduls]) if hduls else np.array([])


def make_subsolar_longitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['sub_solar_lon'] for f in hduls]) if hduls else np.array([])


def make_subspacecraft_latitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['sub_spacecraft_lat'] for f in hduls]) if hduls else np.array([])


def make_subspacecraft_longitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['sub_spacecraft_lon'] for f in hduls]) if hduls else np.array([])


def make_spacecraft_altitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['spacecraft_alt'] for f in hduls]) if hduls else np.array([])


def make_spacecraft_velocity_inertial_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['v_spacecraft_rate_inertial'] for f in hduls]) if hduls else np.array([])


def make_spacecraft_vector(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['v_spacecraft'] for f in hduls]) if hduls else np.array([])


def make_instrument_x_field_of_view(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['vx_instrument'] for f in hduls]) if hduls else np.array([])


def make_instrument_y_field_of_view(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['vy_instrument'] for f in hduls]) if hduls else np.array([])


def make_instrument_z_field_of_view(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['vz_instrument'] for f in hduls]) if hduls else np.array([])


def make_instrument_x_field_of_view_inertial_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['vx_instrument_inertial'] for f in hduls]) if hduls else np.array([])


def make_instrument_y_field_of_view_inertial_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['vy_instrument_inertial'] for f in hduls]) if hduls else np.array([])


def make_instrument_z_field_of_view_inertial_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['vz_instrument_inertial'] for f in hduls]) if hduls else np.array([])


def make_instrument_sun_angle(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['spacecraftgeometry'].data['inst_sun_angle'] for f in hduls]) if hduls else np.array([])


def make_app_flip(hduls: list[hdulist]) -> np.ndarray:
    x_field_of_view = make_instrument_x_field_of_view(hduls)
    spacecraft_velocity_inertial_frame = make_spacecraft_velocity_inertial_frame(hduls)
    try:
        dot = x_field_of_view * spacecraft_velocity_inertial_frame
        app_flip = np.array([np.sum(dot) > 0])
    except IndexError:
        app_flip = np.array([])
    return app_flip


def make_ephemeris_time(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['integration'].data['et'] for f in hduls]) if hduls else np.array([])


def make_mirror_data_number(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['integration'].data['mirror_dn'] for f in hduls]) if hduls else np.array([])


def make_mirror_angle(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['integration'].data['mirror_deg'] for f in hduls]) if hduls else np.array([])


def make_field_of_view(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['integration'].data['fov_deg']for f in hduls]) if hduls else np.array([])


def make_case_temperature(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['integration'].data['case_temp_c'] for f in hduls]) if hduls else np.array([])


def make_integration_time(hduls: list[hdulist]) -> np.ndarray:
    integrations_per_file = get_integrations_per_file(hduls)
    integration_time = [f['observation'].data['int_time'][0] for f in hduls]
    return np.repeat(integration_time, integrations_per_file)


def make_detector_temperature(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate([f['integration'].data['det_temp_c'] for f in hduls]) if hduls else np.array([])


def make_mcp_voltage(hduls: list[hdulist]) -> np.ndarray:
    integrations_per_file = get_integrations_per_file(hduls)
    return np.concatenate([np.repeat(f['observation'].data['mcp_volt'][0], integrations_per_file[c]) for c, f in enumerate(hduls)]) if hduls else np.array([])


def make_mcp_voltage_gain(hduls: list[hdulist]) -> np.ndarray:
    integrations_per_file = get_integrations_per_file(hduls)
    return np.concatenate([np.repeat(f['observation'].data['mcp_gain'][0], integrations_per_file[c]) for c, f in enumerate(hduls)]) if hduls else np.array([])


def make_swath_number(orbit: int, hduls: list[hdulist]) -> np.ndarray:
    mirror_angle = make_mirror_angle(hduls)
    swath_number = compute_swath_number(mirror_angle)

    if orbit in [12093, 14235, 14256, 14838, 15264, 15406]:
        swath_number += 1
    elif orbit in [10381, 14565, 15028, 15192, 15248, 15283, 15425]:
        swath_number += 2
    elif orbit in [14181, 14845, 15035, 15065, 15075, 15156, 15179, 15220, 15351, 15354, 15379, 15397]:
        swath_number += 3
    elif orbit in [3965, 14352, 14361, 14410, 14619, 14872, 15109]:
        swath_number += 4
    elif orbit in [13134, 13229, 14285, 14818, 14823, 15159, 15268, 15288]:
        swath_number += 5
    elif orbit in [14278, 14297, 15070, 15199, 15271, 15281, 15285]:
        swath_number += 6
    elif orbit in [14183, 14327, 14465, 14835, 14837, 14843, 15165, 15394]:
        swath_number += 7

    return swath_number


# TODO: This doesn't account for times when there was a single swath missing in the middle, as with orbit 18150
def compute_swath_number(mirror_angle: np.ndarray) -> np.ndarray:
    """Make the swath number associated with each mirror angle.

    This function assumes the input is all the mirror angles (or, equivalently,
    the field of view) from an orbital segment. Omitting some mirror angles
    may result in nonsensical results. Adding additional mirror angles from
    multiple segments or orbits will certainly result in nonsensical results.

    Parameters
    ----------
    mirror_angle

    Returns
    -------
    np.ndarray
        The swath number associated with each mirror angle.

    Notes
    -----
    This algorithm assumes the mirror in roughly constant step sizes except
    when making a swath jump. It finds the median step size and then uses
    this number to find swath discontinuities. It interpolates between these
    indices and takes the floor of these values to get the integer swath
    number.

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mirror_change = np.diff(mirror_angle)
        threshold = np.abs(np.median(mirror_change)) * 4
        mirror_discontinuities = np.where(np.abs(mirror_change) > threshold)[0] + 1
        if any(mirror_discontinuities):
            n_swaths = len(mirror_discontinuities) + 1
            integrations = range(len(mirror_angle))
            interp_swaths = np.interp(integrations, mirror_discontinuities, range(1, n_swaths), left=0)
            swath_number = np.floor(interp_swaths).astype('int')
        else:
            swath_number = np.zeros(mirror_angle.shape)

    return swath_number


# TODO: would it be easier to set the number of swaths between X and Y to be 6 and Yto Z to be 8? But idk how well this would work with relay swaths
def make_number_of_swaths(orbit: int, hduls: list[hdulist]) -> np.ndarray:
    swath_number = make_swath_number(orbit, hduls)
    number_of_swaths = np.array([swath_number[-1] + 1]) if swath_number.size > 0 else np.array([])
    if orbit in [3115, 3174, 3211, 3229, 3248, 3375, 3488, 4049, 4122, 4141, 4231, 4780, 6525, 11678, 13161, 14208,
                 14275, 15027, 15076, 15150, 15156, 15157, 15267, 15287, 15294, 15310, 15402, 15463, 15491]:
        number_of_swaths += 1
    elif orbit in [3456, 3581, 3721, 6971, 7241, 15029, 15034, 15116, 15123, 15168, 15178, 15219, 15226, 15280, 15308,
                   15327, 15261, 15368, 15383, 15395, 15409, 15429]:
        number_of_swaths += 2
    elif orbit in [14186, 14409, 14845, 15054, 15082, 15089, 15189, 15209, 15274, 15297, 15315, 15400]:
        number_of_swaths += 3
    elif orbit in [7430, 7802, 7876, 8530, 14255, 14439, 15048, 15096, 15103, 15131, 15247]:
        number_of_swaths += 4
    elif orbit in [14836, 14871, 15056, 15227, 15329, 15331, 15422]:
        number_of_swaths += 5
    elif orbit in [13138, 13150, 15374]:
        number_of_swaths += 6
    elif orbit in [14817, 14828, 15172, 15185, 15213, 15345]:
        number_of_swaths += 7
    return number_of_swaths


def make_opportunity_classification(orbit: int, hduls: list[hdulist]) -> np.ndarray:
    mirror_angle = make_mirror_angle(hduls)
    swath_number = make_swath_number(orbit, hduls)
    return compute_opportunity_classification(mirror_angle, swath_number)


def compute_opportunity_classification(mirror_angle: np.ndarray, swath_number: np.ndarray) -> np.ndarray:
    opportunity_integrations = np.empty(swath_number.shape, dtype='bool')
    for sn in np.unique(swath_number):
        angles = mirror_angle[swath_number == sn]
        relay = pu.minimum_mirror_angle in angles and pu.maximum_mirror_angle in angles
        opportunity_integrations[swath_number == sn] = relay
    return opportunity_integrations


def make_failsafe_integrations(hduls: list[hdulist]) -> np.ndarray:
    mcp_voltage = make_mcp_voltage(hduls)
    return np.isclose(mcp_voltage, pu.apoapse_muv_failsafe_voltage)


def make_dayside_integrations(hduls: list[hdulist]) -> np.ndarray:
    failsafe_integrations = make_failsafe_integrations(hduls)
    nightside_integrations = make_nightside_integrations(hduls)
    return np.logical_and(~failsafe_integrations, ~nightside_integrations)


def make_nightside_integrations(hduls: list[hdulist]) -> np.ndarray:
    mcp_voltage = make_mcp_voltage(hduls)
    return mcp_voltage > pu.constants.apoapse_muv_day_night_voltage_boundary


def make_spatial_bin_edges(hduls: list[hdulist]) -> np.ndarray:
    if not hduls:
        return np.array([])
    spatial_pixel_low = np.array([np.squeeze(f['binning'].data['spapixlo']) for f in hduls])
    spatial_pixel_high = np.array([np.squeeze(f['binning'].data['spapixhi']) for f in hduls])
    if not np.all(spatial_pixel_low == spatial_pixel_low[0]) or not np.all(spatial_pixel_high == spatial_pixel_high[0]):
        raise ValueError('The spatial binning is not the same.')
    return np.append(spatial_pixel_low[0], spatial_pixel_high[0, -1] + 1)


def make_spectral_bin_edges(hduls: list[hdulist]) -> np.ndarray:
    if not hduls:
        return np.array([])
    spectral_pixel_low = np.array([np.squeeze(f['binning'].data['spepixlo']) for f in hduls])
    spectral_pixel_high = np.array([np.squeeze(f['binning'].data['spepixhi']) for f in hduls])
    if not np.all(spectral_pixel_low == spectral_pixel_low[0]) or not np.all(spectral_pixel_high == spectral_pixel_high[0]):
        raise ValueError('The spectral binning is not the same.')
    return np.append(spectral_pixel_low[0], spectral_pixel_high[0, -1] + 1)


def make_spatial_bin_width(hduls: list[hdulist]) -> np.ndarray:
    if not hduls:
        return np.array([])
    bin_size = np.array([f['primary'].header['spa_size'] for f in hduls])
    if not np.all(bin_size == bin_size[0]):
        raise ValueError('The spatial bin width is not the same.')
    return np.array([bin_size[0]])


def make_spectral_bin_width(hduls: list[hdulist]) -> np.ndarray:
    if not hduls:
        return np.array([])
    bin_size = np.array([f['primary'].header['spe_size'] for f in hduls])
    if not np.all(bin_size == bin_size[0]):
        raise ValueError('The spectral bin width is not the same.')
    return np.array([bin_size[0]])


def make_detector_raw(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['detector_raw'].data, 3) for f in hduls]) if hduls else np.array([])


def make_detector_dark_subtracted(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['detector_dark_subtracted'].data, 3) for f in hduls]) if hduls else np.array([])


def make_random_uncertainty_dn(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['random_dn_unc'].data, 3) for f in hduls]) if hduls else np.array([])


def make_random_uncertainty_physical(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['random_phy_unc'].data, 3) for f in hduls]) if hduls else np.array([])


def _make_muv_flatfield(spatial_bin_edges: np.ndarray, spectral_bin_edges: np.ndarray) -> np.ndarray:
    original_flatfield = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/instrument/muv_flatfield.npy')

    spatial_bins = spatial_bin_edges.shape[0] - 1
    spectral_bins = spectral_bin_edges.shape[0] - 1

    new_flatfield = np.zeros((spatial_bins, spectral_bins))
    for spatial_bin in range(spatial_bins):
        for spectral_bin in range(spectral_bins):
            new_flatfield[spatial_bin, spectral_bin] = np.mean(
                original_flatfield[spatial_bin_edges[spatial_bin]: spatial_bin_edges[spatial_bin + 1],
                spectral_bin_edges[spectral_bin]: spectral_bin_edges[spectral_bin + 1]])
    return new_flatfield


def _make_gain_correction(dark_subtracted: np.ndarray, spatial_bin_width: int,
                          spectral_bin_width: int, integration_time: np.ndarray,
                          mcp_volt: np.ndarray, mcp_gain: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        voltage_file = File('/mnt/science/mars/missions/maven/instruments/iuvs/instrument/voltage.hdf5')

        voltage = voltage_file['voltage'][:]
        fit_coefficients = voltage_file['fit_coefficients'][:]
        ref_mcp_gain = 50.909455

        normalized_img = dark_subtracted.T / integration_time / spatial_bin_width / spectral_bin_width

        a = np.interp(mcp_volt, voltage, fit_coefficients[:, 0])
        b = np.interp(mcp_volt, voltage, fit_coefficients[:, 1])

        norm_img = np.exp(a + b * np.log(normalized_img))
        return (norm_img / normalized_img * mcp_gain / ref_mcp_gain).T


def make_brightness(dark_subtracted: np.ndarray, spatial_bin_edges: np.ndarray, spectral_bin_edges: np.ndarray,
                    spatial_bin_width: int, spectral_bin_width: int, integration_time: np.ndarray,
                    mcp_voltage: np.ndarray, mcp_voltage_gain: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if dark_subtracted.size > 0:
            # Get the flatfield
            flatfield = _make_muv_flatfield(spatial_bin_edges, spectral_bin_edges)

            # The sensitivity curve is currently 512 elements. Make it (1024,) for simplicity
            sensitivity_curve = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/instrument/muv_sensitivity_curve_observational.npy')[1]
            sensitivity_curve = np.repeat(sensitivity_curve, 2)

            # Get the sensitivity in each spectral bin
            # For array shape reasons, I spread this out over several lines
            rebinned_sensitivity_curve = np.array([np.mean(sensitivity_curve[spectral_bin_edges[i]:spectral_bin_edges[i + 1]]) for i in range(spectral_bin_edges.shape[0] - 1)])
            partial_corrected_brightness = dark_subtracted / rebinned_sensitivity_curve * 4 * np.pi * 10 ** -9 / pu.pixel_angular_size / spatial_bin_width
            partial_corrected_brightness = (partial_corrected_brightness.T / mcp_voltage_gain / integration_time).T

            # Finally, do the voltage gain and flatfield corrections
            voltage_correction = _make_gain_correction(dark_subtracted, spatial_bin_width, spectral_bin_width, integration_time, mcp_voltage, mcp_voltage_gain)
            data = partial_corrected_brightness / flatfield * voltage_correction

            # If the data have negative DNs, then they become NaNs during the voltage correction
            data[np.isnan(data)] = 0
        else:
            data = np.array([])
        return data
