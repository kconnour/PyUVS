from astropy.io import fits
import numpy as np

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
