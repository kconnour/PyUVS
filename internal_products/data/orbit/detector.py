import warnings

from astropy.io import fits
from h5py import File
import numpy as np

import pyuvs as pu
from binning import make_spatial_bin_edges, make_spatial_bin_width, \
    make_spectral_bin_edges, make_spectral_bin_width
from integration import make_integration_time, make_mcp_voltage, \
    make_mcp_voltage_gain

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


def make_detector_raw(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['detector_raw'].data, 3) for f in hduls]) if hduls else np.array([])


def make_detector_dark(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['background_dark'].data, 3) for f in hduls]) if hduls else np.array([])


def make_detector_dark_subtracted(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['detector_dark_subtracted'].data, 3) for f in hduls]) if hduls else np.array([])


def make_random_uncertainty_dn(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['random_dn_unc'].data, 3) for f in hduls]) if hduls else np.array([])


def make_random_uncertainty(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['random_phy_unc'].data, 3) for f in hduls]) if hduls else np.array([])


def make_systematic_uncertainty(hduls: list[hdulist]) -> np.ndarray:
    return np.vstack([add_leading_axis_if_necessary(f['systematic_phy_unc'].data, 3) for f in hduls]) if hduls else np.array([])


def make_brightness(hduls: list[hdulist]) -> np.ndarray:
    dark_subtracted = make_detector_dark_subtracted(hduls)
    if not dark_subtracted.shape[0]:
        return np.array([])
    spatial_bin_edges = make_spatial_bin_edges(hduls)
    spatial_bin_width = make_spatial_bin_width(hduls)[0]
    spectral_bin_edges = make_spectral_bin_edges(hduls)
    spectral_bin_width = make_spectral_bin_width(hduls)[0]
    integration_time = make_integration_time(hduls)
    mcp_voltage = make_mcp_voltage(hduls)
    mcp_voltage_gain = make_mcp_voltage_gain(hduls)

    def make_muv_flatfield() -> np.ndarray:
        original_flatfield = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/instrument/muv_flatfield.npy')  # shape: (1024, 1024)

        spatial_bins = spatial_bin_edges.shape[0] - 1
        spectral_bins = spectral_bin_edges.shape[0] - 1

        new_flatfield = np.zeros((spatial_bins, spectral_bins))
        for spatial_bin in range(spatial_bins):
            for spectral_bin in range(spectral_bins):
                new_flatfield[spatial_bin, spectral_bin] = np.mean(
                    original_flatfield[spatial_bin_edges[spatial_bin]: spatial_bin_edges[spatial_bin + 1],
                    spectral_bin_edges[spectral_bin]: spectral_bin_edges[spectral_bin + 1]])
        return new_flatfield

    def make_gain_correction() -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            voltage_file = File('/mnt/science/mars/missions/maven/instruments/iuvs/instrument/voltage.hdf5')

            voltage = voltage_file['voltage'][:]
            fit_coefficients = voltage_file['fit_coefficients'][:]
            ref_mcp_gain = 50.909455

            normalized_img = dark_subtracted.T / integration_time / spatial_bin_width / spectral_bin_width

            a = np.interp(mcp_voltage, voltage, fit_coefficients[:, 0])
            b = np.interp(mcp_voltage, voltage, fit_coefficients[:, 1])

            norm_img = np.exp(a + b * np.log(normalized_img))
            return (norm_img / normalized_img * mcp_voltage_gain / ref_mcp_gain).T

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Make the flatfield for this binning scheme
        flatfield = make_muv_flatfield()

        # The sensitivity curve is currently 512 elements. Make it (1024,) for simplicity
        sensitivity_curve = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/instrument/muv_sensitivity_curve_observational.npy')[1]
        sensitivity_curve = np.repeat(sensitivity_curve, 2)

        # Get the sensitivity in each spectral bin
        # For array shape reasons, I spread this out over several lines
        rebinned_sensitivity_curve = np.array([np.mean(sensitivity_curve[spectral_bin_edges[i]:spectral_bin_edges[i + 1]]) for i in range(spectral_bin_edges.shape[0] - 1)])
        partial_corrected_brightness = dark_subtracted / rebinned_sensitivity_curve * 4 * np.pi * 10 ** -9 / pu.pixel_angular_size / spatial_bin_width
        partial_corrected_brightness = (partial_corrected_brightness.T / mcp_voltage_gain / integration_time).T

        # Finally, do the voltage gain and flatfield corrections
        voltage_correction = make_gain_correction()
        data = partial_corrected_brightness / flatfield * voltage_correction

        # If the data have negative DNs, then they become NaNs during the voltage correction
        data[np.isnan(data)] = 0
        return data
