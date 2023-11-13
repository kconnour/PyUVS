import warnings
import numpy as np

import pyuvs as pu


def _make_muv_flatfield(spatial_bin_edges: np.ndarray, spectral_bin_edges: np.ndarray) -> np.ndarray:
    original_flatfield = np.load('/mnt/science/data_lake/mars/maven/iuvs/instrument/muv_flatfield_v1.npy')

    spatial_bins = spatial_bin_edges.shape[0] - 1
    spectral_bins = spectral_bin_edges.shape[0] - 1

    new_flatfield = np.zeros((spatial_bins, spectral_bins))
    for spatial_bin in range(spatial_bins):
        for spectral_bin in range(spectral_bins):
            new_flatfield[spatial_bin, spectral_bin] = np.mean(
                original_flatfield[spatial_bin_edges[spatial_bin]: spatial_bin_edges[spatial_bin + 1],
                spectral_bin_edges[spectral_bin]: spectral_bin_edges[spectral_bin + 1]])
    return new_flatfield


def _make_gain_correction(dark_subtracted, spatial_bin_width, spectral_bin_width, integration_time, mcp_volt, mcp_gain):
    """

    Parameters
    ----------
    dark_subtracted: np.ndarray
        The detector dark subtracted
    spatial_bin_width: int
        The number of detector pixels in a spatial bin
    spectral_bin_width: int
        The number of detector pixels in a spectral bin
    integration_time
    mcp_volt
    mcp_gain

    Returns
    -------

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        volt_array = np.load('/mnt/science/data_lake/mars/maven/iuvs/instrument/voltage.npy')
        ab = np.load('/mnt/science/data_lake/mars/maven/iuvs/instrument/voltage_fit_coefficients.npy')
        ref_mcp_gain = 50.909455

        normalized_img = dark_subtracted / integration_time / spatial_bin_width / spectral_bin_width

        a = np.interp(mcp_volt, volt_array, ab[:, 0])
        b = np.interp(mcp_volt, volt_array, ab[:, 1])

        norm_img = np.exp(a + b * np.log(normalized_img))
        return norm_img / normalized_img * mcp_gain / ref_mcp_gain


def make_brightness(dark_subtracted: np.ndarray, spatial_bin_edges: np.ndarray, spectral_bin_edges: np.ndarray,
                    spatial_bin_width: int, spectral_bin_width: int, integration_time: np.ndarray,
                    mcp_voltage: np.ndarray, mcp_voltage_gain: np.ndarray, wavelength_center: np.ndarray) -> np.ndarray:
    """This will make the calibrated brightness (kR) of a data file.

    Parameters
    ----------
    dark_subtracted
    spatial_bin_edges
    spectral_bin_edges
    spatial_bin_width
    spectral_bin_width
    integration_time
    mcp_voltage
    mcp_voltage_gain

    Returns
    -------

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if dark_subtracted.size > 0:
            # Get the flatfield
            flatfield = _make_muv_flatfield(spatial_bin_edges, spectral_bin_edges)

            # The sensitivity curve is currently 512 elements. Make it (1024,) for simplicity
            sensitivity_data = np.load('/mnt/science/data_lake/mars/maven/iuvs/instrument/muv_sensitivity_curve_observational.npy')
            sensitivity_curve = sensitivity_data[1]
            sensitivity_wavelengths = sensitivity_data[0]
            #sensitivity_curve = np.repeat(sensitivity_curve, 2)

            # TODO: I'm still unsure about the sensitivity curve. If I do it without the wavelengths, it's not spatial bin dependent but it seems smarter.
            #  If I use Justin's wavelength centers, I get an answer that's (133, 19) instead of just (19,)

            # Get the sensitivity in each spectral bin
            # For array shape reasons, I spread this out over several lines
            #rebinned_sensitivity_curve = np.array([np.mean(sensitivity_curve[spectral_bin_edges[i]:spectral_bin_edges[i + 1]]) for i in range(spectral_bin_edges.shape[0] - 1)])
            rebinned_sensitivity_curve = np.interp(wavelength_center, sensitivity_wavelengths, sensitivity_curve)
            partial_corrected_brightness = dark_subtracted / rebinned_sensitivity_curve * 4 * np.pi * 10 ** -9 / pu.pixel_angular_size / spatial_bin_width / mcp_voltage_gain / integration_time

            # Finally, do the voltage gain and flatfield corrections
            # Dividing a spatial bin by its width, is the brightness of a detector pixel---which corresponds to the pixel angular size
            voltage_correction = _make_gain_correction(dark_subtracted, spatial_bin_width, spectral_bin_width, integration_time, mcp_voltage, mcp_voltage_gain)
            # TODO: I think this should be multiply by the gain correction, but the model says divide by the gain correction instead
            data = partial_corrected_brightness * voltage_correction / flatfield

            # If the data have negative DNs, then they become NaNs during the voltage correction
            data[np.isnan(data)] = 0
        else:
            data = np.array([])
        return data
