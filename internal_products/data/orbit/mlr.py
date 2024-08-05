from pathlib import Path

from astropy.io import fits
import numpy as np
import statsmodels.api as sm

from binning import make_spectral_bin_width, make_spectral_bin_edges
from detector import make_brightness, make_random_uncertainty


hdulist = fits.hdu.hdulist.HDUList


def fit_muv_templates_to_nightside_data(hduls: list[hdulist]) -> np.ndarray:
    brightness = make_brightness(hduls)
    if not brightness.shape[0]:
        return np.vstack(([], [], [], [], [], [], [], [], []))
    random_uncertainty = make_random_uncertainty(hduls)
    pixels_per_spectral_bin = make_spectral_bin_width(hduls)[0]
    starting_spectral_pixel = make_spectral_bin_edges(hduls)[0]

    # Get the data bins
    n_integrations = brightness.shape[0]
    n_spatial_bins = brightness.shape[1]
    n_spectral_bins = brightness.shape[2]

    # Pad NaNs to the data
    binned_starting_index = starting_spectral_pixel // pixels_per_spectral_bin
    data_shape = (brightness.shape[:-1] + (1024 // pixels_per_spectral_bin,))

    spectra = np.zeros(data_shape) * np.nan
    spectra[..., binned_starting_index: binned_starting_index + n_spectral_bins] = brightness
    uncertainty = np.zeros(data_shape) * np.nan
    uncertainty[..., binned_starting_index: binned_starting_index + n_spectral_bins] = random_uncertainty

    # Rebin the templates to the data's binning
    # NOTE: all templates come from theoretical calculations. The NO curve is
    #  itself an MLR fit of a spectrum to several theoretical curves. I asked
    #  Zac for the coefficients on each curve, and he said they're about 2/3
    #  delta and 1/3 gamma, but the exact coefficient have been lost. The
    #  constant noise is just 1 / sensitivity curve, normalized.
    p = Path('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates')
    templates = np.vstack([
        np.load(p / 'constant_noise.npy'),
        np.genfromtxt(p / 'co-cameron-bands_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'cop_1ng_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'co2p_fdb_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'co2p_uvd_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'n2_vk_calibrated_1024-bins.dat'),
        np.load(p / 'no_nightglow.npy'),
        np.genfromtxt(p / 'o2972_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'solar_continuum_calibrated_1024-bins.dat')
        ])

    reshaped_templates = np.reshape(templates, (templates.shape[0], templates.shape[1]//pixels_per_spectral_bin, pixels_per_spectral_bin))
    binned_templates = np.sum(reshaped_templates, axis=-1).T

    # Fit templates to the data
    brightnesses = np.zeros((len(templates),) + brightness.shape[:-1]) * np.nan
    for integration in range(n_integrations):
        for spatial_bin in range(n_spatial_bins):
            fit = sm.WLS(spectra[integration, spatial_bin, :], binned_templates,
                         weights=1 / uncertainty[integration, spatial_bin, :] ** 2,
                         missing='drop').fit()  # This ignores NaNs
            coeff = fit.params
            for species in range(len(templates)):
                brightnesses[species, integration, spatial_bin] = np.sum(coeff[species] * binned_templates[:, species])

    return brightnesses
