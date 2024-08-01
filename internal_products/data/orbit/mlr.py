from pathlib import Path

from astropy.io import fits
import numpy as np
import statsmodels.api as sm


from binning import make_spectral_bin_width, make_spectral_bin_edges
from detector import make_brightness, make_random_uncertainty


hdulist = fits.hdu.hdulist.HDUList

kR: float = 10**9 / (4 * np.pi)
"""Definition of the kilorayleigh [photons/second/m**2/steradian]."""

'''co_cameron_bands = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates/co_cameron_bands.npy')
cop_1ng = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates/co+_first_negative.npy')
co2p_fdb = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates/co2+_fox_duffendack_barker.npy')
co2p_uvd = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates/co2+_ultraviolet_doublet.npy')
n2vk = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates/nitrogen_vegard_kaplan.npy')
no_nightglow = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates/no_nightglow.npy')
oxygen_2972 = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates/oxygen_2972.npy')
solar_continuum = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates/solar_continuum.npy')'''


'''def fit_muv_templates_to_nightside_data(hduls: list[hdulist]) -> np.ndarray:
    dark_subtracted = make_detector_dark_subtracted(hduls)
    if not dark_subtracted.shape[0]:
        return np.array([])
    uncertainty = make_random_uncertainty_dn(hduls)
    pixels_per_spatial_bin = make_spatial_bin_width(hduls)[0]
    pixels_per_spectral_bin = make_spectral_bin_width(hduls)[0]
    starting_spectral_pixel = make_spectral_bin_edges(hduls)[0]
    integration_time = make_integration_time(hduls)[0]   # 0 cause they're all the same value
    voltage_gain = make_mcp_voltage_gain(hduls)[0]  # 0 cause they're all the same value

    # Get the data bins
    n_integrations = detector_image_dark_subtracted.shape[0]
    n_spatial_bins = detector_image_dark_subtracted.shape[1]
    n_spectral_bins = detector_image_dark_subtracted.shape[2]

    # Compute the wavelengths on the binning scheme used to acquire data
    nominal_wavelength_edges = np.linspace(173.92304103, 341.52213187, num=1025)
    rebinned_wavelength_centers = nominal_wavelength_edges[pixels_per_spectral_bin // 2::pixels_per_spectral_bin]  # this is exact if we assume the wavelength scale is linear
    wavelength_width = (341.52213187 - 173.92304103) / 1025 * pixels_per_spectral_bin

    # Compute the sensitivity curve on this binning scheme
    sensitivity_curve = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/instrument/muv_sensitivity_curve_observational.npy')
    rebinned_sensitivity_curve = np.interp(
        rebinned_wavelength_centers,
        sensitivity_curve[0],
        sensitivity_curve[1])

    # Compute the calibration curve on this binning scheme
    bin_omega = pu.pixel_angular_size * pixels_per_spatial_bin
    rebinned_sensitivity_curve = wavelength_width * voltage_gain * integration_time * kR * \
                                 rebinned_sensitivity_curve * bin_omega

    # Pad NaNs to the data
    spectral_scheme = 1024 // pixels_per_spectral_bin
    binned_starting_index = starting_spectral_index // pixels_per_spectral_bin
    data_shape = (detector_image_dark_subtracted.shape[:-1] + (spectral_scheme,))

    spectra = np.zeros(data_shape) * np.nan
    spectra[..., binned_starting_index: binned_starting_index + n_spectral_bins] = detector_image_dark_subtracted
    uncertainty = np.zeros(data_shape) * np.nan
    uncertainty[..., binned_starting_index: binned_starting_index + n_spectral_bins] = detector_image_uncertainty

    # Rebin the templates to the data's binning
    templates = np.vstack([
        no_nightglow,
        co_cameron_bands,
        co2p_uvd,
        oxygen_2972,
        co2p_fdb,
        n2vk,
        cop_1ng,
        solar_continuum])

    reshaped_templates = np.reshape(templates, (templates.shape[0], templates.shape[1]//pixels_per_spectral_bin, pixels_per_spectral_bin))
    binned_templates = np.sum(reshaped_templates, axis=-1).T
    binned_templates = sm.add_constant(binned_templates)

    # Fit templates to the data
    brightnesses = np.zeros((len(templates) + 1,) + detector_image_dark_subtracted.shape[:-1]) * np.nan   # templates + 1 cause I'm fitting a constant term + each template
    for integration in range(n_integrations):
        for spatial_bin in range(n_spatial_bins):
            fit = sm.WLS(spectra[integration, spatial_bin, :], binned_templates,
                         weights=1 / uncertainty[integration, spatial_bin, :] ** 2,
                         missing='drop').fit()  # This ignores NaNs
            coeff = fit.params
            for species in range(len(templates) + 1):
                brightnesses[species, integration, spatial_bin] = np.sum(coeff[species] * binned_templates[:, species] * wavelength_width / rebinned_sensitivity_curve)

    return brightnesses'''


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
    p = Path('/mnt/science/mars/missions/maven/instruments/iuvs/spectral_templates')
    templates = np.vstack([
        np.genfromtxt(p / 'co-cameron-bands_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'cop_1ng_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'co2p_fdb_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'co2p_uvd_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'n2_vk_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'no_nightglow_delta_bands_calibrated_1024-bins.dat') * 2/3 +
        np.genfromtxt(p / 'no_nightglow_gamma_bands-all_calibrated_1024-bins.dat') * 1/3,
        np.genfromtxt(p / 'o2972_calibrated_1024-bins.dat'),
        np.genfromtxt(p / 'solar_continuum_calibrated_1024-bins.dat')
        ])

    reshaped_templates = np.reshape(templates, (templates.shape[0], templates.shape[1]//pixels_per_spectral_bin, pixels_per_spectral_bin))
    binned_templates = np.sum(reshaped_templates, axis=-1).T
    binned_templates = sm.add_constant(binned_templates)

    # Fit templates to the data
    brightnesses = np.zeros((len(templates) + 1,) + brightness.shape[:-1]) * np.nan   # templates + 1 cause I'm fitting a constant term + each template
    for integration in range(n_integrations):
        for spatial_bin in range(n_spatial_bins):
            fit = sm.WLS(spectra[integration, spatial_bin, :], binned_templates,
                         weights=1 / uncertainty[integration, spatial_bin, :] ** 2,
                         missing='drop').fit()  # This ignores NaNs
            coeff = fit.params
            for species in range(len(templates) + 1):
                brightnesses[species, integration, spatial_bin] = np.sum(coeff[species] * binned_templates[:, species])# * wavelength_width)

    return brightnesses
