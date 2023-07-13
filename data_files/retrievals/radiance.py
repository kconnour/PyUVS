"""This script will make a l1c file for a given orbit using a reference solar spectrum.
"""
import datetime
import math
from pathlib import Path
import warnings
import multiprocessing as mp

from astropy.io import fits
from h5py import File
import numpy as np
from scipy.constants import Planck, speed_of_light
from scipy.integrate import quadrature
from scipy.io import readsav
import mars_time as mt
from netCDF4 import Dataset

import data_files.apsis.pyuvs as pu


def make_gain_correction(det_dark_subtracted, spa_size, spe_size, integration_time, mcp_volt, mcp_gain):
    """

    Parameters
    ----------
    det_dark_subtracted: np.ndarray
        The detector dark subtacted
    spa_size: int
        The number of detector pixels in a spatial bin
    spe_size: int
        The number of detector pixels in a spectral bin
    integration_time
    mcp_volt
    mcp_gain

    Returns
    -------

    """
    volt_array = np.load('/mnt/science/data_lake/mars/maven/iuvs/instrument/voltage.npy')
    ab = np.load('/mnt/science/data_lake/mars/maven/iuvs/instrument/voltage_fit_coefficients.npy')
    ref_mcp_gain = 50.909455

    normalized_img = det_dark_subtracted / integration_time / spa_size / spe_size

    a = np.interp(mcp_volt, volt_array, ab[:, 0])
    b = np.interp(mcp_volt, volt_array, ab[:, 1])

    norm_img = np.exp(a + b * np.log(normalized_img))
    return norm_img / normalized_img * mcp_gain / ref_mcp_gain


def process_file(fileno):
    # Open the .fits file and read in relevant data
    hdul = fits.open(data_files[fileno])
    dds = hdul['detector_dark_subtracted'].data
    sza = hdul['pixelgeometry'].data['pixel_solar_zenith_angle']
    spectral_bin_width: int = int(np.median(hdul['binning'].data['spebinwidth'][0]))  # bins
    spatial_bin_width: int = int(np.median(hdul['binning'].data['spabinwidth'][0]))  # bins
    spectral_bin_low = hdul['binning'].data['spepixlo'][0, :]  # bin number
    spectral_bin_high = hdul['binning'].data['spepixhi'][0, :]  # bin number
    voltage: float = hdul['observation'].data['mcp_volt'][0]
    voltage_gain: float = hdul['observation'].data['mcp_gain'][0]
    integration_time: float = hdul['observation'].data['int_time'][0]

    # Make the voltage correction
    voltage_correction = make_gain_correction(dds, spatial_bin_width, spectral_bin_width, integration_time, voltage, voltage_gain)

    # Read in Justin's wavelengths
    wavelength_center = readsav(str(wavelength_files[fileno]))['wavelength_muv']  # shape: (50, 20)
    wavelength_low = readsav(str(wavelength_files[fileno]))['wavelength_muv_lo']  # shape: (50, 20)
    wavelength_high = readsav(str(wavelength_files[fileno]))['wavelength_muv_hi']  # shape: (50, 20)

    # Make wavelength edges from these data
    wavelength_edges = np.zeros((wavelength_low.shape[0], wavelength_low.shape[1]+1))  # shape: (50, 21)
    wavelength_edges[:, :-1] = wavelength_low
    wavelength_edges[:, -1] = wavelength_high[:, -1]
    spectral_bin_edges = np.concatenate((spectral_bin_low, np.array([spectral_bin_high[-1]])))

    # Don't trust the primary structure--do it myself. This will make kR, not kR/nm
    sensitivity_curve = np.load('/mnt/science/data_lake/mars/maven/iuvs/instrument/muv_sensitivity_curve_manufacturer.npy')   # shape: (512, 2)
    rebinned_sensitivity_curve = np.interp(wavelength_center, sensitivity_curve[:, 0], sensitivity_curve[:, 1]) # shape: wavelength_center.shape
    kR = dds / rebinned_sensitivity_curve * 4 * np.pi * 10**-9 / voltage_gain / pu.pixel_omega / integration_time / spatial_bin_width * voltage_correction

    # Load in the point spread function
    psf = np.load('/mnt/science/data_lake/mars/maven/iuvs/instrument/muv_point_spread_function.npy')

    # Turn the flux into kR
    solar_flux = solar_spectrum * (solar_wavelengths * 10**-9) / (Planck * speed_of_light) * (4 * np.pi / 10**10) / 1000

    # Get the solar flux
    def solar(wav):
        return np.interp(wav, solar_wavelengths, solar_flux)

    def integrate_solar(low, high):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return quadrature(solar, low, high)[0]

    # Make a reflectance array that I'll populate. Then populate it
    reflectance = np.zeros(kR.shape)
    for spabin in range(reflectance.shape[1]):
        # Compute the wavelengths on a 1024 grid (but don't take all 1024 pixels since some are in the keyholes)
        pixel_edges = np.arange(spectral_bin_edges[0], spectral_bin_edges[-1] + 1)
        pixel_edge_wavelengths = np.interp(pixel_edges, spectral_bin_edges, wavelength_edges[spabin, :])

        # Integrate the solar flux / rebin it to the number of spectral bins used in the observation
        intsolar = np.array([integrate_solar(pixel_edge_wavelengths[i], pixel_edge_wavelengths[i+1]) for i in range(len(pixel_edge_wavelengths)-1)])

        # multiply the flux by 1/R**2, where 1/R = radius
        intsolar *= radius ** 2

        # Convolve the flux by the PSF
        convolved_flux = np.convolve(intsolar, psf, mode='same')
        edge_indices = spectral_bin_edges - spectral_bin_edges[0]

        rebinned_solar_flux = np.array([np.sum(convolved_flux[edge_indices[i]: edge_indices[i+1]]) for i in range(reflectance.shape[2])])

        for integration in range(reflectance.shape[0]):
            # This gives reflectance
            #reflectance[integration, spabin, :] = \
            #    kR[integration, spabin, :] * np.pi / np.cos(np.radians(sza[integration, spabin])) / rebinned_solar_flux

            # This gives radiance
            reflectance[integration, spabin, :] = \
                kR[integration, spabin, :] * np.pi / rebinned_solar_flux

    # Save the reflectance
    filename = save_location / block_code / f'{orbit_code}-{fileno}.npy'
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(filename), reflectance)


if __name__ == '__main__':
    for orbit in range(3000, 4000):
        iuvs_data_location = Path('/mnt/science/data_lake/mars/maven/iuvs/production')
        wavelength_location = Path('/mnt/science/data_lake/mars/maven/iuvs/apoapse_wavelengths')
        save_location = Path('/mnt/science/data/mars/maven/iuvs')

        orbit_code = 'orbit' + f'{orbit}'.zfill(5)
        block_code = 'orbit' + f'{math.floor(orbit / 100) * 100}'.zfill(5)

        apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
        mars_sun_distance = apsis_file['apoapse/mars_sun_distance'][orbit]

        # Compute the Mars radius ratio
        earth_sun_distance = 1.496e8  # km
        radius = earth_sun_distance / mars_sun_distance

        # Get the Earth date of the orbit
        mars_year = apsis_file['apoapse/mars_year'][orbit]
        sol = apsis_file['apoapse/sol'][orbit]
        marstime = mt.MarsTime(mars_year, sol)
        dt = mt.marstime_to_datetime(marstime)

        day_index = dt.timetuple().tm_yday - 1

        if (dt - datetime.datetime(2020, 2, 1)).days < 0:
            solstice = Dataset(f'/mnt/science/data_lake/sun/solstice/SORCE_SOLSTICE_L3_HR_V18_{dt.year}.nc')
            solar_wavelengths = solstice.variables['standard_wavelengths'][:]
            solar_spectrum = solstice.variables['irradiance'][day_index, :]
        # TODO: use TSIS spectrum here
        else:
            continue

        data_files = sorted((iuvs_data_location / block_code).glob(f'*apoapse*{orbit_code}*muv*.gz'))
        wavelength_files = sorted((wavelength_location / block_code).glob(f'*apoapse*{orbit_code}*muv*.gz'))

        n_cpus = mp.cpu_count()  # = 8 for my old desktop, 12 for my laptop, 20 for my new desktop
        pool = mp.Pool(n_cpus - 2)  # save one/two just to be safe. Some say it's faster

        # NOTE: if there are any issues in the argument of apply_async, it'll break out of that and move on to the next iteration.
        for filenumber in range(len(data_files)):
            pool.apply_async(process_file, args=(filenumber,))
        pool.close()
        pool.join()
