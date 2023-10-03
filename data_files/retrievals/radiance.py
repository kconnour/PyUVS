#!/home/kyle/repos/PyUVS/venv/bin/python
"""This script will make a l1c file for a given orbit using a reference solar spectrum.
"""
import abc
from datetime import datetime
import math
from pathlib import Path
import warnings
import multiprocessing as mp

from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.io import fits
from h5py import File
import numpy as np
from scipy.constants import Planck, speed_of_light
from scipy.integrate import quadrature
from scipy.io import readsav
from netCDF4 import Dataset

from brightness import make_brightness


class SolarFlux(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_irradiance(self):
        pass

    @abc.abstractmethod
    def get_wavelengths(self):
        pass

    def convert_flux_to_kR(self):
        return self.get_irradiance() * (self.get_wavelengths() * 10**-9) / (Planck * speed_of_light) * (4 * np.pi / 10**10) / 1000

    def integrate_flux(self, low: float, high: float) -> float:
        """Integrate the solar flux and get the kR of each spectral bin.

        Parameters
        ----------
        low
            The low wavelength (nm)
        high
            The high wavelengths (nm)

        Returns
        -------

        """
        wavelengths = self.get_wavelengths()
        kR = self.convert_flux_to_kR()

        def solar(wav):
            return np.interp(wav, wavelengths, kR)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return quadrature(solar, low, high)[0]


class Solstice(SolarFlux):
    def __init__(self, dt: datetime):
        self.dt = dt
        self._base_path = Path('/mnt/science/data_lake/sun/solstice')
        self.dataset = self._load_dataset()
        self._irradiance = self._set_irradiance()
        self._wavelengths = self._set_wavelengths()

    def _load_dataset(self):
        return Dataset(self._base_path / f'SORCE_SOLSTICE_L3_HR_V18_{self.dt.year}.nc')

    def _set_irradiance(self):
        # NOTE: I put the logic in the constructor because I call this method a lot. Each time I call it, it reads in
        #  the wavelengths, which is super duper slow. It may be better to ignore this method and just make a public
        #  attribute.
        return self.dataset.variables['irradiance'][self.dt.timetuple().tm_yday - 1, :]

    def _set_wavelengths(self):
        # NOTE: I put the logic in the constructor because I call this method a lot. Each time I call it, it reads in
        #  the wavelengths, which is super duper slow. It may be better to ignore this method and just make a public
        #  attribute.
        return self.dataset.variables['standard_wavelengths'][:]

    def get_irradiance(self):
        return self._irradiance

    def get_wavelengths(self):
        return self._wavelengths


class TSIS1(SolarFlux):
    def __init__(self, dt: datetime):
        self.dt = dt
        self._base_path = Path('/mnt/science/data_lake/sun/tsis-1')
        self.dataset = self._load_dataset()
        self._irradiance = self._set_irradiance()
        self._wavelengths = self._set_wavelengths()

    def _load_dataset(self):
        return Dataset(self._base_path / f'tsis_ssi_L3_c24h_latest.nc')

    def _set_irradiance(self):
        # TODO: In theory I should adjust this spectrum to match what SOLSTICE would've measured. However, over the
        #  spectral range I care about, the differences are so small that it's not worth the effort right now
        jd = Time(str(self.dt), format='iso').jd
        times = self.dataset['time'][:]
        default_idx = np.abs(times - jd).argmin()
        # Most of the time I can just return the irradiance. But sometimes, the UV is all NaNs. Account for that case here
        # by using the next time the instrument did in fact measure some data. This assumes all the UV is NaNs
        next_valid_index = np.argmax(~np.isnan(self.dataset['irradiance'][default_idx:, 0]))   # This will return 0 if the index has valid data
        return self.dataset['irradiance'][default_idx + next_valid_index, :]

    def _set_wavelengths(self):
        return self.dataset['wavelength'][:]

    def get_irradiance(self):
        # NOTE: I put the logic in the constructor because I call this method a lot. Each time I call it, it reads in
        #  the wavelengths, which is super duper slow. It may be better to ignore this method and just make a public
        #  attribute.
        return self._irradiance

    def get_wavelengths(self):
        # NOTE: I put the logic in the constructor because I call this method a lot. Each time I call it, it reads in
        #  the wavelengths, which is super duper slow. It may be better to ignore this method and just make a public
        #  attribute.
        return self._wavelengths


def make_radiance(orbit: int) -> None:
    ### Start by defining stuff that I only have to do once per orbit

    # Define some paths
    iuvs_data_location = Path('/mnt/science/data_lake/mars/maven/iuvs/production')
    wavelength_location = Path('/mnt/science/data_lake/mars/maven/iuvs/apoapse_wavelengths')
    save_location = Path('/mnt/science/data/mars/maven/iuvs/radiance')

    orbit_code = 'orbit' + f'{orbit}'.zfill(5)
    block_code = 'orbit' + f'{math.floor(orbit / 100) * 100}'.zfill(5)

    # Read in the data paths
    data_files = sorted((iuvs_data_location / block_code).glob(f'*apoapse*{orbit_code}*muv*.gz'))
    wavelength_files = sorted((wavelength_location / block_code).glob(f'*apoapse*{orbit_code}*muv*'))

    # Read in apsis info
    apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
    mars_sun_distance = apsis_file['apoapse/mars_sun_distance'][orbit-1]
    et = apsis_file['apoapse/ephemeris_time'][orbit-1]

    # Define the solar flux
    timestamp = (Time(2000, format='jyear') + TimeDelta(et * u.s)).iso
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    if dt < datetime(2020, 2, 1):
        solar_flux = Solstice(dt)
    else:
        solar_flux = TSIS1(dt)

    earth_sun_distance = 1.496e8  # km
    radius = earth_sun_distance / mars_sun_distance

    # Load in the point spread function
    psf = np.load('/mnt/science/data_lake/mars/maven/iuvs/instrument/muv_point_spread_function.npy')

    for file_number in range(len(data_files)):
        print(f'file number {file_number}')
        # Open the .fits file and read in relevant data
        hdul = fits.open(data_files[file_number])
        dds = hdul['detector_dark_subtracted'].data
        if np.ndim(dds) == 2:
            dds = dds[None, :]
        spatial_bin_width: int = int(np.median(hdul['binning'].data['spabinwidth'][0]))  # bins
        spectral_bin_width: int = int(np.median(hdul['binning'].data['spebinwidth'][0]))  # bins
        spatial_bin_low = hdul['binning'].data['spapixlo'][0, :]  # bin number
        spatial_bin_high = hdul['binning'].data['spapixhi'][0, :]  # bin number
        spectral_bin_low = hdul['binning'].data['spepixlo'][0, :]  # bin number
        spectral_bin_high = hdul['binning'].data['spepixhi'][0, :]  # bin number
        integration_time: float = hdul['observation'].data['int_time'][0]
        voltage: float = hdul['observation'].data['mcp_volt'][0]
        voltage_gain: float = hdul['observation'].data['mcp_gain'][0]

        # Get the wavelengths
        # Read in Justin's wavelengths
        wavelength_low = readsav(str(wavelength_files[file_number]))['wavelength_muv_lo']  # shape: (50, 20)
        wavelength_high = readsav(str(wavelength_files[file_number]))['wavelength_muv_hi']  # shape: (50, 20)
        wavelength_center = readsav(str(wavelength_files[file_number]))['wavelength_muv']  # shape: (50, 20)

        # Make variables for the brightness computation
        spatial_bin_edges = np.concatenate((spatial_bin_low, np.array([spatial_bin_high[-1]])))
        spectral_bin_edges = np.concatenate((spectral_bin_low, np.array([spectral_bin_high[-1]])))

        # Make the brightness
        brightness = make_brightness(dds, spatial_bin_edges, spectral_bin_edges, spatial_bin_width, spectral_bin_width,
                                     integration_time, voltage, voltage_gain, wavelength_center)


        # Make wavelength edges from these data. Note this is clunky and could probably be done in one line,
        #  but it's easiest for now to make an empty array and then populate
        wavelength_edges = np.zeros((wavelength_low.shape[0], wavelength_low.shape[1] + 1))  # shape: (50, 21)
        wavelength_edges[:, :-1] = wavelength_low
        wavelength_edges[:, -1] = wavelength_high[:, -1]

        radiance = np.zeros(dds.shape)
        for spatial_bin in range(radiance.shape[1]):
            print(f'spatial bin {spatial_bin}')
            # Compute the wavelengths on a 1024 grid (but don't take all 1024 pixels since some are in the keyholes)
            # I have to do this because of the PSF
            pixel_edges = np.arange(spectral_bin_edges[0], spectral_bin_edges[-1] + 1)
            pixel_edge_wavelengths = np.interp(pixel_edges, spectral_bin_edges, wavelength_edges[spatial_bin, :])

            # Integrate the solar flux
            integrated_flux = np.array([solar_flux.integrate_flux(pixel_edge_wavelengths[i], pixel_edge_wavelengths[i+1]) for i in range(len(pixel_edge_wavelengths)-1)])
            integrated_flux *= radius ** 2

            # Convolve the flux by the PSF and rebin to IUVS resolution
            convolved_flux = np.convolve(integrated_flux, psf, mode='same')
            edge_indices = spectral_bin_edges - spectral_bin_edges[0]
            rebinned_solar_flux = np.array([np.sum(convolved_flux[edge_indices[i]: edge_indices[i + 1]]) for i in range(radiance.shape[2])])

            for integration in range(radiance.shape[0]):
                radiance[integration, spatial_bin] = brightness[integration, spatial_bin, :] * np.pi / rebinned_solar_flux

        # Save the reflectance
        fn = f'{file_number}'.zfill(2)
        filename = save_location / block_code / f'{orbit_code}-{fn}.npy'
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(filename), radiance)


if __name__ == '__main__':
    n_cpus = mp.cpu_count()  # = 8 for my old desktop, 12 for my laptop, 20 for my new desktop
    pool = mp.Pool(n_cpus - 2)  # save one/two just to be safe. Some say it's faster
    for orb in range(3400, 3401):
        # NOTE: if there are any issues in the argument of apply_async, it'll break out of that and move on to the next iteration.
        make_radiance(orb)
        #pool.apply_async(make_radiance, args=(orb,))
    pool.close()
    pool.join()
