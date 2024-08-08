import abc
import datetime
from pathlib import Path
import warnings

from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy import units as u
from h5py import File
from netCDF4 import Dataset
import numpy as np
from scipy.constants import Planck, speed_of_light
from scipy.integrate import quadrature

from binning import make_spectral_bin_edges
from detector import make_brightness

hdulist = fits.hdu.hdulist.HDUList


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
        self._base_path = Path('/mnt/science/sun/solstice')
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
        self._base_path = Path('/mnt/science/sun/tsis-1')
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
        idx = np.abs(times - jd).argmin()
        return self.dataset['irradiance'][idx, :]

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


def make_radiance(hduls: list[hdulist], file: File) -> np.ndarray:
    """Make the radiance (I/F) for this observation.

    Parameters
    ----------
    hduls
        The list of relevant .fits files
    file
        The .hdf5 file

    Returns
    -------
    np.ndarray
        An array of the radiance for each integration, spatial bin, and spectral
        bin.

    """
    # Make some data. Note that my brightness function returns the brightness in
    # kR, not kR/nm (like in the .fits files). This makes computing the radiance
    # quite a bit simpler
    brightness = make_brightness(hduls)
    spectral_bin_edges = make_spectral_bin_edges(hduls)

    # TODO: read in the wavelength edges. This should be the same shape as
    #  brightness, but with one additional element on the spectral axis
    #  (i.e. if brightness is (500, 133, 19), it should be (500, 133, 20)
    #  Below (commented out) is how I did it from Justin's .sav files
    '''wavelength_center = readsav(justins_files[fileno])['wavelength_muv']  # shape: (50, 20)
    wavelength_low = readsav(justins_files[fileno])['wavelength_muv_lo']  # shape: (50, 20)
    wavelength_high = readsav(justins_files[fileno])['wavelength_muv_hi']  # shape: (50, 20)

    # Make wavelength edges from these data
    wavelength_edges = np.zeros((wavelength_low.shape[0], wavelength_low.shape[1]+1))  # shape: (50, 21)
    wavelength_edges[:, :-1] = wavelength_low
    wavelength_edges[:, -1] = wavelength_high[:, -1]'''

    # Load in the point spread function
    psf = np.load('/mnt/science/mars/missions/maven/instruments/iuvs/instrument/muv_point_spread_function.npy')

    # Get the Mars-sun distance
    earth_sun_distance = 1.496e8  # km
    mars_sun_distance = file['apoapse/apsis/mars_sun_distance'][:]
    distance_ratio = earth_sun_distance / mars_sun_distance

    # Get the solar spectrum that corresponds to the time of this orbit
    et = file['apoapse/apsis/ephemeris_time'][:]
    timestamp = (Time(2000, format='jyear') + TimeDelta(et * u.s)).iso
    dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    # SOLSTICE went end of life in February 2020. Use that for the solar flux
    # before then and TSIS1 for times after that
    if dt < datetime.datetime(2020, 2, 1):
        solar_flux = Solstice(dt)
    else:
        solar_flux = TSIS1(dt)

    # Make a radiance array that I'll populate. Then populate it
    radiance = np.zeros(brightness.shape)
    for spatial_bin in range(radiance.shape[1]):
        # Compute the wavelengths edges on a 1024 grid, but only keep the ones
        # between the first and last spectral bin (this array will have 800 or so elements).
        detector_spectral_pixel_edges = np.arange(spectral_bin_edges[0], spectral_bin_edges[-1] + 1)
        pixel_edge_wavelengths = np.interp(detector_spectral_pixel_edges, spectral_bin_edges, wavelength_edges[spatial_bin, :])

        # Integrate the solar flux and decrease it by 1/R**2 so that it's what
        # SOLSTICE/TSIS1 would've seen if it were at Mars at had IUVS's detector's spectral grid
        integrated_flux = np.array([solar_flux.integrate_flux(pixel_edge_wavelengths[i], pixel_edge_wavelengths[i+1])
                                    for i in range(len(pixel_edge_wavelengths) - 1)])
        integrated_flux *= distance_ratio ** 2

        # Convolve the flux by IUVS's PSF and rebin it to the spectral resolution
        # IUVS used for this file. This is the best approximation I can think of
        # for what IUVS would've seen if it were looking at the sun at the same
        # time it collected data
        convolved_flux = np.convolve(integrated_flux, psf, mode='same')
        edge_indices = spectral_bin_edges - spectral_bin_edges[0]
        rebinned_solar_flux = np.array([np.sum(convolved_flux[edge_indices[i]: edge_indices[i + 1]]) for i in range(radiance.shape[2])])

        # Finally, make the radiance. The radiance as defined by Hapke (2012)
        # which is the one used in the radiative transfer code, is pi * I / F
        for integration in range(radiance.shape[0]):
            radiance[integration, spatial_bin] = brightness[integration, spatial_bin, :] * np.pi / rebinned_solar_flux

    return radiance
