from pathlib import Path
import multiprocessing as mp
import os
from tempfile import mkdtemp
import math

from astropy.io import fits
from h5py import File
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

import pyrt


class Ames:
    def __init__(self):
        self._base_path = Path('/mnt/science/data_lake/mars/gcm/ames/my_generic')

        self._fixed = Dataset(self._base_path / '05344.fixed.nc')
        self._average = Dataset(self._base_path / '05344.atmos_average.nc')
        self._diurnal = Dataset(self._base_path / '05344.atmos_diurn.nc')

        self.latitude = self._fixed.variables['lat'][:]
        self.longitude = self._fixed.variables['lon'][:]

        self.dust_vprof = self._average.variables['dustref']  # shape: (year, pressure level, lat, lon)
        # self.ice_vprof = self._average.variables['cldref']  # shape: (year, pressure level, lat, lon)

        # self.pressure_levels = self._average.variables['pstd']  # shape: (pressure level,). Index 0 is TOA
        self.surface_pressure = self._average.variables['ps']  # shape: (season, time of day, lat, lon)
        self.surface_temperature = self._average.variables['ts']  # shape: (season, time of day, lat, lon)
        self.temperature = self._average.variables['temp'][:]  # shape: (season, time of day, pressure level, lat, lon)
        self.season = self._make_season()
        self.local_time = self._diurnal.variables['time_of_day_24'][:]

        self.ak = self._fixed.variables['ak'][:]
        self.bk = self._fixed.variables['bk'][:]

    def _make_season(self):
        time = self._average.variables['time'][:]
        delta = time[1] - time[0]
        time -= delta / 2
        time -= time[0]
        return time

    def get_seasonal_index(self, sol):
        return np.argmin(np.abs(self.season - sol))

    def get_latitude_index(self, lat: float):
        return np.argmin(np.abs(self.latitude - lat))

    def get_longitude_index(self, lon: float):
        return np.argmin(np.abs(self.longitude - lon))

    def get_local_time_index(self, lon: float, lt):
        return np.argmin(np.abs(self.local_time - (lt - (lon / 360 * 24)) % 24))

    def make_pressure(self):
        return np.multiply.outer(self.surface_pressure, self.bk) + self.ak  # shape: (season, lat, lon, altitude)

    def get_pixel_pressure(self, lat, lon, sol):
        # Make the pressure and temperature vertical profiles from the surface
        lat_idx = self.get_latitude_index(lat)
        lon_idx = self.get_longitude_index(lon)
        season_idx = self.get_seasonal_index(sol)

        surface_pressure = self.surface_pressure[season_idx, lat_idx, lon_idx]
        return np.multiply.outer(surface_pressure, self.bk) + self.ak  # This is a huge performance improvement by not computing all the atmospheric pressures

    def get_pixel_temperature(self, lat, lon, sol):
        lat_idx = self.get_latitude_index(lat)
        lon_idx = self.get_longitude_index(lon)
        season_idx = self.get_seasonal_index(sol)

        surface_temperature = self.surface_temperature[season_idx, lat_idx, lon_idx]

        # Get where the surface index would be
        surface_idx = np.argmin(~self.temperature.mask)

        return np.insert(self.temperature[season_idx, :, lat_idx, lon_idx], surface_idx, surface_temperature)

    def get_pixel_boundary_altitude(self, lat, lon, sol):
        pressure = self.get_pixel_pressure(lat, lon, sol)
        return -np.log(pressure / pressure[-1]) * 10

    def get_dust_profile(self, lat, lon, sol):
        lat_idx = self.get_latitude_index(lat)
        lon_idx = self.get_longitude_index(lon)
        season_idx = self.get_seasonal_index(sol)

        return self._average.variables['dustref'][season_idx, :, lat_idx, lon_idx]

    def get_season_surface_pressure(self, sol):
        season_idx = self.get_seasonal_index(sol)
        return self.surface_temperature[season_idx]


for orbit in range(3400, 3500):
    block = math.floor(orbit / 100) * 100
    block_code = 'orbit' + f'{block}'.zfill(5)
    orbit_code = 'orbit' + f'{orbit}'.zfill(5)
    # Read in external files
    lut = np.load('/mnt/science/data_lake/mars/maven/iuvs/lut.npy')
    radiance_files = sorted(Path(f'/mnt/science/data/mars/maven/iuvs/radiance/{block_code}').glob(f'orbit*{orbit}*'))
    fits_files = sorted(Path(f'/mnt/science/data_lake/mars/maven/iuvs/production/{block_code}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
    hduls = [fits.open(f) for f in fits_files]
    apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')

    # Define save location
    save_location = Path('/mnt/science/data/mars/maven/iuvs/retrievals')

    # Make the sol info
    sol = apsis_file['apoapse/sol'][3453]

    # Make the interpolation
    sza_grid = np.arange(8) * 10
    ea_grid = np.arange(8) * 10
    azimuth_grid = np.arange(19) * 10
    pressure_grid = np.linspace(0.3, 13.3, 10) * 100
    dust_grid = np.arange(21) / 10
    ice_grid = np.arange(11) / 10
    wavelength_grid = np.array([0.210, 0.215, 0.220, 0.272, 0.278, 0.284])

    interp = RegularGridInterpolator((sza_grid, ea_grid, azimuth_grid, pressure_grid, dust_grid, ice_grid, wavelength_grid), lut)

    # Get pressure info
    gcm = Ames()
    season_surface_pressure = gcm.get_season_surface_pressure(sol)
    sfc_interp = RegularGridInterpolator((gcm.latitude, gcm.longitude), season_surface_pressure)

    for counter, radiance in enumerate(radiance_files):
        rad = np.load(radiance)
        szas = hduls[counter]['pixelgeometry'].data['pixel_solar_zenith_angle']
        eas = hduls[counter]['pixelgeometry'].data['pixel_emission_angle']
        pas = hduls[counter]['pixelgeometry'].data['pixel_phase_angle']
        alts = hduls[counter]['pixelgeometry'].data['pixel_corner_mrh_alt'][:, :, 4]
        azimuths = pyrt.azimuth(szas, eas, pas)
        latitudes = hduls[counter]['pixelgeometry'].data['pixel_corner_lat'][:, :, 4]
        longitudes = hduls[counter]['pixelgeometry'].data['pixel_corner_lon'][:, :, 4]

        if rad.shape[-1] == 20:
            wavelength_indices = [1, 2, 3, -8, -7, -6]
        elif rad.shape[-1] == 19:
            wavelength_indices = [1, 2, 3, -7, -6, -5]
        elif rad.shape[-1] == 15:
            wavelength_indices = [1, 2, 3, -3, -2, -1]

        memmap_filename_answer = os.path.join(mkdtemp(), 'myNewFileAnswer.dat')
        answer = np.memmap(memmap_filename_answer, dtype=float, shape=latitudes.shape + (9,), mode='w+') * np.nan # 9 is for dust, ice, error, simulated I/F at 6 wavelengths
        answer0 = np.zeros(latitudes.shape + (9,)) * np.nan

        def fit_dust_and_ice(radiance, sza, ea, aza, psurf) -> float:
            def find_best_fit(guess: np.ndarray):
                dust_guess = guess[0]
                ice_guess = guess[1]
                simulated_spectrum = np.zeros(wavelength_grid.shape) * np.nan
                for c, wav in enumerate(wavelength_grid):
                    input = np.array([sza, ea, aza, psurf, dust_guess, ice_guess, wav])
                    simulated_spectrum[c] = interp(input)
                return np.sum((simulated_spectrum - radiance[wavelength_indices])**2)

            return minimize(find_best_fit, np.array([0.7, 0.2]), method='Nelder-Mead', tol=1e-2, bounds=((0, 2), (0, 1)), options={'adaptive': True}).x

        def make_array(input):
            # I have no idea why this bit is necessary. I couldn't get it to just save the lut after doing the multiprocessing.
            ind0 = input[0]
            ind1 = input[1]
            arr = input[2]

            answer0[ind0, ind1] = arr

        def do_retrieval(integration, spatial_bin):
            print(integration, spatial_bin)
            if alts[integration, spatial_bin] != 0 or szas[integration, spatial_bin] > 70 or eas[integration, spatial_bin] > 70:
                return integration, spatial_bin, np.zeros(9,) * np.nan
            sfc_pressure = sfc_interp(np.array([latitudes[integration, spatial_bin], longitudes[integration, spatial_bin]]))[0]
            answer[integration, spatial_bin, :2] = fit_dust_and_ice(rad[integration, spatial_bin], szas[integration, spatial_bin], eas[integration, spatial_bin], azimuths[integration, spatial_bin], sfc_pressure)

            # Get the spectrum of the best fit answer
            sim_spec = np.zeros(6,)
            for c, wav in enumerate(wavelength_grid):
                sim_spec[c] = interp(np.array([szas[integration, spatial_bin], eas[integration, spatial_bin], azimuths[integration, spatial_bin], sfc_pressure, answer[integration, spatial_bin, 0], answer[integration, spatial_bin, 1], wav]))
            error = np.sum((rad[integration, spatial_bin, wavelength_indices] - sim_spec)**2 / sim_spec)
            #error = np.sum(np.abs((rad[integration, spatial_bin, wavelength_indices] - sim_spec)**2/rad[integration, spatial_bin, wavelength_indices]))

            # Add the spectrum to the array
            answer[integration, spatial_bin, 2] = error
            answer[integration, spatial_bin, 3:] = sim_spec
            return integration, spatial_bin, answer[integration, spatial_bin]

        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus - 1)
        for integration in range(rad.shape[0]):
            for spatial_bin in range(rad.shape[1]):
                pool.apply_async(func=do_retrieval, args=(integration, spatial_bin), callback=make_array)

        # https://www.machinelearningplus.com/python/parallel-processing-python/
        pool.close()
        pool.join()

        # Save the file
        fn = f'{counter}'.zfill(2)
        filename = save_location / block_code / f'{orbit_code}-{fn}.npy'
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(filename), radiance)

