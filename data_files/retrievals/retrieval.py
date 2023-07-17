import time
import os
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
import math
from tempfile import mkdtemp

from astropy.io import fits
from h5py import File
from netCDF4 import Dataset
import numpy as np
from scipy.constants import Boltzmann
from scipy.integrate import quadrature as quad
from scipy.optimize import minimize
from scipy.io import readsav
from scipy.interpolate import interpn

import disort
import pyrt
import mars_time


class Ames:
    def __init__(self):
        self._base_path = Path('/mnt/science/data_lake/mars/gcm/ames/my_generic')
        self._average_path = self._base_path / '05344.atmos_average.nc'
        self._fixed_path = self._base_path / '05344.fixed.nc'

        self._fixed = Dataset(self._fixed_path)
        self._average = Dataset(self._average_path)

        gcm_lat = grid.variables['lat'][:]
        gcm_lon = grid.variables['lon'][:]

        gcm_dust_vprof = yearly_gcm.variables['dustref']  # shape: (year, pressure level, lat, lon)
        gcm_ice_vprof = yearly_gcm.variables['cldref']  # shape: (year, pressure level, lat, lon)

        gcm_pressure_levels = gcm.variables['pstd']  # shape: (pressure level,). Index 0 is TOA
        gcm_surface_pressure = gcm.variables['ps']  # shape: (season, time of day, lat, lon)
        gcm_surface_temperature = gcm.variables['ts']  # shape: (season, time of day, lat, lon)
        gcm_temperature = gcm.variables['temp']  # shape: (season, time of day, pressure level, lat, lon)
        gcm_season = gcm.variables['time'][:]  # idk why the times start from 10,000
        gcm_local_time = gcm.variables['time_of_day_24'][:]

        season_idx = np.argmin(np.abs(gcm_season - 10000 - sol))
        yearly_idx = np.argmin(np.abs(yearly_gcm.variables['time'][:] - 10000 - sol))


class Radprop:
    def __init__(self):
        self._base_path = Path('/mnt/science/data_lake/mars/radiative_properties')



def perform_retrieval(orbit: int):
    orbit_code = f'orbit' + f'{orbit}'.zfill(5)
    block = math.floor(orbit / 100) * 100
    orbit_block = 'orbit' + f'{block}'.zfill(5)

    # Define some paths
    iuvs_data_location = Path('/mnt/science/data_lake/mars/maven/iuvs/production')
    wavelength_location = Path('/mnt/science/data_lake/mars/maven/iuvs/apoapse_wavelengths')
    radiance_location = Path('/mnt/science/data/mars/maven/iuvs/radiance')

    # Get the orbit specific files
    l1b_files = sorted((iuvs_data_location / orbit_block).glob(f'*apoapse*{orbit_code}*muv*.gz'))
    wavelength_files = sorted((wavelength_location / orbit_block).glob(f'*apoapse*{orbit_code}*muv*'))
    radiance_files = sorted((radiance_location / orbit_block).glob(f'{orbit_code}*'))

    # Get apsis info
    apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
    sol = apsis_file['apoapse/sol'][orbit]

    ##############
    # Ames GCM
    ##############
    # Read in the Ames GCM
    grid = Dataset('/media/kyle/McDataFace/ames/sim1/10000.fixed.nc')
    gcm = Dataset('/media/kyle/McDataFace/ames/sim1/c48_big.atmos_diurn_plev-002.nc')
    yearly_gcm = Dataset('/media/kyle/McDataFace/ames/sim1/c48_big.atmos_average_plev-001.nc')
    # print(gcm.variables)

    gcm_lat = grid.variables['lat'][:]
    gcm_lon = grid.variables['lon'][:]

    gcm_dust_vprof = yearly_gcm.variables['dustref']  # shape: (year, pressure level, lat, lon)
    gcm_ice_vprof = yearly_gcm.variables['cldref']  # shape: (year, pressure level, lat, lon)

    gcm_pressure_levels = gcm.variables['pstd']  # shape: (pressure level,). Index 0 is TOA
    gcm_surface_pressure = gcm.variables['ps']  # shape: (season, time of day, lat, lon)
    gcm_surface_temperature = gcm.variables['ts']  # shape: (season, time of day, lat, lon)
    gcm_temperature = gcm.variables['temp']  # shape: (season, time of day, pressure level, lat, lon)
    gcm_season = gcm.variables['time'][:]  # idk why the times start from 10,000
    gcm_local_time = gcm.variables['time_of_day_24'][:]

    season_idx = np.argmin(np.abs(gcm_season - 10000 - sol))
    yearly_idx = np.argmin(np.abs(yearly_gcm.variables['time'][:] - 10000 - sol))

    ##############
    # Aerosol radprop
    ##############
    # Dust
    dust_cext = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/extinction_cross_section.npy')  # (24, 317)
    dust_csca = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/scattering_cross_section.npy')  # (24, 317)
    dust_pmom = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/legendre_coefficients.npy')  # (129, 24, 317)
    dust_wavelengths = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/wavelengths.npy')
    dust_particle_sizes = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/particle_sizes.npy')

    # Ice
    ice_cext = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/extinction_cross_section.npy')  # (24, 317)
    ice_csca = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/scattering_cross_section.npy')  # (24, 317)
    ice_pmom = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/legendre_coefficients.npy')  # (96,)   # For 3 micron particles at 321 nm
    ice_g = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/asymmetry_parameter.npy')
    ice_wavelengths = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/wavelengths.npy')
    ice_particle_sizes = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/particle_sizes.npy')


if __name__ == '__main__':
    #n_cpus = mp.cpu_count()  # = 8 for my old desktop, 12 for my laptop, 20 for my new desktop
    #pool = mp.Pool(n_cpus - 2)  # save one/two just to be safe. Some say it's faster
    for orb in range(3453, 3454):
        # NOTE: if there are any issues in the argument of apply_async, it'll break out of that and move on to the next iteration.
        perform_retrieval(3453)
        #pool.apply_async(make_radiance, args=(orb,))
    #pool.close()
    #pool.join()









if __name__ == '__main__':
    t0 = time.time()
    # Load in all the info from the given file
    orbit: int = 3753
    orbit_code = f'orbit' + f'{orbit}'.zfill(5)
    block = math.floor(orbit / 100) * 100
    orbit_block = 'orbit' + f'{block}'.zfill(5)
    lamber = True

    ##############
    # IUVS data
    ##############

    # Get the l1b file
    l1b_files = sorted(Path(f'/media/kyle/McDataFace/iuvsdata/production/{orbit_block}').glob(f'*apoapse*{orbit}*muv*.gz'))
    # Load in the reflectance
    ff = np.load('/home/kyle/repos/PyUVS-old/pyuvs/anc/flatfields/mid-hi-res-flatfield-my34gds.npy')
    reflectance_files = sorted(Path(f'/home/kyle/iuvs/radiance/{orbit_block}').glob(f'radiance-{orbit_code}*.npy'))
    # Load in the corrected wavelengths
    wavelengths_files = sorted(Path(f'/home/kyle/iuvs/wavelengths/{orbit_block}').glob(f'*{orbit_code}*'))

    # Compute SPICE info
    spice_path = Path('/media/kyle/McDataFace/spice')
    s = Spice(spice_path)
    s.load_spice()
    orbits, all_et = s.find_all_maven_apsis_et('apoapse', endtime=datetime(2017, 5, 29))

    print('found apsis info')
    et = all_et[orbits == orbit][0]
    pf = PositionFinder(et)
    mars_sun_distance = pf.get_mars_sun_distance()

    # Compute the Mars radius ratio
    earth_sun_distance = 1.496e8  # km
    radius = earth_sun_distance / mars_sun_distance

    dt = pf.get_datetime()
    sol = mer.EarthDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second).to_sol()

    ##############
    # Ames GCM
    ##############
    # Read in the Ames GCM
    grid = Dataset('/media/kyle/McDataFace/ames/sim1/10000.fixed.nc')
    gcm = Dataset('/media/kyle/McDataFace/ames/sim1/c48_big.atmos_diurn_plev-002.nc')
    yearly_gcm = Dataset('/media/kyle/McDataFace/ames/sim1/c48_big.atmos_average_plev-001.nc')
    # print(gcm.variables)

    gcm_lat = grid.variables['lat'][:]
    gcm_lon = grid.variables['lon'][:]

    gcm_dust_vprof = yearly_gcm.variables['dustref']  # shape: (year, pressure level, lat, lon)
    gcm_ice_vprof = yearly_gcm.variables['cldref']  # shape: (year, pressure level, lat, lon)

    gcm_pressure_levels = gcm.variables['pstd']  # shape: (pressure level,). Index 0 is TOA
    gcm_surface_pressure = gcm.variables['ps']  # shape: (season, time of day, lat, lon)
    gcm_surface_temperature = gcm.variables['ts']  # shape: (season, time of day, lat, lon)
    gcm_temperature = gcm.variables['temp']  # shape: (season, time of day, pressure level, lat, lon)
    gcm_season = gcm.variables['time'][:]  # idk why the times start from 10,000
    gcm_local_time = gcm.variables['time_of_day_24'][:]

    season_idx = np.argmin(np.abs(gcm_season - 10000 - sol))
    yearly_idx = np.argmin(np.abs(yearly_gcm.variables['time'][:] - 10000 - sol))

    ##############
    # Surface albedo map
    ##############
    band6 = fits.open('/home/kyle/repos/iuvs-ice/map/band6_w.fits')['primary'].data
    band7 = fits.open('/home/kyle/repos/iuvs-ice/map/band7_w.fits')['primary'].data
    hapke_w = np.dstack((band6, band7))

    ##############
    # Surface albedo map
    ##############
    mola = np.load('/home/kyle/repos/iuvs-ice/map/mola-topography.npy')   # This is in meters, not km
    mola_lat = np.linspace(90, -90, num=1440)
    mola_lon = np.linspace(0, 360, num=2880)

    ##############
    # Aerosol radprop
    ##############
    # Dust
    dust_cext = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/extinction_cross_section.npy')  # (24, 317)
    dust_csca = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/scattering_cross_section.npy')  # (24, 317)
    dust_pmom = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/legendre_coefficients.npy')  # (129, 24, 317)
    dust_wavelengths = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/wavelengths.npy')
    dust_particle_sizes = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/particle_sizes.npy')

    # Ice
    ice_cext = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/extinction_cross_section.npy')  # (24, 317)
    ice_csca = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/scattering_cross_section.npy')  # (24, 317)
    ice_pmom = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/legendre_coefficients.npy')  # (96,)   # For 3 micron particles at 321 nm
    ice_g = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/asymmetry_parameter.npy')
    ice_wavelengths = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/wavelengths.npy')
    ice_particle_sizes = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/particle_sizes.npy')
    # ice_pmom = pyrt.decompose_hg(ice_g, 129)

    ##############
    # Surface arrays
    ##############
    n_streams = 8
    n_polar = 1  # defined by IUVS' viewing geometry
    n_azimuth = 1  # defined by IUVS' viewing geometry

    bemst = np.zeros(int(0.5 * n_streams))
    emust = np.zeros(n_polar)
    rho_accurate = np.zeros((n_polar, n_azimuth))
    rhoq = np.zeros((int(0.5 * n_streams), int(0.5 * n_streams + 1), n_streams))
    rhou = np.zeros((n_polar, int(0.5 * n_streams + 1), n_streams))

    # Process file is here
    def process_file(file: int):
        hdul = fits.open(l1b_files[file])
        reflectance = np.load(reflectance_files[file]) / ff
        wavelengths = readsav(wavelengths_files[file])['wavelength_muv'] / 1000  # convert to microns

        # Get the data from the l1b file
        pixelgeometry = hdul['pixelgeometry'].data
        latitude = pixelgeometry['pixel_corner_lat']
        longitude = pixelgeometry['pixel_corner_lon']
        local_time = pixelgeometry['pixel_local_time']
        solar_zenith_angle = pixelgeometry['pixel_solar_zenith_angle']
        emission_angle = pixelgeometry['pixel_emission_angle']
        phase_angle = pixelgeometry['pixel_phase_angle']
        data_ls = hdul['observation'].data['solar_longitude']

        # Make the azimuth angles
        solar_zenith_angle_good = np.where(solar_zenith_angle <= 90, True, False)
        solar_zenith_angle = np.where(solar_zenith_angle <= 90, solar_zenith_angle, 0)
        azimuth = pyrt.azimuth(solar_zenith_angle, emission_angle, phase_angle)

        mu0 = np.cos(np.radians(solar_zenith_angle))
        mu = np.cos(np.radians(emission_angle))

        def column_density(pressure_profile, temperature_profile, altitude, profargs, tempargs):
            """Make the column density.

            Parameters
            ----------
            pressure_profile
            temperature_profile
            altitude
            profargs
            tempargs

            Returns
            -------

            """
            def hydrostatic_profile(alts):
                return pressure_profile(alts, *profargs) / temperature_profile(alts, *tempargs) / Boltzmann

            n = [quad(hydrostatic_profile, altitude[i+1], altitude[i])[0] for i in range(len(altitude) - 1)]
            return np.array(n) * 1000

        def linear_grid_profile(altitude, altitude_grid, profile_grid) -> np.ndarray:
            """Make a profile with linear interpolation between grid points.

            Parameters
            ----------
            altitude: ArrayLike
                The altitudes at which to make the vertical profile. Must be
                1-dimensional and monotonically decreasing.
            altitude_grid: ArrayLike
                The altitudes where ``profile_grid`` is defined. Must be 1-dimensional
                and monotonically decreasing.
            profile_grid: ArrayLike
                The profile at each point in ``altitude_grid``.

            Returns
            -------
            np.ndarray
                1-dimensional array of the profile with shape ``altitude.shape``.

            Raises
            ------
            TypeError
                Raised if any of the inputs cannot be cast to ndarrays.
            ValueError
                Raised if any of the inputs do not have the aforementioned desired
                properties.
            """
            # NOTE: numpy needs linearly increasing arrays, which is opposite of the altitude convention! They don't warn about this
            return np.interp(np.flip(altitude), np.flip(altitude_grid), np.flip(profile_grid))

        global retrieval   # this makes the function global so multiprocessing can pickle it. Very strange...
        def retrieval(integration: int, spatial_bin: int):
            # Exit if the angles are not retrievable
            if (not solar_zenith_angle_good[integration, spatial_bin]) or \
                    solar_zenith_angle[integration, spatial_bin] >= 72 or \
                    emission_angle[integration, spatial_bin] >= 72:
                answer = np.zeros((2,)) * np.nan
                return integration, spatial_bin, answer, np.nan, np.zeros(6,) * np.nan

            pixel_wavs = wavelengths[spatial_bin, :]

            ##############
            # Equation of state
            ##############
            # Get the nearest neighbor values of the lat/lon/lt values (for speed I'm not using linear interpolation)
            pixel_lat = latitude[integration, spatial_bin, 4]
            pixel_lon = longitude[integration, spatial_bin, 4]

            lat_idx = np.argmin(np.abs(gcm_lat - pixel_lat))
            lon_idx = np.argmin(np.abs(gcm_lon - pixel_lon))
            lt_idx = np.argmin(np.abs(gcm_local_time - (local_time[integration, spatial_bin] - (longitude[integration, spatial_bin, 4] / 360 * 24)) % 24))

            # Get 1D profiles of temperature and pressure for this pixel (well, pressures are just fixed by the model to be the same everywhere)
            pixel_temperature = gcm_temperature[season_idx, lt_idx, :, lat_idx, lon_idx]

            # Get where the surface index would be
            surface_idx = np.argmin(~pixel_temperature.mask)

            # Insert the surface values into the temperature and pressure arrays so they're shape (41,) instead of (40,)
            sfc_pressure = gcm_surface_pressure[season_idx, lt_idx, lat_idx, lon_idx]    # This is NN
            sfc_temperature = gcm_surface_temperature[season_idx, lt_idx, lat_idx, lon_idx]    # This is NN
            pixel_pressure = np.insert(gcm_pressure_levels, surface_idx, sfc_pressure)
            pixel_temperature = np.insert(pixel_temperature, surface_idx, sfc_temperature)

            # Get the average altitude from the GCM
            gcm_lat_min = (gcm_lat[lat_idx-1] + gcm_lat[lat_idx]) / 2
            gcm_lat_max = (gcm_lat[lat_idx] + gcm_lat[lat_idx+1]) / 2
            gcm_lon_max = (gcm_lon[lon_idx] + gcm_lon[lon_idx + 1]) / 2
            gcm_lon_min = (gcm_lon[lon_idx-1] + gcm_lon[lon_idx]) / 2
            gcm_mean_alt = np.mean(mola[np.bitwise_and(mola_lat >= gcm_lat_min, mola_lat <= gcm_lat_max),
                                        np.bitwise_and(mola_lon >= gcm_lon_min, mola_lon <= gcm_lon_max)])

            # Get the average altitude of an IUVS pixel (this is hackish, cause I just average the corners and center)
            def get_altitude_of_point(lat_point, lon_point):
                mola_lat_idx = np.argmin(np.abs(mola_lat - lat_point))
                mola_lon_idx = np.argmin(np.abs(mola_lon - lon_point))
                return mola[mola_lat_idx, mola_lon_idx]

            # Get the scale factor to multiply the Rayleigh OD by
            iuvs_mean_alt = np.mean([get_altitude_of_point(latitude[integration, spatial_bin, f], longitude[integration, spatial_bin, f]) for f in range(5)])
            pressure_scale_factor = np.exp((gcm_mean_alt - iuvs_mean_alt) / 10000)

            # Make the altitudes assuming exponentially decreasing pressure and constant scale height
            z = -np.log(pixel_pressure / sfc_pressure) * 10

            # Finally, use these to compute the column density in each "good" layer
            colden = column_density(linear_grid_profile, linear_grid_profile, z[:surface_idx+1],
                                    (z[:surface_idx+1], pixel_pressure[:surface_idx+1]),
                                    (z[:surface_idx+1], pixel_temperature[:surface_idx+1])) * pressure_scale_factor

            ##############
            # Rayleigh scattering
            ##############
            rayleigh_scattering_optical_depth = pyrt.rayleigh_co2_optical_depth(colden, pixel_wavs)
            rayleigh_ssa = np.ones(rayleigh_scattering_optical_depth.shape)
            rayleigh_pmom = pyrt.rayleigh_legendre(rayleigh_scattering_optical_depth.shape[0], pixel_wavs.shape[0])

            #print(np.sum(rayleigh_scattering_optical_depth, axis=0))
            #print(sfc_pressure)
            #raise SystemExit(9)

            rayleigh_column = pyrt.Column(rayleigh_scattering_optical_depth, rayleigh_ssa, rayleigh_pmom)

            ##############
            # Aerosol vertical profiles
            ##############
            # Get the GCM profiles and normalize them
            # I need to account for differences between the GCMs!!!
            # Since pressure is the vertical coordinate, they arrays may not have the same lengths!!!!!
            # So, set the bad values to the last good value
            dust_vprof = gcm_dust_vprof[yearly_idx, :surface_idx, lat_idx, lon_idx]
            dust_vprof[dust_vprof.mask] = dust_vprof[np.argmin(~dust_vprof.mask)-1]
            dust_vprof = dust_vprof / np.sum(dust_vprof)

            ice_vprof = gcm_ice_vprof[yearly_idx, :surface_idx, lat_idx, lon_idx]
            ice_vprof[ice_vprof.mask] = ice_vprof[np.argmin(~ice_vprof.mask) - 1]
            ice_vprof = ice_vprof / np.sum(ice_vprof)

            ##############
            # Surface
            ##############
            # Make empty arrays. These are fine to be empty with Lambert surfaces and they'll be filled with more complicated surfaces later on
            bemst_empty = np.zeros(int(0.5 * n_streams))
            emust_empty = np.zeros(n_polar)
            rho_accurate_empty = np.zeros((n_polar, n_azimuth))
            rhoq_empty = np.zeros(
                         (int(0.5 * n_streams), int(0.5 * n_streams + 1), n_streams))
            rhou_empty = np.zeros((n_polar, int(0.5 * n_streams + 1), n_streams))

            # For Lambert
            clancy = np.linspace(0.014, 0.017, num=100)
            clancy_wavs = np.linspace(0.258, 0.32, num=100)
            sfc = np.interp(wavelengths, clancy_wavs, clancy)

            # For Hapke HG2. The idea is the make arrays that are (n_wavelengths, array_shape). This way I can compute them once for each wavelength
            # and each pixel, and then just call them once.
            #wolff_hapke = np.interp(pixel_wavs, np.linspace(0.258, 0.32, num=100), np.linspace(0.06, 0.08, num=100))
            wolff_hapke = np.array([interpn((np.linspace(-90, 90, num=180),
                                   np.linspace(0, 360, num=360), np.array([0.26, 0.32])),
                                   hapke_w, np.array([pixel_lat, pixel_lon, f]), bounds_error=False,
                                   fill_value=None)[0] for f in pixel_wavs])
            bemst = np.zeros((19,) + bemst_empty.shape)
            emust = np.zeros((19,) + emust_empty.shape)
            rho_accurate = np.zeros((19,) + rho_accurate_empty.shape)
            rhoq = np.zeros((19,) + rhoq_empty.shape)
            rhou = np.zeros((19,) + rhou_empty.shape)

            # Make the surface arrays just once for Hapke HG2
            if not lamber:
                for counter, foobar in enumerate(wolff_hapke):
                    #brdf_arg = np.array([0.8, 0.06, foobar, 0.3, 0.45, 20])  # From Wolff2010
                    brdf_arg = np.array([1, 0.06, foobar, 0.3, 0.7, 20])  # From Wolff2019
                    _rhoq, _rhou, _emust, _bemst, _rho_accurate = \
                        disort.disobrdf(True, mu[integration, spatial_bin], np.pi, mu0[integration, spatial_bin], False, 0.01, False,
                                        rhoq_empty, rhou_empty, emust_empty, bemst_empty, False,
                                        azimuth[integration, spatial_bin], 0, rho_accurate_empty,
                                        6, brdf_arg, 200, nstr=n_streams, numu=1, nphi=1)
                    bemst[counter] = _bemst
                    emust[counter] = _emust
                    rho_accurate[counter] = _rho_accurate
                    rhoq[counter] = _rhoq
                    rhou[counter] = _rhou

            wavelength_indices = [1, 2, 3, -4, -3, -2]
            #wavelength_indices = [-4, -3, -2]

            def simulate_tau(guess: np.ndarray) -> np.ndarray:
                #print(f'guess = {guess}')
                dust_guess = guess[0]
                ice_guess = guess[1]

                simulated_toa_reflectance = np.zeros(len(wavelength_indices))

                # This is a hack to add bounds to the solver
                if np.any(guess < 0):
                    simulated_toa_reflectance[:] = 999999
                    return simulated_toa_reflectance

                for counter, wav_index in enumerate(wavelength_indices):
                    ##############
                    # Dust FSP
                    ##############
                    # Before: pixel_wavs[wav_index]. Now I'm scaling it to 240 nm
                    dust_extinction = pyrt.extinction_ratio_grid(dust_cext, dust_particle_sizes, dust_wavelengths, pixel_wavs[wav_index])
                    dust_reff_grid = np.linspace(1.5, 1.5, num=100)
                    dust_z_reff = np.linspace(100, 0, num=100)
                    dust_reff = np.interp(np.flip(z[:surface_idx]), np.flip(dust_z_reff), np.flip(dust_reff_grid))   # The "bad" altitudes don't get assigned a particle size

                    dust_extinction_grid = pyrt.regrid(dust_extinction, dust_particle_sizes, dust_wavelengths, dust_reff, pixel_wavs)

                    dust_od = pyrt.optical_depth(dust_vprof, colden, dust_extinction_grid, dust_guess)
                    dust_ssa = pyrt.regrid(dust_csca / dust_cext, dust_particle_sizes, dust_wavelengths, dust_reff, pixel_wavs)
                    dust_legendre = pyrt.regrid(dust_pmom, dust_particle_sizes, dust_wavelengths, dust_reff, pixel_wavs)
                    dust_column = pyrt.Column(dust_od, dust_ssa, dust_legendre)

                    ##############
                    # Ice FSP
                    ##############
                    # Before: pixel_wavs[wav_index]. Now I'm scaling it to 240 nm
                    ice_extinction = pyrt.extinction_ratio_grid(ice_cext, ice_particle_sizes, ice_wavelengths, pixel_wavs[wav_index])
                    ice_reff_grid = np.linspace(3, 3, num=100)
                    ice_z_reff = np.linspace(100, 0, num=100)
                    ice_reff = np.interp(np.flip(z[:surface_idx]), np.flip(ice_z_reff), np.flip(ice_reff_grid))   # The "bad" altitudes don't get assigned a particle size

                    ice_extinction_grid = pyrt.regrid(ice_extinction, ice_particle_sizes, ice_wavelengths, ice_reff, pixel_wavs)

                    ice_od = pyrt.optical_depth(ice_vprof, colden, ice_extinction_grid, ice_guess)
                    ice_ssa = pyrt.regrid(ice_csca / ice_cext, ice_particle_sizes, ice_wavelengths, ice_reff, pixel_wavs)
                    #ice_asym = pyrt.regrid(ice_g, ice_particle_sizes, ice_wavelengths, ice_reff, pixel_wavs)
                    #ice_legendre = pyrt.decompose_hg(ice_asym, 129)
                    # For now I only have a 1D array of phase functions so I need to broadcast
                    ice_legendre = np.broadcast_to(ice_pmom, (ice_ssa.shape + ice_pmom.shape))
                    ice_legendre = np.moveaxis(ice_legendre, -1, 0)
                    ice_column = pyrt.Column(ice_od, ice_ssa, ice_legendre)

                    ##############
                    # Total atmosphere
                    ##############
                    atm = rayleigh_column + dust_column + ice_column
                    weight = np.arange(129) * 2 + 1
                    atm.legendre_coefficients = (atm.legendre_coefficients.T / weight).T

                    ##############
                    # Output arrays
                    ##############
                    n_user_levels = atm.optical_depth.shape[0] + 1
                    albedo_medium = np.zeros(n_polar)
                    diffuse_up_flux = np.zeros(n_user_levels)
                    diffuse_down_flux = np.zeros(n_user_levels)
                    direct_beam_flux = np.zeros(n_user_levels)
                    flux_divergence = np.zeros(n_user_levels)
                    intensity = np.zeros((n_polar, n_user_levels, n_azimuth))
                    mean_intensity = np.zeros(n_user_levels)
                    transmissivity_medium = np.zeros(n_polar)

                    # Misc
                    user_od_output = np.zeros(n_user_levels)
                    temper = np.zeros(n_user_levels)
                    h_lyr = np.zeros(n_user_levels)
                    ##############
                    # Call DISORT
                    ##############

                    # The 2nd option of the 2nd line is LAMBER
                    rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
                        disort.disort(True, False, False, False, [False, False, False, False, False],
                                      False, lamber, True, False,
                                      atm.optical_depth[:, wav_index],
                                      atm.single_scattering_albedo[:, wav_index],
                                      atm.legendre_coefficients[:, :, wav_index],
                                      temper, 1, 1, user_od_output,
                                      mu0[integration, spatial_bin], 0,
                                      mu[integration, spatial_bin], azimuth[integration, spatial_bin],
                                      np.pi, 0, sfc[wav_index], 0, 0, 1, 3400000, h_lyr,
                                      rhoq[wav_index], rhou[wav_index], rho_accurate[wav_index], bemst[wav_index], emust[wav_index],
                                      0, '', direct_beam_flux,
                                      diffuse_down_flux, diffuse_up_flux, flux_divergence,
                                      mean_intensity, intensity, albedo_medium,
                                      transmissivity_medium, maxcmu=n_streams, maxulv=n_user_levels, maxmom=128)
                    simulated_toa_reflectance[counter] = uu[0, 0, 0]
                '''fig, ax = plt.subplots()
                ax.scatter(pixel_wavs, reflectance[integration, spatial_bin, :], label='data')
                ax.scatter(pixel_wavs[wavelength_indices], simulated_toa_reflectance, label='simulation')
                ax.legend()
                plt.savefig('/home/kyle/iuvs/test.png')
                raise SystemExit(9)'''
                return simulated_toa_reflectance

            def find_best_fit(guess: np.ndarray):
                simulated_toa_reflectance = simulate_tau(guess)
                '''print(f'{reflectance[integration, spatial_bin, wavelength_indices]} \n'
                      f'{simulated_toa_reflectance} \n'
                      f'{guess}')'''
                return np.sum((simulated_toa_reflectance - reflectance[integration, spatial_bin, wavelength_indices])**2)

            fitted_optical_depth = minimize(find_best_fit, np.array([0.7, 0.2]), method='Nelder-Mead', tol=1e-2, bounds=((0, 2), (0, 1)), options={'adaptive': True}).x
            best_fit_od = np.array(fitted_optical_depth)
            sim = simulate_tau(best_fit_od)
            #error = np.sum((reflectance[integration, spatial_bin, wavelength_indices] - sim)**2 / sim)
            total_error = np.abs(reflectance[integration, spatial_bin, wavelength_indices] - sim) / reflectance[integration, spatial_bin, wavelength_indices]
            error = np.sum(total_error) / len(total_error)  # This is the mean relative error
            if error > 0.1:
                fitted_optical_depth = minimize(find_best_fit, np.array([0.5, 0.1]), method='Nelder-Mead', tol=1e-2, bounds=((0, 2), (0, 1)), options={'adaptive': True}).x
                best_fit_od = np.array(fitted_optical_depth)
                sim = simulate_tau(best_fit_od)
                # error = np.sum((reflectance[integration, spatial_bin, wavelength_indices] - sim)**2 / sim)
                total_error = np.abs(reflectance[integration, spatial_bin, wavelength_indices] - sim) / reflectance[integration, spatial_bin, wavelength_indices]
                error = np.sum(total_error) / len(total_error)  # This is the mean relative error

            '''print(f'data = {reflectance[integration, spatial_bin, wavelength_indices]}')
            print(f'sim = {sim}')
            print(f'answer={fitted_optical_depth}')
            print(f'error = {error}')
            t1 = time.time()
            print(t1 - t0)
            print(pixel_lat, pixel_lon)
            raise SystemExit(9)'''
            return integration, spatial_bin, best_fit_od, error, sim


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make a shared array
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        memmap_filename_dust = os.path.join(mkdtemp(), 'myNewFileDust.dat')
        memmap_filename_ice = os.path.join(mkdtemp(), 'myNewFileIce.dat')
        memmap_filename_error = os.path.join(mkdtemp(), 'myNewFileError.dat')
        memmap_filename_radiance = os.path.join(mkdtemp(), 'myNewFileRadiance.dat')
        retrieved_dust = np.memmap(memmap_filename_dust, dtype=float,
                                   shape=reflectance.shape[:-1], mode='w+') * np.nan
        retrieved_ice = np.memmap(memmap_filename_ice, dtype=float,
                                   shape=reflectance.shape[:-1], mode='w+') * np.nan
        retrieved_error = np.memmap(memmap_filename_error, dtype=float,
                                   shape=reflectance.shape[:-1], mode='w+') * np.nan
        retrieved_radiance = np.memmap(memmap_filename_radiance, dtype=float,
                                   shape=reflectance.shape[:-1] + (6,), mode='w+') * np.nan  # This 6 is for 6 wavelengths

        def make_answer(inp):
            integration = inp[0]
            position = inp[1]
            answer = inp[2]
            err = inp[3]
            simulated_radiance = inp[4]
            retrieved_dust[integration, position] = answer[0]
            retrieved_ice[integration, position] = answer[1]
            retrieved_error[integration, position] = err
            retrieved_radiance[integration, position] = simulated_radiance

        n_cpus = mp.cpu_count()    # = 8 for my old desktop, 12 for my laptop, 20 for my new desktop
        pool = mp.Pool(n_cpus - 1)   # save one/two just to be safe. Some say it's faster
        # NOTE: if there are any issues in the argument of apply_async (here,
        # retrieve_ssa), it'll break out of that and move on to the next iteration.

        for integ in range(reflectance.shape[0]):
            for posit in range(reflectance.shape[1]):
                pool.apply_async(func=retrieval, args=(integ, posit), callback=make_answer)
                #print(f'starting integ {integ} and posti {posit}')
                #retrieval(integ, posit)

        # This is OM in 3453-03
        '''for integ in [25]:
            for posit in [30]:
                retrieval(integ, posit)'''

        # https://www.machinelearningplus.com/python/parallel-processing-python/
        pool.close()
        pool.join()  # I guess this postpones further code execution until the queue is finished
        fileno = f'{file}'.zfill(2)
        np.save(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/{orbit_code}-{fileno}-dust.npy', retrieved_dust)
        np.save(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/{orbit_code}-{fileno}-ice.npy', retrieved_ice)
        np.save(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/{orbit_code}-{fileno}-error.npy', retrieved_error)
        np.save(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/{orbit_code}-{fileno}-simulated_radiance.npy', retrieved_radiance)

    for file in range(len(l1b_files)):
        print(f'starting file {file}')
        process_file(file)

    print(time.time() - t0)