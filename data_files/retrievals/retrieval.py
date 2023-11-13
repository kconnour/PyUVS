import os
from pathlib import Path
import multiprocessing as mp
import math
from tempfile import mkdtemp
import warnings

from astropy.io import fits
from h5py import File
import numpy as np
from scipy.optimize import minimize
from scipy.io import readsav
from scipy.interpolate import RegularGridInterpolator

from ames import Ames


import disort
import pyrt

class Radprop:
    def __init__(self):
        self._base_path = Path('/mnt/science/data_lake/mars/radiative_properties')
        self.hdul = None

    def _get_file_forward_scattering_properties(self) -> np.ndarray:
        return self.hdul['forw'].data

    def _get_file_legendre_coefficients(self) -> np.ndarray:
        return self.hdul['pmom'].data

    def _get_file_phase_function(self) -> np.ndarray:
        return self.hdul['phsfn'].data

    def _get_file_phase_function_reexpansion(self) -> np.ndarray:
        return self.hdul['expansion'].data

    def get_primary(self) -> np.ndarray:
        """Get the primary hdul.

        Returns
        -------
        The primary structure

        Notes
        -----
        This array should contain no relevant info.

        """
        return self.hdul['primary'].data

    def get_particle_sizes(self) -> np.ndarray:
        """Get the particle sizes associated with each of the radiative properties.

        Returns
        -------
        The particle sizes

        Notes
        -----
        The particle sizes are the centers of some particle size distribution.

        """
        return self.hdul['particle_sizes'].data

    def get_wavelengths(self) -> np.ndarray:
        return self.hdul['wavelengths'].data

    def get_scattering_angles(self) -> np.ndarray:
        return self.hdul['scattering_angle'].data

    def get_scattering_cross_sections(self) -> np.ndarray:
        return self._get_file_forward_scattering_properties()[..., 1]

    def get_extinction_cross_sections(self) -> np.ndarray:
        return self._get_file_forward_scattering_properties()[..., 0]

    def get_asymmetry_parameters(self) -> np.ndarray:
        return self._get_file_forward_scattering_properties()[..., 2]

    def get_phase_functions(self) -> np.ndarray:
        return np.moveaxis(self._get_file_phase_function(), 0, -1)

    def get_legendre_coefficients(self) -> np.ndarray:
        return self._get_file_legendre_coefficients()   # This is stupid but it needs a quick fix

    def get_phase_function_reexpansions(self) -> np.ndarray:
        return np.moveaxis(self._get_file_phase_function_reexpansion(), 0, -1)

    def _get_header(self) -> fits.header.Header:
        return self.hdul['primary'].header

    def get_file_creation_date(self) -> str:
        header = self._get_header()
        return header['date']

    def get_history(self) -> str:
        header = self._get_header()
        history = str(header['history'])
        # The ??? currently only applies to the dust file
        return history.replace('???', '--')


class Dust(Radprop):
    def __init__(self):
        super().__init__()
        self.hdul = fits.open(self._base_path / 'dust.fits.gz')


class Ice(Radprop):
    def __init__(self):
        super().__init__()
        self.hdul = fits.open(self._base_path / 'ice.fits.gz')


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

    # GCM
    gcm = Ames()

    # Read in radprop
    # I have no f****** clue why I can't call these methods later on. If I define variables here, the multiprocessing
    # works. If not, it breaks when these are called in parallel. It works when called in series
    dust_radprop = Dust()
    dust_particle_sizes = dust_radprop.get_particle_sizes()
    dust_wavelengths = dust_radprop.get_wavelengths()
    dust_extinction_cross_sections = dust_radprop.get_extinction_cross_sections()
    dust_scattering_cross_sections = dust_radprop.get_scattering_cross_sections()
    dust_legendre_coefficients = dust_radprop.get_legendre_coefficients()

    ice_radprop = Ice()
    ice_particle_sizes = ice_radprop.get_particle_sizes()
    ice_wavelengths = ice_radprop.get_wavelengths()
    ice_extinction_cross_sections = ice_radprop.get_extinction_cross_sections()
    ice_scattering_cross_sections = ice_radprop.get_scattering_cross_sections()
    ice_legendre_coefficients = ice_radprop.get_legendre_coefficients()

    n_streams = 8
    n_polar = 1  # defined by IUVS' viewing geometry
    n_azimuth = 1  # defined by IUVS' viewing geometry

    # Get the surface phase function
    marci_wavelengths = [0.260, 0.320]
    w6 = fits.open('/mnt/science/data_lake/mars/band6_w.fits')
    w7 = fits.open('/mnt/science/data_lake/mars/band7_w.fits')
    w6 = np.roll(w6['primary'].data, 180, axis=1)
    w7 = np.roll(w7['primary'].data, 180, axis=1)

    hapkew_lat = np.linspace(-90, 90, num=180)
    hapkew_lon = np.linspace(0, 360, num=360)

    stacked_hapke_w = np.dstack([w6, w7])
    hapke_w = RegularGridInterpolator((hapkew_lat, hapkew_lon, marci_wavelengths), stacked_hapke_w, bounds_error=False, fill_value=None)

    # Read in the topography
    mola = np.load('/mnt/science/data_lake/mars/maps/mola-topography.npy')
    mola_lat = np.linspace(90, -90, num=1440)
    mola_lon = np.linspace(0, 360, num=2880)

    # TODO remove: warnings.filterwarnings("error")

    def process_file(fileno: int):
        print('start processing')
        hdul = fits.open(l1b_files[fileno])
        wavelengths = readsav(wavelength_files[fileno])['wavelength_muv'] / 1000  # convert to microns
        radiance = np.load(radiance_files[fileno]) * 24/22

        # Get the data from the l1b file
        pixelgeometry = hdul['pixelgeometry'].data
        latitude = pixelgeometry['pixel_corner_lat']
        longitude = pixelgeometry['pixel_corner_lon']
        local_time = pixelgeometry['pixel_local_time']
        solar_zenith_angle = pixelgeometry['pixel_solar_zenith_angle']
        emission_angle = pixelgeometry['pixel_emission_angle']
        phase_angle = pixelgeometry['pixel_phase_angle']
        tangent_altitude = pixelgeometry['pixel_corner_mrh_alt'][..., 4]

        # Make the azimuth angles
        azimuth = pyrt.azimuth(solar_zenith_angle, emission_angle, phase_angle)

        mu0 = np.cos(np.radians(solar_zenith_angle))
        mu = np.cos(np.radians(emission_angle))

        global retrieval  # this makes the function global so multiprocessing can pickle it. Very strange...

        def retrieval(integration: int, spatial_bin: int):
            print(f'start retrieval {integration}, {spatial_bin}')
            # Exit if the pixel is not retrievable
            if solar_zenith_angle[integration, spatial_bin] >= 72 or \
                    emission_angle[integration, spatial_bin] >= 72 or \
                    tangent_altitude[integration, spatial_bin] != 0:
                answer = np.zeros((2,)) * np.nan
                return integration, spatial_bin, answer, np.nan, np.zeros(6, ) * np.nan

            pixel_wavs = wavelengths[spatial_bin, :]

            ##############
            # Equation of state
            ##############
            # Get the nearest neighbor values of the lat/lon/lt values (for speed I'm not using linear interpolation)
            pixel_lat = latitude[integration, spatial_bin, 4]
            pixel_lon = longitude[integration, spatial_bin, 4]
            pixel_lt = local_time[integration, spatial_bin]

            latitude_idx = gcm.get_nearest_latitude_index(pixel_lat)
            longitude_idx = gcm.get_nearest_longitude_index(pixel_lon)
            seasonal_idx = gcm.get_nearest_seasonal_index(sol)
            lt_idx = gcm.get_nearest_local_time_index(pixel_lon, pixel_lt)

            surface_temperature = gcm.surface_temperature[seasonal_idx, lt_idx, latitude_idx, longitude_idx]
            pixel_temperature = np.insert(gcm.atmospheric_temperature[seasonal_idx, lt_idx, :, latitude_idx, longitude_idx], 0, surface_temperature)
            surface_pressure = gcm.surface_pressure[seasonal_idx, lt_idx, latitude_idx, longitude_idx]
            pixel_pressure = gcm.get_atmospheric_pressure(surface_pressure)
            z = -np.log(pixel_pressure / pixel_pressure[-1]) * 10

            # Make the pressure scale factor
            gcm_lat_idx = np.argmin(np.abs(mola_lat - gcm.latitude_bin_centers[latitude_idx]))
            gcm_lon_idx = np.argmin(np.abs(mola_lon - gcm.longitude_bin_centers[longitude_idx]))
            iuvs_lat_idx = np.argmin(np.abs(mola_lat - latitude[integration, spatial_bin, 4]))
            iuvs_lon_idx = np.argmin(np.abs(mola_lon - longitude[integration, spatial_bin, 4]))
            pressure_scale_factor = np.exp((mola[gcm_lat_idx, gcm_lon_idx] - mola[iuvs_lat_idx, iuvs_lon_idx]) / 10000)

            # Finally, use these to compute the column density in each "good" layer
            try:
                colden = pyrt.column_density(pixel_pressure * pressure_scale_factor, pixel_temperature, z)
            except:
                print(f'problem at integration, sb = {integration}, {spatial_bin}')


            ##############
            # Aerosol guesses
            ##############
            gcm_dust = gcm.dust_visible_optical_depth[seasonal_idx, lt_idx, latitude_idx, longitude_idx]
            gcm_ice = gcm.ice_visible_optical_depth[seasonal_idx, lt_idx, latitude_idx, longitude_idx]

            ##############
            # Rayleigh scattering
            ##############
            rayleigh_co2 = pyrt.rayleigh_co2(colden, pixel_wavs)

            ##############
            # Aerosol vertical profiles
            ##############
            dust_vprof = gcm.dust_vertical_profile[seasonal_idx, lt_idx, :, latitude_idx, longitude_idx]
            dust_vprof = dust_vprof / np.sum(dust_vprof)

            ice_vprof = gcm.ice_vertical_profile[seasonal_idx, lt_idx, :, latitude_idx, longitude_idx]
            ice_vprof = ice_vprof / np.sum(ice_vprof)

            ##############
            # Surface
            ##############
            pixel_hapke_w = hapke_w(np.array([pixel_lat, pixel_lon, 0.290]))[0]
            rhoq, rhou, emust, bemst, rho_accurate = pyrt.make_hapkeHG2roughness_surface(True, False, n_polar, n_azimuth, n_streams, mu[integration, spatial_bin], mu0[integration, spatial_bin], azimuth[integration, spatial_bin], 0, np.pi, 200, 1, 0.06, pixel_hapke_w, 0.3, 0.7, 20)

            # Choose a consistent set of wavelengths for all retrievals
            if radiance.shape[-1] == 20:
                wavelength_indices = [1, 2, 3, -8, -7, -6]
            elif radiance.shape[-1] == 19:
                wavelength_indices = [1, 2, 3, -7, -6, -5]
            elif radiance.shape[-1] == 15:
                wavelength_indices = [1, 2, 3, -3, -2, -1]

            def simulate_tau(guess: np.ndarray) -> np.ndarray:
                # print(f'guess = {guess}')
                dust_guess = guess[0]
                ice_guess = guess[1]

                simulated_toa_radiance = np.zeros((len(wavelength_indices),))# * np.nan

                # This is a hack to add bounds to the solver
                if np.any(guess < 0):
                    simulated_toa_radiance[:] = 999999
                    return simulated_toa_radiance

                for counter, wav_index in enumerate(wavelength_indices):
                    ##############
                    # Dust FSP
                    ##############
                    # Get the dust optical depth at 250 nm
                    dust_z_grad = np.linspace(1, 1.5, num=len(z)-1)
                    ext = pyrt.extinction_ratio(dust_extinction_cross_sections, dust_particle_sizes, dust_wavelengths, 0.25)
                    ext = pyrt.regrid(ext, dust_particle_sizes, dust_wavelengths, dust_z_grad, pixel_wavs)
                    dust_optical_depth = pyrt.optical_depth(dust_vprof, colden, ext, dust_guess)
                    dust_single_scattering_albedo = pyrt.regrid(
                        dust_scattering_cross_sections / dust_extinction_cross_sections,
                        dust_particle_sizes, dust_wavelengths, dust_z_grad, pixel_wavs)
                    dust_legendre = pyrt.regrid(dust_legendre_coefficients, dust_particle_sizes,
                                                dust_wavelengths, dust_z_grad, pixel_wavs)
                    dust_column = pyrt.Column(dust_optical_depth, dust_single_scattering_albedo, dust_legendre)

                    ##############
                    # Ice FSP
                    ##############
                    # Get the ice optical depth at 250 nm
                    ice_z_grad = np.linspace(5, 5, num=len(z)-1)
                    ext = pyrt.extinction_ratio(ice_extinction_cross_sections, ice_particle_sizes, ice_wavelengths, 0.25)
                    ext = pyrt.regrid(ext, ice_particle_sizes, ice_wavelengths, ice_z_grad, pixel_wavs)
                    ice_optical_depth = pyrt.optical_depth(ice_vprof, colden, ext, ice_guess)
                    ice_single_scattering_albedo = pyrt.regrid(
                        ice_scattering_cross_sections / ice_extinction_cross_sections,
                        ice_particle_sizes, ice_wavelengths, ice_z_grad, pixel_wavs)
                    ice_legendre = pyrt.regrid(ice_legendre_coefficients, ice_particle_sizes,
                                                ice_wavelengths, ice_z_grad, pixel_wavs)
                    ice_column = pyrt.Column(ice_optical_depth, ice_single_scattering_albedo, ice_legendre)

                    ##############
                    # Total atmosphere
                    ##############
                    atm = rayleigh_co2 + dust_column + ice_column

                    ##############
                    # Output arrays
                    ##############
                    n_user_levels = atm.optical_depth.shape[0] + 1
                    albedo_medium = pyrt.empty_albedo_medium(n_polar)
                    diffuse_up_flux = pyrt.empty_diffuse_up_flux(n_user_levels)
                    diffuse_down_flux = pyrt.empty_diffuse_down_flux(n_user_levels)
                    direct_beam_flux = pyrt.empty_direct_beam_flux(n_user_levels)
                    flux_divergence = pyrt.empty_flux_divergence(n_user_levels)
                    intensity = pyrt.empty_intensity(n_polar, n_user_levels, n_azimuth)
                    mean_intensity = pyrt.empty_mean_intensity(n_user_levels)
                    transmissivity_medium = pyrt.empty_transmissivity_medium(n_polar)

                    # Misc
                    user_od_output = np.zeros(n_user_levels)
                    temper = np.zeros(n_user_levels)
                    h_lyr = np.zeros(n_user_levels)

                    ##############
                    # Call DISORT
                    ##############
                    lamber = True
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
                                      np.pi, 0, 0.01, 0, 0, 1, 3400000, h_lyr,
                                      rhoq, rhou, rho_accurate, bemst, emust,
                                      0, '', direct_beam_flux,
                                      diffuse_down_flux, diffuse_up_flux, flux_divergence,
                                      mean_intensity, intensity, albedo_medium,
                                      transmissivity_medium, maxcmu=n_streams, maxulv=n_user_levels, maxmom=159)
                    simulated_toa_radiance[counter] = uu[0, 0, 0]
                return simulated_toa_radiance

            def find_best_fit(guess: np.ndarray):
                simulated_toa_reflectance = simulate_tau(guess)
                '''print(f'{reflectance[integration, spatial_bin, wavelength_indices]} \n'
                      f'{simulated_toa_reflectance} \n'
                      f'{guess}')'''
                return np.sum((simulated_toa_reflectance - radiance[integration, spatial_bin, wavelength_indices]) ** 2)

            fitted_optical_depth = minimize(find_best_fit, np.array([gcm_dust, gcm_ice]), method='Nelder-Mead', tol=1e-2, bounds=((0, 2), (0, 1))).x
            best_fit_od = np.array(fitted_optical_depth)
            sim = simulate_tau(best_fit_od)
            total_error = np.abs(radiance[integration, spatial_bin, wavelength_indices] - sim) / radiance[integration, spatial_bin, wavelength_indices]
            error = np.sum(total_error) / len(total_error)  # This is the mean relative error
            print(f'data = {radiance[integration, spatial_bin, wavelength_indices]}')
            print(f'sim = {sim}')
            print(integration, spatial_bin, best_fit_od, error)
            raise SystemExit(9)
            return integration, spatial_bin, best_fit_od, error, sim

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make a shared array
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        memmap_filename_dust = os.path.join(mkdtemp(), 'myNewFileDust.dat')
        memmap_filename_ice = os.path.join(mkdtemp(), 'myNewFileIce.dat')
        memmap_filename_error = os.path.join(mkdtemp(), 'myNewFileError.dat')
        memmap_filename_radiance = os.path.join(mkdtemp(), 'myNewFileRadiance.dat')
        retrieved_dust = np.memmap(memmap_filename_dust, dtype=float,
                                   shape=radiance.shape[:-1], mode='w+') * np.nan
        retrieved_ice = np.memmap(memmap_filename_ice, dtype=float,
                                  shape=radiance.shape[:-1], mode='w+') * np.nan
        retrieved_error = np.memmap(memmap_filename_error, dtype=float,
                                    shape=radiance.shape[:-1], mode='w+') * np.nan
        retrieved_radiance = np.memmap(memmap_filename_radiance, dtype=float,
                                       shape=radiance.shape[:-1] + (6,), mode='w+') * np.nan  # This 6 is for 6 wavelengths

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

        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus - 1)  # save one/two just to be safe. Some say it's faster
        # NOTE: if there are any issues in the argument of apply_async (here,
        # retrieve_ssa), it'll break out of that and move on to the next iteration.

        for integ in [14]:#range(radiance.shape[0]):
            for posit in [132]:#range(radiance.shape[1]):
                retrieval(integ, posit)
                #pool.apply_async(func=retrieval, args=(integ, posit), callback=make_answer)
                # print(f'starting integ {integ} and posti {posit}')

        # https://www.machinelearningplus.com/python/parallel-processing-python/
        pool.close()
        pool.join()  # I guess this postpones further code execution until the queue is finished
        fileno = f'{file}'.zfill(2)
        np.save(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}/{orbit_code}-{fileno}-dust.npy', retrieved_dust)
        np.save(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}/{orbit_code}-{fileno}-ice.npy', retrieved_ice)
        np.save(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}/{orbit_code}-{fileno}-error.npy', retrieved_error)
        np.save(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}/{orbit_code}-{fileno}-simulated_radiance.npy', retrieved_radiance)
        del retrieved_dust
        del retrieved_ice
        del retrieved_error
        del retrieved_radiance

    for file in range(len(l1b_files)):
        print(f'starting file {file}')
        process_file(file)


if __name__ == '__main__':
    import time
    t0 = time.time()
    for orb in range(3464, 3465):
        # NOTE: if there are any issues in the argument of apply_async, it'll break out of that and move on to the next iteration.
        perform_retrieval(orb)
    print(time.time() - t0)
