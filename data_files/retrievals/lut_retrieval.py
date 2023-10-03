from pathlib import Path
import multiprocessing as mp
import os
from tempfile import mkdtemp
import math

from astropy.io import fits
from h5py import File
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

import pyrt

from ames import Ames


for orbit in range(3464, 3465):
    block = math.floor(orbit / 100) * 100
    block_code = 'orbit' + f'{block}'.zfill(5)
    orbit_code = 'orbit' + f'{orbit}'.zfill(5)
    # Read in external files
    lut = np.load('/home/kyle/iuvs/lut-16streams.npy')
    radiance_files = sorted(Path(f'/mnt/science/data/mars/maven/iuvs/radiance/{block_code}').glob(f'orbit*{orbit}*'))
    fits_files = sorted(Path(f'/mnt/science/data_lake/mars/maven/iuvs/production/{block_code}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
    hduls = [fits.open(f) for f in fits_files]
    apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')

    # Define save location
    save_location = Path('/mnt/science/data/mars/maven/iuvs/retrievals')

    # Make the sol info
    sol = apsis_file['apoapse/sol'][orbit]

    # Make the interpolation
    sza_grid = np.arange(8) * 10
    ea_grid = np.arange(8) * 10
    azimuth_grid = np.arange(19) * 10
    hapke_w_grid = np.arange(8)/100 + 0.05
    pressure_grid = np.linspace(0.3, 13.3, 10) * 100
    dust_grid = np.arange(21) / 10
    ice_grid = np.arange(11) / 10
    wavelength_grid = np.array([0.210, 0.215, 0.220, 0.272, 0.278, 0.284])

    interp = RegularGridInterpolator((sza_grid, ea_grid, azimuth_grid, hapke_w_grid, pressure_grid, dust_grid, ice_grid, wavelength_grid), lut, bounds_error=False, fill_value=None)

    # Get pressure info
    gcm = Ames()
    seasonal_idx = gcm.get_nearest_seasonal_index(sol)
    season_surface_pressure = gcm.surface_pressure[seasonal_idx, :, :, :]
    sfc_interp = RegularGridInterpolator((gcm.local_time_bin_centers[:], gcm.latitude_bin_centers[:], gcm.longitude_bin_centers[:]), season_surface_pressure, bounds_error=False, fill_value=None)

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

    for counter, radiance_file in enumerate(radiance_files):
        radiance = np.load(radiance_file) * 24/22  # This factor is just so I don't have to reprocess the radiance files
        szas = hduls[counter]['pixelgeometry'].data['pixel_solar_zenith_angle']
        eas = hduls[counter]['pixelgeometry'].data['pixel_emission_angle']
        pas = hduls[counter]['pixelgeometry'].data['pixel_phase_angle']
        alts = hduls[counter]['pixelgeometry'].data['pixel_corner_mrh_alt'][:, :, 4]
        azimuths = pyrt.azimuth(szas, eas, pas)
        latitudes = hduls[counter]['pixelgeometry'].data['pixel_corner_lat'][:, :, 4]
        longitudes = hduls[counter]['pixelgeometry'].data['pixel_corner_lon'][:, :, 4]
        local_time = hduls[counter]['pixelgeometry'].data['pixel_local_time']

        if radiance.shape[-1] == 20:
            wavelength_indices = [1, 2, 3, -8, -7, -6]
        elif radiance.shape[-1] == 19:
            wavelength_indices = [1, 2, 3, -7, -6, -5]
        elif radiance.shape[-1] == 15:
            wavelength_indices = [1, 2, 3, -3, -2, -1]

        memmap_filename_answer = os.path.join(mkdtemp(), 'myNewFileAnswer.dat')
        answer = np.memmap(memmap_filename_answer, dtype=float, shape=latitudes.shape + (3 + len(wavelength_indices),), mode='w+') * np.nan # 9 is for dust, ice, error, simulated I/F at 6 wavelengths
        answer0 = np.zeros(latitudes.shape + (3 + len(wavelength_indices),)) * np.nan

        def fit_dust_and_ice(rad, sza, ea, aza, psurf, pix_hapke_w) -> float:
            def find_best_fit(guess: np.ndarray):
                dust_guess = guess[0]
                ice_guess = guess[1]
                simulated_spectrum = np.zeros(wavelength_grid.shape) * np.nan
                for c, wav in enumerate(wavelength_grid):
                    input = np.array([sza, ea, aza, pix_hapke_w, psurf, dust_guess, ice_guess, wav])
                    simulated_spectrum[c] = interp(input)[0]
                return np.sum((simulated_spectrum - rad[wavelength_indices])**2)

            return minimize(find_best_fit, np.array([0.7, 0.2]), method='Nelder-Mead', bounds=((0, 2), (0, 1))).x
            #return minimize(find_best_fit, np.array([0.7]), method='Nelder-Mead', tol=1e-2, bounds=((0, 2),), options={'adaptive': True}).x

        def make_array(input):
            # I have no idea why this bit is necessary. I couldn't get it to just save the lut after doing the multiprocessing.
            ind0 = input[0]
            ind1 = input[1]
            arr = input[2]

            answer0[ind0, ind1] = arr

        def do_retrieval(integration, spatial_bin):
            print(integration, spatial_bin)
            if alts[integration, spatial_bin] != 0 or szas[integration, spatial_bin] > 70 or eas[integration, spatial_bin] > 70:
                return integration, spatial_bin, np.zeros((3 + len(wavelength_indices),)) * np.nan
            latitude_idx = gcm.get_nearest_latitude_index(latitudes[integration, spatial_bin])
            longitude_idx = gcm.get_nearest_longitude_index(longitudes[integration, spatial_bin])
            lt_idx = gcm.get_nearest_local_time_index(longitudes[integration, spatial_bin], local_time[integration, spatial_bin])
            sfc_pressure = season_surface_pressure[lt_idx, latitude_idx, longitude_idx] * 0.82   # The 0.82 is an empirical factor to account for the fact that I assumed an isothermal profile in the LUT

            pixel_hapke_w = hapke_w(np.array([latitudes[integration, spatial_bin], longitudes[integration, spatial_bin], 0.290]))[0]   # hack... say the hapke w is spatially variable but not spectrally variable. Doing it spectrally would require code redesign
            answer[integration, spatial_bin, :2] = fit_dust_and_ice(radiance[integration, spatial_bin], szas[integration, spatial_bin], eas[integration, spatial_bin], azimuths[integration, spatial_bin], sfc_pressure, pixel_hapke_w)

            # Get the spectrum of the best fit answer
            sim_spec = np.zeros(len(wavelength_indices))
            for c, wav in enumerate(wavelength_grid):
                sim_spec[c] = interp(np.array([szas[integration, spatial_bin], eas[integration, spatial_bin], azimuths[integration, spatial_bin], pixel_hapke_w, sfc_pressure, answer[integration, spatial_bin, 0], answer[integration, spatial_bin, 1], wav]))[0]
            error = np.sum((radiance[integration, spatial_bin, wavelength_indices] - sim_spec)**2)

            # Add the spectrum to the array
            answer[integration, spatial_bin, 2] = error
            answer[integration, spatial_bin, 3:] = sim_spec
            print(answer[integration, spatial_bin, :2])
            print(error)
            print(answer[integration, spatial_bin, :2])
            print(radiance[integration, spatial_bin, wavelength_indices])
            print(sim_spec)
            print(radiance[integration, spatial_bin, wavelength_indices] / sim_spec)
            raise SystemExit(9)
            return integration, spatial_bin, answer[integration, spatial_bin]

        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus - 1)
        for integration in [120]: # range(radiance.shape[0]):
            for spatial_bin in [120]: #range(radiance.shape[1]):
                #pool.apply_async(func=do_retrieval, args=(integration, spatial_bin), callback=make_array)
                do_retrieval(integration, spatial_bin)

        # https://www.machinelearningplus.com/python/parallel-processing-python/
        pool.close()
        pool.join()

        # Save the file
        fn = f'{counter}'.zfill(2)
        filename = save_location / block_code / f'{orbit_code}-{fn}.npy'
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(filename), answer0)
