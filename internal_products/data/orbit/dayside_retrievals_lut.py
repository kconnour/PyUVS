"""This script will create the LUT for use in retrievals. It uses the pyDISORT
and pyRT_DISORT packages from my Github (or, possibly, by the time you read this
they'll belong to Mike Wolff). I suggest making the ice and dust grid have
a larger range and a finer resolution; that way, you can just use nearest
neighbor interpolation when finding the optical depths.

Several parts of this script are held together by duct tape so I suggest
modifying it as little as possible if you choose to use the LUT approach.

--Kyle, August 2024

"""
from pathlib import Path
import time
import os
from tempfile import mkdtemp
import multiprocessing as mp

import numpy as np
from astropy.io import fits
import pyrt
import disort

iuvs_wavelengths = np.array([0.210, 0.215, 0.220, 0.272, 0.278, 0.284]) - 0.005
z = np.linspace(100, 0, num=15)
altitude_midpoint = (z[:-1] + z[1:]) / 2
temperature = np.ones(15,) * 170

################
## Define Radprop
################

class Radprop:
    def __init__(self):
        self._base_path = Path('/media/kyle/iuvs/radiative_properties/wolff')
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
        return self._get_file_legendre_coefficients()  # This is stupid but it needs a quick fix

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
        self.hdul = fits.open(self._base_path / 'dust01-mars045i_all_area_s0780.fits.gz')


class Ice(Radprop):
    def __init__(self):
        super().__init__()
        self.hdul = fits.open(self._base_path / 'ice01-droxtal_050_tmat1_reff_v010.fits.gz')


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

##############
# Output arrays
##############
n_user_levels = z.shape[0]
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

# 8 SZAs (0--70), 8 EAs (0--70), 19 azimuth angles (0--180),
# 8 Hapke w's, 10 Psurf (same ones as Mike), 21 dust OD (0--2), 11 ice OD (0--1),
# 6 wavelengths (3 short wavelength and 3 long wavelength)
memmap_filename_lut = os.path.join(mkdtemp(), 'myNewFileLUT.dat')
memmap_filename_lut0 = os.path.join(mkdtemp(), 'myNewFileLUT0.dat')
lut = np.memmap(memmap_filename_lut, dtype=float, shape=(8, 8, 19, 8, 10, 21, 11, 6), mode='w+') * np.nan
lut0 = np.memmap(memmap_filename_lut0, dtype=float, shape=(8, 8, 19, 8, 10, 21, 11, 6), mode='w+') * np.nan


def process_lut(a, sza, b, ea):
    for c, azimuth in enumerate(np.arange(19) * 10):
        mu0 = np.cos(np.radians(sza))
        mu = np.cos(np.radians(ea))

        for cc, hapke_w in enumerate(np.arange(8)/100 + 0.05):
            clancy_albedo = np.linspace(0.01, 0.015, num=100)
            clancy_wavs = np.linspace(0.258, 0.32, num=100)
            lambert_albedo = np.interp(iuvs_wavelengths, clancy_wavs, clancy_albedo)

            rhoq, rhou, emust, bemst, rho_accurate = pyrt.make_hapkeHG2roughness_surface(True, False, n_polar, n_azimuth, n_streams, mu, mu0, azimuth, 0, np.pi, 200, 1, 0.06, hapke_w, 0.3, 0.7, 20)

            for d, surface_pressure in enumerate(np.linspace(0.3, 13.3, 10)):
                pressure = surface_pressure * 100 * np.exp(-z / 10)
                colden = pyrt.column_density(pressure, temperature, z)
                rayleigh_co2 = pyrt.rayleigh_co2(colden, iuvs_wavelengths)

                dust_profile = pyrt.conrath(altitude_midpoint, 1, 10, 0.01)
                dust_profile = dust_profile / np.sum(dust_profile)
                ice_profile = np.logical_and(20 <= altitude_midpoint, altitude_midpoint <= 30)
                ice_profile = ice_profile / np.sum(ice_profile)

                for e, dust_od in enumerate(np.arange(21) / 10):
                    ##############
                    # Dust FSP
                    ##############
                    # Get the dust optical depth at 250 nm
                    dust_z_grad = np.linspace(1, 1.5, num=len(z) - 1)
                    ext = pyrt.extinction_ratio(dust_extinction_cross_sections, dust_particle_sizes, dust_wavelengths, 0.25)
                    ext = pyrt.regrid(ext, dust_particle_sizes, dust_wavelengths, dust_z_grad, iuvs_wavelengths)
                    dust_optical_depth = pyrt.optical_depth(dust_profile, colden, ext, dust_od)
                    dust_single_scattering_albedo = pyrt.regrid(
                        dust_scattering_cross_sections / dust_extinction_cross_sections,
                        dust_particle_sizes, dust_wavelengths, dust_z_grad, iuvs_wavelengths)
                    dust_legendre = pyrt.regrid(dust_legendre_coefficients, dust_particle_sizes,
                                                dust_wavelengths, dust_z_grad, iuvs_wavelengths)
                    dust_column = pyrt.Column(dust_optical_depth, dust_single_scattering_albedo, dust_legendre)
                    for f, ice_od in enumerate(np.arange(11) / 10):
                        ##############
                        # Ice FSP
                        ##############
                        # Get the ice optical depth at 250 nm
                        ice_z_grad = np.linspace(5, 5, num=len(z) - 1)
                        ext = pyrt.extinction_ratio(ice_extinction_cross_sections, ice_particle_sizes, ice_wavelengths, 0.25)
                        ext = pyrt.regrid(ext, ice_particle_sizes, ice_wavelengths, ice_z_grad, iuvs_wavelengths)
                        ice_optical_depth = pyrt.optical_depth(ice_profile, colden, ext, ice_od)
                        ice_single_scattering_albedo = pyrt.regrid(
                            ice_scattering_cross_sections / ice_extinction_cross_sections,
                            ice_particle_sizes, ice_wavelengths, ice_z_grad, iuvs_wavelengths)
                        ice_legendre = pyrt.regrid(ice_legendre_coefficients, ice_particle_sizes,
                                                   ice_wavelengths, ice_z_grad, iuvs_wavelengths)
                        ice_column = pyrt.Column(ice_optical_depth, ice_single_scattering_albedo, ice_legendre)

                        ##############
                        # Total atmosphere + DISORT call
                        ##############
                        atm = rayleigh_co2 + dust_column + ice_column

                        for g, wavelength in enumerate(iuvs_wavelengths):
                            rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
                                disort.disort(True, False, False, False, [False, False, False, False, False],
                                              False, False, True, False,
                                              atm.optical_depth[:, g],
                                              atm.single_scattering_albedo[:, g],
                                              atm.legendre_coefficients[:, :, g],
                                              temper, 1, 1, user_od_output,
                                              mu0, 0, mu, azimuth,
                                              np.pi, 0, lambert_albedo[g], 0, 0, 1, 3400000, h_lyr,
                                              rhoq, rhou, rho_accurate, bemst, emust,
                                              0, '', direct_beam_flux,
                                              diffuse_down_flux, diffuse_up_flux, flux_divergence,
                                              mean_intensity, intensity, albedo_medium,
                                              transmissivity_medium, maxcmu=n_streams, maxulv=n_user_levels, maxmom=160)

                            lut[a, b, c, cc, d, e, f, g] = uu[0, 0, 0]
    return lut, a, b


def make_lut():
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus - 1)  # save one/two just to be safe. Some say it's faster
    # NOTE: if there are any issues in the argument of apply_async (here,
    # retrieve_ssa), it'll break out of that and move on to the next iteration.

    for a, sza in enumerate(np.arange(8) * 10):
        for b, ea in enumerate(np.arange(8) * 10):
            process_lut(a, sza, b, ea)
            pool.apply_async(func=process_lut, args=(a, sza, b, ea), callback=make_array)

    pool.close()
    pool.join()  # I guess this postpones further code execution until the queue is finished
    np.save(f'/mnt/science/mars/missions/maven/instruments/iuvs/lut.npy', lut0)


def make_array(input):
    # I have no idea why this bit is necessary. I couldn't get it to just save the lut after doing the multiprocessing.
    arr = input[0]
    ind0 = input[1]
    ind1 = input[2]
    lut0[ind0, ind1] = arr[ind0, ind1]


if __name__ == '__main__':
    t0 = time.time()
    make_lut()
    print(time.time() - t0)
