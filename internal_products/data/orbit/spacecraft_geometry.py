from astropy.io import fits
import numpy as np

hdulist = fits.hdu.hdulist.HDUList


def make_subsolar_latitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['sub_solar_lat'] for f in hduls]
    ) if hduls else np.array([])


def make_subsolar_longitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['sub_solar_lon'] for f in hduls]
    ) if hduls else np.array([])


def make_subspacecraft_latitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['sub_spacecraft_lat'] for f in hduls]
    ) if hduls else np.array([])


def make_subspacecraft_longitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['sub_spacecraft_lon'] for f in hduls]
    ) if hduls else np.array([])


def make_spacecraft_altitude(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['spacecraft_alt'] for f in hduls]
    ) if hduls else np.array([])


def make_spacecraft_position_iau_mars_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['v_spacecraft'] for f in hduls]
    ) if hduls else np.array([])


def make_spacecraft_velocity_iau_mars_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['v_spacecraft_rate'] for f in hduls]
    ) if hduls else np.array([])


def make_spacecraft_position_inertial_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['v_spacecraft_inertial'] for f in hduls]
    ) if hduls else np.array([])


def make_spacecraft_velocity_inertial_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['v_spacecraft_rate_inertial'] for f in hduls]
    ) if hduls else np.array([])
