from astropy.io import fits
import numpy as np

from spacecraft_geometry import make_spacecraft_velocity_inertial_frame

hdulist = fits.hdu.hdulist.HDUList


def make_instrument_sun_angle(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['inst_sun_angle'] for f in hduls]
    ) if hduls else np.array([])


def make_app_flip(hduls: list[hdulist]) -> np.ndarray:
    instrument_x_field_of_view_inertial_frame = make_instrument_x_field_of_view_inertial_frame(hduls)
    spacecraft_velocity_inertial_frame = make_spacecraft_velocity_inertial_frame(hduls)
    try:
        dot = instrument_x_field_of_view_inertial_frame * \
              spacecraft_velocity_inertial_frame
        app_flip = np.array([np.sum(dot) > 0])
    except IndexError:
        app_flip = np.array([])
    return app_flip


def make_instrument_x_field_of_view_iau_mars_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['vx_instrument'] for f in hduls]
    ) if hduls else np.array([])


def make_instrument_y_field_of_view_iau_mars_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['vy_instrument'] for f in hduls]
    ) if hduls else np.array([])


def make_instrument_z_field_of_view_iau_mars_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['vz_instrument'] for f in hduls]
    ) if hduls else np.array([])


def make_instrument_x_field_of_view_inertial_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['vx_instrument_inertial'] for f in hduls]
    ) if hduls else np.array([])


def make_instrument_y_field_of_view_inertial_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['vy_instrument_inertial'] for f in hduls]
    ) if hduls else np.array([])


def make_instrument_z_field_of_view_inertial_frame(hduls: list[hdulist]) -> np.ndarray:
    return np.concatenate(
        [f['spacecraftgeometry'].data['vz_instrument_inertial'] for f in hduls]
    ) if hduls else np.array([])
