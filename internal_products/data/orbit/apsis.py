from h5py import File
import numpy as np

from paths import apsis_file_path


file = File(apsis_file_path)


def make_ephemeris_time(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/ephemeris_time'][orbits == orbit]


def make_mars_year(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/mars_year'][orbits == orbit]


def make_sol(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/sol'][orbits == orbit]


def make_solar_longitude(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/solar_longitude'][orbits == orbit]


def make_subsolar_latitude(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/subsolar_latitude'][orbits == orbit]


def make_subsolar_longitude(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/subsolar_longitude'][orbits == orbit]


def make_subspacecraft_latitude(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/subspacecraft_latitude'][orbits == orbit]


def make_subspacecraft_longitude(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/subspacecraft_longitude'][orbits == orbit]


def make_spacecraft_altitude(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/spacecraft_altitude'][orbits == orbit]


def make_subspacecraft_local_time(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/subspacecraft_local_time'][orbits == orbit]


def make_mars_sun_distance(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/mars_sun_distance'][orbits == orbit]


def make_subsolar_subspacecraft_angle(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/subsolar_subspacecraft_angle'][orbits == orbit]


def make_mars_position(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/mars_position'][orbits == orbit]


def make_mars_velocity(orbit: int, apsis: str) -> np.ndarray:
    orbits = file[f'{apsis}/orbit'][:]
    return file[f'{apsis}/mars_velocity'][orbits == orbit]
