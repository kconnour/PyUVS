from pathlib import Path
import warnings

from astropy.io import fits
import numpy as np

import pyuvs as pu
from data import hdulist


def get_segment_orbit_channel_fits_file_paths(iuvs_fits_file_location: Path, segment: str, orbit: int, channel: str) -> list[Path]:
    orbit_block = pu.make_orbit_block(orbit)
    orbit_code = pu.make_orbit_code(orbit)
    return sorted((iuvs_fits_file_location / orbit_block).glob(f'*{segment}*{orbit_code}*{channel}*.gz'))


def remove_hduls_with_oulier_obs_id(hduls: list[hdulist], obs_id: list[int]) -> list[hdulist]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        median_obs_id = np.median(obs_id)
    outlier_files = [f for c, f in enumerate(hduls) if obs_id[c] != median_obs_id]
    for file in outlier_files:
        hduls.remove(file)
    return hduls


def get_apoapse_hduls(iuvs_fits_file_location: Path, orbit: int, channel: str) -> list[hdulist]:
    data_file_paths = get_segment_orbit_channel_fits_file_paths(iuvs_fits_file_location, 'apoapse', orbit, channel)
    hduls = [fits.open(f) for f in data_file_paths]
    obs_id = [f['primary'].header['obs_id'] for f in hduls]
    return remove_hduls_with_oulier_obs_id(hduls, obs_id)


def get_apoapse_muv_hduls(orbit: int, iuvs_fits_file_location: Path) -> list[hdulist]:
    return get_apoapse_hduls(iuvs_fits_file_location, orbit, 'muv')


def get_apoapse_muv_failsafe_hduls(orbit: int, iuvs_fits_file_location: Path) -> list[hdulist]:
    apoapse_hduls = get_apoapse_muv_hduls(orbit, iuvs_fits_file_location)
    mcp_voltage = [f['observation'].data['mcp_volt'][0] for f in apoapse_hduls]
    failsafe = [np.isclose(f, pu.apoapse_muv_failsafe_voltage) for f in mcp_voltage]
    return [f for c, f in enumerate(apoapse_hduls) if failsafe[c]]


def get_apoapse_muv_dayside_hduls(orbit: int, iuvs_fits_file_location: Path) -> list[hdulist]:
    apoapse_hduls = get_apoapse_muv_hduls(orbit, iuvs_fits_file_location)
    mcp_voltage = [f['observation'].data['mcp_volt'][0] for f in apoapse_hduls]
    failsafe = [np.isclose(f, pu.apoapse_muv_failsafe_voltage) for f in mcp_voltage]
    nightside = [f >= pu.apoapse_muv_day_night_voltage_boundary for f in mcp_voltage]
    return [f for c, f in enumerate(apoapse_hduls) if not failsafe[c] and not nightside[c]]


def get_apoapse_muv_nightside_hduls(orbit: int, iuvs_fits_file_location: Path) -> list[hdulist]:
    apoapse_hduls = get_apoapse_muv_hduls(orbit, iuvs_fits_file_location)
    mcp_voltage = [f['observation'].data['mcp_volt'][0] for f in apoapse_hduls]
    nightside = [f >= pu.apoapse_muv_day_night_voltage_boundary for f in mcp_voltage]
    return [f for c, f in enumerate(apoapse_hduls) if nightside[c]]
