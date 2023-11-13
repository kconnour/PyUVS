# This script regrids the IUVS data onto a given GCM grid
import math
import multiprocessing as mp

import numpy as np
from pathlib import Path
from astropy.io import fits

from constants import *


def regrid_ames_data(orbit: int):
    orbit_code = f'orbit' + f'{orbit}'.zfill(5)
    block = math.floor(orbit / 100) * 100
    orbit_block = 'orbit' + f'{block}'.zfill(5)

    base_path = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}')  # Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}')
    # files = sorted(list(set(sorted(base_path.glob(f'{orbit_code}*.npy'))) - set(sorted(base_path.glob(f'{orbit_code}*FF*.npy')))))
    files = sorted(base_path.glob(f'*{orbit_code}*'))
    if not files:
        return

    loadeddata = [np.load(f) for f in files]
    retrievals = np.vstack([f for f in loadeddata if f.shape[1] != 33])
    dust = retrievals[..., 0]
    ice = retrievals[..., 1]
    error = retrievals[..., 2]

    files = sorted(Path(f'/mnt/science/data_lake/mars/maven/iuvs/production/{orbit_block}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
    files = [fits.open(f) for f in files]

    lat = np.vstack([f['pixelgeometry'].data['pixel_corner_lat'] for f in files if f['primary'].data.shape[1] != 33])
    lon = np.vstack([f['pixelgeometry'].data['pixel_corner_lon'] for f in files if f['primary'].data.shape[1] != 33])
    alt = np.vstack([f['pixelgeometry'].data['pixel_corner_mrh_alt'][..., 4] for f in files if f['primary'].data.shape[1] != 33])
    fov = np.concatenate([f['integration'].data['fov_deg'] for f in files if f['primary'].data.shape[1] != 33])

    #######################
    ### Add in the IUVS data in cylindrical map form
    #######################

    # This will regrid the IUVS data onto the GCM 2x2 resolution
    dust_grid = np.zeros((90, 180)) * np.nan
    ice_grid = np.zeros((90, 180)) * np.nan

    for lat_idx, latitude in enumerate(np.arange(90) * 2 - 90):
        for lon_idx, longitude in enumerate(np.arange(180) * 2):
            print(lat_idx, lon_idx)
            lat_mask = np.logical_and(latitude < lat[..., 4], lat[..., 4] < latitude + 2)
            lon_mask = np.logical_and(longitude < lon[..., 4], lon[..., 4] < longitude + 2)
            mask = np.logical_and(lat_mask, lon_mask)

            # The nan mask should eliminate the need to account for off-disk pixels
            dust_grid[lat_idx, lon_idx] = np.nanmean(dust[mask])
            ice_grid[lat_idx, lon_idx] = np.nanmean(ice[mask])

    dust_filename = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/{orbit_block}/{orbit_code}-dust.npy')
    ice_filename = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/{orbit_block}/{orbit_code}-ice.npy')
    dust_filename.parent.mkdir(parents=True, exist_ok=True)
    np.save(dust_filename, dust_grid)
    np.save(ice_filename, ice_grid)


def regrid_pcm_data(orbit: int):
    orbit_code = f'orbit' + f'{orbit}'.zfill(5)
    block = math.floor(orbit / 100) * 100
    orbit_block = 'orbit' + f'{block}'.zfill(5)

    base_path = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}')  # Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}')
    # files = sorted(list(set(sorted(base_path.glob(f'{orbit_code}*.npy'))) - set(sorted(base_path.glob(f'{orbit_code}*FF*.npy')))))
    files = sorted(base_path.glob(f'*{orbit_code}*'))
    if not files:
        return

    loadeddata = [np.load(f) for f in files]
    retrievals = np.vstack([f for f in loadeddata if f.shape[1] != 33])
    dust = retrievals[..., 0]
    ice = retrievals[..., 1]
    error = retrievals[..., 2]

    files = sorted(Path(f'/mnt/science/data_lake/mars/maven/iuvs/production/{orbit_block}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
    files = [fits.open(f) for f in files]

    lat = np.vstack([f['pixelgeometry'].data['pixel_corner_lat'] for f in files if f['primary'].data.shape[1] != 33])
    lon = np.vstack([f['pixelgeometry'].data['pixel_corner_lon'] for f in files if f['primary'].data.shape[1] != 33])

    #lon = np.where(lon > 180, lon-360, lon)

    #######################
    ### Add in the IUVS data in cylindrical map form
    #######################

    # This will regrid the IUVS data onto the GCM lat x lon resolution
    dust_grid = np.zeros((49, 65)) * np.nan   # 3.75 degrees by 5.625
    ice_grid = np.zeros((49, 65)) * np.nan

    gcm_lat = np.flip(np.arange(49) * 3.75 - 90)

    lon_centers = np.linspace(-180, 180, num=65)
    lon_boundaries = (lon_centers[1:] + lon_centers[:-1]) / 2
    lon_boundaries = np.where(lon_boundaries < 0, lon_boundaries + 360, lon_boundaries)
    lon_boundaries = np.roll(lon_boundaries, 32)
    gcm_lon = np.concatenate(([0], lon_boundaries, [360]))

    for lat_idx, latitude in enumerate(gcm_lat):
        for lon_idx in range(len(gcm_lon[:-1])):
            print(lat_idx, lon_idx)
            lat_mask = np.logical_and(latitude < lat[..., 4], lat[..., 4] < latitude + 3.75)
            lon_mask = np.logical_and(gcm_lon[lon_idx] < lon[..., 4], lon[..., 4] < gcm_lon[lon_idx+1])
            mask = np.logical_and(lat_mask, lon_mask)

            # The nan mask should eliminate the need to account for off-disk pixels
            dust_grid[lat_idx, lon_idx] = np.nanmean(dust[mask])
            ice_grid[lat_idx, lon_idx] = np.nanmean(ice[mask])

    dust_filename = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/pcm/{orbit_block}/{orbit_code}-dust.npy')
    ice_filename = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/pcm/{orbit_block}/{orbit_code}-ice.npy')
    dust_filename.parent.mkdir(parents=True, exist_ok=True)
    np.save(dust_filename, dust_grid)
    np.save(ice_filename, ice_grid)


if __name__ == '__main__':
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus - 5)
    for orb in range(9000, 9500):
        pool.apply_async(func=regrid_ames_data, args=(orb,))
        #regrid_pcm_data(orb)

    pool.close()
    pool.join()
