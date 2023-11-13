# This will make a seasonal composite of the IUVS retrievals
import numpy as np
from h5py import File


if __name__ == '__main__':
    apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
    all_orbits = apsis_file['apoapse/orbit'][:]
    sol = apsis_file['apoapse/sol'][:]
    lt = apsis_file['apoapse/subspacecraft_local_time'][:]
    ls = apsis_file['apoapse/solar_longitude'][:]
    subsclon = apsis_file['apoapse/subspacecraft_longitude'][:]
    mars_year = apsis_file['apoapse/mars_year'][:]

    for my in [33, 34, 35]:
        idx = mars_year == my
        yearly_orbits = all_orbits[idx]
        yearly_ls = ls[idx]

        answer = np.zeros((72, 12, 90, 180))    # This is 107 MB
        for lsbin in range(72):   # These are 5 Ls in width
            ls_idx = np.logical_and(yearly_ls < lsbin * 5, (lsbin + 1) * 5 < yearly_ls)
            ls_orbits = yearly_orbits[ls_idx]

            iuvs_ames_dust = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/{orbit_block}/{orbit_code}-dust.npy')
            iuvs_ames_ice = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/{orbit_block}/{orbit_code}-ice.npy')


