from pathlib import Path
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from h5py import File
import mars_time as mt


if __name__ == '__main__':
    apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
    all_orbits = apsis_file['apoapse/orbit'][:]
    sol = apsis_file['apoapse/sol'][:]
    lt = apsis_file['apoapse/subspacecraft_local_time'][:]
    ls = apsis_file['apoapse/solar_longitude'][:]
    subsclon = apsis_file['apoapse/subspacecraft_longitude'][:]
    mars_year = apsis_file['apoapse/mars_year'][:]

    #ames_lt_idx = np.argmin(np.abs(np.arange(24)+0.5 - (lt - (subsclon / 360 * 24)) % 24))
    #ames_sol_idx = int(sol / 668 * 133) - 4   # The -4 is because John sent me a simulation from sol 10 to 10, not 0 to 0

    ames_path = Path('/mnt/science/data_lake/mars/gcm/ames/my30') / 'c48L36_my30.atmos_diurn.nc'
    ames = Dataset(ames_path)

    ames_dust = ames['dodvis'][:]
    ames_ice = ames['taucloud_VIS'][:]

    def regrid_ames(arr):
        # TODO: shift the seasonal axis by 4 to account for John's error
        # arr shape: (133, 24, 90, 180)

        ls_bounds = [mt.MarsTime(30, i).solar_longitude for i in np.arange(133)*5]
        ls_bounds[0] = 0
        ls_target_grid_right_edges = np.arange(360/5) * 5 + 5

        print(ls_bounds)
        print(ls_target_grid_right_edges)

        # testing: i commented this all out to keep the same native grid
        ls_idx = np.array([np.argmax(np.abs(ls_bounds[i] < ls_target_grid_right_edges)) for i in range(len(ls_bounds))])  # This is the bin index each point seasonal point belongs in
        regridded_arr = np.zeros((72, 24, 90, 180)) * np.nan  # 72 Ls bins 5 degrees each, 4 LT bins 3 hours each, 90 lat, 180 lon
        for i in range(72):
            idx = ls_idx == i
            if np.sum(idx) == 1:
                regridded_arr[i] = arr[idx]
            else:
                regridded_arr[i] = np.mean(arr[idx], axis=0)

        # Make a diagonalized array
        diag = np.zeros((24, 180))
        for i in range(24):
            low = int(i/24*180)
            high = int((i+1)/24*180)
            diag[i, low:high] = 1

        # Roll the diagonalized array to only pick out the LTs of interest and put them the regridded array
        reshaped_arr = np.moveaxis(regridded_arr, 2, 1)   # shape: (72, 90, 24, 180)    testing:i changed regridded_arr to arr
        answer = np.zeros((72, 24, 90, 180))   # testing: i changed 72 to 133
        for i in range(24):
            roll_amount = int(-i/24 * 180)
            rolled_diag = np.flip(np.roll(diag, roll_amount, axis=1), axis=1)   # I have no fucking goddamn fucking clue why I need the fucking flip but it fucking works with it
            data = reshaped_arr * rolled_diag
            answer[:, i, :, :] = np.sum(data, axis=-2)

        return answer

    def regrid_ames_test():
        foo = np.empty((133, 24, 90, 180))
        for i in range(24):
            foo[:, i] = i

        bar = regrid_ames(foo)
        assert bar[:, 0] == np.zeros((133, 90, 180))
        assert bar[:, 1] == np.zeros((133, 90, 180)) + 1
        assert bar[:, 2] == np.zeros((133, 90, 180)) + 2


    fig, ax = plt.subplots(3, 2)

    # Add in Ames dust and ice
    ames_X = np.arange(73) * 5
    ames_Y = np.arange(91) * 2 - 90

    ames_regridded_dust = regrid_ames(ames_dust)
    ames_regridded_ice = regrid_ames(ames_ice)

    ames_dust_to_plot = np.zeros((4, 72, 90))
    ames_ice_to_plot = np.zeros((4, 72, 90))
    for lt in range(4):
        ames_dust_to_plot[lt] = np.mean(np.mean(ames_regridded_dust[:, lt*3+6: (lt+1)*3+6], axis=-1), axis=1)
        ames_ice_to_plot[lt] = np.mean(np.mean(ames_regridded_ice[:, lt * 3 + 6: (lt + 1) * 3 + 6], axis=-1), axis=1)

    ax[1, 0].pcolormesh(ames_X, ames_Y, ames_dust_to_plot[0].T, vmin=0, vmax=2, cmap='cividis')
    ax[1, 1].pcolormesh(ames_X, ames_Y, ames_ice_to_plot[0].T, vmin=0, vmax=1, cmap='viridis')

    plt.savefig('/home/kyle/thesis/season.png')
