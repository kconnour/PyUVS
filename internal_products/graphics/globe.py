import warnings

from h5py import File
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pyuvs as pu
from paths import apsis_file_path, iuvs_images_location


def plot_apoapse_mars_map(orbit: int) -> None:
    # Get the Mars map and info
    mars_map_file = File('/mnt/science/mars/maps/mars_surface.hdf5')
    mars_map = mars_map_file['map'][:]
    latitude_boundaries = mars_map_file['latitude_boundaries'][:]
    longitude_boundaries = mars_map_file['longitude_boundaries'][:]

    # Get the apsis info
    apsis_file = File(apsis_file_path)
    orbits = apsis_file['apoapse/orbit'][:]
    subspacecraft_latitude = apsis_file['apoapse/subspacecraft_latitude'][orbits == orbit][0]
    subspacecraft_longitude = apsis_file['apoapse/subspacecraft_longitude'][orbits == orbit][0]
    spacecraft_altitude = apsis_file['apoapse/spacecraft_altitude'][orbits == orbit][0]

    # Make a bounding box such that the image is 4000 km x 4000 km
    rmars = 3400 * 10 ** 3
    image_width = 4000 * 10 ** 3
    corner_pos = (1 - rmars / image_width) / 2
    bbox = (corner_pos, corner_pos, 1 - 2 * corner_pos, 1 - 2 * corner_pos)

    # Make properties of the image
    fig = plt.figure(figsize=(7, 7), facecolor=(0, 0, 0, 0))
    globe = ccrs.Globe(semimajor_axis=rmars, semiminor_axis=rmars)
    with warnings.catch_warnings(action='ignore', category=UserWarning):
        projection = ccrs.NearsidePerspective(
            central_latitude=subspacecraft_latitude,
            central_longitude=subspacecraft_longitude,
            satellite_height=spacecraft_altitude * 10 ** 3,
            globe=globe)
    transform = ccrs.PlateCarree(globe=globe)
    axis = plt.axes(bbox, projection=projection)

    # Plot the image
    longitude_grid, latitude_grid = np.meshgrid(
        longitude_boundaries, latitude_boundaries)
    axis.pcolormesh(longitude_grid, latitude_grid, mars_map,
                    transform=transform, rasterized=True)

    # Save the graphic
    save_location = iuvs_images_location / 'apoapse' / 'globe-topography'
    orbit_code = pu.make_orbit_code(orbit)
    filename = f'{orbit_code}_globe.png'
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    plot_apoapse_mars_map(3400)
