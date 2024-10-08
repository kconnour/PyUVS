from pathlib import Path
import warnings

from h5py import File
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pyuvs as pu
from paths import apsis_file_path


def checkerboard() -> np.ndarray:
    """Create an 5-degree-size RGB checkerboard array.

    Parameters
    ----------
    None

    Returns
    -------
    np.ndarray
        The checkerboard grid.

    """
    return np.repeat(np.kron([[0.67, 0.33] * 36, [0.33, 0.67] * 36] * 18, np.ones((5, 5)))[:, :, None], 3, axis=2)


def plot_apoapse_muv_dayside_globe_upright(orbit: int) -> None:
    """Plot an upright globe of how Mars looked from apoapsis from a given
    orbit.

    Parameters
    ----------
    orbit
        The orbit number.

    Returns
    -------
    None

    Notes
    -----
    Matteo specified that he wants the following properties for the image:
    - upright globe with only dayside info
    - checkerboard background (on-disk) and black background (off-disk)
    - HEQ coloring
    - the custom filename he provided
    - 145 DPI

    """
    # Load in the relevant data
    file_path = Path('/media/kyle/iuvs/data/')
    orbit_block = pu.make_orbit_block(orbit)
    orbit_code = pu.make_orbit_code(orbit)

    f = File(file_path / orbit_block / f'{orbit_code}.hdf5')
    dayside_integrations = f['apoapse/muv/integration/dayside_integrations'][:]
    opportunity_integrations = f['apoapse/integration/opportunity_classification'][:]
    dayside_science_integrations = np.logical_and(dayside_integrations, ~opportunity_integrations)

    # If we didn't take dayside data on a given orbit, don't try to make a globe for that orbit.
    if np.sum(dayside_science_integrations) == 0:
        return

    brightness = f['apoapse/muv/dayside/detector/brightness'][~opportunity_integrations[dayside_integrations]]
    tangent_altitude = f['apoapse/muv/dayside/spatial_bin_geometry/tangent_altitude'][:]
    solar_zenith_angle = f['apoapse/muv/dayside/spatial_bin_geometry/solar_zenith_angle'][:]
    latitude = f['apoapse/muv/dayside/spatial_bin_geometry/latitude'][~opportunity_integrations[dayside_integrations]]
    longitude = f['apoapse/muv/dayside/spatial_bin_geometry/longitude'][~opportunity_integrations[dayside_integrations]]
    swath_number = f['apoapse/integration/swath_number'][dayside_integrations]
    subspacecraft_latitude = f['apoapse/apsis/subspacecraft_latitude'][0]
    subspacecraft_longitude = f['apoapse/apsis/subspacecraft_longitude'][0]
    spacecraft_altitude = f['apoapse/apsis/spacecraft_altitude'][0]

    # Make a bounding box such that the image represents 8000 km x 8000 km
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

    # Plot black all over the globe to represent points where no data were taken
    axis.imshow(checkerboard()*0, transform=transform, extent=(-180, 180, -90, 90))

    # Colorize the image (turn kR to RGB)
    mask = np.logical_and(tangent_altitude[..., 4] == 0, solar_zenith_angle <= 102)
    #mask = np.logical_and(tangent_altitude[..., 4] == 0, solar_zenith_angle >= 90)
    image = pu.graphics.histogram_equalize_detector_image(brightness, mask=mask) / 255
    # Plot the RGB values onto the globe, each swath at a time. I don't do them
    #  all at once, otherwise there would be rogue integrations between the
    #  swath boundaries.
    for swath in np.unique(swath_number):
        swath_indices = swath_number == swath

        latitude_grid = pu.graphics.make_swath_geographic_grid(latitude[swath_indices])
        longitude_grid = pu.graphics.make_swath_geographic_grid(longitude[swath_indices])

        swath_image = image[swath_indices]

        axis.pcolormesh(longitude_grid, latitude_grid, swath_image,
                        linewidth=0, edgecolors='none', transform=transform,
                        rasterized=True, zorder=swath)


        axis.contour(longitude[..., -1][swath_indices], latitude[..., -1][swath_indices], solar_zenith_angle[swath_indices], [90, 100],
                     transform=transform, colors='red', linewidths=0.5, zorder=swath)

    filename = f'{orbit_code}_globe_heq'

    # Save the graphic
    save_location = Path('/mnt/science/images/high_altitude_plume_rebuttal')
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    plot_apoapse_muv_dayside_globe_upright(8889)
