from pathlib import Path

from h5py import File
import matplotlib.pyplot as plt
import numpy as np

import pyuvs as pu


def make_apoapse_muv_cylindrical_map(orbit: int) -> None:
    """Make cylindrical maps for Gil

    Parameters
    ----------
    orbit
        The orbit number.

    Returns
    -------
    None

    Notes
    -----
    Gil wants the following:
    - An image in a 2:1 aspect ratio with no ticks or anything
    - black background
    - the lowest DPI possible where info won't be lost

    I don't know how exactly to compute the proper DPI, so I've taken a guess.
    A pixel can be as little as 7km across, so 2 * 3.14 * 3400 / 7 = ~3000.
    If I make a figure of width 10in and 300 DPI, I'd have 3000 dots at the
    equator. For most times this will be overkill, but it's what he wanted...

    """
    # Load in the relevant data
    file_path = Path('/media/kyle/iuvs/data/')
    orbit_block = pu.make_orbit_block(orbit)
    orbit_code = pu.make_orbit_code(orbit)

    f = File(file_path / orbit_block / f'{orbit_code}.hdf5')
    dayside = f['apoapse/muv/integration/dayside_integrations'][:]
    dayside_integrations = f['apoapse/muv/integration/dayside_integrations'][:]
    opportunity_integrations = f['apoapse/integration/opportunity_classification'][:]
    dayside_science_integrations = np.logical_and(dayside_integrations, ~opportunity_integrations)
    if np.sum(dayside_science_integrations) == 0:
        return

    opportunity_swaths = f['apoapse/integration/opportunity_classification'][:]
    brightness = f['apoapse/muv/dayside/detector/brightness'][~opportunity_swaths[dayside]]
    tangent_altitude = f['apoapse/muv/dayside/spatial_bin_geometry/tangent_altitude'][:]
    solar_zenith_angle = f['apoapse/muv/dayside/spatial_bin_geometry/solar_zenith_angle'][:]
    latitude = f['apoapse/muv/dayside/spatial_bin_geometry/latitude'][~opportunity_swaths[dayside]]
    longitude = f['apoapse/muv/dayside/spatial_bin_geometry/longitude'][~opportunity_swaths[dayside]]

    # Colorize the data
    dayside_on_disk_pixels = np.logical_and(tangent_altitude[..., 4] == 0, solar_zenith_angle <= 102)
    colored_data = pu.graphics.histogram_equalize_detector_image(brightness, mask=dayside_on_disk_pixels) / 255
    dayside_off_disk_pixels = np.all(tangent_altitude != 0, axis=-1)
    colored_data[dayside_off_disk_pixels] = np.nan

    # Find the pixels that wrap around the planet
    wrapped_pixels = np.logical_and(
        np.any(longitude > 300, axis=-1),
        np.any(longitude < 60, axis=-1))

    # Setup the figure
    fig = plt.figure(figsize=(10, 5), facecolor=(0, 0, 0, 1))
    ax = fig.add_axes((0, 0, 1, 1))

    # Plot data, integration by integration. The strategy is to plot all the
    #  pixels that do not wrap around the planet, then shift everything to the
    #  left and plot the left pixels, then shift everything right and plot the
    #  right pixels
    for integration in range(brightness.shape[0]):
        if not integration:
            continue

        latitude_grid = pu.graphics.make_single_integration_geographic_grid(latitude[integration])
        longitude_grid = pu.graphics.make_single_integration_geographic_grid(longitude[integration])

        # Account for times when our data have NaNs (which isn't often and idk why they're there in the first place)
        if np.any(np.isnan(latitude_grid)):
            plt.close(fig)
            return

        # Plot the non-wrapped pixels
        image = np.copy(colored_data[integration])
        image[wrapped_pixels[integration]] = np.nan
        ax.pcolormesh(longitude_grid, latitude_grid, image[None, :], linewidth=0, edgecolors='none', rasterized=True, shading='flat')

        # Plot the left side pixels
        longitude_grid[longitude_grid > 90] -= 360
        image = np.copy(colored_data[integration])
        image[~wrapped_pixels[integration]] = np.nan
        ax.pcolormesh(longitude_grid, latitude_grid, image[None, :], linewidth=0, edgecolors='none', rasterized=True)

        # Plot the right side pixels
        longitude_grid += 360
        ax.pcolormesh(longitude_grid, latitude_grid, image[None, :], linewidth=0, edgecolors='none', rasterized=True)

    # Configure the plot parameters
    ax.set_frame_on(False)
    ax.set_ylim(-90, 90)
    ax.set_xlim(0, 360)
    ax.set_yticks([])
    ax.set_xticks([])

    # Save the graphic
    save_location = Path('/mnt/science/images/gil/cylindrical')
    filename = f'{orbit_code}_cylindrical-map.png'
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    for orb in range(16400, 17200):
        print(orb)
        try:
            make_apoapse_muv_cylindrical_map(orb)
        except IndexError:
            print(f"{orb} failed")
