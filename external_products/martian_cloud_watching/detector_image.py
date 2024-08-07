from pathlib import Path

from h5py import File
import matplotlib.pyplot as plt
import numpy as np

import pyuvs as pu
from _filename import make_filename


file_path = Path('/media/kyle/iuvs/data/')


def setup_figure(n_swaths: int, angular_width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    field_of_view = (pu.constants.maximum_mirror_angle - pu.constants.minimum_mirror_angle) * 2
    width = n_swaths * angular_width / field_of_view * height
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    return fig, ax


def setup_square_figure():
    fig = plt.figure(figsize=(7, 7), dpi=145)
    ax = fig.add_axes((0, 0, 1, 1))
    return fig, ax


def make_histogram_equalized_detector_image(orbit: int, save_location: Path) -> None:
    """Plot a HEQ detector image.

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
    - 145 DPI
    - black background

    """
    orbit_block = pu.orbit.make_orbit_block(orbit)
    orbit_code = pu.orbit.make_orbit_code(orbit)

    f = File(file_path / orbit_block / f'{orbit_code}.hdf5')

    dayside = f['apoapse/muv/integration/dayside_integrations'][:]
    opportunity_swaths = f['apoapse/integration/opportunity_classification'][:]
    dayside_science_integrations = np.logical_and(dayside, ~opportunity_swaths)

    if np.sum(dayside_science_integrations) == 0:
        return

    brightness = f['apoapse/muv/dayside/detector/brightness'][~opportunity_swaths[dayside]]
    swath_number = f['apoapse/integration/swath_number'][:][dayside_science_integrations]
    n_swaths = f['apoapse/integration/number_of_swaths'][:][0]
    tangent_altitude = f['apoapse/muv/dayside/spatial_bin_geometry/tangent_altitude'][:][..., 4]
    field_of_view = f['apoapse/integration/field_of_view'][:][dayside_science_integrations]
    solar_zenith_angle = f['apoapse/muv/dayside/spatial_bin_geometry/solar_zenith_angle'][:]
    spatial_bin_edges = f['apoapse/muv/dayside/binning/spatial_bin_edges'][:]
    app_flip = f['apoapse/instrument_geometry/app_flip'][0]

    solar_zenith_angle[tangent_altitude != 0] = np.nan

    mask = np.logical_and(tangent_altitude == 0, solar_zenith_angle <= 102)
    image = pu.graphics.histogram_equalize_detector_image(brightness, mask=mask) / 255

    angular_width = (spatial_bin_edges[-1] - spatial_bin_edges[0]) / 1024 * pu.constants.angular_detector_width

    # TODO: change as needed
    # fig, ax = setup_figure(n_swaths, angular_width, 6)
    fig, ax = setup_square_figure()

    # Plot the image
    n_spatial_bins = image.shape[1]
    for swath in np.unique(swath_number):
        swath_indices = swath_number == swath
        fov = field_of_view[swath_indices]
        x, y = pu.graphics.make_swath_grid(fov, n_spatial_bins, swath, angular_width)
        rgb_image = image[swath_indices]
        rgb_image = np.fliplr(rgb_image) if app_flip else rgb_image
        ax.pcolormesh(x, y, rgb_image, linewidth=0, edgecolors='none',
                      rasterized=True)

    # Plot the SZA = 90 terminator line
    n_spatial_bins = solar_zenith_angle.shape[1]
    spatial_bin_centers = np.arange(n_spatial_bins) + 0.5

    for swath in np.unique(swath_number):
        swath_indices = swath_number == swath
        # orbit 4361 required this
        if np.sum(swath_indices) < 2:
            continue
        sza = solar_zenith_angle[swath_indices]
        sza = np.fliplr(sza) if app_flip else sza
        ax.contour((spatial_bin_centers / n_spatial_bins + swath) * angular_width, field_of_view[swath_indices],
                     sza, [90], colors='red', linewidths=0.5)

    ax.set_xlim(0, angular_width * n_swaths)
    ax.set_ylim(pu.constants.minimum_mirror_angle * 2, pu.constants.maximum_mirror_angle * 2) if app_flip \
        else ax.set_ylim(pu.constants.maximum_mirror_angle * 2, pu.constants.minimum_mirror_angle * 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('k')

    # Get info I need for the filename
    solar_longitude = f['apoapse/apsis/solar_longitude'][:][0]
    subsolar_subspacecraft_angle = f['apoapse/apsis/subsolar_subspacecraft_angle'][:][0]
    spatial_bin_width = f['apoapse/muv/dayside/binning/spatial_bin_edges'][:].shape[0] - 1
    spectral_bin_width = f['apoapse/muv/dayside/binning/spectral_bin_edges'][:].shape[0] - 1

    filename = make_filename(orbit_code, solar_longitude, subsolar_subspacecraft_angle, spatial_bin_width, spectral_bin_width, 'heq', 'ql')
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save)
    plt.close(fig)


def make_square_root_scaled_detector_image(orbit: int, save_location: Path) -> None:
    """Plot a detector image with square root color scaling.

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
    - 145 DPI
    - black background

    """
    orbit_block = pu.orbit.make_orbit_block(orbit)
    orbit_code = pu.orbit.make_orbit_code(orbit)

    f = File(file_path / orbit_block / f'{orbit_code}.hdf5')

    dayside = f['apoapse/muv/integration/dayside_integrations'][:]
    opportunity_swaths = f['apoapse/integration/opportunity_classification'][:]
    dayside_science_integrations = np.logical_and(dayside, ~opportunity_swaths)

    if np.sum(dayside_science_integrations) == 0:
        return

    brightness = f['apoapse/muv/dayside/detector/brightness'][~opportunity_swaths[dayside]]
    swath_number = f['apoapse/integration/swath_number'][:][dayside_science_integrations]
    n_swaths = f['apoapse/integration/number_of_swaths'][:][0]
    tangent_altitude = f['apoapse/muv/dayside/spatial_bin_geometry/tangent_altitude'][:][..., 4]
    field_of_view = f['apoapse/integration/field_of_view'][:][dayside_science_integrations]
    solar_zenith_angle = f['apoapse/muv/dayside/spatial_bin_geometry/solar_zenith_angle'][:]
    spatial_bin_edges = f['apoapse/muv/dayside/binning/spatial_bin_edges'][:]
    app_flip = f['apoapse/instrument_geometry/app_flip'][0]

    solar_zenith_angle[tangent_altitude != 0] = np.nan

    mask = np.logical_and(tangent_altitude == 0, solar_zenith_angle <= 102)
    try:
        image = pu.graphics.square_root_scale_detector_image(brightness, mask=mask) / 255
    except IndexError:
        return

    angular_width = (spatial_bin_edges[-1] - spatial_bin_edges[0]) / 1024 * pu.constants.angular_detector_width

    # TODO: change as needed
    #fig, ax = setup_figure(n_swaths, angular_width, 6)
    fig, ax = setup_square_figure()

    # Plot the image
    n_spatial_bins = image.shape[1]
    for swath in np.unique(swath_number):
        swath_indices = swath_number == swath
        fov = field_of_view[swath_indices]
        x, y = pu.graphics.make_swath_grid(fov, n_spatial_bins, swath, angular_width)
        rgb_image = image[swath_indices]
        rgb_image = np.fliplr(rgb_image) if app_flip else rgb_image
        ax.pcolormesh(x, y, rgb_image, linewidth=0, edgecolors='none',
                      rasterized=True)

    # Plot the SZA = 90 terminator line
    n_spatial_bins = solar_zenith_angle.shape[1]
    spatial_bin_centers = np.arange(n_spatial_bins) + 0.5

    for swath in np.unique(swath_number):
        swath_indices = swath_number == swath
        # orbit 4361 required this
        if np.sum(swath_indices) < 2:
            continue
        sza = solar_zenith_angle[swath_indices]
        sza = np.fliplr(sza) if app_flip else sza
        ax.contour((spatial_bin_centers / n_spatial_bins + swath) * angular_width, field_of_view[swath_indices],
                   sza, [90], colors='red', linewidths=0.5)

    ax.set_xlim(0, angular_width * n_swaths)
    ax.set_ylim(pu.constants.minimum_mirror_angle * 2, pu.constants.maximum_mirror_angle * 2) if app_flip \
        else ax.set_ylim(pu.constants.maximum_mirror_angle * 2, pu.constants.minimum_mirror_angle * 2)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_facecolor('k')

    # Get info I need for the filename
    solar_longitude = f['apoapse/apsis/solar_longitude'][:][0]
    subsolar_subspacecraft_angle = f['apoapse/apsis/subsolar_subspacecraft_angle'][:][0]
    spatial_bin_width = f['apoapse/muv/dayside/binning/spatial_bin_edges'][:].shape[0] - 1
    spectral_bin_width = f['apoapse/muv/dayside/binning/spectral_bin_edges'][:].shape[0] - 1

    filename = make_filename(orbit_code, solar_longitude, subsolar_subspacecraft_angle, spatial_bin_width, spectral_bin_width, 'sqrt', 'ql')
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    print(filename)
    plt.savefig(save)
    plt.close(fig)
