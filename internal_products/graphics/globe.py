from pathlib import Path
import warnings

from h5py import File
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pyuvs as pu
from paths import apsis_file_path, iuvs_images_location


def plot_apoapse_mars_map_upright(orbit: int) -> None:
    """Plot an upright globe of how Mars looked at apoapsis

    Parameters
    ----------
    orbit
        The orbit number

    Returns
    -------
    None

    Notes
    -----
    This globe is "upright", i.e. north is up. The pole is somewhere along the
    vertical line that goes through the center of the image, such that the
    view of the planet is maximized from the viewing location.

    """
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
    filename = f'{orbit_code}_globe-topography.png'
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150)
    plt.close(fig)


def plot_apoapse_muv_dayside_globe_upright(orbit: int) -> None:
    """Plot an upright globe of the apoapse MUV data with HEQ coloring.

    Parameters
    ----------
    orbit
        The orbit number

    Returns
    -------
    None

    Notes
    -----
    This globe is "upright", i.e. north is up. The pole is somewhere along the
    vertical line that goes through the center of the image, such that the
    view of the planet is maximized from the viewing location.

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
        print('here')
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

    # Plot black all over the globe to represent points where no data were taken
    axis.imshow(np.zeros((360, 180, 3)), transform=transform, extent=(0, 360, -90, 90))

    # Colorize the image (turn kR to RGB)
    mask = np.logical_and(tangent_altitude[..., 4] == 0, solar_zenith_angle <= 102)
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
                        rasterized=True)

    # Save the graphic
    save_location = iuvs_images_location / 'apoapse' / 'muv' / 'globe-heq-upright-dayside'
    orbit_code = pu.make_orbit_code(orbit)
    filename = f'{orbit_code}_globe-heq-upright-dayside.png'
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150)
    plt.close(fig)


def plot_apoapse_muv_dayside_globe_rotated(orbit: int) -> None:
    """Plot a rotated globe of the apoapse MUV data with HEQ coloring.

    Parameters
    ----------
    orbit
        The orbit number

    Returns
    -------
    None

    Notes
    -----
    This globe is rotated such that Mars is oriented how IUVS saw it.

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
        print('here')
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
    mars_position = f['apoapse/apsis/mars_position'][0]
    mars_velocity = f['apoapse/apsis/mars_velocity'][0]

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
            central_latitude=0,
            central_longitude=0,
            satellite_height=spacecraft_altitude * 10 ** 3,
            globe=globe)
    transform = pu.graphics.compute_rotated_transform(
        mars_position, mars_velocity, subspacecraft_latitude, subspacecraft_longitude, globe)
    axis = plt.axes(bbox, projection=projection)

    # Plot black all over the globe to represent points where no data were taken
    axis.imshow(np.zeros((360, 180, 3)), transform=transform, extent=(-180, 180, -90, 90))

    # Colorize the image (turn kR to RGB)
    mask = np.logical_and(tangent_altitude[..., 4] == 0, solar_zenith_angle <= 102)
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
                        rasterized=True)

    # Save the graphic
    save_location = iuvs_images_location / 'apoapse' / 'muv' / 'globe-heq-rotated-dayside'
    orbit_code = pu.make_orbit_code(orbit)
    filename = f'{orbit_code}_globe-heq-rotated-dayside.png'
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150)
    plt.close(fig)


def plot_apoapse_muv_daynight_globe_rotated_no_nightglow(orbit: int) -> None:
    """Plot a rotated globe of the apoapse MUV data with HEQ coloring and
    NO nightglow data included.

    Parameters
    ----------
    orbit
        The orbit number

    Returns
    -------
    None

    Notes
    -----
    This globe is rotated such that Mars is oriented how IUVS saw it.

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
        print('here')
        return

    # Load in datasets for the dayside image
    brightness = f['apoapse/muv/dayside/detector/brightness'][~opportunity_integrations[dayside_integrations]]
    tangent_altitude = f['apoapse/muv/dayside/spatial_bin_geometry/tangent_altitude'][:]
    solar_zenith_angle = f['apoapse/muv/dayside/spatial_bin_geometry/solar_zenith_angle'][:]
    latitude = f['apoapse/muv/dayside/spatial_bin_geometry/latitude'][~opportunity_integrations[dayside_integrations]]
    longitude = f['apoapse/muv/dayside/spatial_bin_geometry/longitude'][~opportunity_integrations[dayside_integrations]]
    swath_number = f['apoapse/integration/swath_number'][dayside_integrations]
    subspacecraft_latitude = f['apoapse/apsis/subspacecraft_latitude'][0]
    subspacecraft_longitude = f['apoapse/apsis/subspacecraft_longitude'][0]
    spacecraft_altitude = f['apoapse/apsis/spacecraft_altitude'][0]
    mars_position = f['apoapse/apsis/mars_position'][0]
    mars_velocity = f['apoapse/apsis/mars_velocity'][0]

    # Load in datasets for the nightside image
    species_brightness = f[f'apoapse/muv/nightside/species/no_nightglow'][:]
    nightside_spatial_bin_vector = f['apoapse/muv/nightside/spatial_bin_geometry/bin_vector'][:]  # (integration, spatial bin, 5, 3)
    spacecraft_position = f['apoapse/spacecraft_geometry/iau_mars_frame/spacecraft_position'][~dayside_science_integrations]
    instrument_y_field_of_view = f['apoapse/instrument_geometry/iau_mars_frame/instrument_y_field_of_view'][~dayside_science_integrations]
    app_flip = f['apoapse/instrument_geometry/app_flip'][0]

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
            central_latitude=0,
            central_longitude=0,
            satellite_height=spacecraft_altitude * 10 ** 3,
            globe=globe)
    transform = pu.graphics.compute_rotated_transform(
        mars_position, mars_velocity, subspacecraft_latitude, subspacecraft_longitude, globe)
    axis = plt.axes(bbox, projection=projection)

    # Plot black all over the globe to represent points where no data were taken
    axis.imshow(np.zeros((360, 180, 3)), transform=transform, extent=(-180, 180, -90, 90))

    # Colorize the image (turn kR to RGB)
    mask = np.logical_and(tangent_altitude[..., 4] == 0, solar_zenith_angle <= 102)
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
                        rasterized=True)

    # Create the nightside axis on top of the already existing axis
    nightside_ax = plt.axes((0, 0, 1, 1))
    nightside_ax.set_xticks([])
    nightside_ax.set_xlim(-4000, 4000)
    nightside_ax.set_yticks([])
    nightside_ax.set_ylim(-4000, 4000)
    nightside_ax.set_frame_on('False')
    nightside_ax.patch.set_visible(False)

    # This uses Justin's method of seam blending the nightside data by defining
    #  a grid beforehand and just averaging the pixels that fell within that
    #  pixel
    pixel_angle = np.arccos(np.dot(nightside_spatial_bin_vector[0, 0, 0, :],
                                   nightside_spatial_bin_vector[0, 0, 1, :]))
    if np.isnan(pixel_angle):
        print('was nan')
        pixel_angle = np.radians(10 / nightside_spatial_bin_vector.shape[1])  # this is the angular slit width / the number of spatial bins
    pixel_size = 2 * np.tan(pixel_angle) * spacecraft_altitude

    # dimensions of pixel grid and width of a pixel in kilometers
    xsize_night = int(8000 / pixel_size)
    ysize_night = int(8000 / pixel_size)

    # arrays to hold projected data (total and count for averaging after every data point placed)
    total_night = np.zeros((ysize_night, xsize_night))
    count_night = np.zeros((ysize_night, xsize_night))

    # Calculate the pixel position at apoapsis projected to a plane through the center of Mars
    for integration in range(species_brightness.shape[0]):
        norm = spacecraft_position[integration] / np.linalg.norm(spacecraft_position[integration])
        vx = np.cross(instrument_y_field_of_view[integration], norm)
        print(integration)
        for spatial_bin in range(species_brightness.shape[1]):
            for pixel_corner in range(5):
                try:
                    vpixcorner = nightside_spatial_bin_vector[integration, spatial_bin, pixel_corner, :]
                    vdiff = spacecraft_position[integration] - (np.dot(spacecraft_position[integration], vpixcorner) * vpixcorner)

                    x = int(np.dot(vdiff, vx) * np.linalg.norm(vdiff) /
                            np.linalg.norm(
                                [np.dot(vdiff, vx), np.dot(vdiff, instrument_y_field_of_view[integration])]) / pixel_size + xsize_night / 2)
                    y = int(np.dot(vdiff, instrument_y_field_of_view[integration]) * np.linalg.norm(vdiff) /
                            np.linalg.norm(
                                [np.dot(vdiff, vx), np.dot(vdiff, instrument_y_field_of_view[integration])]) / pixel_size + ysize_night / 2)

                    # make sure they fall within the grid...
                    if (x >= 0) and (x < 8000) and (y >= 0) and (y < 8000):
                        # put the value in the grid
                        total_night[y, x] += species_brightness[integration, spatial_bin]
                        count_night[y, x] += 1
                except (IndexError, ValueError):
                    pass

    # calculate the average of the nightside grid
    total_night[count_night == 0] = np.nan
    night_grid = total_night / count_night

    # rotate if not beta-flipped
    if app_flip:
        night_grid = np.rot90(night_grid, k=2, axes=(0, 1))

    # meshgrids for data dislpay
    x_night, y_night = np.meshgrid(np.linspace(-4000, 4000, xsize_night), np.linspace(-4000, 4000, ysize_night))

    # place the nightside grid
    nightside_ax.pcolormesh(x_night, y_night, night_grid, cmap=pu.graphics.no_colormap,
                            norm=matplotlib.colors.SymLogNorm(linthresh=1, vmin=0, vmax=10), rasterized=True)

    # Save the graphic
    save_location = iuvs_images_location / 'apoapse' / 'muv' / 'globe-heq-rotated-nightside'
    orbit_code = pu.make_orbit_code(orbit)
    filename = f'{orbit_code}_globe-heq-rotated-nightside-no_nightglow.png'
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150)
    plt.close(fig)


def plot_apoapse_muv_daynight_globe_rotated_aurora(orbit: int) -> None:
    """Plot a rotated globe of the apoapse MUV data with HEQ coloring and
    aurora data included.

    Parameters
    ----------
    orbit
        The orbit number

    Returns
    -------
    None

    Notes
    -----
    This globe is rotated such that Mars is oriented how IUVS saw it.

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
        print('here')
        return

    # Load in datasets for the dayside image
    brightness = f['apoapse/muv/dayside/detector/brightness'][~opportunity_integrations[dayside_integrations]]
    tangent_altitude = f['apoapse/muv/dayside/spatial_bin_geometry/tangent_altitude'][:]
    solar_zenith_angle = f['apoapse/muv/dayside/spatial_bin_geometry/solar_zenith_angle'][:]
    latitude = f['apoapse/muv/dayside/spatial_bin_geometry/latitude'][~opportunity_integrations[dayside_integrations]]
    longitude = f['apoapse/muv/dayside/spatial_bin_geometry/longitude'][~opportunity_integrations[dayside_integrations]]
    swath_number = f['apoapse/integration/swath_number'][dayside_integrations]
    subspacecraft_latitude = f['apoapse/apsis/subspacecraft_latitude'][0]
    subspacecraft_longitude = f['apoapse/apsis/subspacecraft_longitude'][0]
    spacecraft_altitude = f['apoapse/apsis/spacecraft_altitude'][0]
    mars_position = f['apoapse/apsis/mars_position'][0]
    mars_velocity = f['apoapse/apsis/mars_velocity'][0]

    # Load in datasets for the nightside image
    species_brightness = f[f'apoapse/muv/nightside/species/co_cameron_bands'][:] + f[f'apoapse/muv/nightside/species/co2+_uvd'][:]
    nightside_spatial_bin_vector = f['apoapse/muv/nightside/spatial_bin_geometry/bin_vector'][:]  # (integration, spatial bin, 5, 3)
    spacecraft_position = f['apoapse/spacecraft_geometry/iau_mars_frame/spacecraft_position'][~dayside_science_integrations]
    instrument_y_field_of_view = f['apoapse/instrument_geometry/iau_mars_frame/instrument_y_field_of_view'][~dayside_science_integrations]
    app_flip = f['apoapse/instrument_geometry/app_flip'][0]

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
            central_latitude=0,
            central_longitude=0,
            satellite_height=spacecraft_altitude * 10 ** 3,
            globe=globe)
    transform = pu.graphics.compute_rotated_transform(
        mars_position, mars_velocity, subspacecraft_latitude, subspacecraft_longitude, globe)
    axis = plt.axes(bbox, projection=projection)

    # Plot black all over the globe to represent points where no data were taken
    axis.imshow(np.zeros((360, 180, 3)), transform=transform, extent=(-180, 180, -90, 90))

    # Colorize the image (turn kR to RGB)
    mask = np.logical_and(tangent_altitude[..., 4] == 0, solar_zenith_angle <= 102)
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
                        rasterized=True)

    # Create the nightside axis on top of the already existing axis
    nightside_ax = plt.axes((0, 0, 1, 1))
    nightside_ax.set_xticks([])
    nightside_ax.set_xlim(-4000, 4000)
    nightside_ax.set_yticks([])
    nightside_ax.set_ylim(-4000, 4000)
    nightside_ax.set_frame_on('False')
    nightside_ax.patch.set_visible(False)

    # This uses Justin's method of seam blending the nightside data by defining
    #  a grid beforehand and just averaging the pixels that fell within that
    #  pixel
    pixel_angle = np.arccos(np.dot(nightside_spatial_bin_vector[0, 0, 0, :],
                                   nightside_spatial_bin_vector[0, 0, 1, :]))
    if np.isnan(pixel_angle):
        pixel_angle = np.radians(10 / nightside_spatial_bin_vector.shape[1])  # this is the angular slit width / the number of spatial bins
    pixel_size = 2 * np.tan(pixel_angle) * spacecraft_altitude

    # dimensions of pixel grid and width of a pixel in kilometers
    xsize_night = int(8000 / pixel_size)
    ysize_night = int(8000 / pixel_size)

    # arrays to hold projected data (total and count for averaging after every data point placed)
    total_night = np.zeros((ysize_night, xsize_night))
    count_night = np.zeros((ysize_night, xsize_night))

    # Calculate the pixel position at apoapsis projected to a plane through the center of Mars
    for integration in range(species_brightness.shape[0]):
        norm = spacecraft_position[integration] / np.linalg.norm(spacecraft_position[integration])
        vx = np.cross(instrument_y_field_of_view[integration], norm)
        print(integration)
        for spatial_bin in range(species_brightness.shape[1]):
            for pixel_corner in range(5):
                try:
                    vpixcorner = nightside_spatial_bin_vector[integration, spatial_bin, pixel_corner, :]
                    vdiff = spacecraft_position[integration] - (np.dot(spacecraft_position[integration], vpixcorner) * vpixcorner)

                    x = int(np.dot(vdiff, vx) * np.linalg.norm(vdiff) /
                            np.linalg.norm(
                                [np.dot(vdiff, vx), np.dot(vdiff, instrument_y_field_of_view[integration])]) / pixel_size + xsize_night / 2)
                    y = int(np.dot(vdiff, instrument_y_field_of_view[integration]) * np.linalg.norm(vdiff) /
                            np.linalg.norm(
                                [np.dot(vdiff, vx), np.dot(vdiff, instrument_y_field_of_view[integration])]) / pixel_size + ysize_night / 2)

                    # make sure they fall within the grid...
                    if (x >= 0) and (x < 8000) and (y >= 0) and (y < 8000):
                        # put the value in the grid
                        total_night[y, x] += species_brightness[integration, spatial_bin]
                        count_night[y, x] += 1
                except (IndexError, ValueError):
                    pass

    # calculate the average of the nightside grid
    total_night[count_night == 0] = np.nan
    night_grid = total_night / count_night

    # rotate if not beta-flipped
    if app_flip:
        night_grid = np.rot90(night_grid, k=2, axes=(0, 1))

    # meshgrids for data dislpay
    x_night, y_night = np.meshgrid(np.linspace(-4000, 4000, xsize_night), np.linspace(-4000, 4000, ysize_night))

    # place the nightside grid
    nightside_ax.pcolormesh(x_night, y_night, night_grid, cmap=pu.graphics.co2p_colormap,
                            norm=matplotlib.colors.SymLogNorm(linthresh=1, vmin=0, vmax=4), rasterized=True)

    # Save the graphic
    save_location = iuvs_images_location / 'apoapse' / 'muv' / 'globe-heq-rotated-nightside'
    orbit_code = pu.make_orbit_code(orbit)
    filename = f'{orbit_code}_globe-heq-rotated-nightside-aurora.png'
    save = save_location / pu.make_orbit_block(orbit) / filename
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150)
    plt.close(fig)
