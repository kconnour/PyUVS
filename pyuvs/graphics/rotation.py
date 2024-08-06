import cartopy.crs as ccrs
import numpy as np


def compute_rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Compute the rotation matrix associated with counterclockwise rotation a
    bout the given axis by theta radians. To transform a vector, calculate its
    dot-product with the rotation matrix.

    Parameters
    ----------
    axis
        The rotation axis in Cartesian coordinates. Does not have to be a
        unit vector.
    theta : float
        The angle [radians] to rotate about the rotation axis. Positive angles
        rotate counter-clockwise.

    Returns
    -------
    matrix
        The 3D rotation matrix with dimensions (3,3).

    Notes
    -----
    Zac wrote all of this.

    """
    # convert the axis to a numpy array and normalize it
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # calculate components of the rotation matrix elements
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    # build the rotation matrix
    matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                       [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                       [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    # return the rotation matrix
    return matrix


def compute_rotated_transform(
        mars_position: np.ndarray,
        mars_velocity: np.ndarray,
        subspacecraft_latitude: float,
        subspacecraft_longitude: float,
        globe: ccrs.Globe) -> ccrs.RotatedPole:
    """Compute the rotation transform.

    Parameters
    ----------
    mars_position
        The 3-element vector of Mars's position
    mars_velocity
        The 3-element vector of Mars's velocity
    subspacecraft_latitude
        The sub-spacecraft latitude
    subspacecraft_longitude
        The sub-spacecraft longitude
    globe
        The globe that this rotated transform belongs to

    Returns
    -------
    A Cartopy rotated transform

    Notes
    -----
    "This took me weeks of cobbling together code snippets I found online and
    I don't know how it works" ---Zac (paraphrased)

    However, Kyle took the original code and tidied it up to what you see here
    (not that it's particularly tidy).

    """
    # Convert the inputs to meters
    mars_position = mars_position * 10 ** 3
    mars_velocity = mars_velocity * 10 ** 3

    # Define some constants or something
    north_polar_vector = [0, 0, 1]  # North pole unit vector in IAU basis
    G = 6.673e-11 * 6.4273e23
    h = np.cross(mars_position, mars_velocity)
    n = h / np.linalg.norm(h)
    ev = np.cross(mars_velocity, h) / G - mars_position / np.linalg.norm(mars_position)
    evn = ev / np.linalg.norm(ev)
    b = np.cross(evn, n)

    # when hovering over the sub-spacecraft point unrotated (the meridian of the point is a straight vertical line,
    # this is the exact view when using cartopy's NearsidePerspective or Orthographic with central_longitude and
    # central latitude set to the sub-spacecraft point), calculate the angle by which the planet must be rotated
    # about the sub-spacecraft point
    angle = np.arctan2(np.dot(north_polar_vector, -b), np.dot(north_polar_vector, n))

    # first, rotate the pole to a different latitude given the subspacecraft latitude
    # cartopy's RotatedPole uses the location of the dateline (-180) as the lon_0 coordinate of the north pole
    pole_lat = 90 + subspacecraft_latitude
    pole_lon = -180

    # convert pole latitude to colatitude (for spherical coordinates)
    # also convert to radians for use with numpy trigonometric functions
    phi = pole_lon * np.pi / 180
    theta = (90 - pole_lat) * np.pi / 180

    # calculate the Cartesian vector pointing to the pole
    polar_vector = [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]

    # by rotating the pole, the observer's sub-point in cartopy's un-transformed coordinates is (0,0)
    # the rotation axis is therefore the x-axis
    rotation_axis = [1, 0, 0]

    # rotate the polar vector by the calculated angle
    rotated_polar_vector = np.dot(compute_rotation_matrix(rotation_axis, -angle), polar_vector)

    # get the new polar latitude and longitude after the rotation, with longitude offset to dateline
    rotated_polar_lon = np.arctan(rotated_polar_vector[1] / rotated_polar_vector[0]) * 180 / np.pi - 180
    if subspacecraft_latitude < 0:
        rotated_polar_lat = 90 - np.arccos(rotated_polar_vector[2] / np.linalg.norm(rotated_polar_vector)) * 180 / np.pi
    else:
        rotated_polar_lat = 90 + np.arccos(rotated_polar_vector[2] / np.linalg.norm(rotated_polar_vector)) * 180 / np.pi

    # calculate a RotatedPole transform for the rotated pole position
    transform = ccrs.RotatedPole(pole_latitude=rotated_polar_lat, pole_longitude=rotated_polar_lon,
                                 central_rotated_longitude=0)

    # transform the viewer (0,0) point
    tcoords = transform.transform_point(0, 0, ccrs.PlateCarree())

    # find the angle by which the planet needs to be rotated about it's rotated polar axis and calculate a new
    # RotatedPole transform including this angle rotation
    rot_ang = subspacecraft_longitude - tcoords[0]
    return ccrs.RotatedPole(pole_latitude=rotated_polar_lat, pole_longitude=rotated_polar_lon,
                            central_rotated_longitude=rot_ang, globe=globe)
