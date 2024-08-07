

def make_filename(orbit_code: str, solar_longitude: float, subsolar_subspacecraft_angle: float,
                  spatial_bin_width: int, spectral_bin_width: int, coloring_code: str, image_type: str) -> str:
    """ Make the filenames for images associated with the "Martian Cloud Watching" project.

    Parameters
    ----------
    orbit_code
        The orbit code.
    solar_longitude
        The solar longitude of the apsis of this observation.
    subsolar_subspacecraft_angle
        The angle between the subsolar point and the subspacecraft point.
    spatial_bin_width
        The number of pixels in a spatial bin.
    spectral_bin_width
        The number of pixels in a spectral bin.
    coloring_code
        The "code" used to create the coloring for this observation. Examples are "heq" or "sqrt" but can be anything.
    image_type
        The type of image. Examples are "ql" or "globe" but can be anything.

    Returns
    -------
    The filename given the input values.

    """
    return f'{orbit_code}-' \
           f'Ls{solar_longitude * 10:04.0f}-' \
           f'angle{subsolar_subspacecraft_angle * 10:04.0f}-' \
           f'binning{spatial_bin_width:04}x{spectral_bin_width:04}-' \
           f'{coloring_code}-' \
           f'{image_type}.png'
