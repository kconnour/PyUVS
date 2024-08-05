import numpy as np


def make_equidistant_spectral_cutoff_indices(n_spectral_bins: int) -> tuple[int, int]:
    """Make indices such that the input spectral bins are in 3 equally spaced
    color channels.

    Parameters
    ----------
    n_spectral_bins
        The number of spectral bins.

    Returns
    -------
    The blue-green and the green-red cutoff indices.

    Examples
    --------
    Get the wavelength cutoffs for some common apoapse MUV spectral binning
    schemes.

    >>> make_equidistant_spectral_cutoff_indices(15)
    (5, 10)

    >>> make_equidistant_spectral_cutoff_indices(19)
    (6, 13)

    >>> make_equidistant_spectral_cutoff_indices(20)
    (7, 13)

    """
    blue_green_cutoff = round(n_spectral_bins / 3)
    green_red_cutoff = round(n_spectral_bins * 2 / 3)
    return blue_green_cutoff, green_red_cutoff


def turn_detector_image_to_3_channels(image: np.ndarray) -> np.ndarray:
    """Turn a detector image into 3 channels by coadding over the spectral
    dimension.

    Parameters
    ----------
    image
        The image to turn into 3 channels. This is assumed to be 3 dimensional
        and have a shape of (n_integrations, n_spatial_bins, n_spectral_bins).

    Returns
    -------
    A co-added image with shape (n_integrations, n_spatial_bins, 3).

    """
    n_spectral_bins = image.shape[2]
    blue_green_cutoff, green_red_cutoff = \
        make_equidistant_spectral_cutoff_indices(n_spectral_bins)

    red = np.sum(image[..., green_red_cutoff:], axis=-1)
    green = np.sum(image[..., blue_green_cutoff:green_red_cutoff], axis=-1)
    blue = np.sum(image[..., :blue_green_cutoff], axis=-1)

    return np.dstack([red, green, blue])


def histogram_equalize_grayscale_image(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Histogram equalize a grayscale image.

    Parameters
    ----------
    image
        The image to histogram equalize. This is assumed to be 2-dimensional
        (2 spatial dimensions) but can have any dimensionality.
    mask
        A mask of booleans where ``False`` values are excluded from the
        histogram equalization scaling. This must have the same shape as
        ``image``.

    Returns
    -------
    A histogram equalized array with the same shape as the inputs with values
    ranging from 0 to 255.

    See Also
    --------
    histogram_equalize_rgb_image: Histogram equalize a 3-color-channel image.

    Notes
    -----
    I could not get the scikit-learn algorithm to work so I created this.
    The algorithm works like this:

    1. Sort all data used in the coloring.
    2. Use these sorted values to determine the 256 left bin cutoffs.
    3. Linearly interpolate each value in the grid over 256 RGB values and the
       corresponding data values.
    4. Take the floor of the interpolated values since I'm using left cutoffs.

    """
    sorted_values = np.sort(image[mask], axis=None)
    left_cutoffs = np.array([sorted_values[int(i / 256 * len(sorted_values))]
                             for i in range(256)])
    rgb = np.linspace(0, 255, num=256)
    return np.floor(np.interp(image, left_cutoffs, rgb))


def histogram_equalize_rgb_image(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Histogram equalize an RGB image.

    Parameters
    ----------
    image
        The image to histogram equalize. This is assumed to be 3-dimensional
        (the first 2 being spatial and the last being spectral). The last
        dimension as assumed to have a length of 3. Indices 0, 1, and 2
        correspond to R, G, and B, respectively.
    mask
        A mask of booleans where ``False`` values are excluded from the
        histogram equalization scaling. This must have the same shape as the
        first N-1 dimensions of ``image``.

    Returns
    -------
    A histogram equalized array with the same shape as the inputs with values
    ranging from 0 to 255.

    See Also
    --------
    histogram_equalize_grayscale_image: Histogram equalize a
    single-color-channel image.

    """
    red = histogram_equalize_grayscale_image(image[..., 0], mask=mask)
    green = histogram_equalize_grayscale_image(image[..., 1], mask=mask)
    blue = histogram_equalize_grayscale_image(image[..., 2], mask=mask)
    return np.dstack([red, green, blue])


def histogram_equalize_detector_image(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Histogram equalize a detector image.

    Parameters
    ----------
    image
        The image to histogram equalize. This is assumed to be 3-dimensional
        (the first 2 being spatial and the last being spectral).
    mask
        A mask of booleans where ``False`` values are excluded from the
        histogram equalization scaling. This must have the same shape as the
        first N-1 dimensions of ``image``.

    Returns
    -------
    Histogram equalized detector image, where the output array has a shape of
    (M, N, 3).

    """
    coadded_image = turn_detector_image_to_3_channels(image)
    return histogram_equalize_rgb_image(coadded_image, mask=mask)
