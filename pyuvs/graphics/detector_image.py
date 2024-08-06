import numpy as np


def make_swath_grid(field_of_view: np.ndarray, n_spatial_bins: int, swath_number: int, angular_size: float) \
        -> tuple[np.ndarray, np.ndarray]:
    """Make a swath grid of mirror angles and spatial bins.

    Parameters
    ----------
    field_of_view
        The instrument's field of view.
    swath_number
        The swath number.
    n_spatial_bins
        The number of spatial bins.
    angular_size
        The angular size of one horizontal element. This is likely the angular
        detector width or the angular size of the observation, but can be any
        value.

    Returns
    -------
    The swath grid.

    """
    # Make the spatial bin edges (horizontal dimension)
    spatial_bin_angular_edges = np.linspace(angular_size * swath_number, angular_size * (swath_number + 1), num=n_spatial_bins+1)

    # Make the field of view edges (vertical dimension)
    n_integrations = field_of_view.size
    # orbit 4361 requires this check
    if n_integrations == 1:
        field_of_view_edges = np.linspace(field_of_view[0], field_of_view[0], num=2)
    else:
        mean_angle_difference = np.mean(np.diff(field_of_view))
        field_of_view_edges = np.linspace(
            field_of_view[0] - mean_angle_difference / 2,
            field_of_view[-1] + mean_angle_difference / 2,
            num=n_integrations + 1)
    return np.meshgrid(spatial_bin_angular_edges, field_of_view_edges)


def make_single_integration_geographic_grid(grid: np.ndarray) -> np.ndarray:
    """The geographic grid for a single integration

    Parameters
    ----------
    grid
        The latitude or longitude of the integration

    Returns
    -------
    An grid of the corners of the integration.

    """
    grid_corners = np.zeros((2, grid.shape[0] + 1)) * np.nan
    grid_corners[0, :-1] = grid[:, 0]
    grid_corners[1, :-1] = grid[:, 1]
    grid_corners[0, -1] = grid[-1, 2]
    grid_corners[1, -1] = grid[-1, 3]
    return grid_corners


def make_swath_geographic_grid(grid: np.ndarray) -> np.ndarray:
    """The geographic grid for a swath

    Parameters
    ----------
    grid
        The latitude or longitude of the integration

    Returns
    -------
    An grid of the corners of the swath.

    Notes
    -----
    An integration is not contiguous! In other words, the top of one integration
    will not be at the same geographic point as the bottom of the next
    integration. However, it can be close, and this function is a useful
    approximation for when this is true.

    """
    n_integrations = grid.shape[0]
    n_spatial_bin = grid.shape[1]
    grid_corners = np.zeros((n_integrations + 1, n_spatial_bin + 1)) * np.nan

    grid_corners[:-1, :-1] = grid[..., 0]
    grid_corners[-1, :-1] = grid[-1, :, 1]
    grid_corners[:-1, -1] = grid[:, -1, 2]
    grid_corners[-1, -1] = grid[-1, -1, 3]
    return grid_corners
