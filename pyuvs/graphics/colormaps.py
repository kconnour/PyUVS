import matplotlib


no_colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'NO', ((0, 0, 0), (0, 0.5, 0), (0.61, 0.8, 0.2), (1, 1, 1)), N=256)
"""This is the NO nightglow black --> green --> yellow-green --> white colormap.
Identical to IDL #8."""

co2p_colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'CO2p', ((0, 0, 0), (0.7255, 0.0588, 0.7255), (1, 1, 1)), N=256)
"""This is the CO2+ black --> pink --> white coloramp."""

co_colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'CO', ((0, 0, 0), (0.722, 0.051, 0), (1, 1, 1)), N=256)
"""This is the CO Cameron band black --> red --> white colormap. Idential to
IDL #3."""

h_colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'H', ((0, 0, 0), (0, 0.204, 0.678), (1, 1, 1)), N=256)
"""This is the Hydrogen Lyman-alpha black --> blue --> white colormap. Idential
to IDL #1."""
