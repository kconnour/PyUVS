from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.binning import make_spatial_bin_edges, \
    make_spatial_bin_width, make_spectral_bin_edges, make_spectral_bin_width
from internal_products.data.orbit.file.compression import compression, \
    compression_opts


hdulist = fits.hdu.hdulist.HDUList


def add_spatial_bin_edges_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'spatial_bin_edges',
        data=make_spatial_bin_edges(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.bin_number


def add_spatial_bin_width_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'spatial_bin_width',
        data=make_spatial_bin_width(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.bins


def add_spectral_bin_edges_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'spectral_bin_edges',
        data=make_spectral_bin_edges(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.bin_number


def add_spectral_bin_width_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'spectral_bin_width',
        data=make_spectral_bin_width(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.bins
