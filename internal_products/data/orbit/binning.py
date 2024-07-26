from h5py import Group

from compression import compression, compression_opts
from data import hdulist, make_spatial_bin_edges, make_spatial_bin_width, \
    make_spectral_bin_edges, make_spectral_bin_width
import units


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
