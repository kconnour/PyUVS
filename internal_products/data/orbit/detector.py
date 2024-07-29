from h5py import Group

from compression import compression, compression_opts
from data import hdulist, make_detector_raw, make_detector_dark_subtracted, \
    make_brightness, make_spatial_bin_edges, make_spatial_bin_width, \
    make_spectral_bin_edges, make_spectral_bin_width, make_integration_time, \
    make_mcp_voltage, make_mcp_voltage_gain, make_random_uncertainty_dn, \
    make_random_uncertainty_physical
import units


def add_raw_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'raw',
        data=make_detector_raw(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.data_number


def add_dark_subtracted_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'dark_subtracted',
        data=make_detector_dark_subtracted(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.data_number


def add_brightness_to_file(group: Group, hduls: list[hdulist]) -> None:
    dark_subtracted = make_detector_dark_subtracted(hduls)
    spatial_bin_edges = make_spatial_bin_edges(hduls)
    spatial_bin_width = make_spatial_bin_width(hduls)[0]
    spectral_bin_edges = make_spectral_bin_edges(hduls)
    spectral_bin_width = make_spectral_bin_width(hduls)[0]
    integration_time = make_integration_time(hduls)
    mcp_voltage = make_mcp_voltage(hduls)
    mcp_voltage_gain = make_mcp_voltage_gain(hduls)

    data = make_brightness(
        dark_subtracted, spatial_bin_edges, spectral_bin_edges,
        spatial_bin_width, spectral_bin_width, integration_time, mcp_voltage,
        mcp_voltage_gain)

    dataset = group.create_dataset(
        'brightness',
        data=data,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness


def add_random_uncertainty_dn_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'random_uncertainty_dn',
        data=make_random_uncertainty_dn(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.data_number


def add_random_uncertainty_physical_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'random_uncertainty_physical',
        data=make_random_uncertainty_physical(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness + '/nm'
