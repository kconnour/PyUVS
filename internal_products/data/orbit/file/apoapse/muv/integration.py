from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.integration import \
    make_detector_temperature, make_mcp_voltage, make_mcp_voltage_gain, \
    make_apoapse_muv_failsafe_integrations, \
    make_apoapse_muv_dayside_integrations, \
    make_apoapse_muv_nightside_integrations
from internal_products.data.orbit.file.compression import compression, \
    compression_opts


hdulist = fits.hdu.hdulist.HDUList


def add_detector_temperature_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'detector_temperature',
        data=make_detector_temperature(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.temperature


def add_mcp_voltage_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'mcp_voltage',
        data=make_mcp_voltage(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.voltage


def add_mcp_voltage_gain_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'mcp_voltage_gain',
        data=make_mcp_voltage_gain(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.voltage


def add_failsafe_integrations_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'failsafe_integrations',
        data=make_apoapse_muv_failsafe_integrations(hduls),
        compression=compression,
        compression_opts=compression_opts)


def add_dayside_integrations_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'dayside_integrations',
        data=make_apoapse_muv_dayside_integrations(hduls),
        compression=compression,
        compression_opts=compression_opts)


def add_nightside_integrations_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'nightside_integrations',
        data=make_apoapse_muv_nightside_integrations(hduls),
        compression=compression,
        compression_opts=compression_opts)
