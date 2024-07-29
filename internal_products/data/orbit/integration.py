from h5py import Group

from compression import compression, compression_opts
from data import hdulist, make_ephemeris_time, make_mirror_data_number, \
    make_mirror_angle, make_field_of_view, make_case_temperature, \
    make_integration_time, make_swath_number, make_number_of_swaths, \
    make_opportunity_classification, make_detector_temperature, \
    make_mcp_voltage, make_mcp_voltage_gain, make_failsafe_integrations, \
    make_dayside_integrations, make_nightside_integrations
import units


def add_ephemeris_time_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'ephemeris_time',
        data=make_ephemeris_time(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.ephemeris_time


def add_mirror_data_number_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'mirror_data_number',
        data=make_mirror_data_number(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.data_number


def add_mirror_angle_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'mirror_angle',
        data=make_mirror_angle(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_field_of_view_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'field_of_view',
        data=make_field_of_view(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_case_temperature_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'case_temperature',
        data=make_case_temperature(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.temperature


def add_integration_time_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'integration_time',
        data=make_integration_time(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.integration_time


def add_swath_number_to_file(group: Group, hduls: list[hdulist], orbit: int) -> None:
    dataset = group.create_dataset(
        'swath_number',
        data=make_swath_number(orbit, hduls),
        compression=compression,
        compression_opts=compression_opts)


def add_number_of_swaths_to_file(group: Group, hduls: list[hdulist], orbit: int) -> None:
    dataset = group.create_dataset(
        'number_of_swaths',
        data=make_number_of_swaths(orbit, hduls),
        compression=compression,
        compression_opts=compression_opts)


def add_opportunity_classification_to_file(group: Group, hduls: list[hdulist], orbit: int) -> None:
    dataset = group.create_dataset(
        'opportunity_classification',
        data=make_opportunity_classification(orbit, hduls),
        compression=compression,
        compression_opts=compression_opts)


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
        data=make_failsafe_integrations(hduls),
        compression=compression,
        compression_opts=compression_opts)


def add_dayside_integrations_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'dayside_integrations',
        data=make_dayside_integrations(hduls),
        compression=compression,
        compression_opts=compression_opts)


def add_nightside_integrations_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'nightside_integrations',
        data=make_nightside_integrations(hduls),
        compression=compression,
        compression_opts=compression_opts)
