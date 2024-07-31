from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.integration import make_ephemeris_time, \
    make_mirror_data_number, make_mirror_angle, make_field_of_view, \
    make_case_temperature, make_integration_time, make_apoapse_swath_number, \
    make_apoapse_number_of_swaths, make_apoapse_opportunity_classification
from internal_products.data.orbit.file.compression import compression, \
    compression_opts


hdulist = fits.hdu.hdulist.HDUList


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
        data=make_apoapse_swath_number(orbit, hduls),
        compression=compression,
        compression_opts=compression_opts)


def add_number_of_swaths_to_file(group: Group, hduls: list[hdulist], orbit: int) -> None:
    dataset = group.create_dataset(
        'number_of_swaths',
        data=make_apoapse_number_of_swaths(orbit, hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['comment'] = ('This is the number of expected swaths in the '
        'sequence, assuming IUVS collected data during all of them')


def add_opportunity_classification_to_file(group: Group, hduls: list[hdulist], orbit: int) -> None:
    dataset = group.create_dataset(
        'opportunity_classification',
        data=make_apoapse_opportunity_classification(orbit, hduls),
        compression=compression,
        compression_opts=compression_opts)
