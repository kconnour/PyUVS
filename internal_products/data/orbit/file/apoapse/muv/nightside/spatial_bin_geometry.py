from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.spatial_bin_geometry import \
    make_spatial_bin_latitude, make_spatial_bin_longitude, \
    make_spatial_bin_tangent_altitude, make_spatial_bin_tangent_altitude_rate, \
    make_spatial_bin_line_of_sight, make_spatial_bin_solar_zenith_angle, \
    make_spatial_bin_emission_angle, make_spatial_bin_phase_angle, \
    make_spatial_bin_zenith_angle, make_spatial_bin_local_time, \
    make_spatial_bin_right_ascension, make_spatial_bin_declination, \
    make_spatial_bin_vector
from internal_products.data.orbit.file.compression import compression, \
    compression_opts


hdulist = fits.hdu.hdulist.HDUList


def add_latitude_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'latitude',
        data=make_spatial_bin_latitude(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.latitude


def add_longitude_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'longitude',
        data=make_spatial_bin_longitude(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.longitude


def add_tangent_altitude_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'tangent_altitude',
        data=make_spatial_bin_tangent_altitude(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.distance


def add_tangent_altitude_rate_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'tangent_altitude_rate',
        data=make_spatial_bin_tangent_altitude_rate(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.velocity


def add_line_of_sight_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'line_of_sight',
        data=make_spatial_bin_line_of_sight(hduls),
        compression=compression,
        compression_opts=compression_opts)


def add_solar_zenith_angle_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'solar_zenith_angle',
        data=make_spatial_bin_solar_zenith_angle(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_emission_angle_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'emission_angle',
        data=make_spatial_bin_emission_angle(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_phase_angle_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'phase_angle',
        data=make_spatial_bin_phase_angle(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_zenith_angle_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'zenith_angle',
        data=make_spatial_bin_zenith_angle(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_local_time_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'local_time',
        data=make_spatial_bin_local_time(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.local_time


def add_right_ascension_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'right_ascension',
        data=make_spatial_bin_right_ascension(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_declination_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'declination',
        data=make_spatial_bin_declination(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_bin_vector_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'bin_vector',
        data=make_spatial_bin_vector(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
