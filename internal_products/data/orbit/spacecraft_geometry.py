from h5py import Group

from compression import compression, compression_opts
from data import hdulist, make_subsolar_latitude, make_subsolar_longitude, \
    make_subspacecraft_latitude, make_subspacecraft_longitude, \
    make_spacecraft_altitude, make_spacecraft_velocity_inertial_frame, \
    make_spacecraft_vector
import units


def add_subsolar_latitude_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'subsolar_latitude',
        data=make_subsolar_latitude(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.latitude


def add_subsolar_longitude_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'subsolar_longitude',
        data=make_subsolar_longitude(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.longitude


def add_subspacecraft_latitude_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'subspacecraft_latitude',
        data=make_subspacecraft_latitude(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.latitude


def add_subspacecraft_longitude_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'subspacecraft_longitude',
        data=make_subspacecraft_longitude(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.longitude


def add_spacecraft_altitude_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'spacecraft_altitude',
        data=make_spacecraft_altitude(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.altitude
    dataset.attrs['comment'] = 'Position of MAVEN above Mars\'s reference ellipsoid'


def add_spacecraft_velocity_inertial_frame_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'spacecraft_velocity_inertial_frame',
        data=make_spacecraft_velocity_inertial_frame(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.velocity


def add_spacecraft_vector_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'spacecraft_vector',
        data=make_spacecraft_vector(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.altitude
    dataset.attrs['comment'] = 'Position of MAVEN relative to Mars\'s center of mass'
