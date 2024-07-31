from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.spacecraft_geometry import \
    make_subsolar_latitude, make_subsolar_longitude, \
    make_subspacecraft_latitude, make_subspacecraft_longitude, \
    make_spacecraft_altitude
from internal_products.data.orbit.file import comment
from internal_products.data.orbit.file.compression import compression, \
    compression_opts


hdulist = fits.hdu.hdulist.HDUList


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
    dataset.attrs['unit'] = units.distance
    dataset.attrs['comment'] = comment.spacecraft_altitude
