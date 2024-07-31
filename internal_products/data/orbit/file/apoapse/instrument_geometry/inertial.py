from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.instrument_geometry import \
    make_instrument_x_field_of_view_inertial_frame, \
    make_instrument_y_field_of_view_inertial_frame, \
    make_instrument_z_field_of_view_inertial_frame
from internal_products.data.orbit.file import comment
from internal_products.data.orbit.file.compression import compression, \
    compression_opts

hdulist = fits.hdu.hdulist.HDUList
frame = 'Inertial'


def add_instrument_x_field_of_view_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_x_field_of_view',
        data=make_instrument_x_field_of_view_inertial_frame(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
    dataset.attrs['frame'] = frame
    dataset.attrs['comment'] = comment.instrument_x_field_of_view


def add_instrument_y_field_of_view_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_y_field_of_view',
        data=make_instrument_y_field_of_view_inertial_frame(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
    dataset.attrs['frame'] = frame
    dataset.attrs['comment'] = comment.instrument_y_field_of_view


def add_instrument_z_field_of_view_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_z_field_of_view',
        data=make_instrument_z_field_of_view_inertial_frame(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
    dataset.attrs['frame'] = frame
    dataset.attrs['comment'] = comment.instrument_z_field_of_view
