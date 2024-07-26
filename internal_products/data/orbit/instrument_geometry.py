from h5py import Group

from data import hdulist, make_instrument_x_field_of_view, \
    make_instrument_y_field_of_view, make_instrument_z_field_of_view, \
    make_instrument_x_field_of_view_inertial_frame, \
    make_instrument_y_field_of_view_inertial_frame, \
    make_instrument_z_field_of_view_inertial_frame, make_instrument_sun_angle, \
    make_app_flip
from compression import compression, compression_opts

import units


# TODO: I should really specify the frame of these 3 datasets
def add_instrument_x_field_of_view_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_x_field_of_view',
        data=make_instrument_x_field_of_view(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
    dataset.attrs['comment'] = \
        ('Direction of IUVS\'s field of view X axis, including scan mirror '
         'rotation. This is the instrument spatial direction (parallel to the '
         'slit).')


def add_instrument_y_field_of_view_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_y_field_of_view',
        data=make_instrument_y_field_of_view(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
    dataset.attrs['comment'] = \
        ('Direction of IUVS\'s field of view Y axis, including scan mirror '
         'rotation. This is the instrument scan direction (perpendicular to '
         'the slit).')


def add_instrument_z_field_of_view_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_z_field_of_view',
        data=make_instrument_z_field_of_view(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
    dataset.attrs['comment'] = \
        ('Direction of IUVS\'s field of view Z axis, including scan mirror '
         'rotation. This is in the direction of IUVS\'s boresight')


def add_instrument_x_field_of_view_inertial_frame_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_x_field_of_view_inertial_frame',
        data=make_instrument_x_field_of_view_inertial_frame(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
    dataset.attrs['comment'] = \
        ('Direction of IUVS\'s field of view X axis, including scan mirror '
         'rotation. This is the instrument spatial direction (parallel to the '
         'slit).')


def add_instrument_y_field_of_view_inertial_frame_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_y_field_of_view_inertial_frame',
        data=make_instrument_y_field_of_view_inertial_frame(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
    dataset.attrs['comment'] = \
        ('Direction of IUVS\'s field of view Y axis, including scan mirror '
         'rotation. This is the instrument scan direction (perpendicular to '
         'the slit).')


def add_instrument_z_field_of_view_inertial_frame_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_z_field_of_view_inertial_frame',
        data=make_instrument_z_field_of_view_inertial_frame(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.unit_vector
    dataset.attrs['comment'] = \
        ('Direction of IUVS\'s field of view Z axis, including scan mirror '
         'rotation. This is in the direction of IUVS\'s boresight')


def add_instrument_sun_angle_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'instrument_sun_angle',
        data=make_instrument_sun_angle(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_app_flip_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'app_flip',
        data=make_app_flip(hduls),
        compression=compression,
        compression_opts=compression_opts)
