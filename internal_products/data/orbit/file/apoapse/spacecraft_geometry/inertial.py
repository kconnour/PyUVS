from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.spacecraft_geometry import \
    make_spacecraft_position_inertial_frame, \
    make_spacecraft_velocity_inertial_frame
from internal_products.data.orbit.file.compression import compression, \
    compression_opts
from internal_products.data.orbit.file import comment

hdulist = fits.hdu.hdulist.HDUList
frame = 'Inertial'


def add_spacecraft_position_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'spacecraft_position',
        data=make_spacecraft_position_inertial_frame(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.distance
    dataset.attrs['frame'] = frame
    dataset.attrs['comment'] = comment.spacecraft_position


def add_spacecraft_velocity_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'spacecraft_velocity',
        data=make_spacecraft_velocity_inertial_frame(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.velocity
    dataset.attrs['frame'] = frame
    dataset.attrs['comment'] = comment.spacecraft_velocity
