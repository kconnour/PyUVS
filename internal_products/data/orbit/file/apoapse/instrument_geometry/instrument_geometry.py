from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.instrument_geometry import \
    make_instrument_sun_angle, make_app_flip
from internal_products.data.orbit.file.compression import compression, \
    compression_opts

hdulist = fits.hdu.hdulist.HDUList


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
    dataset.attrs['comment'] = ('True if the APP is flipped; False otherwise. '
        'In principle, the APP can be rotated at any angle, but in practice '
        'the engineering team only rotates the APP either 0 or 180 degrees.')
