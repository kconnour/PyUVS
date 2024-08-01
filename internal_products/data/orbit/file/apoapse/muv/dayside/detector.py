from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.detector import make_random_uncertainty, \
    make_systematic_uncertainty, make_brightness
from internal_products.data.orbit.file.compression import compression, \
    compression_opts


hdulist = fits.hdu.hdulist.HDUList


def add_random_uncertainty_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'random_uncertainty',
        data=make_random_uncertainty(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness + '/nm'


def add_systematic_uncertainty_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'systematic_uncertainty',
        data=make_systematic_uncertainty(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness + '/nm'


def add_brightness_to_file(group: Group, hduls: list[hdulist]) -> None:
    dataset = group.create_dataset(
        'brightness',
        data=make_brightness(hduls),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness
