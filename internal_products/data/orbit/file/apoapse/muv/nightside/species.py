from astropy.io import fits
from h5py import Group

from internal_products.data import units
from internal_products.data.orbit.mlr import fit_muv_templates_to_nightside_data
from internal_products.data.orbit.file.compression import compression, \
    compression_opts


hdulist = fits.hdu.hdulist.HDUList


def add_mlr_fits_to_file(group: Group, hduls: list[hdulist]) -> None:
    mlr_fits = fit_muv_templates_to_nightside_data(hduls)

    constant = mlr_fits[0]
    co_cameron_bands = mlr_fits[1]
    cop_1ng = mlr_fits[2]
    co2p_fdb = mlr_fits[3]
    co2p_uvd = mlr_fits[4]
    n2vk = mlr_fits[5]
    no_nightglow = mlr_fits[6]
    oxygen_2972 = mlr_fits[7]
    solar_continuum = mlr_fits[8]

    dataset = group.create_dataset(
        'constant_fit_term',
        data=constant,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness

    dataset = group.create_dataset(
        'co_cameron_bands',
        data=co_cameron_bands,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness

    dataset = group.create_dataset(
        'co+_1negative',
        data=cop_1ng,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness

    dataset = group.create_dataset(
        'co2+_fdb',
        data=co2p_fdb,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness

    dataset = group.create_dataset(
        'co2+_uvd',
        data=co2p_uvd,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness

    dataset = group.create_dataset(
        'n2_vk',
        data=n2vk,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness

    dataset = group.create_dataset(
        'no_nightglow',
        data=no_nightglow,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness

    dataset = group.create_dataset(
        'oxygen_2972',
        data=oxygen_2972,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness

    dataset = group.create_dataset(
        'solar_continuum',
        data=solar_continuum,
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.brightness
