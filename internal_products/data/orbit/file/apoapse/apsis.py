from h5py import Group

from internal_products.data.orbit.apsis import make_ephemeris_time, \
    make_mars_year, make_sol, make_solar_longitude, make_subsolar_latitude, \
    make_subsolar_longitude, make_subspacecraft_latitude, \
    make_subspacecraft_longitude, make_spacecraft_altitude, \
    make_subspacecraft_local_time, make_mars_sun_distance, \
    make_subsolar_subspacecraft_angle, make_mars_position, make_mars_velocity
from internal_products.data.orbit.file.compression import compression, \
    compression_opts
from internal_products.data import units

apsis = 'apoapse'


def add_ephemeris_time_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'ephemeris_time',
        data=make_ephemeris_time(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.ephemeris_time


def add_mars_year_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'mars_year',
        data=make_mars_year(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.mars_year


def add_sol_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'sol',
        data=make_sol(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.sol


def add_solar_longitude_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'solar_longitude',
        data=make_solar_longitude(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.solar_longitude


def add_subsolar_latitude_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'subsolar_latitude',
        data=make_subsolar_latitude(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.latitude


def add_subsolar_longitude_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'subsolar_longitude',
        data=make_subsolar_longitude(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.longitude


def add_subspacecraft_latitude_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'subspacecraft_latitude',
        data=make_subspacecraft_latitude(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.latitude


def add_subspacecraft_longitude_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'subspacecraft_longitude',
        data=make_subspacecraft_longitude(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.longitude


def add_spacecraft_altitude_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'spacecraft_altitude',
        data=make_spacecraft_altitude(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.distance


def add_subspacecraft_local_time_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'subspacecraft_local_time',
        data=make_subspacecraft_local_time(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.local_time


def add_mars_sun_distance_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'mars_sun_distance',
        data=make_mars_sun_distance(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.distance


def add_subsolar_subspacecraft_angle_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'subsolar_subspacecraft_angle',
        data=make_subsolar_subspacecraft_angle(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.angle


def add_mars_position_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'mars_position',
        data=make_mars_position(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.distance


def add_mars_velocity_to_file(group: Group, orbit: int) -> None:
    dataset = group.create_dataset(
        'mars_velocity',
        data=make_mars_velocity(orbit, apsis),
        compression=compression,
        compression_opts=compression_opts)
    dataset.attrs['unit'] = units.velocity
