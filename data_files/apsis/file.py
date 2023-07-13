from h5py import File
import mars_time as mt
import numpy as np
import spiceypy

import pyuvs as pu
from apsis import compute_maven_apsis_et


file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5', mode='a')  # 'a' means to read/write if exists, create otherwise

# Load in the proper kernels
spiceypy.kclear()
pu.furnish_ck_files()
pu.furnish_lsk_files()
pu.furnish_pck_files()
pu.furnish_sclk_files()
pu.furnish_spk_files()

print('loaded kernels')

for segment in ['apoapse', 'periapse']:
    print(f'starting {segment}')

    # Make the file structure
    apsis = file.create_group(segment)

    # Add the ephemeris times of the orbit
    ephemeris_times = compute_maven_apsis_et(segment)
    dataset = apsis.create_dataset('ephemeris_time', data=ephemeris_times)
    dataset.attrs['unit'] = 'Seconds since J2000'

    # Add orbits
    orbits = np.arange(len(ephemeris_times)) + 1
    apsis.create_dataset('orbit', data=orbits)

    # Add Mars year
    datetimes = [spiceypy.et2datetime(et) for et in ephemeris_times]
    mars_time = [mt.datetime_to_marstime(dt) for dt in datetimes]
    mars_year = np.array([i.year for i in mars_time])
    apsis.create_dataset('mars_year', data=mars_year)

    # Add sol
    sol = np.array([i.sol for i in mars_time])
    apsis.create_dataset('sol', data=sol)

    # Add solar longitude
    solar_longitude = np.array([pu.compute_solar_longitude(et) for et in ephemeris_times])
    apsis.create_dataset('solar_longitude', data=solar_longitude)

    # Add sub-solar latitude
    subsolar_latitude = np.array([pu.compute_subsolar_point(et)[0] for et in ephemeris_times])
    dataset = apsis.create_dataset('subsolar_latitude', data=subsolar_latitude)
    dataset.attrs['unit'] = 'Degrees [N]'

    # Add sub-solar longitude
    subsolar_longitude = np.array([pu.compute_subsolar_point(et)[1] for et in ephemeris_times])
    dataset = apsis.create_dataset('subsolar_longitude', data=subsolar_longitude)
    dataset.attrs['unit'] = 'Degrees [E]'

    # Add sub-spacecraft latitude
    subspacecraft_latitude = np.array([pu.compute_subspacecraft_point(et)[0] for et in ephemeris_times])
    dataset = apsis.create_dataset('subspacecraft_latitude', data=subspacecraft_latitude)
    dataset.attrs['unit'] = 'Degrees [N]'

    # Add sub-spacecraft longitude
    subsolar_longitude = np.array([pu.compute_subspacecraft_point(et)[1] for et in ephemeris_times])
    dataset = apsis.create_dataset('subspacecraft_longitude', data=subsolar_longitude)
    dataset.attrs['unit'] = 'Degrees [E]'

    # Add spacecraft altitude
    spacecraft_altitude = np.array([pu.compute_spacecraft_altitude(et) for et in ephemeris_times])
    dataset = apsis.create_dataset('spacecraft_altitude', data=spacecraft_altitude)
    dataset.attrs['unit'] = 'km'

    # Add subspacecraft local time
    subspacecraft_local_time = np.array([pu.compute_subspacecraft_local_time(et) for et in ephemeris_times])
    dataset = apsis.create_dataset('subspacecraft_local_time', data=subspacecraft_local_time)
    dataset.attrs['unit'] = 'hours'

    # Add Mars-sun distance
    mars_sun_distance = np.array([pu.compute_mars_sun_distance(et) for et in ephemeris_times])
    dataset = apsis.create_dataset('mars_sun_distance', data=mars_sun_distance)
    dataset.attrs['unit'] = 'km'

    # Add subsolar_subspacecraft_angle
    angle = pu.haversine((subsolar_latitude, subsolar_longitude), (subspacecraft_latitude, subsolar_longitude))
    dataset = apsis.create_dataset('subsolar_subspacecraft_angle', data=angle)
    dataset.attrs['unit'] = 'Degrees'
