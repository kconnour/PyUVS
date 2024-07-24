"""This script will create the MAVEN apsis file for all orbits from the start
of the mission until the latest time with kernels."""
from h5py import File
import mars_time as mt
import numpy as np
import spiceypy

import paths
import angle
import kernels
import mars_geometry
import maven_geometry


if __name__ == '__main__':
    # Make an empty file if it doesn't exist; overwrite it otherwise
    file = File(paths.apsis_file_path, mode='w')

    # Load in the proper SPICE kernels
    kernels.furnish_leap_second_kernel()
    kernels.furnish_mars_kernel()
    kernels.furnish_maven_orbit_kernels()
    kernels.furnish_planetary_constants_kernel()
    kernels.furnish_maven_frame_kernel()
    print('loaded kernels')

    # Populate the arrays for each segment
    for segment in ['apoapse', 'periapse']:
        print(f'starting {segment}')

        # Make the file structure
        apsis = file.create_group(segment)

        # Create data at apses to populate the arrays with
        ephemeris_times = maven_geometry.compute_maven_apsis_ephemeris_times(segment)
        datetimes = [spiceypy.et2datetime(et) for et in ephemeris_times]
        mars_times = [mt.datetime_to_marstime(dt) for dt in datetimes]
        mars_years = np.array([i.year for i in mars_times])
        sols = np.array([i.sol for i in mars_times])

        orbits = np.arange(len(ephemeris_times)) + 1

        solar_longitude = np.array([mars_geometry.compute_solar_longitude(et) for et in ephemeris_times])
        subsolar_latitude = np.array([mars_geometry.compute_subsolar_point(et)[0] for et in ephemeris_times])
        subsolar_longitude = np.array([mars_geometry.compute_subsolar_point(et)[1] for et in ephemeris_times])
        mars_sun_distance = np.array([mars_geometry.compute_mars_sun_distance(et) for et in ephemeris_times])

        subspacecraft_latitude = np.array([maven_geometry.compute_subspacecraft_point(et)[0] for et in ephemeris_times])
        subspacecraft_longitude = np.array([maven_geometry.compute_subspacecraft_point(et)[1] for et in ephemeris_times])
        spacecraft_altitude = np.array([maven_geometry.compute_spacecraft_altitude(et) for et in ephemeris_times])
        subspacecraft_local_time = np.array([maven_geometry.compute_subspacecraft_local_time(et) for et in ephemeris_times])
        subsolar_subspacecraft_angle = angle.haversine((subsolar_latitude, subsolar_longitude), (subspacecraft_latitude, subspacecraft_longitude))
        mars_vector = np.vstack([maven_geometry.compute_mars_state(et)[0] for et in ephemeris_times])
        mars_velocity = np.vstack([maven_geometry.compute_mars_state(et)[1] for et in ephemeris_times])

        # Add ephemeris times
        dataset = apsis.create_dataset('ephemeris_time', data=ephemeris_times)
        dataset.attrs['unit'] = 'Seconds since J2000'

        # Add orbits
        apsis.create_dataset('orbit', data=orbits)

        # Add Mars years
        apsis.create_dataset('mars_year', data=mars_years)

        # Add sols
        dataset = apsis.create_dataset('sol', data=sols)
        dataset.attrs['unit'] = 'Day of the Martian year'

        # Add solar longitudes
        dataset = apsis.create_dataset('solar_longitude', data=solar_longitude)
        dataset.attrs['unit'] = 'Degrees'

        # Add sub-solar latitudes
        dataset = apsis.create_dataset('subsolar_latitude', data=subsolar_latitude)
        dataset.attrs['unit'] = 'Degrees [N]'

        # Add sub-solar longitudes
        dataset = apsis.create_dataset('subsolar_longitude', data=subsolar_longitude)
        dataset.attrs['unit'] = 'Degrees [E]'

        # Add sub-spacecraft latitudes
        dataset = apsis.create_dataset('subspacecraft_latitude', data=subspacecraft_latitude)
        dataset.attrs['unit'] = 'Degrees [N]'

        # Add sub-spacecraft longitudes
        dataset = apsis.create_dataset('subspacecraft_longitude', data=subspacecraft_longitude)
        dataset.attrs['unit'] = 'Degrees [E]'

        # Add spacecraft altitudes
        dataset = apsis.create_dataset('spacecraft_altitude', data=spacecraft_altitude)
        dataset.attrs['unit'] = 'km'

        # Add subspacecraft local times
        dataset = apsis.create_dataset('subspacecraft_local_time', data=subspacecraft_local_time)
        dataset.attrs['unit'] = 'hours'

        # Add Mars-sun distances
        dataset = apsis.create_dataset('mars_sun_distance', data=mars_sun_distance)
        dataset.attrs['unit'] = 'km'

        # Add sub-solar sub-spacecraft angles
        dataset = apsis.create_dataset('subsolar_subspacecraft_angle', data=subsolar_subspacecraft_angle)
        dataset.attrs['unit'] = 'Degrees'

        # Add Mars position vectors
        dataset = apsis.create_dataset('mars_vector', data=mars_vector)
        dataset.attrs['unit'] = 'km'

        # Add Mars velocity vectors
        dataset = apsis.create_dataset('mars_velocity', data=mars_velocity)
        dataset.attrs['unit'] = 'km/s'

    file.close()
