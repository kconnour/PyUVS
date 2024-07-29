"""This script will create a data file for each MAVEN orbit, populating it
with data."""
from h5py import File

import pyuvs as pu
from paths import orbit_file_path, iuvs_fits_files_location
import fits

import binning
import detector
import integration
import spacecraft_geometry
import spatial_bin_geometry
import instrument_geometry


def make_data_file(orbit: int) -> None:
    # Set up the file
    filename = orbit_file_path / pu.orbit.make_orbit_block(orbit) / f'{pu.make_orbit_code(orbit)}.hdf5'
    filename.parent.mkdir(parents=True, exist_ok=True)
    file = File(filename, mode='w')    # create file and overwrite it if it exists
    file.attrs['orbit'] = orbit

    # Fill the file with data
    for segment in ['apoapse']:
        match segment:
            case 'apoapse':
                apoapse = file.create_group('apoapse')

                # Get data from this to work with. For FUV/MUV independent data, just choose either channel
                apoapse_hduls = fits.get_apoapse_muv_hduls(orbit, iuvs_fits_files_location)

                # Add integration datasets to file
                apoapse_integration = apoapse.create_group('integration')
                integration.add_ephemeris_time_to_file(apoapse_integration, apoapse_hduls)
                integration.add_mirror_data_number_to_file(apoapse_integration, apoapse_hduls)
                integration.add_mirror_angle_to_file(apoapse_integration, apoapse_hduls)
                integration.add_field_of_view_to_file(apoapse_integration, apoapse_hduls)
                integration.add_case_temperature_to_file(apoapse_integration, apoapse_hduls)
                integration.add_integration_time_to_file(apoapse_integration, apoapse_hduls)
                integration.add_swath_number_to_file(apoapse_integration, apoapse_hduls, orbit)
                integration.add_number_of_swaths_to_file(apoapse_integration, apoapse_hduls, orbit)
                integration.add_opportunity_classification_to_file(apoapse_integration, apoapse_hduls, orbit)

                # Add spacecraft geometry datasets to file
                apoapse_spacecraft_geometry = apoapse.create_group('spacecraft_geometry')
                spacecraft_geometry.add_subsolar_latitude_to_file(apoapse_spacecraft_geometry, apoapse_hduls)
                spacecraft_geometry.add_subsolar_longitude_to_file(apoapse_spacecraft_geometry, apoapse_hduls)
                spacecraft_geometry.add_subspacecraft_latitude_to_file(apoapse_spacecraft_geometry, apoapse_hduls)
                spacecraft_geometry.add_subspacecraft_longitude_to_file(apoapse_spacecraft_geometry, apoapse_hduls)
                spacecraft_geometry.add_spacecraft_altitude_to_file(apoapse_spacecraft_geometry, apoapse_hduls)
                spacecraft_geometry.add_spacecraft_velocity_inertial_frame_to_file(apoapse_spacecraft_geometry, apoapse_hduls)
                spacecraft_geometry.add_spacecraft_vector_to_file(apoapse_spacecraft_geometry, apoapse_hduls)

                # Add instrument geometry datasets to file
                apoapse_instrument_geometry = apoapse.create_group('instrument_geometry')
                instrument_geometry.add_instrument_sun_angle_to_file(apoapse_instrument_geometry, apoapse_hduls)
                instrument_geometry.add_app_flip_to_file(apoapse_instrument_geometry, apoapse_hduls)

                for frame in ['iau_mars', 'inertial']:
                    match frame:
                        case 'iau_mars':
                            apoapse_instrument_geometry_iau_mars = apoapse_instrument_geometry.create_group('iau_mars_frame')
                            instrument_geometry.add_instrument_x_field_of_view_to_file(apoapse_instrument_geometry_iau_mars, apoapse_hduls)
                            instrument_geometry.add_instrument_y_field_of_view_to_file(apoapse_instrument_geometry_iau_mars, apoapse_hduls)
                            instrument_geometry.add_instrument_z_field_of_view_to_file(apoapse_instrument_geometry_iau_mars, apoapse_hduls)
                        case 'inertial':
                            apoapse_instrument_geometry_inertial = apoapse_instrument_geometry.create_group('inertial_frame')
                            instrument_geometry.add_instrument_x_field_of_view_inertial_frame_to_file(apoapse_instrument_geometry_inertial, apoapse_hduls)
                            instrument_geometry.add_instrument_y_field_of_view_inertial_frame_to_file(apoapse_instrument_geometry_inertial, apoapse_hduls)
                            instrument_geometry.add_instrument_z_field_of_view_inertial_frame_to_file(apoapse_instrument_geometry_inertial, apoapse_hduls)

                for channel in ['muv']:
                    match channel:
                        case 'muv':
                            apoapse_muv = apoapse.create_group('muv')

                            # Get apoapse MUV data to work with
                            apoapse_muv_hduls = fits.get_apoapse_muv_hduls(orbit, iuvs_fits_files_location)

                            # Add MUV-specific integration datasets to file
                            apoapse_muv_integration = apoapse_muv.create_group('integration')
                            integration.add_detector_temperature_to_file(apoapse_muv_integration, apoapse_muv_hduls)
                            integration.add_mcp_voltage_to_file(apoapse_muv_integration, apoapse_muv_hduls)
                            integration.add_mcp_voltage_gain_to_file(apoapse_muv_integration, apoapse_muv_hduls)
                            integration.add_failsafe_integrations_to_file(apoapse_muv_integration, apoapse_muv_hduls)
                            integration.add_dayside_integrations_to_file(apoapse_muv_integration, apoapse_muv_hduls)
                            integration.add_nightside_integrations_to_file(apoapse_muv_integration, apoapse_muv_hduls)

                            for experiment in ['failsafe', 'dayside', 'nightside']:
                                match experiment:
                                    case 'failsafe':
                                        apoapse_muv_failsafe = apoapse_muv.create_group('failsafe')

                                        # Get apoapse MUV failsafe data to work with
                                        apoapse_muv_failsafe_hduls = fits.get_apoapse_muv_failsafe_hduls(orbit, iuvs_fits_files_location)

                                        # Add binning datasets to file
                                        apoapse_muv_failsafe_binning = apoapse_muv_failsafe.create_group('binning')
                                        binning.add_spatial_bin_edges_to_file(apoapse_muv_failsafe_binning, apoapse_muv_failsafe_hduls)
                                        binning.add_spectral_bin_edges_to_file(apoapse_muv_failsafe_binning, apoapse_muv_failsafe_hduls)
                                        binning.add_spatial_bin_width_to_file(apoapse_muv_failsafe_binning, apoapse_muv_failsafe_hduls)
                                        binning.add_spectral_bin_width_to_file(apoapse_muv_failsafe_binning, apoapse_muv_failsafe_hduls)

                                        # Add bin geometry datasets to file
                                        apoapse_muv_failsafe_spatial_bin_geoemtry = apoapse_muv_failsafe.create_group('spatial_bin_geometry')
                                        spatial_bin_geometry.add_latitude_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_longitude_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_tangent_altitude_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_tangent_altitude_rate_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_line_of_sight_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_solar_zenith_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_emission_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_phase_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_zenith_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_local_time_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_right_ascension_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_declination_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)
                                        spatial_bin_geometry.add_bin_vector_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_failsafe_hduls)

                                        # Add detector datasets to file
                                        apoapse_muv_failsafe_detector = apoapse_muv_failsafe.create_group('detector')
                                        detector.add_raw_to_file(apoapse_muv_failsafe_detector, apoapse_muv_failsafe_hduls)
                                        detector.add_dark_subtracted_to_file(apoapse_muv_failsafe_detector, apoapse_muv_failsafe_hduls)
                                        detector.add_brightness_to_file(apoapse_muv_failsafe_detector, apoapse_muv_failsafe_hduls)

                                    case 'dayside':
                                        apoapse_muv_dayside = apoapse_muv.create_group('dayside')

                                        # Get apoapse MUV dayside data to work with
                                        apoapse_muv_dayside_hduls = fits.get_apoapse_muv_dayside_hduls(orbit, iuvs_fits_files_location)

                                        # Add binning datasets to file
                                        apoapse_muv_failsafe_binning = apoapse_muv_dayside.create_group('binning')
                                        binning.add_spatial_bin_edges_to_file(apoapse_muv_failsafe_binning, apoapse_muv_dayside_hduls)
                                        binning.add_spectral_bin_edges_to_file(apoapse_muv_failsafe_binning, apoapse_muv_dayside_hduls)
                                        binning.add_spatial_bin_width_to_file(apoapse_muv_failsafe_binning, apoapse_muv_dayside_hduls)
                                        binning.add_spectral_bin_width_to_file(apoapse_muv_failsafe_binning, apoapse_muv_dayside_hduls)

                                        # Add bin geometry datasets to file
                                        apoapse_muv_failsafe_spatial_bin_geoemtry = apoapse_muv_dayside.create_group('spatial_bin_geometry')
                                        spatial_bin_geometry.add_latitude_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_longitude_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_tangent_altitude_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_tangent_altitude_rate_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_line_of_sight_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_solar_zenith_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_emission_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_phase_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_zenith_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_local_time_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_right_ascension_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_declination_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)
                                        spatial_bin_geometry.add_bin_vector_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_dayside_hduls)

                                        # Add detector datasets to file
                                        apoapse_muv_dayside_detector = apoapse_muv_dayside.create_group('detector')
                                        detector.add_raw_to_file(apoapse_muv_dayside_detector, apoapse_muv_dayside_hduls)
                                        detector.add_dark_subtracted_to_file(apoapse_muv_dayside_detector, apoapse_muv_dayside_hduls)
                                        detector.add_brightness_to_file(apoapse_muv_dayside_detector, apoapse_muv_dayside_hduls)

                                    case 'nightside':
                                        apoapse_muv_nightside = apoapse_muv.create_group('nightside')

                                        # Get apoapse MUV nightside data to work with
                                        apoapse_muv_nightside_hduls = fits.get_apoapse_muv_nightside_hduls(orbit, iuvs_fits_files_location)

                                        # Add binning datasets to file
                                        apoapse_muv_failsafe_binning = apoapse_muv_nightside.create_group('binning')
                                        binning.add_spatial_bin_edges_to_file(apoapse_muv_failsafe_binning, apoapse_muv_nightside_hduls)
                                        binning.add_spectral_bin_edges_to_file(apoapse_muv_failsafe_binning, apoapse_muv_nightside_hduls)
                                        binning.add_spatial_bin_width_to_file(apoapse_muv_failsafe_binning, apoapse_muv_nightside_hduls)
                                        binning.add_spectral_bin_width_to_file(apoapse_muv_failsafe_binning, apoapse_muv_nightside_hduls)

                                        # Add bin geometry datasets to file
                                        apoapse_muv_failsafe_spatial_bin_geoemtry = apoapse_muv_nightside.create_group('spatial_bin_geometry')
                                        spatial_bin_geometry.add_latitude_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_longitude_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_tangent_altitude_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_tangent_altitude_rate_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_line_of_sight_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_solar_zenith_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_emission_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_phase_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_zenith_angle_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_local_time_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_right_ascension_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_declination_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)
                                        spatial_bin_geometry.add_bin_vector_to_file(apoapse_muv_failsafe_spatial_bin_geoemtry, apoapse_muv_nightside_hduls)

                                        # Add detector datasets to file
                                        apoapse_muv_nightside_detector = apoapse_muv_nightside.create_group('detector')
                                        detector.add_raw_to_file(apoapse_muv_nightside_detector, apoapse_muv_nightside_hduls)
                                        detector.add_dark_subtracted_to_file(apoapse_muv_nightside_detector, apoapse_muv_nightside_hduls)
                                        detector.add_brightness_to_file(apoapse_muv_nightside_detector, apoapse_muv_nightside_hduls)
