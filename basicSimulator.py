from Constellation import *
from GroundSegment import *

delta_t = 10                                                                        # Time to rotate the constellation in seconds

N, h, inclination, walker_star = commercial_constellation_parameters("Kepler")      # Get the parameters for the constellation
constellation = Constellation(N, h, inclination, walker_star)                       # CReate the constellation

ground_segment = GroundSegment(KSAT)                                                # Get the parameters for the ground stations

constellation.rotate(delta_t)                                                       # Rotate the constellation by delta_t seconds

ground_segment.rotate(delta_t)                                                      # Rotate the ground stations by delta_t seconds

plot3D(constellation.satellites, ground_segment.ground_stations)                    # Plot the constellation and ground stations

inter_satellite_links = InterSatelliteLinks(constellation.satellites, "RF")         # Generate the inter-satelltie links

ground_to_satellite_links = GroundtoSatelliteLinks(constellation.satellites, ground_segment.ground_stations) # Generate the ground to satellite links


inter_satellite_links.plot_rates()                                                  # Plot the rates for the inter-satellite links

ground_to_satellite_links.plot_rates()                                              # Plot the rates for the ground to satellite links (uplink and downlink)


plt.show()