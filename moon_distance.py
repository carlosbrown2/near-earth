from astroquery.jplhorizons import Horizons
import astropy.units as u
import matplotlib.pyplot as plt
from numpy import mean

# Query for NEO ephemerides (The moon)
neo = Horizons(id='301', location="399", epochs={'start': '2023-01-01', 'stop': '2023-03-01', 'step': '1d'})
neo_ephem = neo.ephemerides()

# Calculate the distance between NEO and Earth
distance = neo_ephem['delta'].to(u.km)# - earth_ephem['delta'].to(u.km)
mean_y = distance.mean().value
# Plot the distance
fig, ax = plt.subplots()
ax.plot(neo_ephem['datetime_jd'], distance, 'k')
ax.set_xlabel('Time (JD)')
ax.set_ylabel('Distance (km)')
# Mean distance from earth
plt.axhline(mean_y, linestyle='--', color='gray')
plt.title("Distance from Earth to Moon")
plt.show()
