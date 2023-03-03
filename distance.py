from astroquery.jplhorizons import Horizons
import astropy.units as u
import matplotlib.pyplot as plt
import pdb

# Query for NEO ephemerides
distance_dict = {"Mars": 499, "Moon":301}
body="Mars"

neo = Horizons(id=distance_dict.get(body), location="399", epochs={'start': '2020-01-01', 'stop': '2023-03-01', 'step': '1d'})
neo_ephem = neo.ephemerides()

# Calculate the distance between NEO and Earth
distance = neo_ephem['delta'].to(u.km)# - earth_ephem['delta'].to(u.km)
pdb.set_trace()
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
