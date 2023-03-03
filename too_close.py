from astroquery.jplhorizons import Horizons
import astropy.units as u
import matplotlib.pyplot as plt
import pdb
# Query for NEO ephemerides
distance_dict = {"Mars": 499, "Moon":301}
body="Mars"
plot_threshold = True

# km threshold distance
threshold = 7500000
neo = Horizons(id=distance_dict.get(body), location="399", epochs={'start': '2023-01-01', 'stop': '2023-03-01', 'step': '1d'})
neo_ephem = neo.ephemerides()

# Calculate the distance between NEO and Earth
distance = neo_ephem['delta'].to(u.km)# - earth_ephem['delta'].to(u.km)
mean_y = distance.mean().value
max_y = distance.max().value
min_y = distance.min().value
text_adjust = (max_y - min_y) / 40

# pdb.set_trace()
close_dist_list = [x.value for x in distance if x.value < threshold]
# Plot the distance
fig, ax = plt.subplots()
ax.plot(neo_ephem['datetime_jd'], distance, 'k')
ax.set_xlabel('Time (JD)')
ax.set_ylabel('Distance (km)')
# Mean distance from earth
plt.axhline(mean_y, linestyle='--', color='gray')
plt.text(neo_ephem['datetime_jd'][0], mean_y+text_adjust, f"Mean distance: {mean_y:.2f} km", color='gray')
if plot_threshold:
    plt.axhline(threshold, linestyle='--', color='red')
    plt.text(neo_ephem['datetime_jd'][0], threshold+text_adjust, f"Threshold distance: {threshold:.2f} km", color='red')
plt.title(f"Distance from Earth to {body}")
plt.show()
