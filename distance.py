from astroquery.jplhorizons import Horizons
import astropy.units as u
import matplotlib.pyplot as plt
import pdb
from datetime import date, timedelta
import pandas as pd
import re
from tqdm import tqdm
from numpy import argmin

df = pd.read_csv('sbdb_query_results.csv')
spkids = df.loc[:, 'spkid'].to_list()

# Query for ephemerides
distance_dict = {"Mars": 499, "Moon":301, 'Bus':'1000010'}
body="Bus"

# Dates
epoch_length = 365
today = date.today().strftime('%Y-%m-%d')
future_date = (date.today() + timedelta(days=epoch_length)).strftime('%Y-%m-%d')

neos = []
for spkid in tqdm(spkids):
    neo = Horizons(id=spkid, location="399", id_type='designation', epochs={'start': today, 'stop': future_date, 'step': '1d'})
    try:
        neo_ephem = neo.ephemerides()
    except Exception as e:
        print('error thrown, trying record number...')
        records = str(e).split('\n')
        pattern = '^[^\d]*(\d+)'
        matches = re.findall(pattern, records[-2])
        try:
            neo = Horizons(id=matches[0], location="399", id_type=None, epochs={'start': today, 'stop': future_date, 'step': '1d'})
            neo_ephem = neo.ephemerides()
        except:
            continue

    # Calculate the distance between NEO and Earth
    distance = neo_ephem['delta'].to(u.au)# - earth_ephem['delta'].to(u.km)
    mean_y = distance.mean().value
    min_dist = distance.min()
    if min_dist.value <= 1.3:
        idx = argmin(distance)
        t = neo_ephem['datetime_jd'][idx]
        neos.append({'spkid':spkid, 'distance': min_dist.value, 'time': t})

df_out = pd.DataFrame(neos)
df_out.to_csv('neos.csv', index=False)



# # Plot the distance
# fig, ax = plt.subplots()
# ax.plot(neo_ephem['datetime_jd'], distance, 'k')
# ax.set_xlabel('Time (JD)')
# ax.set_ylabel('Distance (km)')
# # Mean distance from earth
# plt.axhline(mean_y, linestyle='--', color='gray')
# plt.title("Distance from Earth to Moon")
# plt.show()
