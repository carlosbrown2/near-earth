from astroquery.jplhorizons import Horizons
import astropy.units as u
import matplotlib.pyplot as plt
import pdb
from datetime import date, timedelta
import pandas as pd
import re
from tqdm import tqdm
from numpy import argmin
from typing import List, Dict



# Dates
years = 1
epoch_length = 365 * years



class Distance:
    def __init__(self, spkids, location, epoch_length) -> None:
        self.spkids = spkids
        self.location = location
        self.epoch_length = epoch_length
        self.today = date.today().strftime('%Y-%m-%d')
        self.future_date = (date.today() + timedelta(days=epoch_length)).strftime('%Y-%m-%d')    

    def calculate_ephemerid(self, spkid):
        neo = Horizons(id=spkid, location=self.location, id_type='designation', epochs={'start': self.today, 'stop': self.future_date, 'step': '1d'})
        try:
            neo_ephem = neo.ephemerides()
        except Exception as e:
            print('error thrown, trying record number approach...')
            records = str(e).split('\n')
            pattern = r'^[^\d]*(\d+)'
            if len(records) >= 3 and records[-1] == '':
                matches = re.findall(pattern, records[-2])
            elif len(records) == 2 and records[-1] != '':
                try:
                    matches = re.findall(pattern, records[-1])
                except:
                    return 1
            try:
                neo = Horizons(id=matches[0], location="399", id_type=None, epochs={'start': self.today, 'stop': self.future_date, 'step': '1d'})
                neo_ephem = neo.ephemerides()
            except:
                return 1
        return neo_ephem
    
    def check_approaches(self) -> List[Dict]:
        bad_neos = []
        for spkid in tqdm(self.spkids):
            neo_ephem = self.calculate_ephemerid(spkid)
                # Calculate the distance between NEO and Earth
            distance = neo_ephem['delta'].to(u.km)# - earth_ephem['delta'].to(u.km)
            mean_y = distance.mean().value
            min_dist = distance.min()
            # object comes within one earth radius of observer
            if min_dist.value <= 6378.136:
                idx = argmin(distance)
                t = neo_ephem['datetime_jd'][idx]
                bad_neos.append({'spkid':spkid, 'distance': min_dist.value, 'time': t})
        return bad_neos

if __name__ == '__main__':
    df = pd.read_csv('sbdb_query_results.csv')
    spkids = df.loc[df.neo == 'Y', 'spkid'].to_list()
    print(f'There are {len(spkids)} NEOs to process')
    dist = Distance(spkids, "399", 730)
    neos = dist.check_approaches()
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
