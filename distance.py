from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
import pdb
from datetime import date, timedelta
import pandas as pd
import re
from tqdm import tqdm
from numpy import argmin
from typing import List, Dict, AnyStr



# Dates
years = 1
epoch_length = 365 * years



class Distance:
    def __init__(self, spkids: List, location: AnyStr, epoch_length: int) -> None:
        self.spkids = spkids
        self.location = location
        self.epoch_length = epoch_length
        self.today = date.today().strftime('%Y-%m-%d')
        self.future_date = (date.today() + timedelta(days=epoch_length)).strftime('%Y-%m-%d')

    def calculate_ephemeris(self, spkid: AnyStr) -> Table:
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
            neo_ephem = self.calculate_ephemeris(spkid)
                # Calculate the distance between NEO and Earth
            distance = neo_ephem['delta'].to(u.km) - neo_ephem['r_3sigma']# - earth_ephem['delta'].to(u.km)
            min_dist = distance.min()
            # object comes within one earth radius of observer
            if min_dist.value <= 6378.136:
                idx = argmin(distance)
                t = neo_ephem['datetime_jd'][idx]
                bad_neos.append({'spkid':spkid, 'distance': min_dist.value, 'time': t})
        return bad_neos
    
    def plot_ephemeris(self, ephem: Table, feature: AnyStr, title: AnyStr) -> None:
        # Plot the distance
        fig, ax = plt.subplots()
        ax.plot(ephem['datetime_jd'].to(u.km), ephem[feature], 'k')
        ax.set_xlabel('Time (JD)')
        ax.set_ylabel('Distance (km)')
        plt.title(title)
        plt.show()



if __name__ == '__main__':
    # Read in export from SBDB
    df = pd.read_csv('sbdb_query_results.csv')
    spkids = df.loc[df.neo == 'Y', 'spkid'].to_list()
    print(f'There are {len(spkids)} NEOs to process')
    dist = Distance(spkids, "399", 730)
    neos = dist.check_approaches()
    df_out = pd.DataFrame(neos)
    df_out.to_csv('neos.csv', index=False)
