import requests
from random import sample
import pdb

body_list = range(1000001, 54433326)

# test = sample(body_list, 100)
test = ['1000010']

result_list = []
for body in test:
    query = f'https://ssd-api.jpl.nasa.gov/sbdb.api?spk={str(body)}'
    result = requests.get(query)
    if result.status_code == 200:
        result_list.append(result.text)

pdb.set_trace()