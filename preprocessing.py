import json
import os

from utils import *

json_path = "data/vn_administrative_location.json"
with open(json_path) as f:
    locations = json.load(f)
LIST_OF_PROVINCES = []
for location in locations:
    LIST_OF_PROVINCES.append(location["name"])

if __name__ == "__main__":
    full_locations_json = {}
    full_locations_street_json = {}
    similarities = []

    for province in locations:
        province_name = province["name"]
        districts = province["districts"]
        
        for district in districts:
            wards = district["wards"]
            streets = district["streets"]
            for ward in wards:
                for street in streets:
                    location_info1 = street["name"]+', '+ward["name"]+', '+district["name"]+', '+province["name"]
                    full_locations_street_json[norm(location_info1)] = location_info1

                location_info2 = ward["name"]+', '+district["name"]+', '+province["name"]
                full_locations_json[norm(location_info2)] = location_info2

    with open("data/full_locations_street.json", 'w') as fp:
        json.dump(full_locations_street_json, fp)

    with open("data/full_locations.json", 'w') as fp:
        json.dump(full_locations_json, fp)