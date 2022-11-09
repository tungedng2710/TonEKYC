import easyocr
import cv2 
import matplotlib.pyplot as plt
import re
import unidecode
from datetime import datetime
import numpy as np
import math
import os 
import json
from difflib import SequenceMatcher
from itertools import combinations

READER = easyocr.Reader(['vi'])
json_path = "data/vn_administrative_location.json"
with open(json_path) as f:
    LOCATIONS = json.load(f)
LIST_OF_PROVINCES = []
for location in LOCATIONS:
    LIST_OF_PROVINCES.append(location["name"])
with open("data/full_locations.json") as f:
    FULL_LOCATIONS_DICT = json.load(f)
with open("data/full_locations_street.json") as f:
    FULL_LOCATIONS_STREET_DICT = json.load(f)


def get_ocr_results(image_path):
    raw_results = READER.readtext(image_path)
    detections = {}
    for result in raw_results:
        coordinates = result[0]
        top_left = tuple(coordinates[0])
        top_right = tuple(coordinates[1])
        bottom_right = tuple(coordinates[2])
        bottom_left = tuple(coordinates[3])
        center = (int((top_left[0] + top_right[0]) / 2), 
                int((top_left[1] + bottom_left[1]) / 2))
        keypoints = [top_left, top_right, bottom_right, bottom_left, center]

        detections[result[1]] = {
            "keypoints": keypoints,
            "confidence": result[2] 
        }
    return detections


def dist(x, y):
    return math.sqrt( (y[0] - x[0])**2 + (y[1] - x[1])**2 )

def get_nearest_right(p1, list_of_p2):
    '''
    p1, p2 have (x, y) format
    '''
    vd = []
    for i in range(len(list_of_p2)):
        if list_of_p2[i][0] > p1[0]:
            vd.append(dist(p1, list_of_p2[i]))
        else:
            vd.append(math.inf) # assign a very large number (infinity)
    return list_of_p2[np.argmin(np.array(vd))]

def check_id_number(text: str = None):
    if text == None:
        return False
    if text.isnumeric():
        if len(text) == 12:
            return True
        else:
            return False
    return False

def check_name(text: str = None):
    if text == None:
        return False
    no_accent = unidecode.unidecode(text)
    if no_accent.isupper():
        return True
    else:
        return False

def check_date_format(text: str = None):
    text = re.sub('[^A-Za-z0-9]+', '', text)
    # text = re.findall(r'\d+', text)[-1]
    format = "%d%m%Y"
    res = True
    try:
        res = bool(datetime.strptime(text, format))
    except ValueError:
        res = False
    return res

def get_datetime(text: str = None):
    for txt in text.split():
        if check_date_format(txt):
            return txt
        else:
            return None
    
def norm(text: str = None):
    text = unidecode.unidecode(text).lower().replace(' ', '')
    text = re.sub('[^A-Za-z0-9]+', '', text)
    return text

def similar(s1: str = None, 
            s2: str = None):    
    return SequenceMatcher(None, s1, s2).ratio()

def argmax(lst: list = None):
    return lst.index(max(lst))

def count_word(sentence: str = ""):
    return len(sentence.split())

def get_nearest_province(location_info, LIST_OF_PROVINCES):
    potential_name = norm(location_info)[-10:]
    similarities = []
    for province in LIST_OF_PROVINCES:
        similarity = similar(norm(potential_name), norm(province))
        similarities.append(similarity)
    return LIST_OF_PROVINCES[argmax(similarities)]


def refine_location_information(sample: str = "", 
                                get_street_info: bool = False):
    full_locations = []
    similarities = []

    if get_street_info:
        for key in FULL_LOCATIONS_STREET_DICT.keys():
            similarities.append(similar(norm(sample), norm(key)))
            full_locations.append(FULL_LOCATIONS_STREET_DICT[key])
    else:
        for key in FULL_LOCATIONS_DICT.keys():
            similarities.append(similar(norm(sample), norm(key)))
            full_locations.append(FULL_LOCATIONS_DICT[key])
    
    return full_locations[argmax(similarities)]


def get_all_substring(string: str = ""):
    return [string[x:y] for x, y in combinations(range(len(string) + 1), r = 2)]    

def get_substring_reverse(string: str = ""):
    reversed_string = string[::-1]
    substrings = []
    sub = ""
    for c in reversed_string:
        sub += c
        substrings.append(sub[::-1])
    return substrings


def refine_ocr_results(results: dict = None):
    assert results is not None
    # Refine place of origin info
    results["Place of origin"] = refine_location_information(results["Place of origin"])
    
    # # Refine place of residence info
    # similarity = -1
    # src = results["Place of residence"]
    # for key in FULL_LOCATIONS_STREET_DICT.keys():
    #     for substring in get_substring_reverse:
    #         if similar(norm(substring), norm(key)) > similarity:
    #             similarity = similar(norm(substring), norm(key))
    #         else:
    #             refined = FULL_LOCATIONS_STREET_DICT[key]
    # results["Place of residence"] = refined
    
    return results

def perspective_transoform(image, source_points):
    dest_points = np.float32([[0,0], [3000,0], [3000,1800], [0,1800]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (3000, 1800))
    # plt.subplots(figsize = (10, 10))
    # plt.imshow(dst[:, :, ::-1])
    # plt.axis('off')
    cv2.imwrite("aligned.jpg", dst)

if __name__ == "__main__":
    pass