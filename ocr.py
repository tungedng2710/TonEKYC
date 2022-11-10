import os
from utils import *
import numpy as np
from utils.ocr_utils import *

def match_keys_values(detections: dict = None,
                      image: np.ndarray = None):
    '''
    image loaded by opencv (cv2); shape (height, width, depth) ((y, x, d))
    keypoints' coordinate format is (x, y)
    '''
    assert detections is not None
    assert image is not None

    results = {}

    for text in detections.keys():
        keypoints = detections[text]["keypoints"] #keypoints = [top_left, top_right, bottom_right, bottom_left, center]
        if check_id_number(text):
            results["No."] = text
            landmark1 = keypoints[-2]

    left = image[:, :landmark1[0]]
    right = image[:, landmark1[0]:]
    temp = detections.copy()

    for text in detections.keys():
        center = detections[text]["keypoints"][-1]
        if center[1] < landmark1[1]: # Remove 
            temp.pop(text)
    
    for text in temp.keys():
        center = detections[text]["keypoints"][-1]
        if center[0] < landmark1[0]: # examine left part
            for txt in text.split():
                if check_date_format(txt):
                    txt = re.sub('[^A-Za-z0-9]+', '', txt)
                    results["Date of expiry"] = datetime.strptime(txt, '%d%m%Y').strftime('%d/%m/%Y')
                    landmark2 = detections[text]["keypoints"][2] # bottom_right point of DoB
        else: # examine right part
            for txt in text.split():
                if check_date_format(txt):
                    txt = re.sub('[^A-Za-z0-9]+', '', txt)
                    results["Date of birth"] = datetime.strptime(txt, '%d%m%Y').strftime('%d/%m/%Y')
                    landmark3 = detections[text]["keypoints"][-2] # bottom_left point of DoB
        if check_name(text):
            results["Full name"] = text
    
    smaller = image[landmark3[1]:, landmark2[0]:]
    cv2.imwrite("smaller.jpg", smaller)
    detections_smaller = get_ocr_results("smaller.jpg")
    for text in detections_smaller.keys():
        norm_text = re.sub('[^A-Za-z0-9]+', '', text).lower()
        if "origi" in norm_text:
            landmark4 = detections_smaller[text]["keypoints"][0] # top_left
        if ("resi" in norm_text) or ("dence" in norm_text):
            landmark5 = detections_smaller[text]["keypoints"][0] # top_left

    smaller_up = smaller[:landmark4[1], :]
    cv2.imwrite("smaller.jpg", smaller_up)
    detections_smaller = get_ocr_results("smaller.jpg")
    list_of_kpts = []
    for text in detections_smaller.keys():
        list_of_kpts.append(detections_smaller[text]["keypoints"][-1])

    for text in detections_smaller.keys():
        list_of_kpts1 = list_of_kpts.copy()
        list_of_kpts2 = list_of_kpts.copy()
        if "sex" in re.sub('[^A-Za-z0-9]+', '', text).lower():
            p1 = detections_smaller[text]["keypoints"][-1]
            nearest = get_nearest_right(p1, list_of_kpts1)
            results["Sex"] = [i for i in detections_smaller if detections_smaller[i]["keypoints"][-1]==nearest][0].split(' ', 1)[0]
        if "nation" in re.sub('[^A-Za-z0-9]+', '', text).lower():
            p1 = detections_smaller[text]["keypoints"][-1]
            nearest = get_nearest_right(p1, list_of_kpts2)
            results["Nationality"] = [i for i in detections_smaller if detections_smaller[i]["keypoints"][-1]==nearest][0]

    smaller_up = smaller[:landmark5[1], :]
    cv2.imwrite("smaller.jpg", smaller_up)
    detections_smaller = get_ocr_results("smaller.jpg")
    line = ""
    for text in detections_smaller.keys():
        line += ' '+text
    results["Place of origin"] = line.split("origin")[1].replace(':','').lstrip()

    smaller_down = smaller[landmark5[1]:, :]
    cv2.imwrite("smaller.jpg", smaller_down)
    detections_smaller = get_ocr_results("smaller.jpg")
    line = ""
    for text in detections_smaller.keys():
        line += ' '+text

    similarities = []
    for word in line.split():
        similarities.append(similar(word, "residence"))
    word = line.split()[argmax(similarities)]
    line = line.replace(word, "residence")

    results["Place of residence"] = line.split("residence")[1]
    os.remove("smaller.jpg")

    return refine_ocr_results(results)

