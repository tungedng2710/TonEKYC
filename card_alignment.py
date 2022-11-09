
import cv2
from matplotlib import pyplot as plt
import numpy as np
from utils.alignment import *
import torch
from utils.find_nearest_box import NearestBox
import os
import time
import argparse
from utils import detect_face

def getCenterRatios(img, centers):
    """
    Calculates the position of the centers of all boxes 
    in the ID card image and Unet Mask relative to the width and height of the image 
    and returns these ratios as a numpy array.
    """
    if(len(img.shape) == 2):
        img_h, img_w = img.shape
        ratios = np.zeros_like(centers, dtype=np.float32)
        for i, center in enumerate(centers):
            ratios[i] = (center[0]/img_w, center[1]/img_h)
        return ratios
    else :
        img_h, img_w,_ = img.shape
        ratios = np.zeros_like(centers, dtype=np.float32)
        for i, center in enumerate(centers):
            ratios[i] = (center[0]/img_w, center[1]/img_h)
        return ratios


def matchCenters(ratios1, ratios2):
    """
    It takes the ratio of the centers of the regions 
    included in the mask and CRAFT result on the image 
    and maps them according to the absolute distance. 
    Returns the index of the centers with the lowest absolute difference accordingly
    """

    bbb0 = np.zeros_like(ratios2)
    bbb1 = np.zeros_like(ratios2)
    bbb2 = np.zeros_like(ratios2)
    bbb3 = np.zeros_like(ratios2)

    for i , r2 in enumerate(ratios2):
        bbb0[i] = abs(ratios1[0] - r2)
        bbb1[i] = abs(ratios1[1] - r2)
        bbb2[i] = abs(ratios1[2] - r2)
        bbb3[i] = abs(ratios1[3] - r2)

    sum_b0 = np.sum(bbb0, axis = 1)
    sum_b0 = np.reshape(sum_b0, (-1, 1))
    arg_min_b0 = np.argmin(sum_b0, axis=0)

    sum_b1 = np.sum(bbb1, axis = 1)
    sum_b1 = np.reshape(sum_b1, (-1, 1))
    arg_min_b1 = np.argmin(sum_b1, axis=0)

    sum_b2 = np.sum(bbb2, axis = 1)
    sum_b2 = np.reshape(sum_b2, (-1, 1))
    arg_min_b2 = np.argmin(sum_b2, axis=0)

    sum_b3 = np.sum(bbb3, axis = 1)
    sum_b3 = np.reshape(sum_b3, (-1, 1))
    arg_min_b3 = np.argmin(sum_b3, axis=0)

    return np.squeeze(arg_min_b0), np.squeeze(arg_min_b1), np.squeeze(arg_min_b2),np.squeeze(arg_min_b3)         



def getCenterOfMasks(thresh):
    """
    Find centers of 4 boxes in mask from top to bottom with unet model output and return them
    """
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by size from smallest to largest
    contours = sorted(contours, key = cv2.contourArea, reverse=False)
    
    contours = contours[-4:] # get the 4 largest contours

    #print("size of cnt", [cv2.contourArea(cnt) for cnt in contours])
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    
    # Sort the 4 largest regions from top to bottom so that we filter the relevant regions
    (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda b:b[1][1], reverse=False))
    
    detected_centers = []
 
    for contour in cnts:
        (x,y,w,h) = cv2.boundingRect(contour)
        #cv2.rectangle(thresh, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cX = round(int(x) + w/2.0)
        cY = round(int(y) + h/2.0)
        detected_centers.append((cX, cY))
        #cv2.circle(thresh, (cX, cY), 7, (255, 0, 0), -1)

    return np.array(detected_centers)


def getBoxRegions(regions):
    """
    The coordinates of the texts on the id card are converted 
    to x, w, y, h type and the centers and coordinates of these boxes are returned.
    """
    boxes = []
    centers = []
    for box_region in regions:

        x1,y1, x2, y2, x3, y3, x4, y4 = np.int0(box_region.reshape(-1))
        x = min(x1, x3)
        y = min(y1, y2)
        w = abs(min(x1,x3) - max(x2, x4))
        h = abs(min(y1,y2) - max(y3, y4))

        cX = round(int(x) + w/2.0)
        cY = round(int(y) + h/2.0)
        centers.append((cX, cY))
        bbox = (int(x), w, int(y), h)
        boxes.append(bbox)

    #print("number of detected boxes", len(boxes))
    return np.array(boxes), np.array(centers)

def align_image(img_path: str = None):
    nearestBox = NearestBox(distance_thresh = 60, draw_line=False)
    face_detector = detect_face.face_factory(face_model = "ssd")
    findFaceID = face_detector.get_face_detector()
    try:
        img = cv2.imread(img_path)
        img1 = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

        final_img = findFaceID.changeOrientationUntilFaceFound(img1, 60)
        
        if(final_img is None):
            print(f"No face detected in identity card {filename}")

        final_img = correctPerspective(final_img)
        final_img = cv2.resize(final_img, (1712, 1078))
        cv2.imwrite("temp.jpg", final_img[:,:,::-1])
        return True
    except:
        print("Something wrong!")
        return False

if '__main__' == __name__:
    
    parser = argparse.ArgumentParser(description = 'Identity Card Information Extractiion')
    parser.add_argument('--folder_name', default="images", type=str, help='folder that contain tc id images')
    parser.add_argument('--neighbor_box_distance', default = 50, type = float, help='Nearest box distance threshold')
    parser.add_argument('--face_recognition',  default = "ssd", type = str,   help='face detection algorithm')
    parser.add_argument('--ocr_method',  default = "EasyOcr", type = str,   help='Type of ocr method for converting images to text')
    parser.add_argument('--rotation_interval', default = 30,   type = int, help='Face search interval for rotation matrix')
    args = parser.parse_args()
    
    Folder = args.folder_name # identity card images folder
    
    use_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    
    # model = UnetModel(Res34BackBone(), use_cuda)
    nearestBox = NearestBox(distance_thresh = args.neighbor_box_distance, draw_line=False)
    face_detector = detect_face.face_factory(face_model = args.face_recognition)
    findFaceID = face_detector.get_face_detector()
    for filename in sorted(os.listdir(Folder)):
        try:
            img = cv2.imread(os.path.join(Folder,filename))
            img1 = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    
            final_img = findFaceID.changeOrientationUntilFaceFound(img1, args.rotation_interval)
            
            # cv2.imwrite("final1.jpg", final_img[:,:,::-1])
            if(final_img is None):
                print(f"No face detected in identity card {filename}")
                break

            final_img = correctPerspective(final_img)
            final_img = cv2.resize(final_img, (1712, 1078))
            cv2.imwrite("data/aligned/"+filename, final_img[:,:,::-1])
        except:
            pass
   
        

