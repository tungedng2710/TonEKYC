import cv2
import os
import json
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

class Card():
    def __init__(self,
                 root_dir: str = None,
                 annotation_path: str = None,
                 group_id: int = 0):
        assert root_dir is not None
        assert annotation_path is not None

        self.root_dir = root_dir
        self.annotation_path = annotation_path
        self.group_id = group_id
        self.top_left = list(np.zeros(2))
        self.top_right = list(np.zeros(2))
        self.bottom_right = list(np.zeros(2))
        self.bottom_left = list(np.zeros(2))
        self.bbox = list(np.zeros(4))

        with open(annotation_path) as f:
            annotation = json.load(f)
        shapes = annotation["shapes"]
        corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
        for polygon in shapes:
            if polygon["group_id"] is None:
                polygon["group_id"] = 0
            if int(polygon["group_id"]) == group_id:
                label = polygon["label"]
                if label in corners:
                    setattr(self, label, polygon["points"][0])
                elif label == "card":
                    self.x1y1 = polygon["points"][0]
                    self.x2y2 = polygon["points"][1]
                    self.bbox = self.x1y1+self.x2y2 # bounding box is x1x2y1y2 format
                else:
                    pass
            else:
                pass
        self.center = [(self.top_left[0]+self.bottom_right[0])/2.0,
                       (self.top_left[1]+self.bottom_right[1])/2.0]
        self.keypoints
        self.img_path = os.path.join(self.root_dir, annotation["imagePath"])
        self.image = self.load(self.img_path)
        self.cropped_image = self.image.crop(tuple(self.bbox))

    def load(self, img_path: str = None):
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert('RGB')
        return img

    def transform(self):
        
        pass

    def visualize(self, 
                  keypoints: bool = False,
                  bbox: bool = False):
        if (keypoints is False) and (bbox is False):
            print("Nothing to show!")
            print("Set the argument 'keypoints' or 'bbox' is True to visualize the sample")
        image = cv2.imread(self.img_path)
        green_bgr = (0, 255, 0)
        blue_bgr = (255, 0, 0)
        red_bgr = (0, 0, 255)
        yellow_bgr = (0, 255, 255)
        pink_bgr = (204, 0, 204)
        if bbox:
            start_point = tuple(map(int, self.x1y1))
            end_point = tuple(map(int, self.x2y2))
            thickness = 2
            cv2.rectangle(image, start_point, end_point, green_bgr, thickness)
        if keypoints:
            points = [self.top_left, self.top_right, self.bottom_right, self.bottom_left, self.center]
            colors = [green_bgr, blue_bgr, red_bgr, yellow_bgr, pink_bgr]
            for i in range(5):
                point = tuple(map(int, points[i]))
                radius = 3
                thickness = 10
                color = colors[i]
                image = cv2.circle(image, point, radius, color, thickness)
        plt.imshow(image[:,:,::-1])
        plt.show()


if __name__ == "__main__":
    root_dir = "data/cccd_kpts"
    annotation_path = "data/cccd_kpts/5.json"
    card1 = Card(root_dir=root_dir, 
                annotation_path=annotation_path, 
                group_id=0)
    card1.visualize(bbox=True, keypoints=True)
    # print(card1.top_left, card1.top_right)