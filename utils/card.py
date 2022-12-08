import cv2
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from PIL import Image

class Card():
    """
    Annotation must be created by labelme: https://github.com/wkentaro/labelme
    Structure of data folders:
    root_dir_name
    --abcxyz.jpg (or any image extension such as png, jpeg)
    --abcxyz.json  

    Required arguments:
    - root_dir (str): path to data folder
    - annotation_path (str): path to annotation file (.json)
    """
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
        corners = ["top_left", "top_right", "bottom_right", "bottom_left"]
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
        self.skeleton = [[0,1], [1,2], [2,3], [3,0], [0,4], [1,4], [2,4], [3,4]]
        self.keypoints = [tuple(self.top_left), 
                          tuple(self.top_right), 
                          tuple(self.bottom_right), 
                          tuple(self.bottom_left), 
                          tuple(self.center)]

        extension = annotation["imagePath"].split('.')[1]
        if extension != "jpg":
            annotation["imagePath"] = annotation["imagePath"].replace(extension, "jpg")
        self.image_path = os.path.join(self.root_dir, annotation["imagePath"])
        self.image_name = annotation["imagePath"]
        self.image = self.load(self.image_path)
        self.image_size = self.image.size
        self.cropped_image = self.image.crop(tuple(self.bbox))

    def load(self, image_path: str = None):
        """
        Load image with PIL, auto convert to RGB mode
        """
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert('RGB')
        return img

    def convert_to_opencv(self):
        open_cv_image = np.array(self.image)
        image = open_cv_image[:, :, ::-1].copy() # RGB to BGR 
        return image

    def visualize(self, 
                  bbox: bool = True,
                  keypoints: bool = True,
                  skeleton: bool = False):
        """
        Draw annotation and show the sample image

        Arguments:
        - keypoints (bool): draw and show keypoints
        - box (bool): draw and show bounding box
        """
        image = self.convert_to_opencv()
        green_bgr = (0, 255, 0)
        blue_bgr = (255, 0, 0)
        red_bgr = (0, 0, 255)
        yellow_bgr = (0, 255, 255)
        pink_bgr = (204, 0, 204)
        points = self.keypoints
        if bbox:
            start_point = tuple(map(int, self.x1y1))
            end_point = tuple(map(int, self.x2y2))
            thickness = 2
            cv2.rectangle(image, start_point, end_point, green_bgr, thickness)
        if keypoints:
            colors = [green_bgr, blue_bgr, red_bgr, yellow_bgr, pink_bgr]
            for i in range(5):
                point = tuple(map(int, points[i]))
                radius = 7
                thickness = 20
                color = colors[i]
                image = cv2.circle(image, point, radius, color, thickness)
        if skeleton:
            for joint in self.skeleton:
                start_point = tuple(map(int, points[joint[0]]))
                end_point = tuple(map(int, points[joint[1]]))
                navy_bgr = (128, 0, 0)
                thickness = 3
                cv2.line(image, start_point, end_point, navy_bgr, thickness) 

        plt.imshow(image[:,:,::-1])
        plt.show()
    
    def augment(self, 
                background_dir: str = "./data/background",
                max_card_width: float = 1000.0,
                max_image_width: float = 2000.0,
                angles: list = None,
                save_image: bool = False):
        if not os.path.exists(background_dir):
            return self.image
        if save_image:
            save_dir = "data/augmented_images"
            if not os.path.exists((save_dir)):
                os.makedirs(save_dir)
        if angles is None:
            angles = [i*10 for i in range(-3,4)]
        self.augmented_images = []

        cropped_image = self.cropped_image
        cropped_image_size = cropped_image.size
        scale = max_card_width / cropped_image_size[0]
        cropped_image = cropped_image.resize((int(max_card_width), int(cropped_image_size[1]*scale)))

        for file_name in os.listdir(background_dir):
            new_start_point = (400, 200)
            for angle in angles:
                background = self.load(os.path.join(background_dir, file_name))
                background_size = background.size
                scale = max_image_width / background_size[0]
                background = background.resize((int(max_image_width), int(background_size[1]*scale)))

                # Rotate the card
                mask = Image.new('L', cropped_image.size, 255)
                front = cropped_image.rotate(angle, expand=True)
                # Paste the rotated card on background
                mask = mask.rotate(angle, expand=True)
                background.paste(front, new_start_point, mask)
                self.augmented_images.append(background)
                if save_image:
                    saved_path = os.path.join(save_dir, file_name.split('.')[0]+'_'+str(angle)+'_'+self.image_name)
                    background.save(saved_path)


def rotate_points(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

if __name__ == "__main__":
    root_dir = "data/cards"
    annotation_path = "data/cards/16.json"
    card = Card(root_dir=root_dir, 
                annotation_path=annotation_path, 
                group_id=0)
    # print(card.visualize(skeleton=True))
    # card.augment(angles=[10], save_image=True)
    # card.visualize(skeleton=True)
    print(card.bbox)