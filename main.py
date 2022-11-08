from utils import *
from ocr import *
import argparse
import json
import cv2
from card_alignment import align_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/examples/test.jpg",
                        help="path to the test image")
    args = parser.parse_args()

    # image = cv2.imread(args.image)
    align_image(args.image)
    # resized = cv2.resize(image, (856, 540), interpolation = cv2.INTER_AREA)
    # image_path = "resized.jpg"
    # cv2.imwrite(image_path, resized)

    detections = get_ocr_results("temp.jpg")
    image = cv2.imread("temp.jpg")

    results = match_keys_values(detections, image)
    with open("data/ocr_results.json", 'w') as fp:
        json.dump(results, fp)
    os.remove("temp.jpg")
    print(results)

