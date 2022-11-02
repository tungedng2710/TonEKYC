from utils import *
from ocr import *
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/examples/test.jpg",
                        help="path to the test image")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    # resized = cv2.resize(image, (856, 540), interpolation = cv2.INTER_AREA)
    # image_path = "resized.jpg"
    # cv2.imwrite(image_path, resized)

    detections = get_ocr_results(args.image)
    image = cv2.imread(args.image)

    results = match_keys_values(detections, image)
    with open("data/ocr_results.json", 'w') as fp:
        json.dump(results, fp)
    print(results)

