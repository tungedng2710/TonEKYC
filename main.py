from utils import *
from feat_ext_templatebase import *
import argparse
import json
import cv2
import pandas as pd
from card_alignment import align_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/examples/test.jpg",
                        help="path to the test image")
    parser.add_argument("--savejson", action='store_true', help="save OCR result to a JSON file")
    parser.add_argument("--savecsv", action='store_true', help="save OCR result to a CSV file")
    args = parser.parse_args()

    flag = align_image(args.image)
    
    if flag:
        detections = get_ocr_results("temp.jpg")
        image = cv2.imread("temp.jpg")

        results = match_keys_values(detections, image)
        os.remove("temp.jpg")
        
        if args.savejson:
            with open("data/ocr_results.json", 'w') as fp:
                json.dump(results, fp)
        
        if args.savecsv:
            to_df = {}
            to_df["keys"] = results.keys()
            to_df["values"] = [results[idx] for idx in results.keys()]
            results_df = pd.DataFrame(to_df)
            results_df.to_csv("data/ocr_results.csv", index=False)
            
        for key in results.keys():
            print("{}: {}".format(key, results[key]))
    else:
        print("Card alignment has been failed!")
