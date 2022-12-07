import os
import pandas as pd

from utils.ocr_utils import get_ocr_results

if __name__ == "__main__":
    result = get_ocr_results("data/cards/1.jpg")
    print(result)
