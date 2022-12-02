import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import pandas as pd

from utils.ocr_utils import get_ocr_results

if __name__ == "__main__":
    print("Done")
    pass
    
