#-----------------------------------------------------------------------------#
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
#-----------------------------------------------------------------------------#
import os
import pandas as pd
import urllib

from utils.ocr_utils import get_ocr_results

if __name__ == "__main__":
    print("Done")
    pass
    
