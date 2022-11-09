# TonEKYC: a naive Vietnamese identity card reader 

## Instant usage
**Prerequisite**
* python 3.9 or higher
* Ubuntu 18 or higher
To extract information from an image of identity card, just run the script below
```bat
python3 main.py --image [path/to/image] 
```
if you want to dump the results into ```json``` or ```csv``` file, just add argument ```--savejson``` and ```--savecsv``` into the script, respectively.

## Card alignment
Key-Information extractor requires an aligned card. Some traditional digital image processing methods are applied to perspective transform raw images. Alignment is integrated into the given pipeline.

Disclaiming: The card alignment process is still quite silly and naive because I'm researching deep models to perform it. Therefore, I have used Dlib and Haar face detection model to do it instead, it is better to choose a rotation angle of less than 30 degrees and the card should be put on a dark background.

## Text detection and OCR
The OCR comes from the [EasyOCR](https://github.com/JaidedAI/EasyOCR), which is a vigorous OCR library supporting variety languages such as Vietnamese.  

Due to time limitation, I have just used rulebase method to extract the information. Naturally, it can't cover all situation so I'm researching some methods based on Graph Neural Network for the KIE problems.
