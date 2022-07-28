# Aadhar-Text-Detection-and-Extraction

script is written python it extracts info from the Aadhar card and can be extended to PAN Card 
- It uses different Image processing Algorithms like thresholding, Gaussian blurring to preprocess the image for data extraction.
- It uses Homography Shift Algorithm to best Align the input Image accord to a specific template so as to Increase the probabilities of the detected text
- It uses EasyOcr library to detect both English and Hindi text present in the image and finally it extracts the English part of the data and saves it into a csv file if the data didn’t exist already
