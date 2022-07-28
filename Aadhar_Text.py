import cv2
import easyocr
import copy
import imutils
import numpy as np
import re
import codecs,string
import pandas as pd
import codecs,string
import cv2
import matplotlib.pyplot as plt
import os.path
import csv

def aadhar_test_detection_script(img, template): 
    reader = easyocr.Reader(['en']);
    
    imgAligned = align_images(img, template);
    
    if len(reader.readtext(imgAligned)) != 0 :
        img = imgAligned
           
    blur = blurr(img, None);
    thres_blur = thresholding(blur, 'gray'); 
    noiseless = coloredNoiseless(blur, None);
    
    aadhar_gender = reader.readtext(thres_blur);
    name_dob = reader.readtext(noiseless);
    
    # aadhar_gender = reader.readtext(img);
    # name_dob = reader.readtext(noiseless);
    # name_dob = aadhar_gender;
    
    aadhar_gender_text = '';
    for i in aadhar_gender :
        if i[2]*100>50.00 :
            aadhar_gender_text += i[1] + ' ';
    
    
    name_dob_text = '';
    for i in name_dob :
        if i[2]*100>50.00 :
            name_dob_text += i[1] + ' ';
    
    match = re.search(r'(\d+/\d+/\d+)',name_dob_text)
    if match == None : 
        match = re.search(r'(\d+/\d+/\d+)',aadhar_gender_text)
    
    data = {};
    name_dob_data(name_dob_text, data);
    id_sex_data(aadhar_gender_text, data);
    if match != None : 
        data['Date of Birth'] = match.group(1)
    
    return data

def align_images(image, template, maxFeatures=500, keepPercent=0.2,debug=True):
    
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY);
    orb = cv2.ORB_create(maxFeatures);
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None);
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None);
        # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING;
    matcher = cv2.DescriptorMatcher_create(method);
    matches = matcher.match(descsA, descsB, None);

    matches = sorted(matches, key=lambda x:x.distance);
        # keep only the top matches
    keep = int(len(matches) * keepPercent);
    matches = matches[:keep];
        # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
        matches, None);
        matchedVis = imutils.resize(matchedVis, width=1000)
        # display(matchedVis);
            
    ptsA = np.zeros((len(matches), 2), dtype="float");
    ptsB = np.zeros((len(matches), 2), dtype="float");
        # loop over the top matches
    for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt;
        ptsB[i] = kpsB[m.trainIdx].pt;
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC);
        
    # use the homography matrix to align the images
    (h, w) = template.shape[:2];
    aligned = cv2.warpPerspective(image, H, (w, h));
    # return the aligned image
    # display(aligned, None)
    return aligned



def name_dob_data(text, data):
    res = text.split()
    name = ""
    dob = ""
    titleWord = ["government", "of", "ol", "0f", "0l", "01", "govemment", "governmenl", "inia", "india", "1dia"]
    # print(res)
    for i in res : 
        temp = ""
        flag = True
        for x in titleWord : 
            if x == i.lower() :
                flag = False
                break
                
        if flag  :
            for j in i :
                if j == '/' or j <= '9' and j >= '0':
                    temp = "";
                    break;
                temp = temp + j
            if temp == "" : 
                break;
            name += temp + " ";
            
    if len(name) != 0 :
        name = name[:-1]
    
            
    for i in res : 
        for j in i :
            if j == '/' or j <= '9' and j >= '0':
                dob = i;
                break;
        if dob != "" : 
            break;
   
    data['Name'] = name
    data['Date of Birth'] = dob

def id_sex_data(text, data):
    res = text.split()
    a_id = ""
    sex = ""
    dob = ""
    if 'female' in text.lower():
        sex = "FEMALE"
    else:
        sex = "MALE"
    
    for word in res:
        if len(word) == 4 and word.isdigit():
            a_id = a_id + word + ' '
    a_id = a_id[:-1]
    
    for i in res : 
        for j in i :
            if j == '/' or j <= '9' and j >= '0':
                dob = i;
                break;
        if dob != "" : 
            break;

    data['ID'] = a_id
    data['Sex'] = sex
    if data['Date of Birth'] == "" :
        data['Date of Birth'] = dob
    
def bwNoiseless(img, cmp): 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noiseless_image_bw = cv2.fastNlMeansDenoising(img, None, 20, 7, 21) 
    return noiseless_image_bw

def coloredNoiseless(img, cmp): 
    noiseless = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21) 
    return noiseless

def blurr(img, cmp): 
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    return blurred

def thresholding(img, cmp=None): 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    return thresh



