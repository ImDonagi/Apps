# imports
import streamlit as st
import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

# ----------------------------

# Functions:

# Segmenting an image using k-means
def segment_image_kmeans(img, k=3, attempts=10): 

    # Convert MxNx3 image into Kx3 where K=MxN
    pixel_values  = img.reshape((-1,3))  #-1 reshape means, in this case MxN

    #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)
    
    return segmented_image

# Take an image, and return a resized version that fits our page
def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    
    else:
        r = width/float(w)
        dim = (width, int(h*r))
        
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized
# ----------------------------

# Interface:

st.set_page_config(page_title='Project- Guy Donagi', layout = 'wide')


header = st.container()
input = st.container()
output = st.container()

with header:
    st.title("Area Calculating Using AruCo Markers")
    st.text("This small project was built as a part of \"Intro To Image Processing\" course in the Faculty of Agriculture.\nIt's quite simple:\n")
    st.text("*  Upload an image containing an object and an AruCo marker to the \"Input Image\" section.")
    st.text("*  The area of the object will be presented in the \"Calculated Area\" section.")

with input:
    st.header("Input Values:")

    sel_col = st.container()
    disp_col = st.container()
    
    k = sel_col.slider('Please choose number of clusters:', value=3, min_value=2, max_value=10)
    attempts = sel_col.slider('Please choose number of attempts:', value=7, min_value=1, max_value=10)
with output:
    st.header("Calculated Area:")
