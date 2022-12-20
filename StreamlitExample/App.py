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
    
    return segmented_image, centers

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

# Environment variables:

DEMO_IMAGE = 'StreamlitExample/bananas.jpeg'

# ----------------------------

# Interface:

st.set_page_config(page_title='Project- Guy Donagi', layout = 'wide')


header = st.container()
input = st.container()
segmented = st.container()

with header:
    st.title("Area Calculating Using AruCo Markers")
    st.text("This small project was built as a part of \"Intro To Image Processing\" course in the Faculty of Agriculture.\nIt's quite simple:\n")
    st.text("*  Upload an image containing an object and an AruCo marker to the \"Input Image\" section.")
    st.text("*  The area of the object will be presented in the \"Output\" section.")

with input:
    sel_col, disp_col = st.columns(2)
    
    sel_col.subheader("Input Values:")
    img_file = sel_col.file_uploader("Please upload an image", type=['jpg', 'jpeg', 'png'])
    k = sel_col.slider('Please choose number of clusters:', value=4, min_value=2, max_value=10)
    attempts = sel_col.slider('Please choose number of attempts:', value=7, min_value=1, max_value=10)
    
    if img_file is not None:
        image = io.imread(img_file)
    else:
        image = io.imread(DEMO_IMAGE)
    
    disp_col.subheader("Your Image:")
    
    disp_col.image(image)
    
with segmented:
    st.header("We're Getting There...")
    
    img_col, area_col = st.columns(2)
    
    segmented_image, centers = segment_image_kmeans(image, k=k, attempts=attempts)
    
    img_col.subheader('Segmented Image:')
    img_col.image(segmented_image, use_column_width=True)
    
    area_col.subheader('Select The Desired Object:')
    area_col.text(centers)
    area_col.text("Now that you have successfully segmented your image into the different objects it holds,\nselect the one whose area you want to calculate.\nEvery option in the select box is actually an RGB color that represents a segment in the image.\nIn order to understand which color is which, [use this](https://www.rapidtables.com/web/color/RGB_Color.html)")
    obj = area_col.selectbox("", options=centers)
