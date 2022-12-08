# import libs
import streamlit as st
import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

# vars
DEMO_IMAGE = 'bananas.jpeg' # a demo image for the segmentation page, if none is uploaded
favicon = 'favicon.png'

# main page
st.set_page_config(page_title="Guy's app", page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
st.title("Guy's App")

# side bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
        width: 350px
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
        width: 350px
        margin-left: -350px
    }    
    </style>
    
    """,
    unsafe_allow_html=True,


)

st.sidebar.title('Segmentation Sidebar')
st.sidebar.subheader('Site Pages')

st.markdown(
    """
    <h1>hello</h1>
    """
)

