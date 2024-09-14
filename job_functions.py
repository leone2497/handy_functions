import pandas as pd
import streamlit as st
import easyocr
import numpy as np
from glob import glob 
from PIL import Image

st.title("Handy Functions")
st.sidebar.title("Functions")
function_choice = st.sidebar.selectbox("Choose a function", ["Extract text from image","Join files"])

if function_choice == "Extract text from image":
    st.subheader("Extract text from image")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            img_np = np.array(image)
            
            reader = easyocr.Reader(['en', 'it'], gpu=False)
            result = reader.readtext(img_np)
            
            st.subheader("Extracted Text:")
            for _, text, _ in result:
                st.text(text)
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    else:
        st.info("Please upload an image file.")
else function_choice == "Join files":
    st.subheader("Join files")
    uploaded_file = st.file_uploader("Upload files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            img_np = np.array(image)
            
            reader = easyocr.Reader(['en', 'it'], gpu=False)
            result = reader.readtext(img_np)
            
            st.subheader("Extracted Text:")
            for _, text, _ in result:
                st.text(text)
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    else:
        st.info("Please upload an file.")

    



