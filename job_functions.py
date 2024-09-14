import pandas as pd
import matplotlib as mt
import streamlit as st
import easyocr
import numpy as np
from glob import glob 

st.title("Handy functions")
st.sidebar.title("Functions")
Functions = st.sidebar.selectbox("Choose a function", ["Extract text from image"])

if action == "Extract text from image":
  st.subheader("Extract text from image")
  uploaded_file = st.file_uploader("Upload image.", type=["jpg", "png", "jpeg"])
  image = Image.open(uploaded_file)
  img_np = np.array(image)
  reader = easyocr.Reader(['en', 'it'], gpu=False)
  result = reader.readtext(img_np)
  for _, text, _ in result:
    print(text)
    



