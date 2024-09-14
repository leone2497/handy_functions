import pandas as pd
import matplotlib as mt
import streamlit as st
import easyocr
import numpy as np
from glob import glob 

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
image = Image.open(uploaded_file)
img_np = np.array(image)
reader = easyocr.Reader(['en', 'it'], gpu=False)
result = reader.readtext(img_np)
