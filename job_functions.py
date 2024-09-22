import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib as mt
import easyocr
import numpy as np
from glob import glob 
from PIL import Image

st.title("Handy Functions")
st.sidebar.title("Functions")
function_choice = st.sidebar.selectbox("Choose a function", ["Extract text from image", "Join files", "Analisys"])

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
elif function_choice == "Join files":
    st.subheader("Join files")
    uploaded_files = st.file_uploader("Upload files", type=["xls", "xlsx"], accept_multiple_files=True)
    
    if uploaded_files:
        combined_df = pd.DataFrame() 
        
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_excel(uploaded_file)  
                combined_df = pd.concat([combined_df, df], ignore_index=True)  
            except Exception as e:
                st.error(f"Error reading the Excel file {uploaded_file.name}: {e}")
        
        if not combined_df.empty:
            st.subheader("Combined Excel Content:")
            st.write(combined_df)  # Display the combined DataFrame
        else:
            st.info("Uploaded files are empty or could not be read.")
    else:
        st.info("Please upload one or more Excel files.")
elif function_choice == "Analisys":
    st.subheader("Types of analisys")
    analisys_choice = st.sidebar.selectbox("Choose analisys", ["Data visualization analisys", "Statistical analisys"])
    file_to_analise = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])
    df = pd.read_excel(uploaded_file)
    st.write("Columns in the uploaded file:")
    st.write(df.columns.tolist())
    if analisys_choice == "Data visualization analisys":
        st.subheader("Data visualization analisys")
        graph = st.sidebar.selectbox("Choose type of visualization", ["Hystogram", "Lines"])
        if graph== "Hystogram":
            first_variable = st.selectbox("Select x variable:", df.columns.tolist())
            sns.histplot(df, first_variable='Price', kde=True, log_scale=True)
            st.pyplot(plt)
            plt.close()
            
        
        
        
