import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import easyocr
import numpy as np
from glob import glob 
from PIL import Image
import torch
from transformers import pipeline
import json
from moviepy.editor import VideoFileClip  # Import moviepy for extracting audio from MP4

st.title("Handy Functions")
st.sidebar.title("Functions")
function_choice = st.sidebar.selectbox("Choose a function", ["Extract text from image", "Join files", "Analysis", "Transcribe Audio"])

# Extract text from image function
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

# Join files function
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

# Analysis function
elif function_choice == "Analysis":
    st.subheader("Types of analysis")
    analysis_choice = st.sidebar.selectbox("Choose analysis", ["Data visualization analysis", "Statistical analysis"])
    file_to_analyze = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

    if file_to_analyze is not None:
        if file_to_analyze.name.endswith('.csv'):
            df = pd.read_csv(file_to_analyze)
        else:
            df = pd.read_excel(file_to_analyze)

        st.write("Columns in the uploaded file:")
        st.write(df.columns.tolist())
        st.write(df)
        
        # Apply filters
        Filter_2 = st.sidebar.selectbox("Choose filter that converts non-numeric values into a unique count", df.columns.tolist())
        values_unique_filter = df[Filter_2].value_counts()
        min_value, max_value = st.slider(
            'Select a range of values based on non-numeric columns',
            min_value=float(values_unique_filter.min()),
            max_value=float(values_unique_filter.max()),
            value=(float(values_unique_filter.min()), float(values_unique_filter.max())))
        filtered_df = df[df[Filter_2].map(values_unique_filter) >= min_value] 
        filtered_df = filtered_df[filtered_df[Filter_2].map(values_unique_filter) <= max_value]
        df = filtered_df
        st.write(df)

        Filter_1 = st.sidebar.selectbox("Choose filter", df.columns.tolist())
        if pd.api.types.is_numeric_dtype(df[Filter_1]):
            min_value, max_value = st.slider('Select a range of values', min_value=float(df[Filter_1].min()), max_value=float(df[Filter_1].max()), value=(float(df[Filter_1].min()), float(df[Filter_1].max())))
            df = df[(df[Filter_1] >= min_value) & (df[Filter_1] <= max_value)]
            st.write(df)
        else:
            st.write("The selected column is not numeric. Please choose a numeric column.")
        
        # Select columns to see unique values
        columns_for_unique = st.multiselect("Select columns to see unique values:", df.columns.tolist())
        
        if columns_for_unique:
            for column in columns_for_unique:
                unique_values = df[column].unique()
                st.write(f"Unique values in '{column}':")
                st.write(unique_values)
        else:
            st.info("Please select at least one column to view unique values.")
        
        # Data visualization analysis
        if analysis_choice == "Data visualization analysis":
            st.subheader("Data visualization analysis")
            graph = st.sidebar.selectbox("Choose type of visualization", ["Histogram", "Lines"])
            if graph == "Histogram":
                first_variable = st.selectbox("Select x variable:", df.columns.tolist())
                plt.figure(figsize=(10, 6))
                sns.histplot(df[first_variable], kde=True, log_scale=True)
                st.pyplot(plt)
                plt.close()
                
        # Statistical analysis
        elif analysis_choice == "Statistical analysis":
            Operation = st.sidebar.selectbox("Select analysis", ["Summary", "Count unique value"])
            if Operation == "Summary":
                first_variable = st.selectbox("Select variable for statistical summary:", df.columns.tolist())
                st.write("Statistical summary of the dataset:")
                st.write(df[first_variable].describe())
            elif Operation == "Count unique value":
                first_variable_v1 = st.selectbox("Select variable unique values counting:", df.columns.tolist())
                st.write("Statistical summary of the dataset:")
                values = df[first_variable_v1].value_counts()
                st.write(values)

# New function: Transcribe Audio using the ASR model
elif function_choice == "Transcribe Audio":
    st.subheader("Transcribe Audio")
    uploaded_audio = st.file_uploader("Upload audio or video file", type=["wav", "mp3", "flac", "mp4"])
    
    if uploaded_audio is not None:
        # Check if it's an mp4 file and extract audio if needed
        if uploaded_audio.name.endswith(".mp4"):
            st.info("Extracting audio from MP4 file...")
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_audio.getbuffer())
            
            video = VideoFileClip("temp_video.mp4")
            audio = video.audio
            audio.write_audiofile("temp_audio.wav")
            audio_file = "temp_audio.wav"
        else:
            audio_file = uploaded_audio
        
        # Use Whisper model for transcription
        model_name = "openai/whisper-large-v3"
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        st.info("Transcribing the audio, please wait...")
        transcription = pipe(audio_file)["text"]
        st.subheader("Transcription:")
        st.write(transcription)
    else:
        st.info("Please upload an audio or video file.")
