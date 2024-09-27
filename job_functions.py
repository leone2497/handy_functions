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
import os
import subprocess
from moviepy.editor import VideoFileClip
import imageio_ffmpeg as ffmpeg

# Function to check if FFmpeg is installed
def check_ffmpeg_installed():
    return ffmpeg.get_ffmpeg_exe() is not None

# Function to extract audio from video using moviepy
def extract_audio_from_video(video_file):
    try:
        video = VideoFileClip(video_file)
        audio_file = "temp_audio.wav"
        audio = video.audio
        audio.write_audiofile(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Error processing the video file with moviepy: {e}")
        return None

# Function to extract audio using ffmpeg CLI as a fallback
def extract_audio_ffmpeg(video_file):
    audio_file = "temp_audio.wav"
    command = ["ffmpeg", "-i", video_file, "-q:a", "0", "-map", "a", audio_file]
    try:
        subprocess.run(command, check=True)
        return audio_file
    except subprocess.CalledProcessError as e:
        st.error(f"Error processing the video file with ffmpeg: {e}")
        return None

# Set up the Streamlit app
st.title("Handy Functions")
st.sidebar.title("Functions")
function_choice = st.sidebar.selectbox("Choose a function", 
    ["Extract text from image", "Join files", "Analysis", "Transcribe Audio"])

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
        df = pd.read_csv(file_to_analyze) if file_to_analyze.name.endswith('.csv') else pd.read_excel(file_to_analyze)

        st.write("Columns in the uploaded file:")
        st.write(df.columns.tolist())
        st.write(df)
        
        # Apply filters
        Filter_2 = st.sidebar.selectbox("Choose filter that converts non-numeric values into a unique count", df.columns.tolist())
        values_unique_filter = df[Filter_2].value_counts()
        
        min_value, max_value = st.slider(
            'Select a range of values based on non-numeric columns',
            min_value=values_unique_filter.min(),
            max_value=values_unique_filter.max(),
            value=(values_unique_filter.min(), values_unique_filter.max()))
        
        filtered_df = df[(df[Filter_2].map(values_unique_filter) >= min_value) & 
                          (df[Filter_2].map(values_unique_filter) <= max_value)]
        
        st.write(filtered_df)

        Filter_1 = st.sidebar.selectbox("Choose filter", df.columns.tolist())
        if pd.api.types.is_numeric_dtype(filtered_df[Filter_1]):
            min_value, max_value = st.slider('Select a range of values', 
                                              min_value=filtered_df[Filter_1].min(), 
                                              max_value=filtered_df[Filter_1].max(), 
                                              value=(filtered_df[Filter_1].min(), filtered_df[Filter_1].max()))
            filtered_df = filtered_df[(filtered_df[Filter_1] >= min_value) & (filtered_df[Filter_1] <= max_value)]
            st.write(filtered_df)
        else:
            st.warning("The selected column is not numeric. Please choose a numeric column.")
        
        # Select columns to see unique values
        columns_for_unique = st.multiselect("Select columns to see unique values:", filtered_df.columns.tolist())
        
        if columns_for_unique:
            for column in columns_for_unique:
                unique_values = filtered_df[column].unique()
                st.write(f"Unique values in '{column}': {unique_values}")
        else:
            st.info("Please select at least one column to view unique values.")
        
        # Data visualization analysis
        if analysis_choice == "Data visualization analysis":
            st.subheader("Data visualization analysis")
            graph = st.sidebar.selectbox("Choose type of visualization", ["Histogram", "Lines"])
            if graph == "Histogram":
                first_variable = st.selectbox("Select x variable:", filtered_df.columns.tolist())
                plt.figure(figsize=(10, 6))
                sns.histplot(filtered_df[first_variable], kde=True, log_scale=True)
                st.pyplot(plt)
                plt.close()
                
        # Statistical analysis
        elif analysis_choice == "Statistical analysis":
            Operation = st.sidebar.selectbox("Select analysis", ["Summary", "Count unique value"])
            if Operation == "Summary":
                first_variable = st.selectbox("Select variable for statistical summary:", filtered_df.columns.tolist())
                st.write("Statistical summary of the dataset:")
                st.write(filtered_df[first_variable].describe())
            elif Operation == "Count unique value":
                first_variable_v1 = st.selectbox("Select variable unique values counting:", filtered_df.columns.tolist())
                st.write("Statistical summary of the dataset:")
                values = filtered_df[first_variable_v1].value_counts()
                st.write(values)

# New function: Transcribe Audio using the ASR model
elif function_choice == "Transcribe Audio":
    st.subheader("Transcribe Audio")
    
    # Check if FFmpeg is installed
    if not check_ffmpeg_installed():
        st.error("FFmpeg is not installed. Please install FFmpeg to use this feature.")
    else:
        uploaded_audio = st.file_uploader("Upload audio or video file", type=["wav", "mp3", "flac", "mp4"])
        
        if uploaded_audio is not None:
            # Save the uploaded file temporarily
            temp_file_path = f"temp_video{os.path.splitext(uploaded_audio.name)[1]}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            
            # Check if it's an mp4 file and extract audio if needed
            audio_file = None
            if uploaded_audio.name.endswith(".mp4"):
                st.info("Extracting audio from MP4 file...")
                audio_file = extract_audio_from_video(temp_file_path) or extract_audio_ffmpeg(temp_file_path)
                if audio_file is None:
                    st.error("Audio extraction failed; unable to transcribe.")
                    os.remove(temp_file_path)  # Remove temp video file
                    st.stop()  # Stop execution to avoid further errors
            else:
                audio_file = temp_file_path
            
            # Use Whisper model for transcription
            model_name = "openai/whisper-large-v3"
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            st.info("Transcribing the audio, please wait...")
            try:
                transcription = pipe(audio_file)["text"]
                st.subheader("Transcription:")
                st.write(transcription)
            except Exception as e:
                st.error(f"Error during transcription: {e}")
            finally:
                os.remove(temp_file_path)  # Remove temp video file
                if audio_file and audio_file == "temp_audio.wav":
                    os.remove(audio_file)  # Remove temp audio file if created
        else:
            st.info("Please upload an audio or video file.")
