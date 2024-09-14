import pandas as pd
import matplotlib as mt
import streamlit as st
import easyocr


reader = easyocr.Reader(['en', 'it'], gpu=False)
