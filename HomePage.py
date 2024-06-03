import streamlit as st
import pandas as pd
import numpy as np
import os

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the current file's directory
os.chdir(current_file_directory)

# For displaying vextronics in the middle of the webpage. This is custom HTML that centers the text. 
st.markdown("<h1 style='text-align: center; color: Black;'>Vextronics</h1>", unsafe_allow_html=True)

# Create 3 columns and place the image in the middle one
col1, col2, col3 = st.columns(3)
col2.image("Images/vextronics.png")
