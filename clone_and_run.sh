#!/bin/bash

# Define repo for cleanup
GITHUB_REPO="AgricultureRobotics"

# Clean up any existing repository (optional)
rm -rf $GITHUB_REPO

# FOR DEBUGGING:
# echo "GitHub token: ${GITHUB_TOKEN}"
# echo "Github clone: https://$GITHUB_TOKEN:x-oauth-basic@github.com/NickBak08/AgricultureRobotics.git"
# Clone the repository

# Clone git repository
git clone https://$GITHUB_TOKEN:x-oauth-basic@github.com/NickBak08/AgricultureRobotics.git

# Go inside program folder
cd AgricultureRobotics

# Run program and redirect to the good port
streamlit run HomePage.py --server.port=8501 --server.address=0.0.0.0