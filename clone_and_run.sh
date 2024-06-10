#!/bin/bash
# Define repo for cleanup
GITHUB_REPO="AgricultureRobotics"
# Clean up any existing repository (optional)
rm -rf $GITHUB_REPO
# FOR DEBUGGING:
#echo "GitHub token: ${GITHUB_TOKEN}"
# echo "Github clone: https://$GITHUB_TOKEN:x-oauth-basic@github.com/NickBak08/AgricultureRobotics.git"
# Clone the repository
# Clone git repository
git clone https://$GITHUB_TOKEN:x-oauth-basic@github.com/NickBak08/AgricultureRobotics.git
# Check if the directory exists
# Go inside program folder
#mkdir -p AgricultureRobotics
echo $(pwd)
echo $(ls)
echo $(ls AgricultureRobotics)

# Run program and redirect to the good port
streamlit run /app/AgricultureRobotics/HomePage.py --server.port=8501 --server.address=0.0.0.0