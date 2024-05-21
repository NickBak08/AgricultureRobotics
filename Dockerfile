# Use an official Python runtime as a base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential \
curl \
software-properties-common \
git && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements to the work folder
COPY requirements.txt /app/

# Install pip installations
RUN pip install --no-cache-dir -r requirements.txt
# 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy shell script to folder and run it
# To clone git everytime docker file is run
COPY clone_and_run.sh /app/
RUN chmod +x /app/clone_and_run.sh

# Add env variable github token for cloning in the bash file
ARG GITHUB_TOKEN

# Expose the port you want to use
EXPOSE 8501

# Checking if port is available
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Can have 1 entrypoint in docker file this command will be run when docker run is used
ENTRYPOINT ["/app/clone_and_run.sh"]

