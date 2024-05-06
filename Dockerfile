# Use an official Python runtime as a base image
FROM python:3.12-slim

# Set an environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
    RUN apt-get update && apt-get install -y build-essential \
    curl \
    software-properties-common \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app
ARG git_user
ARG git_token
# Clone a GitHub repository (example repository)
RUN git clone https://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:x-oauth-basic@github.com/NickBak08/AgricultureRobotics.git .

# Install Python libraries using requirements.txt (if applicable)
# You can copy a local requirements file or specify packages directly
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
# WORKDIR /AgricultureRobotics
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the default command to run when the container starts
RUN echo $(ls)

ENTRYPOINT ["streamlit", "run", "HomePage.py", "--server.port=8501", "--server.address=0.0.0.0"]

