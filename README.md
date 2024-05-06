# AgricultureRobotics
The repository for the TU/e project agriculture robotics in collaboration with VeXtronics.

## Rules
- Make a new branch for every subsystem (pathplanning, computervision, simulation)
- When implementing a new feature create a new branch in your subbranch (computervision/yolo)
- Commit often not only when you want to push your code
- When merging asks others to review the code

# Installing CVAT Information
For installing CVAT on windows you need to follow the following steps
[Link to installing instructions](https://opencv.github.io/cvat/docs/administration/basics/installation/)

# Using docker to run the program:
1) Install docker: https://docs.docker.com/desktop/install/windows-install/
2) Make sure you have installed the most updated version of google chrome
3) Add personal access token to GitHub account (Settings -> developer settings -> personal access tokens).
4) Make sure repo and write:packages boxes are checked when creating personal access token
5) Clone repository onto your system
6) In powershell go to the folder you just cloned and type: ``` docker build -t streamlit . ``` (Can take a couple of minutes)
7) When it is build run: ``` docker run -it -p 8501:8501 -e GITHUB_TOKEN="YOUR_TOKEN" streamlit ```
8) In google chrome type localhost:8501
