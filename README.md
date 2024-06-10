# AgricultureRobotics
The repository for the TU/e project agriculture robotics in collaboration with VeXtronics.
Program is tested on windows 11. 

## Contributers
- Hao Zhou
- Hidde Huitema
- Liz Li
- Alicja Obszyńska
- Nick Bakker

## Rules contributers
- Make a new branch for every subsystem (pathplanning, computervision, simulation)
- When implementing a new feature create a new branch in your subbranch (computervision/yolo)
- Commit often not only when you want to push your code
- When merging asks others to review the code

## General information
- You can find the files for the different parts of the pipeline in their corresponding folders.
- The GUI can be found in pages and then the file seeder.py.
- For installation see  <strong>Using docker to run the program</strong>.
- Most important files are:
  - ComputerVision/field_prediction.py
  - PathPlanning/pathplanning.py
  - Simulation/LQR_pathtracking.py
  - Simulation/MPC_pathtracking.py
  - pages/seeder.py
- Dataset can be found under ComputerVision/YOLO/ultralytics-main/datasets/images
# Using docker to run the program:
Docker is the easiest way to run the pipeline. Follow the steps below and the program can be run. 
1) Install docker: https://docs.docker.com/desktop/install/windows-install/
2) Make sure to also install wsl on windows otherwise docker won't work.
3) Start docker desktop to activate the docker deamon.
4) Make sure you have installed the most updated version of google chrome
5) Add personal access token to GitHub account (Settings -> developer settings -> personal access tokens).
6) Make sure repo and write:packages boxes are checked when creating personal access token
7) Clone repository onto your system ``` git clone "link to repository" ```
8) In powershell go to the folder (inside AgricultureRobotics) and type: ``` docker build -t streamlit . ``` (Can take a couple of minutes)
9) When it is build run: ``` docker run -it -p 8501:8501 -e GITHUB_TOKEN="YOUR_TOKEN" streamlit ```
10) In google chrome type localhost:8501

# Run program without docker:
To run the program without docker, make sure an environment is made on your computer with the packages from the requirements.txt. 
1) Clone the repo onto your system.
2) Go inside AgricultureRobotics in your terminal
3) Type ``` streamlit run HomePage.py ```
4) You will get erros these are probably errors that you miss packages. Install all these packages and it should work. 

# Installing CVAT Information
For installing CVAT on windows you need to follow the following steps
[Link to installing instructions](https://opencv.github.io/cvat/docs/administration/basics/installation/)

# Acknowledgements
The authors would like to thank Yannick Heuts and Erik Vermeer for their supervision during the project and their supporting feedback during the weekly meetings. 
