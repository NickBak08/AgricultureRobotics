# AgricultureRobotics
The repository for the TU/e project agriculture robotics in collaboration with VeXtronics.
Program is tested on windows 11. 

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
2) Make sure to also install wsl on windows otherwise docker won't work.
3) Make sure you have installed the most updated version of google chrome
4) Add personal access token to GitHub account (Settings -> developer settings -> personal access tokens).
5) Make sure repo and write:packages boxes are checked when creating personal access token
6) Clone repository onto your system ``` git clone "link to repository" ```
7) In powershell go to the folder (inside AgricultureRobotics) and type: ``` docker build -t streamlit . ``` (Can take a couple of minutes)
8) When it is build run: ``` docker run -it -p 8501:8501 -e GITHUB_TOKEN="YOUR_TOKEN" streamlit ```
9) In google chrome type localhost:8501

# Run program without docker:
For faster testing, don't need to build docker everytime you change something
1) Clone the repo onto your system.
2) Go inside AgricultureRobotics in your terminal
3) Type ``` streamlit run HomePage.py ```
4) You will get erros these are probably errors that you miss packages. Install all these packages and it should work. 
