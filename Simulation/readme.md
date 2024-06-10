Welcome to the readme of the Simulation part.

If you want to run this code, make sure all the packages from requirements.txt in the main git page are installed in the active environment.

The structure of this folder is as follows:

1. 'MPC_pathtracking.py'  and 'LQR_pathtracking.py' contain the major functions to simulate path tracking and planting using different algorithms. They load data from Pandas Dataframe file and output the visualized control result.
2. 'Best_path' contains one test path can be used to test the control algorithm
3. '_init_.py' is used for making the docker image and the GUI.

To run the simulation separately from the rest of the pipeline, chose the desired algorithm and run the corresponding pathtracking.py file. Parameters for the simulation can be changed at the bottom of .py file.
