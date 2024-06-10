Welcome to the readme of the Pathplanning algorithm part.

If you want to run this code, make sure all the packages from requirements.txt in the main git page are installed in the active environment.

The structure of this folder is as follows:

1. 'pathplanning.py' contains the main functions for generating a path, it loads data from a JSON file, generates headlands, plans a path and plants seeds.
2. 'dubins_curves.py' contains the code for generating dubins curves between 2 points of a given heading
3. '/data' contains different field geometries that can be used to test the pathplanning algorithm
4. '__init__.py' is used for making the docker image and the GUI.

To run the pathplanning algorithm separately from the rest of the pipeline, simply make sure that there is a valid field geometry present in the /data/field_geometry folder and run the pathplanning file (note, depending on the size of the field and some of the parameters settings, this might take a long time). Parameters for the path planning can be changed at the bottom of pathplanning.py