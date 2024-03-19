# AgricultureRobotics
The repository for the TU/e project agriculture robotics in collaboration with VeXtronics.

## Rules
- Make a new branch for every subsystem (pathplanning, computervision, simulation)
- When implementing a new feature create a new branch in your subbranch (computervision/yolo)
- Commit often not only when you want to push your code
- When merging asks others to review the code

# List python packages
List here all the python packages being used
- numpy
- Pillow
- request
- dvc
- dvc-gdrive

# Installing CVAT Information
For installing CVAT on windows you need to follow the following steps
[Link to installing instructions](https://opencv.github.io/cvat/docs/administration/basics/installation/)

# Using DVC for data sharing
Github is not made to share data it is made for text files. To fix this DVC is used. With DVC you can push images to a cloud provider and pull them. This is especially handy when multiple users need to use the same data.

The first step is to install dvc and dvc-gdrive (For now make sure you pulled the computer vision branch and you work in it)!
``` pip install dvc ```
``` pip install dvc-gdrive ```

After that you can use dvc pull to download the data and you can use push when you want to upload new data. 
