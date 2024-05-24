import streamlit as st
from ComputerVision.field_prediction import predict_field, predict_json, filter_json 
from PathPlanning.pathplanning import pathplanning,load_data
from Simulation.MPC_pathtracking import track_path
from Simulation.PID_pathtracking import path_tracking
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

# These are session states you can see it as global variables. 
# Streamlit gets a trigger when a value is changed 
if 'area_field' not in st.session_state:
    st.session_state["area_field"] = 'value'

if 'simulate_path' not in st.session_state:
    st.session_state['simulate_path'] = 'value'

if 'clicked_predict' not in st.session_state:
    st.session_state.clicked_predict = False

if 'clicked_path' not in st.session_state:
    st.session_state.clicked_path = False

if 'show_field' not in st.session_state:
    st.session_state.show_field = False

if 'clicked_simulation' not in st.session_state:
    st.session_state.clicked_simulation = False

if 'select_pixel' not in st.session_state:
    st.session_state.select_pixel = False

# When uploading a new field the GUI will be reset by settings all session_states to False
# These functions need to be defined above there GUI component. 
def reset_gui():
    '''Callback function for reseting GUI'''
    st.session_state.clicked_predict = False
    st.session_state.clicked_path = False
    st.session_state.show_field = False
    st.session_state.clicked_simulation = False
    st.session_state.select_pixel = False
# Creating the title for uploading images
st.title("Upload image field")

# Set the confidence on which you want to filter
confidence_model = st.number_input("Model confidence",0.0,1.0,step=0.01,value=0.6,on_change=None)

# File uploader for an image
uploaded_file = st.file_uploader("Choose a image",on_change=reset_gui)

# Show detected fields when button is pressed
if st.session_state.clicked_predict:
    if uploaded_file.getvalue() is not None:
        st.image(uploaded_file)
        image_predict = Image.open(uploaded_file)
        st.image("./test/test1.png")

# Button for predicting the fields
def predict_button():
    '''Callback function for detecting fields in input image'''
    image_predict = Image.open(uploaded_file)
    predict_field(image_predict,confidence_model)
    st.session_state.clicked_predict = True

st.button("Detect fields",on_click=predict_button)

# Title for setting the scaling factor
st.title("Input size field")

def choose_pixel_image():
    '''Callback function for showing the pixels that are chosen'''
    pil_image = Image.open(uploaded_file).convert('RGB')
    open_cv_image = np.array(pil_image)
    print(open_cv_image.shape)
    open_cv_image = cv2.circle(open_cv_image, (st.session_state['x_pixel1'],st.session_state['y_pixel1']), 5, color=(0, 0, 255), thickness=-1)
    open_cv_image = cv2.circle(open_cv_image, (st.session_state['x_pixel2'],st.session_state['y_pixel2']), 5, color=(0, 0, 255), thickness=-1)
    cv2.imwrite("filter/select_pixels.png",open_cv_image)
    st.session_state.select_pixel = True

if st.session_state.select_pixel:
    st.image("./filter/select_pixels.png")

x_pixel1 = st.number_input("X pixel 1",min_value=0,value=1,on_change=choose_pixel_image,key="x_pixel1")
y_pixel1 = st.number_input("Y pixel 1",min_value=0,value=1,on_change=choose_pixel_image,key="y_pixel1")
x_pixel2 = st.number_input("X pixel 2",min_value=0,value=640,on_change=choose_pixel_image,key="x_pixel2")
y_pixel2 = st.number_input("Y pixel 2",min_value=0,value=640,on_change=choose_pixel_image,key="y_pixel2")
distance = st.number_input("Distance in meters",min_value=0.1,value=1.0,key="distance_meters")


# Title for the second part of the GUI selecting which field you want to use. 
st.title("Select Field")

# When value is changed of the number inputs below run this function
def select_field():
    '''Callback function for selecting which field to use'''
    distance_pixels = 1
    distance_pixels = math.sqrt((st.session_state['x_pixel2'] - st.session_state['x_pixel1']) ** 2 + (st.session_state['y_pixel2'] - st.session_state['y_pixel1']) ** 2)
    #print("distance pixels: ",distance_pixels)
    scale_pixels_per_meter = st.session_state['distance_meters']/distance_pixels
    #print("Scale pixel: ",scale_pixels_per_meter)
    predict_json(scale_pixels_per_meter,st.session_state['approx'])
    filter_json(st.session_state['fieldnr'])

    data_path ="./filter/filtered_geojson.json"

    field = load_data(data_path,scale_pixels=1)
    st.session_state["area_field"] = str(field.area.to_numpy()[0])
    fig, ax = plt.subplots()
    field.plot(ax=ax, color='lightblue', edgecolor='black')
    plt.savefig("filter/check_field.png")

    # Uncomment if you want to see pixel plot
    # fig1, ax1 = plt.subplots()
    # field_pixel.plot(ax=ax1, color='lightblue', edgecolor='black')
    # plt.savefig("filter/check_field_pixel.png")
    st.session_state.show_field = True

# Parameters for selecting a field
approximate_poly = st.number_input("Fine tune edges",value=0.001,on_change=select_field,step=0.001,min_value=0.0,key="approx",format="%.5f")
field_number = st.number_input("Field_number",value=1,on_change=select_field,key="fieldnr")

# When parameters are filled in show selected field
if st.session_state.show_field:
    st.image("./filter/check_field.png")
    print((st.session_state["area_field"]))
    st.text(f"Area size: {str(st.session_state["area_field"])} m^2")
    

# For inputing the parameters of the dynamics of the tractor
st.title("Parameters path planning")
turning_radius = st.number_input("Turning radius", value=1)
distance_AB = st.number_input("Distance between AB lines",value=1)
headland_size = st.number_input("Headland size",value=1)

# Button to run the path planning algorithm
def plan_path():
    '''Callback function for path planning button'''
    data_path ="./filter/filtered_geojson.json"
    include_obs = False
    turning_rad = turning_radius
    distance = distance_AB
    field,field_headlands,best_path,sp, swaths_clipped,base, total_path, bases = pathplanning(data_path=data_path,include_obs=include_obs,turning_rad=turning_rad,tractor_width=2,seed_distance=5,plotting=True,interpolation_dist=0.5)
    st.session_state['simulate_path'] = best_path
    st.session_state.clicked_path = True
    
st.button("Plan path",on_click=plan_path)

# show the planned path
if st.session_state.clicked_path:
    st.title("Path planning result")
    st.image("./filter/path_field.png")

# Title for the simulation part with parameters
st.title("Simulation_result")
velocity_model = st.number_input("Velocity seeder",value=1.0)
iteration_number = st.number_input("Iteration limit",value=500)
option = st.selectbox("Which controller to use for simulation?",("PID","MPC"))

def simulate_path():
    '''Callback function for simulating path button'''
    if option == "PID":
        path_tracking(st.session_state['simulate_path'],velocity_model,iteration_number)
    elif option == "MPC":
        track_path(st.session_state['simulate_path'],velocity_model,iteration_number)
    st.session_state.clicked_simulation = True

st.button("Simulate path test",on_click=simulate_path)

# Show the simulation result
if st.session_state.clicked_simulation:
    if option == "PID":
        st.image("./filter/simulation_field.png")
    elif option == "MPC":
        st.image("./filter/simulation_field_mpc.png")
    else:
        st.write("Wrong simulation chosen or error")
