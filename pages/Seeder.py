import streamlit as st
from ComputerVision.field_prediction import predict_field, predict_json, filter_json 
from PathPlanning.pathplanning import pathplanning,load_data
from Simulation.MPC_pathtracking import track_path
from Simulation.pathtracking import path_tracking
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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


st.title("Upload image field")
confidence_model = st.number_input("Model confidence",0.0,1.0,step=0.01,value=0.6,on_change=None)
bytes_data = None
uploaded_file = st.file_uploader("Choose a image")

# Show detected fields
if st.session_state.clicked_predict:
    if uploaded_file.getvalue() is not None:
        st.image(uploaded_file)
        image_predict = Image.open(uploaded_file)
        st.image("./test/test1.png")


# Button for detecting fields
def button_clicked():
    image_predict = Image.open(uploaded_file)
    predict_field(image_predict,confidence_model)
    st.session_state.clicked_predict = True

st.button("Detect fields",on_click=button_clicked)

st.title("Select Field")

def value_change():
    predict_json(st.session_state['scale'],st.session_state['approx'])
    filter_json(st.session_state['fieldnr'])
    data_path ="./filter/filtered_geojson.json"
    field = load_data(data_path)
    fig, ax = plt.subplots()
    field.plot(ax=ax, color='lightblue', edgecolor='black')
    plt.savefig("filter/check_field.png")
    st.session_state.show_field = True


scale_field = st.number_input("Scale field",value=100,on_change=value_change,key="scale")
approximate_poly = st.number_input("Fine tune edges",value=0.001,on_change=value_change,step=0.001,min_value=0.0,key="approx",format="%.5f")
field_number = st.number_input("Field_number",value=1,on_change=value_change,key="fieldnr")

if st.session_state.show_field:
    st.image("./filter/check_field.png")
# For inputing the parameters of the dynamics of the tractor
st.title("Parameters path planning")
turning_radius = st.number_input("Turning radius", value=1)
distance_AB = st.number_input("Distance between AB lines",value=1)
headland_size = st.number_input("Headland size",value=1)

# Button for detecting fields
def plan_path():
    data_path ="./filter/filtered_geojson.json"
    include_obs = False
    turning_rad = turning_radius
    distance = distance_AB
    field, best_path = pathplanning(data_path,include_obs,turning_rad,distance,True,headland_size,2)
    st.session_state['simulate_path'] = best_path
    st.session_state.clicked_path = True
    
st.button("Plan path",on_click=plan_path)

if st.session_state.clicked_path:
    st.title("Path planning result")
    st.image("./filter/path_field.png")

st.title("Simulation_result")
velocity_model = st.number_input("Velocity seeder",value=1.0)
iteration_number = st.number_input("Iteration limit",value=500)
option = st.selectbox("Which controller to use for simulation?",("PID","MPC"))

def simulate_path():
    if option == "PID":
        path_tracking(st.session_state['simulate_path'],velocity_model)
    elif option == "MPC":
        track_path(st.session_state['simulate_path'],velocity_model,iteration_number)
    st.session_state.clicked_simulation = True

st.button("Simulate path test",on_click=simulate_path)

if st.session_state.clicked_simulation:
    if option == "PID":
        st.image("./filter/simulation_field.png")
    elif option == "MPC":
        st.image("./filter/simulation_field_mpc.png")
    else:
        st.write("Wrong simulation chosen or error")