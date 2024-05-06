
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree

MAX_STEER = np.deg2rad(90.0)  # maximum steering angle [rad]
dt = 0.1  # time step 
L = 2 
Kp = 2.89  # Adjust based on further tests or systematic tuning
Ki = 0.015
Kd = 25
class KinematicModel:

  def __init__(self, x, y, yaw, v, L, dt):
    self.x = x
    self.y = y
    self.yaw = yaw
    self.v = v
    self.L = L # Distance between front and rear axles
    self.dt = dt  # discrete model

  def update_state(self, a, delta_f):
    delta_f = max(-MAX_STEER, min(delta_f, MAX_STEER))
    self.x = self.x+self.v*math.cos(self.yaw)*self.dt
    self.y = self.y+self.v*math.sin(self.yaw)*self.dt
    self.yaw = self.yaw+self.v/self.L*math.tan(delta_f)*self.dt
    self.v = self.v+a*self.dt

  def get_state(self):
    return self.x, self.y, self.yaw, self.v
  
class PID_posion:

    def __init__(self, kp, ki, kd, target, upper=1., lower=-1.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err = 0  # error
        self.err_last = 0 # previous error
        self.err_all = 0 # sum of error
        self.target = target
        self.upper = upper # upper bound of output
        self.lower = lower # lower bound of output
        self.value = 0

    def cal_output(self, state):
        self.err = self.target - state
        self.value = self.kp * self.err + self.ki * \
            self.err_all + self.kd * (self.err - self.err_last)
        self.update()
        return self.value

    def update(self):
        self.err_last = self.err
        self.err_all = self.err_all + self.err
        if self.value > self.upper:
            self.value = self.upper
        elif self.value < self.lower:
            self.value = self.lower

    def auto_adjust(self, Kpc, Tc):
        self.kp = Kpc * 0.6
        self.ki = Ki
        self.kd = self.kp * (0.125 * Tc)
        # self.ki = self.kp / (0.5 * Tc)
        # self.kd = self.kp * (0.125 * Tc)
        return self.kp, self.ki, self.kd

    def set_pid(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        self.err = 0
        self.err_last = 0
        self.err_all = 0

    def set_target(self, target):
        self.target = target

    def adjust_for_curvature(self, curvature):
        if curvature > 0.1:
            self.kp = self.kp*0.9  # Reduce Kp
            self.kd = self.kd*1.14  # Increase Kd

def cal_target_index(robot_state,refer_path):

    dists = []
    for xy in refer_path:
        dis = np.linalg.norm(robot_state-xy)
        dists.append(dis)

    min_index = np.argmin(dists)
    return min_index

def calculate_path_curvature(path, index):
    """
    Calculate the curvature of the path at a specific index using finite differences.

    Args:
    path (np.array): An array of shape (n, 2), where n is the number of waypoints, containing x and y coordinates.
    index (int): The index in the path array at which to calculate the curvature.

    Returns:
    float: The curvature of the path at the given index.
    """
    # Ensure index is within the valid range
    if index < 1 or index >= len(path) - 1:
        return 0  # Curvature is not defined for the first and last points

    # Points before, at, and after the index
    x0, y0 = path[index - 1]
    x1, y1 = path[index]
    x2, y2 = path[index + 1]

    # First derivatives (central difference)
    dx1 = (x2 - x0) / 2
    dy1 = (y2 - y0) / 2

    # Second derivatives (central difference)
    dx2 = x2 - 2 * x1 + x0
    dy2 = y2 - 2 * y1 + y0

    # Curvature calculation
    num = abs(dx1 * dy2 - dy1 * dx2)
    denom = (dx1**2 + dy1**2)**1.5

    # Prevent division by zero
    if denom == 0:
        return 0

    curvature = num / denom
    return curvature

def path_tracking(refer_path, speed, iteration_limit, stop_threshold=0.5):
    """
    Use PID controller to track the reference path
    
    attrs:
        refer_path (pandas DataFrame): a ideal path produced by path planning
        speed(float): the speed of the tracktor
    returns:
        position (pandas DataFrame): a Pandas DF that contains the actual path the trackor takes
    """
    
    # Input validation
    if not isinstance(refer_path, pd.DataFrame):
        raise TypeError("refer_path must be a pandas DataFrame")
    if refer_path.empty:
        raise ValueError("refer_path cannot be empty")
    
    x_points = np.array(refer_path['x'])
    y_points = np.array(refer_path['y'])
    #commands = np.array(refer_path['command']) 
    refer_tra = np.column_stack((x_points, y_points))

    # Initial state
    refer_tree = KDTree(refer_tra) # reference trajectory for searching cloest track point
    final_point = refer_tra[-1]
    start_x = x_points[0] 
    start_y = y_points[0]
    if len(refer_path) > 1:
        dy = y_points[1] - y_points[0]
        dx = x_points[1] - x_points[0]
        start_yaw = np.arctan2(dy, dx)
    else:
        start_yaw = 0 
    x_ = []
    y_ = []
    errors = []
    accumulated_error = 0
    correct_passes = 0
    i = 1
    L = 2
    time_step = 0.1
    tracktor = KinematicModel(start_x,start_y,start_yaw,speed,L,time_step)
    PID = PID_posion(kp = Kp, ki = Ki, kd = Kd, target=0, upper=np.pi/6, lower=-np.pi/6)
  
    #while i in range(1, len(refer_tra) - 1):
    while i in range(1, iteration_limit):
        robot_state = np.zeros(2)
        robot_state[0] = tracktor.x
        robot_state[1] = tracktor.y
        distance, ind = refer_tree.query(robot_state)
        alpha = math.atan2(refer_tra[ind, 1]-robot_state[1], refer_tra[ind, 0]-robot_state[0])
        l_d = np.linalg.norm(refer_tra[ind]-robot_state)
        theta_e = alpha-tracktor.yaw
        e_y = -l_d*math.sin(theta_e)
        squared_error = e_y ** 2
        accumulated_error += squared_error * time_step  
        errors.append(abs(e_y))
        if abs(e_y) < 0.4:
            correct_passes += 1  
        #curvature = calculate_path_curvature(refer_tra, i)
        #PID.adjust_for_curvature(curvature)
        delta_f = PID.cal_output(e_y)
        tracktor.update_state(0,delta_f) 
        x_.append(tracktor.x)
        y_.append(tracktor.y)
        # plt.cla()
        # plt.plot(refer_tra[:, 0], refer_tra[:, 1], '-.b', linewidth=1.0)
        # plt.plot(x_, y_, "-r", label="trajectory")
        # plt.plot(refer_tra[ind, 0], refer_tra[ind, 1], "go", label="target")
        # # plt.axis("equal")
        # plt.grid(True)
        # plt.pause(0.001)
        distance_to_final_point = np.linalg.norm([tracktor.x - final_point[0], tracktor.y - final_point[1]])
        if distance_to_final_point < stop_threshold:
            print(f"Stopping path tracking as the vehicle is within {stop_threshold} units of the final path point.")
            break
        i += 1
        
    position = pd.DataFrame({
        'x': x_,
        'y': y_
    })

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    #accuracy = (correct_passes / i) * 100

    print(f"Mean Error: {mean_error}")
    print(f"Max Error: {max_error}")
    #print(f"Accuracy: {accuracy}%")
    print(f"Accumulated Squared Error: {accumulated_error}")

    # show the path tracking result
    plt.figure(figsize=(8, 8))
    plt.plot(refer_path['x'], refer_path['y'], '-.b', linewidth=1.0, label='Reference Path')
    plt.plot(x_,y_,'r')
    plt.xlabel('X')  
    plt.ylabel('Y') 
    plt.title('Path Tracking') 
    plt.show()
    
    return position

data_path ='D:/5ARIP10 Team Project/working space/AgricultureRobotics/Simulation/actualpath.csv'
speed = 2
iteration_limit = 5000
refer_path = pd.read_csv(data_path)
actual_path = path_tracking(refer_path, speed, iteration_limit, stop_threshold=0.5)
# The refer_path used here is numpy array, the refer_path to the function should be pandas dataframe, so while testing with path planning ignore this cell
# refer_path = np.zeros((1000, 2))
# refer_path[:,0] = np.linspace(0, 100, 1000) 
# refer_path[:,1] = 2*np.sin(refer_path[:,0]/3.0)
# refer_path_df = pd.DataFrame(refer_path, columns=['x', 'y'])

# center_x = 10  
# center_y = 0   
# radius = 10    

# line_length = 20  

# theta = np.linspace(0, 1.25*np.pi, 500)  
# x_circle = center_x - radius * np.cos(theta)  
# y_circle = center_y + radius * np.sin(theta)  
# end_x = x_circle[-1]
# end_y = y_circle[-1]

# y_line = np.linspace(end_y, end_y - line_length, 500)  
# x_line = np.full_like(y_line, end_x)  

# x = np.concatenate([x_circle, x_line])
# y = np.concatenate([y_circle, y_line])
# data = pd.DataFrame({
#     'x': x,
#     'y': y
# })

# reversed_data = data.iloc[::-1].reset_index(drop=True)

# speed = 2

# position = path_tracking(reversed_data, speed,iteration_limit)
#position1 = path_tracking(data,speed,iteration_limit)