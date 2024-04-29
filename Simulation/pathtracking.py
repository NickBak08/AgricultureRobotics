import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class KinematicModel:

  def __init__(self, x, y, yaw, v, L, dt):
    self.x = x
    self.y = y
    self.yaw = yaw
    self.v = v
    self.L = L # Distance between front and rear axles
    self.dt = dt  # discrete model

  def update_state(self, a, delta_f):
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
        self.ki = self.kp / (0.5 * Tc)
        self.kd = self.kp * (0.125 * Tc)
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


def path_tracking(refer_path, speed, stop_threshold=0.5):
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
    
    refer_path_np = refer_path.to_numpy()
   
    # Initial state
    refer_tree = KDTree(refer_path_np) # reference trajectory for searching cloest track point
    final_point = refer_path_np[-1]
    start_x = refer_path_np[0, 0] 
    start_y = refer_path_np[0, 1]
    if len(refer_path) > 1:
        dy = refer_path_np[1, 1] - refer_path_np[0, 1]
        dx = refer_path_np[1, 0] - refer_path_np[0, 0]
        start_yaw = np.arctan2(dy, dx)
    else:
        start_yaw = 0 
    x_ = []
    y_ = []
    errors = []
    accumulated_error = 0
    correct_passes = 0
    i = 0
    L = 2
    time_step = 0.1
    tracktor = KinematicModel(start_x,start_y,start_yaw,speed,L,time_step)
    PID = PID_posion(kp = 2, ki = 0.01, kd = 30, target=0, upper=np.pi/6, lower=-np.pi/6)
    rows, columns = refer_path_np.shape
    while i in range(rows):
        robot_state = np.zeros(2)
        robot_state[0] = tracktor.x
        robot_state[1] = tracktor.y
        distance, ind = refer_tree.query(robot_state)
        alpha = math.atan2(refer_path_np[ind, 1]-robot_state[1], refer_path_np[ind, 0]-robot_state[0])
        l_d = np.linalg.norm(refer_path_np[ind]-robot_state)
        theta_e = alpha-tracktor.yaw
        e_y = -l_d*math.sin(theta_e)
        squared_error = e_y ** 2
        accumulated_error += squared_error * time_step  
        errors.append(abs(e_y))
        if abs(e_y) < 0.4:
            correct_passes += 1  

        delta_f = PID.cal_output(e_y)
        tracktor.update_state(0,delta_f) 
        x_.append(tracktor.x)
        y_.append(tracktor.y)
        
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
    plt.figure(1)
    plt.plot(refer_path_np[:, 0], refer_path_np[:, 1], '-.b', linewidth=1.0)
    plt.plot(x_,y_,'r')
    plt.xlabel('X')  
    plt.ylabel('Y') 
    plt.title('Path Tracking') 
    plt.show()
    
    return position

# The refer_path used here is numpy array, the refer_path to the function should be pandas dataframe, so while testing with path planning ignore this cell
refer_path = np.zeros((1000, 2))
refer_path[:,0] = np.linspace(0, 100, 1000) 
refer_path[:,1] = 2*np.sin(refer_path[:,0]/3.0)
refer_path_df = pd.DataFrame(refer_path, columns=['x', 'y'])

speed = 2

position = path_tracking(refer_path_df, speed)