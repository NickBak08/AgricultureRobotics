import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree

MAX_STEER = np.deg2rad(90.0)  # maximum steering angle [rad]
MAX_ACC = 0.5
dt = 0.1  # time step 
L = 2 
Kp = 2.9  
Ki = 0.015
Kd = 27
lam = 0.1
c = 0.9

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
        self.integral = 0

    def cal_output(self, state):
        self.err = self.target - state
        self.integral += self.err *dt
        derivative = (self.err - self.err_last) / dt
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

    def reset(self):
        self.err = 0
        self.err_last = 0
        self.err_all = 0

    def adjust_for_curvature(self, curvature):
        if curvature > 0.1:
            self.kp = 2.8  # Reduce Kp
            self.kd = 28  # Increase Kd

def cal_target_index(robot_state, refer_path, l_d,ind):

    max_index = min(ind + 50, len(refer_path) - 1)
    search_range = refer_path[ind:max_index + 1]
    dists = np.linalg.norm(search_range - robot_state, axis=1)

    min_index = np.argmin(dists)
    min_index += ind

    delta_l = np.linalg.norm(refer_path[min_index]-robot_state)

    while l_d > delta_l and (min_index+1) < len(refer_path):
        delta_l = np.linalg.norm(refer_path[min_index+1]-robot_state)
        min_index += 1
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

def adjust_lookahead_distance(path, current_index):
    curvature = calculate_path_curvature(path, current_index)
    if curvature < 0.01:  
        return 1.0  
    else:
        return 0.5  

def calculate_seeding_positions(x_points, y_points, commands):
    seed_positions = []
    
    for i in range(len(commands)):
        if commands[i] == 1:
            if i > 0:
                yaw = math.atan2(y_points[i] - y_points[i - 1], x_points[i] - x_points[i - 1])
            else:
                yaw = 0  
                
            spread_distance = 0.2
            
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)
            x_left = x_points[i] + spread_distance * sin_yaw
            y_left = y_points[i] - spread_distance * cos_yaw
            x_right = x_points[i] - spread_distance * sin_yaw
            y_right = y_points[i] + spread_distance * cos_yaw
            
            seed_positions.append([x_points[i], y_points[i], x_left, y_left, x_right, y_right])
    
    seed_positions_array = np.array(seed_positions)
    return seed_positions_array

class PID_Speed:
    def __init__(self, kp, ki, kd, target_speed):
        """
        Initialize the PID Controller for speed.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            target_speed (float): The target speed for the PID controller.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_speed = target_speed
        self.integral = 0
        self.previous_error = 0
        self.last_time = None

    def compute(self, current_speed):
        """
        Compute the control signal based on the current speed.

        Args:
            current_speed (float): The current speed of the vehicle.
            current_time (float): The current time in seconds.

        Returns:
            float: The control output which is the acceleration/deceleration command.
        """
        # Calculate PID errors
        error = self.target_speed - current_speed
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = error

        # Compute PID output
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return output

    def reset(self):
        """
        Reset the integral and previous error of the PID controller.
        """
        self.integral = 0
        self.previous_error = 0
        self.last_time = None

def path_tracking(refer_path, speed, iteration_limit, stop_threshold=1,angle_tolerance=np.deg2rad(30)):
    """
    Use PID controller to track the reference path
    
    attrs:
        refer_path (pandas DataFrame): a ideal path produced by path planning
        speed(float): the speed of the tractor
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
    commands = np.array(refer_path['command']) 
    passed_measurements = 0
    total_measurements = np.sum(commands)
    x_ = []
    y_ = []
    target_indices = []
    refer_tra = np.column_stack((x_points, y_points))

    # Initial state
    final_point = refer_tra[-1]
    start_x = x_points[0] 
    start_y = y_points[0]
    if len(refer_path) > 1:
        dy = y_points[1] - y_points[0]
        dx = x_points[1] - x_points[0]
        start_yaw = np.arctan2(dy, dx)
    else:
        start_yaw = 0 
    x_, y_, errors = [], [], []
    accumulated_error = 0
    velocities = []
    i = 1
    L = 2
    time_step = 0.1
    tractor = KinematicModel(start_x,start_y,start_yaw,speed,L,time_step)
    PID = PID_posion(kp = Kp, ki = Ki, kd = Kd, target=0, upper=np.pi/6, lower=-np.pi/6)
    pid = PID_Speed(kp=0.1, ki=0.01, kd=0.05, target_speed=2)
    ind = 0
    
    def update_plot(frame):
        plt.cla()
        plt.plot(refer_tra[:, 0], refer_tra[:, 1], "-.b", linewidth=1.0, label="Reference Path")
        plt.plot(x_[:frame], y_[:frame], "-r", label="Tracked path")
        # plt.plot(reference_path.refer_path[target_ind, 0], reference_path.refer_path[target_ind, 1], "go", label="target")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.title('PID Path Tracking')
        plt.legend()
        plt.grid(True)

    while i in range(1, iteration_limit):
        robot_state = np.zeros(2)
        robot_state[0] = tractor.x
        robot_state[1] = tractor.y
        x_.append(tractor.x)
        y_.append(tractor.y)

        curvature = calculate_path_curvature(refer_tra, ind)
        c = adjust_lookahead_distance(refer_tra, ind)
        ind = cal_target_index(robot_state, refer_tra, c, ind)
        target_indices.append(ind)
        alpha = math.atan2(refer_tra[ind, 1]-robot_state[1], refer_tra[ind, 0]-robot_state[0])
        target_yaw = (alpha + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
        yaw_diff = abs(target_yaw - tractor.yaw)
        yaw_diff = min(yaw_diff, 2 * np.pi - yaw_diff)

        l_d = np.linalg.norm(refer_tra[ind]-robot_state)
        theta_e = alpha-tractor.yaw
        e_y = -l_d*math.sin(theta_e)
        squared_error = e_y ** 2
        accumulated_error += squared_error * time_step  
        errors.append(abs(e_y))

        if commands[ind] == 1 and (l_d) <= 1.15 and yaw_diff <= angle_tolerance:
            passed_measurements += 1
            commands[ind] = 0

        curvature = calculate_path_curvature(refer_tra, ind)

        #PID.adjust_for_curvature(curvature)
        delta_f = PID.cal_output(e_y)
        pid_speed_adjustment = pid.compute(tractor.v)
        acceleration = max(-MAX_ACC, min(pid_speed_adjustment, MAX_ACC))
        tractor.update_state(acceleration, delta_f) 
        x_.append(tractor.x)
        y_.append(tractor.y)
        velocities.append(tractor.v)

        distance_to_final_point = np.linalg.norm([tractor.x - final_point[0], tractor.y - final_point[1]])
        if distance_to_final_point < stop_threshold:
            print(f"Stopping path tracking as the vehicle is within {stop_threshold} units of the final path point.")
            break
        i += 1
    #fig = plt.figure(figsize=(8, 8))
    #animation = FuncAnimation(fig, update_plot, frames=len(x_), interval=100)

    position = pd.DataFrame({
        'x': x_,
        'y': y_
    })

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    measurement_accuracy = passed_measurements / total_measurements * 100
    print(f"Measurement accuracy: {measurement_accuracy:.2f}%")
    #print(f"passed and total: {passed_measurements,total_measurements}")
    #print(f"Mean Error: {mean_error}")
    #print(f"Max Error: {max_error}")
    #print(f"Accumulated Squared Error: {accumulated_error}")

    # show the path tracking result
    plt.figure(figsize=(8, 8))
    plt.plot(refer_path['x'], refer_path['y'], '-.b', linewidth=1.0, label='Reference Path')
    plt.plot(x_,y_,'r')
    plt.xlabel('X')  
    plt.ylabel('Y') 
    plt.title('Path Tracking') 
    plt.show()
    
    # plt.figure(figsize=(8, 4))
    # plt.plot(velocities, '-b', label='Velocity')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Velocity (m/s)')
    # plt.title('Velocity Profile Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return position

data_path ='D:/5ARIP10 Team Project/working space/AgricultureRobotics/Simulation/bestpath.csv'
speed = 3.75
iteration_limit = 30000
refer_path = pd.read_csv(data_path)
actual_path = path_tracking(refer_path, speed, iteration_limit, stop_threshold=0.5)
