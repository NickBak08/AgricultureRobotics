import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree
from matplotlib.colors import Normalize

MAX_STEER = np.deg2rad(70.0)  # maximum steering angle [rad]
MAX_ACC = 0.5
MAX_VEL = 2
N=100 
EPS = 1e-4 
Q = np.eye(3)*3
R = np.eye(2)*2.
dt=0.5 
L= 2 
Kp = 1.0
# Change the font in all plots
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

class KinematicModel:
    """
    Build the KinematicModel of the tractor
    
    Attributes:
        x : A float of x-coordinate of the tractor.
        y : A float of y-coordinate of the tractor.
        yaw : A float of yaw angle of the tractor.
        v : A float of velocity of the tractor.
        L : A float of wheelbase of the tractor (fixed).
        dt : A float of time step.        
        a : A float of acceleration of the tractor.
        delta_f : A float of steering angle of the tractor.
    """
    def __init__(self, x, y, yaw, v, L, dt):
        """Initialize the kinematic model."""
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L 
        self.dt = dt  

    def update_state(self, a, delta_f):
        """ Update the state of the kinematic model based on acceleration and steering angle."""
        delta_f = max(-MAX_STEER, min(delta_f, MAX_STEER))
        self.x = self.x+self.v*math.cos(self.yaw)*self.dt
        self.y = self.y+self.v*math.sin(self.yaw)*self.dt
        self.yaw = self.yaw+self.v/self.L*math.tan(delta_f)*self.dt
        self.v = self.v+a*self.dt

    def get_state(self):
        """Get the current state of the kinematic model."""
        return self.x, self.y, self.yaw, self.v
  
    def state_space(self, ref_delta, ref_yaw):
        """
        Define the state space matrices for the kinematic model.

        Args:
            ref_delta (float): Reference steering angle.
            ref_yaw (float): Reference yaw angle.

        Returns:
            tuple: (A, B, C) State space matrices.
        """
        A = np.matrix([
            [1.0, 0.0, -self.v*self.dt*math.sin(ref_yaw)],
            [0.0, 1.0, self.v*self.dt*math.cos(ref_yaw)],
            [0.0, 0.0, 1.0]])

        B = np.matrix([
            [self.dt*math.cos(ref_yaw), 0],
            [self.dt*math.sin(ref_yaw), 0],
            [self.dt*math.tan(ref_delta)/self.L, self.v*self.dt /(self.L*math.cos(ref_delta)*math.cos(ref_delta))]
        ])

        return A, B
    
class MyReferencePath:
    """ 
    Caculate the reference control vector based on a reference path
    
    Attributes:
        x_points : A numpy array of x-coordinates of the reference path.
        y_points : A lnumpy array of y-coordinates of the reference path.
        x : A float of current x-coordinate of the tractor.
        y : A float of current y-coordinate of the tractor.
        robot_state : A tuple of Current state (x, y, psi, v).
    """
    def __init__(self, x_points, y_points):
        """  
        Initialize reference path.
        
        Args:
            x_points (numpy array): List of x-coordinates.
            y_points (numpy array): List of y-coordinates.           
        """
        self.refer_path = np.zeros((len(x_points), 4))
        self.refer_path[:, 0] = x_points
        self.refer_path[:, 1] = y_points 
         # Set target point to the last point in the reference path
        self.target_point = self.refer_path[-1, :2] 

         # Calculate tangent and curvature at each point
        for i in range(len(self.refer_path)):
            if i == 0:
                dx = self.refer_path[i+1, 0] - self.refer_path[i, 0]
                dy = self.refer_path[i+1, 1] - self.refer_path[i, 1]
                ddx = self.refer_path[2, 0] + \
                    self.refer_path[0, 0] - 2*self.refer_path[1, 0]
                ddy = self.refer_path[2, 1] + \
                    self.refer_path[0, 1] - 2*self.refer_path[1, 1]
            elif i == (len(self.refer_path)-1):
                dx = self.refer_path[i, 0] - self.refer_path[i-1, 0]
                dy = self.refer_path[i, 1] - self.refer_path[i-1, 1]
                ddx = self.refer_path[i, 0] + \
                    self.refer_path[i-2, 0] - 2*self.refer_path[i-1, 0]
                ddy = self.refer_path[i, 1] + \
                    self.refer_path[i-2, 1] - 2*self.refer_path[i-1, 1]
            else:
                dx = self.refer_path[i+1, 0] - self.refer_path[i, 0]
                dy = self.refer_path[i+1, 1] - self.refer_path[i, 1]
                ddx = self.refer_path[i+1, 0] + \
                    self.refer_path[i-1, 0] - 2*self.refer_path[i, 0]
                ddy = self.refer_path[i+1, 1] + \
                    self.refer_path[i-1, 1] - 2*self.refer_path[i, 1]
            self.refer_path[i, 2] = math.atan2(dy, dx) # caculate the corresponding yaw angle
            self.refer_path[i, 3] = (
                ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))  # caculate the curvature
            
    def calc_ref_trajectory(self, robot_state,index):
        """
        Calculate reference vector.

        Args:
            robot_state (tuple): Current state of the robot (x, y, psi, v).
            num_steps (int) : The predict horizon

        Returns:
            ind (int) : Index of current reference point.
            ref_delta (numpy array) : Reference control inputs.
            ref_yaw (numpy array) : Reference state inputs.
        """
        max_index = min(index + 25, len(self.refer_path) - 1)
        if max_index < index:
            index = max_index 
        search_range = self.refer_path[index:max_index + 1]
        dists = np.linalg.norm(search_range - robot_state, axis=1)
        ind = np.argmin(dists)
        ind += index
        curv = self.refer_path[ind, 3]
        ref_yaw = self.refer_path[ind,2]
         # Control inputs (velocity and steering angle)
        ref_delta = math.atan2(L*curv, 1)
 
        return ind, ref_delta,ref_yaw

def lqr(robot_state, refer_path, s0, A, B, Q, R):
    """
    Linear Quadratic Regulator (LQR) controller implementation.
    
    Parameters:
    - robot_state: The current state of the tractor.
    - refer_path: The reference path the tractor should follow.
    - s0: The current index on the reference path.
    - A: The state transition matrix.
    - B: The control input matrix.
    - Q: The state cost matrix.
    - R: The control cost matrix.
    
    Returns:
    - u_star: The optimal control input deviation.
    """
    def cal_Ricatti(A,B,Q,R):
        """ Calculate the Riccati matrix to solve optimization problem."""
        Qf=Q
        P=Qf
        for t in range(N):
            P_=Q+A.T@P@A-A.T@P@B@np.linalg.pinv(R+B.T@P@B)@B.T@P@A
            if(abs(P_-P).max()<EPS):
                break
            P=P_
        return P_
    
    x=robot_state[0:3]-refer_path[s0,0:3]
    P = cal_Ricatti(A,B,Q,R)
    K = -np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
    u = K @ x
    u_star = u 
    return u_star

class PIDController:
    def __init__(self, kp, ki, kd, target_speed):
        """
        Initialize the PID Controller.

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

def add_imu_noise(true_acceleration, true_angular_velocity):
    """
    Adds noise to the true acceleration and angular velocity to simulate IMU sensor noise.
    
    Parameters:
    - true_acceleration: The true acceleration value.
    - true_angular_velocity: The true angular velocity value.
    
    Returns:
    - noisy_acceleration: The acceleration value with added noise.
    - noisy_angular_velocity: The angular velocity value with added noise.
    """
    acc_std_dev = 0.01
    gyro_std_dev =  0.01
    acc_noise = np.random.normal(0, acc_std_dev)
    gyro_noise = np.random.normal(0, gyro_std_dev)
    noisy_acceleration = true_acceleration + acc_noise
    noisy_angular_velocity = true_angular_velocity + gyro_noise
    return noisy_acceleration, noisy_angular_velocity

def track_path(refer_df, initial_speed, target_speed, iteration_limit, noise = None,save_picture = None,save_animation = None):
    """
    Track a reference path using LQR.

    Args:
        refer_df (DataFrame): DataFrame containing reference path points.
        initial_speed (float): Initial speed of the tractor.
        target_speed (float): Target speed of the tractor
        iteration_limit (int): Maximum number of iterations.
        noise (bool) : Add noise to the system
        save_picture (bool) : Show the animation of pathtracking and save the result pictures

    Returns:
        x_ (numpy array): x-coordinate of actual path
        y_ (numpy array): y-coordinate of actual path
        plant_points_x (numpy array): x-coordinate of planting points
        plant_points_y (numpy array): y-coordinate of planting points
        command (numpy array) : planting result
        measurement_accuracy (float) : the accuracy of planting seed
        final_index (int) : the last correctly passed target point
    """
    # Input validation
    if not isinstance(refer_df, pd.DataFrame):
        raise TypeError("Input Reference path must be a pandas DataFrame")
    if refer_df.empty:
        raise ValueError("Input Reference path cannot be empty")
    
    # Form the reference path and initialize the tractor
    x_points = np.array(refer_df['x'])
    y_points = np.array(refer_df['y'])
    commands = np.array(refer_df['command'])
    reference_path = MyReferencePath(x_points, y_points)
    goal = reference_path.target_point
    print('goal',goal)
    start_x = reference_path.refer_path[0, 0]
    start_y = reference_path.refer_path[0, 1]
    start_yaw = np.arctan2(reference_path.refer_path[1, 1] - start_y,
                           reference_path.refer_path[1, 0] - start_x)
    L = 2.0  
    dt = 0.1  
    tractor = KinematicModel(start_x, start_y, start_yaw, initial_speed, L, dt)   
    pid = PIDController(kp=0.2, ki=0.01, kd=0.05, target_speed =target_speed)

    min_distance_to_update = 2.0
    x_ = []
    y_ = []
    velocities = []
    nearest_distances = []
    plant_points_x = []
    plant_points_y = []
    plant_points_yaw = []
    passed_measurements = 0
    index = 1

    # Create a function to update the animation at each frame
    def update_plot(frame):
        plt.cla()
        plt.plot(reference_path.refer_path[:, 0], reference_path.refer_path[:, 1], "-.b", linewidth=1.0, label="course")
        plt.plot(x_[:frame], y_[:frame], "-r", label="trajectory")
        plt.plot(reference_path.refer_path[target_ind, 0], reference_path.refer_path[target_ind, 1], "go", label="target")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.legend()
        plt.grid(True)

    # Main loop for tracking the path
    for i in range(iteration_limit):
        # Get current state of the vehicle
        robot_state = np.zeros(4)
        robot_state[0] = tractor.x
        robot_state[1] = tractor.y
        robot_state[2] = tractor.yaw
        robot_state[3] = tractor.v
        x0 = robot_state[0:3]

        # Calculate reference trajectory
        target_ind, ref_delta,ref_yaw = reference_path.calc_ref_trajectory(
            robot_state,index)

        # Use LQR to calculate reference steering angle
        A, B = tractor.state_space(ref_delta,ref_yaw)
        u_star = lqr(robot_state, reference_path.refer_path, target_ind, A, B, Q, R)
        acceleration = u_star[0,0]
        acceleration = max(-MAX_ACC, min(acceleration, MAX_ACC))
        steering_angle = u_star[0,1]
        delta = ref_delta + steering_angle

        # Use PID to calculate acceleration
        pid_speed_adjustment = pid.compute(tractor.v)
        acceleration = max(-MAX_ACC, min(pid_speed_adjustment, MAX_ACC))
        
        # Record the current state of the tractor
        x_.append(tractor.x)
        y_.append(tractor.y)
        velocities.append(tractor.v)


        yaw_difference = np.abs((tractor.yaw - ref_yaw + np.pi) % (2 * np.pi) - np.pi)
        alpha = math.atan2(reference_path.refer_path[target_ind, 1]-robot_state[1], reference_path.refer_path[target_ind, 0]-robot_state[0])
        theta_e = alpha-tractor.yaw
        
        distance_to_target = np.linalg.norm(robot_state[0:2] - reference_path.refer_path[target_ind, :2])
        e_y = -distance_to_target*math.sin(theta_e)
        nearest_distances.append(distance_to_target)

        index = target_ind


        # Check if tracktor passed the planting point (distance <= 0.08)
        if commands[target_ind] == 1 and  distance_to_target <= 0.08 and yaw_difference <= 0.3 * np.pi:
            passed_measurements += 1
            plant_points_x.append(tractor.x)
            plant_points_y.append(tractor.y)
            plant_points_yaw.append(tractor.yaw)
            commands[target_ind] = 2

        # Check if the goal is reached
        if np.linalg.norm(robot_state[0:2]-goal) <= 1:
            print("reach goal")
            break

        if noise == True:
            acceleration, delta = add_imu_noise(acceleration, delta)
        # Update the tractor state
        tractor.update_state(acceleration, delta) 

    
    if (save_animation == True):
        fig = plt.figure(figsize=(8, 8))
        animation = FuncAnimation(fig, update_plot, frames=len(x_), interval=100)
        animation.save('./filter/path_tracking_animation.gif', writer='pillow') 

    # Calculate the accuracy
    measurement_accuracy = 0    
    
    total_measurements = sum(1 for x in commands[:index + 1] if x != 0)
    measurement_accuracy = passed_measurements / total_measurements * 100
    print(f"Measurement accuracy: {measurement_accuracy:.2f}%")
    print(f"passed and total: {passed_measurements,total_measurements}")

    # Visualize the final result
    plt.figure(figsize=(8, 8))
    plt.plot(refer_df['x'], refer_df['y'], '-.b', linewidth=1.0, label='Reference Path')
    plt.plot(x_, y_, '-r', label='Tracked Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path Tracking Result')
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
    if save_picture == True:
        plt.savefig('./filter/Path_Tracking_Result.svg', format='svg',bbox_inches='tight')

    plt.figure(figsize=(8, 4))
    plt.plot(velocities, '-b', label='Velocity')
    plt.xlabel('Time Steps')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Profile Over Time')
    plt.legend()
    plt.grid(True)
    if save_picture == True:
        plt.savefig('./filter/Velocity_Profile_Over_Time.png',bbox_inches='tight')
    plt.show()

    return x_, y_ , plant_points_x, plant_points_y, plant_points_yaw, commands,measurement_accuracy,index

def calculate_seeding_positions(x_points, y_points, commands,final_index, spread_distance,save_picture = None):
    seed_positions = []
    right_positions = []

    for i in range(final_index):
        if commands[i] == 1:
            if i > 0:
                yaw = math.atan2(y_points[i] - y_points[i - 1], x_points[i] - x_points[i - 1])
            else:
                yaw = 0  
                
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)
            x_left = x_points[i] + spread_distance * sin_yaw
            y_left = y_points[i] - spread_distance * cos_yaw
            x_right = x_points[i] - spread_distance * sin_yaw
            y_right = y_points[i] + spread_distance * cos_yaw
            seed_positions.append([x_left, y_left])
            seed_positions.append([x_right, y_right])
        if commands[i] == 2:
            if i > 0:
                yaw = math.atan2(y_points[i] - y_points[i - 1], x_points[i] - x_points[i - 1])
            else:
                yaw = 0                 
            
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)
            x_left = x_points[i] + spread_distance * sin_yaw
            y_left = y_points[i] - spread_distance * cos_yaw
            x_right = x_points[i] - spread_distance * sin_yaw
            y_right = y_points[i] + spread_distance * cos_yaw
            right_positions.append([x_left, y_left])
            right_positions.append([x_right, y_right])     
    seed_positions_array = np.array(seed_positions)
    right_positions_array = np.array(right_positions)

    x_coords = seed_positions_array[:, 0]
    y_coords = seed_positions_array[:, 1]
    x_coords1 = right_positions_array[:, 0]
    y_coords1 = right_positions_array[:, 1]
    
    # Visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='red', marker='o', s=1, label=' Inaccurate planted seeds')
    plt.scatter(x_coords1, y_coords1, c='blue', marker='o', s=1, label='Accurate planted seeds')
    plt.title('Seeding Positions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    if save_picture == True:
        plt.savefig('Seeding Positions.png',bbox_inches='tight')
    plt.show()
    return seed_positions_array,right_positions_array

# data_path ='D:/5ARIP10 Team Project/working space/AgricultureRobotics/Simulation/bestpath-1.csv'


# NOTICE
# Use the pathplanning result as the refer_path, set iteration_limit a large number (20000 or more).
# Lower target speed can result in better performance, but it takes lots of time.
# May you can run the whole simulation before and upload the result picture in GUI for the simulation result

# refer_path = pd.read_csv(data_path) # Change to the path plan result
# x_points = np.array(refer_path['x'])
# y_points = np.array(refer_path['y'])
# initial_speed = 1
# target_speed = 1.5
# iteration_limit = 1000
# noise = None
# save_picture = None
# save_animation = None #save the animation would take a long time to complete, dont recommand to use it in low target speed(less than 10),its hard to see tractor moving

# # Use this function to see the path tracking result
# x_,y_,plant_points_x,plant_points_y,plant_points_yaw, command, accuracy, final_ind = track_path(refer_path, initial_speed, target_speed, iteration_limit,noise,save_picture,save_animation)

# # Use this function to see the planting result
# spread_distance = 2 # distance between tractor and seed
# seed_positions_array,correct_positions_array = calculate_seeding_positions(x_points, y_points, command,final_ind,spread_distance,save_picture)
