import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cvxpy
import math
import pandas as pd

# mpc parameters
NX = 3  # x = [x,y,yaw]
NU = 2  # u = [v,delta_f]
T = 10  # predict horizon
R = np.diag([0.1, 0.1])  # input cost matrix
Q = np.diag([1, 1, 1])  # state cost matrix
Qf = Q  # final state matrix
dt = 0.1  # time step 
L = 2  
 
MAX_STEER = np.deg2rad(60.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(60.0)  # maximum steering speed [rad/s]
MAX_ACC = 2

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
        """
        Initialize the kinematic model.

        Args:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
            yaw (float): Initial orientation.
            v (float): Initial velocity.
            L (float): Distance between front and rear axles.
            dt (float): Time step.
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L
        self.dt = dt

    def update_state(self, a, delta_f):
        """
        Update the state of the kinematic model based on acceleration and steering angle.

        Args:
            a (float): Acceleration.
            delta_f (float): Steering angle.
        """
        self.x = self.x+self.v*math.cos(self.yaw)*self.dt
        self.y = self.y+self.v*math.sin(self.yaw)*self.dt
        delta_f = max(-MAX_STEER, min(delta_f, MAX_STEER))
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

        C = np.eye(3)
        return A, B, C

class MyReferencePath:
    """ 
    Caculate the reference control vector based on a reference path
    
    Attributes:
        x_points : A numpy array of x-coordinates of the reference path.
        y_points : A lnumpy array of y-coordinates of the reference path.
        x : A float of current x-coordinate of the tractor.
        y : A float of current y-coordinate of the tractor.
        robot_state : A tuple of Current state (x, y, yaw, v).
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

    def calc_track_error(self, x, y,index):
        """
        Calculate tracking error of current point.

        Args:
        x (float): Current x-coordinate.
        y (float): Current y-coordinate.

        Returns:
        curv (float): Current curvature.
        e (float) : Tracking error.
        ind (int) : Index of current reference point.
        """
        def normalize_angle(angle):
            """Normalize angle to be between -pi and pi."""
            while angle > np.pi:
                angle -= 2.0 * np.pi

            while angle < -np.pi:
                angle += 2.0 * np.pi

            return angle
        
        max_index = min(index + 5, len(self.refer_path) - 1)
        if max_index < index:
            index = max_index 
        search_range = self.refer_path[index:max_index + 1]
        d_x = [point[0] - x for point in search_range]
        d_y = [point[1] - y for point in search_range]
        # d_x = [self.refer_path[i, 0]-x for i in range(len(self.refer_path))]
        # d_y = [self.refer_path[i, 1]-y for i in range(len(self.refer_path))]
        d = [np.sqrt(d_x[i]**2+d_y[i]**2) for i in range(len(d_x))]
        
        ind = np.argmin(d)  
        ind += index

        yaw = self.refer_path[ind, 2]
        curv = self.refer_path[ind, 3]


        return  curv, ind

    def calc_ref_trajectory(self, robot_state,index):
        """
        Calculate reference vector.

        Args:
            robot_state (tuple): Current state of the robot (x, y, yaw, v).
            num_steps (int) : The predict horizon
            index (int) : The index of lastest reference point

        Returns:
            xref (numpy array) : Reference state inputs.
            ind (int) : Index of current reference point.
            dref (numpy array) : Reference control inputs.
        """
        curv, ind = self.calc_track_error(
            robot_state[0], robot_state[1],index)
        xref = np.zeros((NX, T + 1))
        uref = np.zeros((NU, T))
        ncourse = len(self.refer_path)
         # Control inputs (velocity and steering angle)
        ref_delta = math.atan2(L*curv, 1)
        uref[0, :] = robot_state[3]
        uref[1, :] = ref_delta
        ncourse = len(self.refer_path)
        i = 0
        # State reference
        for i in range(T + 1):
            if (ind + i) < ncourse:
                xref[0, i] = self.refer_path[ind + i, 0]
                xref[1, i] = self.refer_path[ind + i, 1]
                xref[2, i] = self.refer_path[ind + i, 2]
            else:
                xref[0, i] = self.refer_path[ncourse - 1, 0]
                xref[1, i] = self.refer_path[ncourse - 1, 1]
                xref[2, i] = self.refer_path[ncourse - 1, 2]
        return xref, ind, uref, curv

class PIDController:
    """ 
    Build a PID controller to control the speed of tractor
    
    Attributes:
        kp : Proportional gain.
        ki : Integral gain.
        kd :  Derivative gain.
        target_speed : A float of target speed of the tractor.
        integral : The accumulated integral error.
        previous_error : The error from the previous step.
    """
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

def linear_mpc_control(xref, x0, uref, tractor,num_steps):
    """
    Perform Model Predictive Control (MPC) using a linearized model.

    Parameters:
    xref (numpy array): Reference trajectory.
    x0 (numpy array): Initial state.
    uref (numpy array): Reference control inputs.
    tractor (KinematicModel): Instance of the KinematicModel class.

    Returns:
    Optimal velocity, optimal steering angle, optimal x-coordinate, optimal y-coordinate, optimal yaw angle.
    """
    def get_nparray_from_matrix(x):
        """Convert a matrix to a flattened numpy array."""
        return np.array(x).flatten()
    
    # Define optimization variables
    T = num_steps
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))
    cost = 0.0  
    constraints = []  

    t = 0
    # Loop through predict horizon
    for t in range(T):
        # Quadratic cost function for control inputs
        cost += cvxpy.quad_form(u[:, t]-uref[:, t], R)

        if t != 0:
            # Quadratic cost function for state deviation from reference
            cost += cvxpy.quad_form(x[:, t] - xref[:, t], Q)

        A, B, C = tractor.state_space(uref[1, t], xref[2, t])
        constraints += [x[:, t + 1]-xref[:, t+1] == A @
                        (x[:, t]-xref[:, t]) + B @ (u[:, t]-uref[:, t])]

    # Final state cost
    cost += cvxpy.quad_form(x[:, T] - xref[:, T], Qf)
    # Constraints
    constraints += [(x[:, 0]) == x0]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]
    # Define and solve optimization problem
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.SCS, verbose=False)

    # Check if optimization was successful
    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        opt_x = get_nparray_from_matrix(x.value[0, :])
        opt_y = get_nparray_from_matrix(x.value[1, :])
        opt_yaw = get_nparray_from_matrix(x.value[2, :])
        opt_v = get_nparray_from_matrix(u.value[0, :])
        opt_delta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = None, None, None, None, None,

    return opt_v, opt_delta, opt_x, opt_y, opt_yaw

def track_path(refer_df, initial_speed, target_speed,iteration_limit,save_picture = None,save_animation = None):
    """
    Track a reference path using Model Predictive Control (MPC).

    Args:
        refer_df (DataFrame): DataFrame containing reference path points.
        speed (float): Initial speed of the vehicle.
        iteration_limit (int): Maximum number of iterations.
        noise (bool) : Add noise to the system
        save_picture (bool) : Save the result pictures
        save_animation (bool) : Save the result animations

    Returns:
        x_ (numpy array): x-coordinate of actual path
        y_ (numpy array): y-coordinate of actual path
        measurement_accuracy (float) : the accuracy of planting seed
        plant_points_x (numpy array): x-coordinate of planting points
        plant_points_y (numpy array): y-coordinate of planting points
        plant_points_yaw (numpy array): yaw angle at planting points
        command (numpy array) : planting result
        measurement_accuracy (float): the accuracy of planting seed
        lateral_error (numpy array) : the lateral error between tractor and planting points
        longitudinal_error (numpy array) :the longitudinal error between tractor and planting points
        distance (numpy array): the distance between tractor and planting points
    """
    def adjust_num_steps(curvature):
        """
        Adjust the number of steps based on the curvature of the path.

        Args:
            curvature (float): Curvature of the path.

        Returns:
            num_steps (int): Adjusted number of steps.
        """
        if abs(curvature) < 0.01:  
            num_steps = 10  
        else: 
            num_steps = 3
        return num_steps
    
    # Input validation
    if not isinstance(refer_df, pd.DataFrame):
        raise TypeError("Input Reference path must be a pandas DataFrame")
    if refer_df.empty:
        raise ValueError("Input Reference path cannot be empty")
    start_time = time.time()
    print('Simulation start')
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
     
    tractor = KinematicModel(start_x, start_y, start_yaw, initial_speed, L, dt)   
    pid = PIDController(kp=0.5, ki=0.001, kd=0.2, target_speed=target_speed)
    x_, y_, state_x, state_y, state_yaw, output_delta, output_velocity, target_indices= [], [], [], [], [], [], [], []
    passed_measurements = 0
    index = 1
    lateral_error = []
    longitudinal_error = []
    distance = []
    travel_distance = []
    velocities = []
    plant_points_x = []
    plant_points_y = []
    plant_points_yaw = []
    overshoot = 0
    # Create a function to update the plot at each frame
    def update_plot(frame):
        plt.cla()
        plt.plot(reference_path.refer_path[:, 0], reference_path.refer_path[:, 1], "-.b", linewidth=1.0, label="Reference Path")
        plt.plot(x_[:frame], y_[:frame], "-r", label="Tracked path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.title('MPC Path Tracking')
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
        xref, target_ind, uref, curvature = reference_path.calc_ref_trajectory(
            robot_state,index)
        target_indices.append(target_ind)
        # Moving block
        num_steps = adjust_num_steps(curvature)
        # Perform MPC control
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = linear_mpc_control(
            xref, x0, uref, tractor,num_steps)
        state_x.append(tractor.x)
        state_y.append(tractor.y)
        state_yaw.append(tractor.yaw)
        output_delta.append(opt_delta[0])
        # output_velocity.append(opt_v)
        # Update vehicle state (assume the acceleration is zero)
        pid_speed_adjustment = pid.compute(tractor.v)
        acceleration = max(-MAX_ACC, min(pid_speed_adjustment, MAX_ACC))
        tractor.update_state(acceleration, opt_delta[0])  
        index = target_ind
        x_.append(tractor.x)
        y_.append(tractor.y)
        velocities.append(tractor.v * 3.6)

        # Calculate distance traveled
        if len(x_) > 1:
            travel_distance.append(travel_distance[-1] + np.sqrt((x_[-1] - x_[-2])**2 + (y_[-1] - y_[-2])**2))
        else:
            travel_distance.append(0)
        # Calculate overshoot
        if tractor.v > target_speed and tractor.v - target_speed > overshoot:
            overshoot = tractor.v - target_speed

        alpha = math.atan2(reference_path.refer_path[target_ind, 1]-robot_state[1], reference_path.refer_path[target_ind, 0]-robot_state[0])
        theta_e = alpha-tractor.yaw
        distance_to_target = np.linalg.norm(robot_state[0:2] - reference_path.refer_path[target_ind, :2])
        e_y = -distance_to_target*math.sin(theta_e)
        e_x = distance_to_target * math.cos(theta_e)
        # Check if tracktor passed the planting point (distance <= 0.08)
        if commands[target_ind] == 1 and distance_to_target <= 0.08:
            passed_measurements += 1
            plant_points_x.append(tractor.x)
            plant_points_y.append(tractor.y)
            plant_points_yaw.append(tractor.yaw)
            lateral_error.append(e_y)
            longitudinal_error.append(e_x)
            distance.append(distance_to_target)
            commands[target_ind] = 2

         # Check if the goal is reached
        if np.linalg.norm(robot_state[0:2]-goal) <= 1:
            print("reach goal")
            break

    total_measurements = sum(1 for j in commands[:index + 1] if j != 0)
    measurement_accuracy = passed_measurements / total_measurements * 100
    print(f"Measurement accuracy: {measurement_accuracy:.2f}%")
    print(f"passed and total: {passed_measurements,total_measurements}")
    print(f"Measurement accuracy: {measurement_accuracy:.2f}%")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation ran for {elapsed_time:.2f} seconds.")
    overshoot_percentage = (overshoot / target_speed) * 100
    print(f"Overshoot of velocity: {overshoot:.2f} m/s ({overshoot_percentage:.2f}%)")


    # Visualize the final result
    # Change the font in all plots
    #plt.rcParams["font.family"] = "serif"
    #plt.rcParams["font.serif"] = ["Times New Roman"]
    if (save_animation == True):
        fig = plt.figure(figsize=(8, 8))
        animation = FuncAnimation(fig, update_plot, frames=len(x_), interval=100)
        animation.save('path_tracking_animation.gif', writer='pillow')

    plt.figure(figsize=(8, 8))
    plt.plot(refer_df['x'], refer_df['y'], '-.b', linewidth=1.0, label='Reference Path')
    plt.plot(x_, y_, '-r', label='Tracked Path')
    plt.xlabel('X(meters)')
    plt.ylabel('Y(meters)')
    plt.title('Path Tracking Result')
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
    if save_picture == True:
        plt.savefig('Path Tracking Result.svg', format='svg',bbox_inches='tight')


    plt.figure(figsize=(8, 8))
    plt.plot(lateral_error,'-b')
    plt.xlabel('Time Steps')
    plt.ylabel('Lateral Error')
    plt.title('Lateral Error Over Time')


    plt.figure(figsize=(8, 8))
    plt.plot(longitudinal_error,'-b')
    plt.xlabel('Time Steps')
    plt.ylabel('Longitudinal Error')
    plt.title('Longitudinal Error Over Time')

    plt.figure(figsize=(8, 4))
    plt.plot(travel_distance,velocities, '-b', label='Velocity')
    plt.xlabel('Distance (m)')
    plt.ylabel('Velocity (km/h)')
    plt.title('Velocity Over Distance')
    plt.legend()
    plt.grid(True)
    if save_picture == True:
        plt.savefig('Velocity Profile Over Time.png',bbox_inches='tight')
    plt.show()

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
        plt.xlabel('X(meters)')
        plt.ylabel('Y(meters)')
        plt.legend()
        plt.grid(True)
        if save_picture == True:
            plt.savefig('Seeding Positions.png',bbox_inches='tight')
        plt.show()
        return seed_positions_array,right_positions_array
    spread_distance = 2 # distance between tractor and seed
    seed_positions_array,correct_positions_array = calculate_seeding_positions(x_points, y_points, commands,index,spread_distance,save_picture)
    return x_, y_ ,plant_points_x, plant_points_y, plant_points_yaw, commands, measurement_accuracy,lateral_error,longitudinal_error,distance


# data_path ='Best_path'

# # NOTICE
# # Use the pathplanning result as the refer_path, set iteration_limit a large number (20000 or more).
# # Lower target speed can result in better performance, but it takes lots of time.
# refer_path = pd.read_csv(data_path) # Change to the path plan result
# initial_speed = 0
# target_speed = 1.5 # (m/s)
# iteration_limit = 1000
# noise = None
# save_picture = None
# save_animation = None #save the animation would take a long time to complete, dont recommand to use it in low target speed(less than 10),its hard to see tractor moving

# # Use this function to see the Simulation result
# x_,y_, plant_points_x, plant_points_y, plant_points_yaw, commands,accuracy,lateral_error,longitudinal_error,distance = track_path(refer_path,initial_speed,target_speed,iteration_limit)



