import numpy as np
import matplotlib
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
 
MAX_STEER = np.deg2rad(70.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(70.0)  # maximum steering speed [rad/s]
MAX_ACC = 0.5

def get_nparray_from_matrix(x):
    """Convert a matrix to a flattened numpy array."""
    return np.array(x).flatten()

def normalize_angle(angle):
    """Normalize angle to be between -pi and pi."""
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

class KinematicModel:
    """
    Build the KinematicModel of the tractor
    
    Attributes:
        x : A float of initial x-coordinate of the tractor.
        y : A float of initial y-coordinate of the tractor.
        yaw : A float of initial yaw angle of the tractor.
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
        Update the state of the kinematic model.

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

    def calc_track_error(self, x, y):
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
        d_x = [self.refer_path[i, 0]-x for i in range(len(self.refer_path))]
        d_y = [self.refer_path[i, 1]-y for i in range(len(self.refer_path))]
        d = [np.sqrt(d_x[i]**2+d_y[i]**2) for i in range(len(d_x))]
        ind = np.argmin(d)  

        yaw = self.refer_path[ind, 2]
        curv = self.refer_path[ind, 3]
        angle = normalize_angle(yaw - math.atan2(d_y[ind], d_x[ind]))
        e = d[ind]  
        if angle < 0:
            e *= -1
        return e, curv, ind

    def calc_ref_trajectory(self, robot_state):
        """
        Calculate reference vector.

        Args:
            robot_state (tuple): Current state of the robot (x, y, psi, v).
            num_steps (int) : The predict horizon

        Returns:
            xref (numpy array) : Reference state inputs.
            ind (int) : Index of current reference point.
            dref (numpy array) : Reference control inputs.
        """
        e,curv, ind = self.calc_track_error(
            robot_state[0], robot_state[1])
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

def adjust_num_steps(curvature):
    """
    Adjust the number of steps based on the curvature of the path.

    Args:
        curvature (float): Curvature of the path.

    Returns:
        num_steps (int): Adjusted number of steps.
    """
    if abs(curvature) < 0.1:  
        num_steps = 10  
    else: 
        num_steps = 3
    return num_steps

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



def track_path(refer_df, speed, iteration_limit):
    """
    Track a reference path using Model Predictive Control (MPC).

    Args:
        refer_df (DataFrame): DataFrame containing reference path points.
        speed (float): Initial speed of the vehicle.
        iteration_limit (int): Maximum number of iterations.

    Returns:
        x_ (numpy array): x-coordinate of actual path
        y_ (numpy array): y-coordinate of actual path
        measurement_accuracy (float) : the accuracy of planting seed
    """
    # Input validation
    if not isinstance(refer_df, pd.DataFrame):
        raise TypeError("refer_path must be a pandas DataFrame")
    if refer_df.empty:
        raise ValueError("refer_path cannot be empty")
    
    x_points = np.array(refer_df['x'])
    y_points = np.array(refer_df['y'])
    commands = np.array(refer_df['command'])
    reference_path = MyReferencePath(x_points, y_points)
    goal = reference_path.target_point

    start_x = reference_path.refer_path[0, 0]
    start_y = reference_path.refer_path[0, 1]
    start_yaw = np.arctan2(reference_path.refer_path[1, 1] - start_y,
                           reference_path.refer_path[1, 0] - start_x)
    L = 2.0  
    dt = 0.1  
    tractor = KinematicModel(start_x, start_y, start_yaw, speed, L, dt)   
    pid = PIDController(kp=0.1, ki=0.01, kd=0.05, target_speed=10.0)
    x_ = []
    y_ = []
    passed_measurements = 0
    total_measurements = np.sum(commands)
    state_x = []
    state_y = []
    state_yaw = []
    output_delta = []
    target_indices = []
    output_velocity = []
    # Create a function to update the plot at each frame
    def update_plot(frame):
        plt.cla()
        plt.plot(reference_path.refer_path[:, 0], reference_path.refer_path[:, 1], "-.b", linewidth=1.0, label="Reference Path")
        plt.plot(x_[:frame], y_[:frame], "-r", label="Tracked path")
        # if frame < len(target_indices):
        #     current_target = target_indices[frame]
        #     plt.plot(reference_path.refer_path[current_target, 0], reference_path.refer_path[current_target, 1], "go", label="target")
        # plt.plot(reference_path.refer_path[target_ind, 0], reference_path.refer_path[target_ind, 1], "go", label="target")
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
            robot_state)
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

        x_.append(tractor.x)
        y_.append(tractor.y)

        # Check if tracktor passed the planting point (distance <= 0.5)
        if commands[target_ind] == 1 and np.linalg.norm(robot_state[0:2] - reference_path.refer_path[target_ind, :2]) <= 0.5:
            passed_measurements += 1
            commands[target_ind] = 0

         # Check if the goal is reached
        if np.linalg.norm(robot_state[0:2]-goal) <= 1:
            print("reach goal")
            break
    #fig = plt.figure(figsize=(8, 8))
    #animation = FuncAnimation(fig, update_plot, frames=len(x_), interval=100)

    # Save the animation as a GIF
    #animation.save('D:/5ARIP10 Team Project/working space/AgricultureRobotics/Simulation/MPC_path_tracking_animation.gif', writer='pillow')   
        
    measurement_accuracy = passed_measurements / total_measurements * 100
    print(f"Measurement accuracy: {measurement_accuracy:.2f}%")
    input_df = pd.DataFrame({
        'robot_x':state_x,
        'robot_y':state_y,
        'robot_yaw':state_yaw
    })
    output_df = pd.DataFrame({
        'steering angle':output_delta
    })
    # input_df.to_csv('D:/5ARIP10 Team Project/working space/AgricultureRobotics/Simulation/input.csv',index=True)
    # output_df.to_csv('D:/5ARIP10 Team Project/working space/AgricultureRobotics/Simulation/output.csv',index=True)
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
    #plt.show()
    plt.savefig("filter/simulation_field_mpc.png")

    return x_, y_ , measurement_accuracy

def store_path_data(x_, y_):
    path_df = pd.DataFrame({
        'x': x_,
        'y': y_
    })
    return path_df


def path_tracking_result(data_path,speed,iteration_limit):

    refer_path = pd.read_csv(data_path)
    x_,y_, accuracy = track_path(refer_path,speed,iteration_limit)
    path_df = store_path_data(x_, y_)
    return path_df

# data_path ='D:/5ARIP10 Team Project/working space/AgricultureRobotics/Simulation/bestpath.csv'
# speed = 5
# iteration_limit = 7000

# actual_path = path_tracking_result(data_path,speed,iteration_limit)



