#from celluloid import Camera 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cvxpy
import math
import pandas as pd

# mpc parameters
NX = 3  
NU = 2  
T = 8  
R = np.diag([0.1, 0.1])  
Q = np.diag([1, 1, 1])  
Qf = Q  


dt = 0.1  
L = 2  
 
MAX_STEER = np.deg2rad(70.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(70.0)  # maximum steering speed [rad/s]


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


class KinematicModel_3:
    def __init__(self, x, y, psi, v, L, dt):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.L = L
        self.dt = dt

    def update_state(self, a, delta_f):
        self.x = self.x+self.v*math.cos(self.psi)*self.dt
        self.y = self.y+self.v*math.sin(self.psi)*self.dt
        delta_f = max(-MAX_STEER, min(delta_f, MAX_STEER))
        self.psi = self.psi+self.v/self.L*math.tan(delta_f)*self.dt
        self.v = self.v+a*self.dt

    def get_state(self):
        return self.x, self.y, self.psi, self.v

    def state_space(self, ref_delta, ref_yaw):
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
    def __init__(self, x_points, y_points):

        self.refer_path = np.zeros((len(x_points), 4))
        self.refer_path[:, 0] = x_points
        self.refer_path[:, 1] = y_points 
        self.target_point = self.refer_path[-1, :2] 
       
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
            self.refer_path[i, 2] = math.atan2(dy, dx) 
            self.refer_path[i, 3] = (
                ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))  

    def calc_track_error(self, x, y):

        d_x = [self.refer_path[i, 0]-x for i in range(len(self.refer_path))]
        d_y = [self.refer_path[i, 1]-y for i in range(len(self.refer_path))]
        d = [np.sqrt(d_x[i]**2+d_y[i]**2) for i in range(len(d_x))]
        s = np.argmin(d)  

        yaw = self.refer_path[s, 2]
        k = self.refer_path[s, 3]
        angle = normalize_angle(yaw - math.atan2(d_y[s], d_x[s]))
        e = d[s]  
        if angle < 0:
            e *= -1

        return e, k, yaw, s

    def calc_ref_trajectory(self, robot_state, dl=1.0):
        e, k, ref_yaw, ind = self.calc_track_error(
            robot_state[0], robot_state[1])

        xref = np.zeros((NX, T + 1))
        dref = np.zeros((NU, T))
        ncourse = len(self.refer_path)

        # xref[0, 0] = self.refer_path[ind, 0]
        # xref[1, 0] = self.refer_path[ind, 1]
        # xref[2, 0] = self.refer_path[ind, 2]

        ref_delta = math.atan2(L*k, 1)
        dref[0, :] = robot_state[3]
        dref[1, :] = ref_delta
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
        # travel = 0.0

        # for i in range(T + 1):
        #     travel += abs(robot_state[3]) * dt
        #     dind = int(round(travel / dl))

        #     if (ind + dind) < ncourse:
        #         xref[0, i] = self.refer_path[ind + dind, 0]
        #         xref[1, i] = self.refer_path[ind + dind, 1]
        #         xref[2, i] = self.refer_path[ind + dind, 2]

        #     else:
        #         xref[0, i] = self.refer_path[ncourse - 1, 0]
        #         xref[1, i] = self.refer_path[ncourse - 1, 1]
        #         xref[2, i] = self.refer_path[ncourse - 1, 2]

        return xref, ind, dref


def normalize_angle(angle):

    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def linear_mpc_control(xref, x0, delta_ref, ugv):

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0  
    constraints = []  
    t = 0
    for t in range(T):
        cost += cvxpy.quad_form(u[:, t]-delta_ref[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(x[:, t] - xref[:, t], Q)

        A, B, C = ugv.state_space(delta_ref[1, t], xref[2, t])
        constraints += [x[:, t + 1]-xref[:, t+1] == A @
                        (x[:, t]-xref[:, t]) + B @ (u[:, t]-delta_ref[:, t])]


    cost += cvxpy.quad_form(x[:, T] - xref[:, T], Qf)

    constraints += [(x[:, 0]) == x0]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

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

    refer_path_np = refer_df.to_numpy()
    x_points = refer_path_np[:, 0]
    y_points = refer_path_np[:, 1]
    reference_path = MyReferencePath(x_points, y_points)
    goal = reference_path.target_point

    start_x = reference_path.refer_path[0, 0]
    start_y = reference_path.refer_path[0, 1]
    start_yaw = np.arctan2(reference_path.refer_path[1, 1] - start_y,
                           reference_path.refer_path[1, 0] - start_x)
    L = 2.0  
    dt = 0.1  
    ugv = KinematicModel_3(start_x, start_y, start_yaw, speed, L, dt)   

    x_ = []
    y_ = []
    fig = plt.figure(1)

    #camera = Camera(fig)

    for i in range(iteration_limit):
        robot_state = np.zeros(4)
        robot_state[0] = ugv.x
        robot_state[1] = ugv.y
        robot_state[2] = ugv.psi
        robot_state[3] = ugv.v
        x0 = robot_state[0:3]
        xref, target_ind, dref = reference_path.calc_ref_trajectory(
            robot_state)
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = linear_mpc_control(
            xref, x0, dref, ugv)
        ugv.update_state(0, opt_delta[0])  

        x_.append(ugv.x)
        y_.append(ugv.y)

        plt.cla()
        plt.plot(reference_path.refer_path[:, 0], reference_path.refer_path[:,
                 1], "-.b",  linewidth=1.0, label="course")
        plt.plot(x_, y_, "-r", label="trajectory")
        plt.plot(reference_path.refer_path[target_ind, 0],
                 reference_path.refer_path[target_ind, 1], "go", label="target")
        # plt.axis("equal")
        # plt.grid(True)
        #plt.pause(0.001)

        if np.linalg.norm(robot_state[0:2]-goal) <= 0.1:
            print("reach goal")
            break
    # animation = camera.animate()
    # animation.save('trajectory.gif')
    plt.figure(figsize=(10, 6))
    plt.plot(refer_df['x'], refer_df['y'], '-.b', linewidth=1.0, label='Reference Path')
    plt.plot(x_, y_, '-r', label='Tracked Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path Tracking MPC')
    plt.legend()
    plt.axis("equal")
    # plt.grid(True)
    plt.savefig("filter/simulation_field_mpc.png")


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
# refer_df = pd.DataFrame(data)
# speed = 1.5
# iteration_limit = 500




# track_path(refer_df,speed,iteration_limit)