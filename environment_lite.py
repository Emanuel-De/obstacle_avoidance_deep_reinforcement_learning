# coding=utf-8
import time
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class cls_Environment:
    """
    The environment class
    """
    def __init__(self):
        """init function, called then object is created"""
        self._t0 = time.clock()  # get current time

        # Runtime variables
        self._runtime_max = 0  # initialise variable for the number of steps until success
        self._stepsize = 0.1  # step size. The sampling rate how often the agent "sees" the environment between its
        #  actions
        self._ode_iterations = 100  # variable for the number of iteration the odeint does.
        self._ode_variables = 10  # variable to define the end of an array for the odeint results.

        # definition of road
        self._road_width_right = -10  # total road with in forward direction
        self._road_width_left = 0  # total road with in oncoming direction
        self._number_of_lanes_right = 2  # number of lanes in the forward direction
        self._number_of_lanes_left = 0  # number of lanes in the oncoming direction
        self._obstacle_width = 0  # negativ because directional vector of obstacle is negativ

        # car characteristic
        self._car_length_front = 1.6  # length from the center of gravity to the front wheels of the car
        self._car_length_back = 1.4  # length from the center of gravity to the back wheels of the car
        self._car_side_slip_rigidity_back = 150000  # car side slip rigidity in the back
        self._side_slip_rigidity_front = self._car_side_slip_rigidity_back * 0.6  # car side slip rigidity in the front
        self._car_inertia = 4000 #1000  # car inertia
        self._car_mass = 2000 #500 # mass of the car
        self._msr = 50.0  # max sensor range, how far the agent can "see"

        # sensor information
        self._number_of_sections = 13  # number of sections per side
        self._sections = [6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6]  # array of sensors indexes
        self._angle = 3  # degress of how wide each sections is

        # initial values
        self._init_lane = 0.0  # initialise the variable for the initial lane
        self._agent_pos_global_init = np.array([0.0, self._init_lane])  # get the agents initial position using the
        # init_lane function
        self._x_value_obstacle = 0  # initialise the variable for the x coordiante of the obstacle
        self._free_yaw_befor_after_obst = 30  # define how many maters before the obstacle the agent is allowed to
        # have a greater yaw angle
        self._y_values_obstacle1 = 0  # initialise a variable for the y coordiante of the obstacle
        self._y_values_obstacle2 = 0  # initialise a variable for the y coordiante of the obstacle
        self._y_values_obstacle = 0  # initialise a variable for the y coordiante of the obstacle
        self._agent_yaw_global_init = 0.0  # variable for the initial yaw angle
        self._speed_init = 5.0  # variable for the initial speed in m/s
        self._delta_speed = 0.5  # variable for the rate with which the agent changes the speed. Only
        # relevant to experiments with speed control
        self._steering_angle_init = 0  # variable for the initial steering angle
        self._delta_steering_angle = 0.5  # variable for the rate with which the agent changes the
        # steering angle

        # decay options
        self._speed_decay = 0  # speed decay rate. How much the agent is slowed down at each step
        self._speed_decay_offset = 1  # value to make sure speed is not negativ
        # ^^ agent cant deal with this concept. doesnt learn to speed up.

        # changeable variables
        self._agent_pos_global = self._agent_pos_global_init  # initialise a variable for the global position of
        # the agent
        self._agent_yaw_global = 0.0  # initialise a variable for the current yaw angle in rad
        self._agent_speed = self._speed_init  # initialise a variable for the current speed
        self._steering_angle = self._steering_angle_init  # initialise a variable for the current steering angle
        self._current_runtime = 0.0  # counter for the number of steps

        # win, lose parameters
        self._dist_to_crash_right = -10.0  # define how far right the agent is allowed to go before crashing
        self._dist_to_crash_left = 0.0  # define how far left the agent is allowed to go before crashing
        self._dist_to_crash_x = 7.1  # distance to obstacle after which agent crashes in [m]
        self._dist_to_crash_y = 0.2  # dist left and right of obst that cause crash in [m]
        self._win_reward = 100  # reward granted if agent wins
        self._lose_reward = -100  # reward granted if agent lose
        self._min_dist_to_success = 10  # distance to cover to be able to win
        self._speed_crash = 3.0  # min speed. if agent goes slower then crash
        self._speed_crash_max = 7.0 # max speed. if agent goes faster then crash
        self._yawangle_crash = 1.5708  # max yaw angle in rad before the obstacle (1.5708 = 90°)
        # defined in self._free_yaw_befor_after_obst
        self._yawangle_crash_open_road = 1.5708 #0.139626  # 0.174533  # max yaw on the open road (0.174533 = 10°; 0.139626 = 8°)

        # misc variables
        self._agent_pos_global_holder = np.array([[0, 0]])  # initialise holder to hold the agents position in order
        # to be plot the path the agent took
        self._max_speed = 16.0  # agent cant go faster than this value
        self._min_speed = 1.0  # agent cant fo slower than this
        self._max_steering_angle = [-20.0, 20.0]  # max and min steering angle. agent cant steer further than this

    def reset(self):
        """reset function to reset the environment and deliver initial state"""
        self._agent_pos_global_holder = np.array([[0, 0]])  # reset holder to hold the agents position in order
        # to be plot the path the agent took
        self._current_runtime = 0  # reset counter for the number of steps

        self._runtime_max = np.random.randint(30, 45)  # randomly (within limits) define how many steps the agent has
        # to take for it to be successful for this episode
        self._init_lane = self._pos_agent_initial_lane()  # value in y direction where the agent and the obst
        # are at init
        self._agent_pos_global_init = np.array([0, self._init_lane])  # put agent in starting place
        self._agent_pos_global = self._agent_pos_global_init  # variable for the current position of the agent.
        # Initialised with the initial position
        self._obstacle_width = - (np.random.randint(2, 5) + np.random.random())  # randomly (within limits) define how
        # wide the obstacle should be for this episode
        self._x_value_obstacle = np.random.randint(30, 120)  # randomly (within limits) define how far away the obstacle
        # should be
        # randomly define in what lane the obstacle should be
        self._y_values_obstacle1 = self._pos_agent_initial_lane() - (self._obstacle_width / 2)
        self._y_values_obstacle2 = self._init_lane - (self._obstacle_width / 2) #
        self._y_values_obstacle = np.random.choice(a=[self._y_values_obstacle1, self._y_values_obstacle2], p=[0.8, 0.2])
        self._agent_yaw_global = 0.0 # current yaw angle of the agent
        self._agent_speed = self._speed_init  # current speed initialised with the initial speed
        self._steering_angle = self._steering_angle_init  # current steering angle initialised with the initial
        # steering angle

        dist_vec = self._calc_dist_to_boundary()  # calculate the distance to the surroundings. Boundary and obstacle
        s0 = self._build_state(dist_vec)  # build the state with dist measurements

        # append position to holder. delete the first entry to delete the move from [0,0] to init position
        self._agent_pos_global_holder = np.append(self._agent_pos_global_holder, [self._agent_pos_global_init], axis=0)
        self._agent_pos_global_holder = np.delete(self._agent_pos_global_holder, 0, 0)

        runtime = self._get_run_time()  # get current time

        return s0, self._agent_pos_global, dist_vec, runtime  # return values to caller

    def step(self, nn_action):
        """step function to "move" through the episode"""
        _ = self._execute_action(nn_action)  # call function to execute the action

        dist_vec = self._calc_dist_to_boundary()  # calculate the distances to surroundings

        done, done_reason = self._check_done(dist_vec)  # check if ep is terminated

        s_next = self._build_state(dist_vec)  # build the next state

        reward = self._calc_reward(done)  # calc the reward for transition

        # some housekeeping and setting the done return variable
        if done == 0:
            done_info = 0
            done = False
        elif done != 0:
            if done == 1: done_info = 1
            elif done == -1: done_info = -1
            else: done_info = 0
            done = True

        self._current_runtime += self._stepsize  # increment the number of steps taken

        self._agent_pos_global_holder = np.append(self._agent_pos_global_holder, [self._agent_pos_global], axis=0)
        # append position to be able to plot

        runtime = self._get_run_time()  # get current time

        return s_next, reward, done, self._agent_pos_global, dist_vec, runtime, done_info, done_reason
        # return the next state, the reward "done" information and some debug and stat variables.

    def render(self):
        """function to render the path of the agent"""
        space_on_side = 2

        def fcn(p_vec, gamma, d_vec):
            point = p_vec + gamma * d_vec
            return point

        point = [0, 0]
        road_right = np.array([[0, self._dist_to_crash_right]])
        road_left = np.array([[0, self._dist_to_crash_left]])

        if (self._agent_pos_global[0] + self._msr) < self._x_value_obstacle:
            length = self._x_value_obstacle + 10
        else:
            length = self._agent_pos_global[0] + self._msr
        gamma = 1
        while gamma < length:
            point = fcn(np.array([0, self._dist_to_crash_right]), gamma, np.array([1, 0]))
            road_right = np.append(road_right, [point], axis=0)
            point = fcn(np.array([0, self._dist_to_crash_left]), gamma, np.array([1, 0]))
            road_left = np.append(road_left, [point], axis=0)
            gamma += 1

        p_obst = np.array([self._x_value_obstacle, self._y_values_obstacle])
        obst = np.array([[self._x_value_obstacle, self._y_values_obstacle]])
        gamma = 0.2
        while gamma < abs(self._obstacle_width):
            point = fcn(p_obst, gamma, np.array([0, -1]))
            obst = np.append(obst, [point], axis=0)
            gamma += 0.2

        ray_holder = []

        yawangle = self._agent_yaw_global
        p_agent = self._agent_pos_global

        for idx, s in enumerate(self._sections):
            alpha = math.radians(s * self._angle) + yawangle
            d_agent_ray = np.array([np.cos(alpha), np.sin(alpha)])
            ray =  np.array([p_agent])
            point = p_agent
            gamma = 1

            while (np.sqrt((point[0]-p_agent[0])**2 + (point[1]-p_agent[1])**2)) < self._msr:
                temp = np.sqrt((point[0]-p_agent[0])**2 + (point[1]-p_agent[1])**2)
                point = fcn(p_agent, gamma, d_agent_ray)
                ray = np.append(ray, [point], axis=0)
                gamma += 1
            ray_holder.append(ray)

        plt.xlim(self._dist_to_crash_left + space_on_side, self._dist_to_crash_right - space_on_side)

        for ray in ray_holder:
            plt.plot(ray[:, 1], ray[:, 0], 'b')

        plt.plot(self._agent_pos_global_holder[:, 1], self._agent_pos_global_holder[:, 0], 'g')
        plt.plot(road_right[:, 1], road_right[:, 0], 'r')
        plt.plot(road_left[:, 1], road_left[:, 0], 'r')
        plt.plot(obst[:, 1], obst[:, 0], 'r')
        plt.grid()
        plt.show()

        return 0

    def _calc_reward(self, done):
        """calculate the reward"""
        # done == 0: all ok; done == -1: fail; done == 1: win

        # on crash reward: lose reward
        if done == -1:
            reward = self._lose_reward

        # reward if no crash and time is up and a min of progress has been made
        elif done == 1:
            reward = self._win_reward

        else:
            reward = 0

        return reward

    def _build_state(self, dist_vec):
        """build state from x, y coordinates to matrix vector"""
        number_var = self._number_of_sections + 6
        nn_input_vec = np.full(number_var, 0, dtype=np.float32)

        length_dist_vec = len(dist_vec[1])
        for idx in range(length_dist_vec):
            nn_input_vec[idx] = dist_vec[1, idx]

        def curr_lane():
            """function to find current lane"""
            if 0 > self._agent_pos_global[1] > self._road_width_right:
                curr_lane_type = 0
            elif 0 < self._agent_pos_global[1] < self._road_width_left:
                curr_lane_type = 10
            else:
                curr_lane_type = 20
            return curr_lane_type

        def lane_left():
            """function to find left lane"""
            if self._number_of_lanes_left == 0:
                lane_to_left = self._agent_pos_global[1] - (self._road_width_right / self._number_of_lanes_right)
            else:
                lane_to_left = self._agent_pos_global[1] + (self._road_width_left / self._number_of_lanes_left)
            if 0 > lane_to_left > self._road_width_right:
                lane_type_left = 0
            elif 0 < lane_to_left < self._road_width_left:
                lane_type_left = 10
            else:
                lane_type_left = 20
            return lane_type_left

        def lane_right():
            """function to find right lane"""
            lane_to_right = self._agent_pos_global[1] + (self._road_width_right / self._number_of_lanes_right)
            if 0 > lane_to_right > self._road_width_right:
                lane_type_right = 0
            elif 0 < lane_to_right < self._road_width_left:
                lane_type_right = 10
            else:
                lane_type_right = 20
            return lane_type_right

        nn_input_vec[length_dist_vec + 0] = curr_lane()  # current lane type
        nn_input_vec[length_dist_vec + 1] = lane_left()  # lane type left
        nn_input_vec[length_dist_vec + 2] = lane_right()  # lane type right
        nn_input_vec[length_dist_vec + 3] = self._agent_speed  # current speed
        nn_input_vec[length_dist_vec + 4] = self._steering_angle  # current steering angle
        nn_input_vec[length_dist_vec + 5] = self._agent_yaw_global  # Yaw angle

        return nn_input_vec

    def _check_done(self, dist_vec):
        """check if episode has to terminate"""
        # done == 0: all ok; done == -1: fail; done == 1: win
        p_obst = np.array([self._x_value_obstacle, self._y_values_obstacle])
        reason = 'unknown'
        done = 0

        # Check if dist to obst is too low
        if ((p_obst[1] + self._dist_to_crash_y) > self._agent_pos_global[1] >
            (p_obst[1] + self._obstacle_width - self._dist_to_crash_y)) and ((p_obst[0] - self._dist_to_crash_x) <
                                                                             self._agent_pos_global[0] < p_obst[0]):
            done = -1
            return done, 'crash obst'

        # hit left or right road limit
        if (self._agent_pos_global[1] > self._dist_to_crash_left) or (self._agent_pos_global[1] <
                                                                      self._dist_to_crash_right):
            done = -1
            return done, 'hit boundry'

        # limit yaw angle in general but allow for greater yaw angle around obstacle
        if (self._x_value_obstacle - self._free_yaw_befor_after_obst) < self._agent_pos_global[0] < \
                (self._x_value_obstacle + self._free_yaw_befor_after_obst):
            if (self._agent_yaw_global > self._yawangle_crash) or (self._agent_yaw_global < (-self._yawangle_crash)):
                done = -1
                return done, 'yaw angle around obst'
        elif (self._agent_yaw_global > self._yawangle_crash_open_road) or (self._agent_yaw_global <
                                                                           (-self._yawangle_crash_open_road)):
                done = -1
                return done, 'yaw angle open road'

        # if agent becomes too slow
        if self._agent_speed < self._speed_crash:
            done = -1
            return done, 'speed to low'

        # if agent becomes too fast
        if self._agent_speed > self._speed_crash_max:
            done = -1
            return done, 'speed to high'

        # if min dist is not covered when time is up
        if (self._current_runtime > self._runtime_max) and (self._agent_pos_global[0] < self._min_dist_to_success):
            done = -1
            return done, 'min dist to win not coverd at runtime'

        if (self._current_runtime > self._runtime_max) and (self._agent_pos_global[0] > self._min_dist_to_success) \
                and (done != -1):
            done = 1
            return done, 'WIN'

        return done, reason

    def _execute_action(self, action):
        """function to execute the action. calling the function to calc motion"""
        if action == 3: #0
            if self._agent_speed < self._max_speed:
                self._agent_speed += self._delta_speed
            delta_x, delta_y, psi = self._execute_motion(self._agent_speed, self._steering_angle)
        elif action == 4: #1
            if self._agent_speed > self._min_speed:
                self._agent_speed -= self._delta_speed
            delta_x, delta_y, psi = self._execute_motion(self._agent_speed, self._steering_angle)
        elif action == 1: #2 # left (positiv angle)
            if self._steering_angle < self._max_steering_angle[1]:
                self._steering_angle += self._delta_steering_angle
            delta_x, delta_y, psi = self._execute_motion(self._agent_speed, self._steering_angle)
        elif action == 2: #3 # right (negativ angle)
            if self._steering_angle > self._max_steering_angle[0]:
                self._steering_angle -= self._delta_steering_angle
            delta_x, delta_y, psi = self._execute_motion(self._agent_speed, self._steering_angle)
        elif action == 0: #4
            delta_x, delta_y, psi = self._execute_motion(self._agent_speed, self._steering_angle)
        else:
            delta_x, delta_y, psi = 0, 0, 0

        if self._agent_speed > (self._speed_decay + self._speed_decay_offset):
            self._agent_speed -= self._speed_decay
            self._agent_speed = float("{0:.4f}".format(self._agent_speed))

        # update the global class variables
        delta_pos = self._cos_transformation(delta_x, delta_y, self._agent_yaw_global)
        self._agent_pos_global = delta_pos
        self._agent_yaw_global += psi

        return 1

    def _cos_transformation(self, local_x, local_y, psi):
        """function for coordinate transformation"""
        T = np.array([[np.cos(psi), -np.sin(psi), self._agent_pos_global[0]],
                      [np.sin(psi), np.cos(psi), self._agent_pos_global[1]],
                      [0, 0, 1]])

        pos_loc = np.array([local_x, local_y, 1])
        pos_global = T.dot(pos_loc)

        return_val = np.array([pos_global[0], pos_global[1]])
        return return_val

    def _calc_dist_to_boundary(self):
        """function to calculate the distance to the surroundings"""
        dist_holder = np.zeros([2, self._number_of_sections])

        p_obst = np.array([self._x_value_obstacle, self._y_values_obstacle])
        p_agent = self._agent_pos_global

        yawangle = self._agent_yaw_global

        d_agent_orth = np.array([-np.sin(yawangle), np.cos(yawangle)]) # positiv 90° rotation

        p_road_limit_left = np.array([0, self._dist_to_crash_left])
        p_road_limit_right = np.array([0, self._dist_to_crash_right])

        b_obst = p_obst - p_agent
        b_road_left = p_road_limit_left - p_agent
        b_road_right = p_road_limit_right - p_agent

        def calc_intersection(a, b):
            """calculate the x, y of the intersection of two lines"""
            try:
                multipliers = np.linalg.solve(a, b)
                intersection = p_agent + d_agent_ray * multipliers[0]
                return intersection
            except np.linalg.linalg.LinAlgError:
                return np.array(['e', 'e'])

        def calc_dist(intersection):
            """calculate the distance to the intersection"""
            sensor_ray = intersection - p_agent
            dist = np.sqrt(sensor_ray[0]**2 + sensor_ray[1]**2)
            return dist

        def calc_intersec_side(intersection):
            """calculate if the intersection is ahed or behind"""
            side = d_agent_orth[1] * (intersection[0] - p_agent[0]) + d_agent_orth[0] * (intersection[1] - p_agent[1])
            return side

        def append_dist_obst(idx, dist):
            """append the dist to obstacle to the holder"""
            if dist < self._msr:
                dist_holder[0, idx] = dist
            else:
                dist_holder[0, idx] = self._msr

        def append_dist(idx, dist):
            """append the dist to boundary to the holder"""
            if dist < self._msr:
                dist_holder[1, idx] = dist
            else:
                dist_holder[1, idx] = self._msr

            if dist_holder[1, idx] > dist_holder[0, idx]:
                dist_holder[1, idx] = dist_holder[0, idx]

        for idx, s in enumerate(self._sections):
            alpha = math.radians(s * self._angle) + yawangle
            d_agent_ray = np.array([np.cos(alpha), np.sin(alpha)])

            a_obst = np.array([[np.cos(alpha), 0], [np.sin(alpha), 1]])

            intersection_obst = calc_intersection(a_obst, b_obst)
            if intersection_obst[0] != 'e' and (calc_intersec_side(intersection_obst) > 0):
                if p_obst[1] > intersection_obst[1] > (p_obst[1] + self._obstacle_width):
                    dist = calc_dist(intersection_obst)
                    append_dist_obst(idx, dist)
                else: dist_holder[0, idx] = self._msr
            else: dist_holder[0, idx] = self._msr

            a_road = np.array(np.array([[np.cos(alpha), -1], [np.sin(alpha), 0]]))
            intersection_road_left = calc_intersection(a_road, b_road_left)
            if intersection_road_left[0] != 'e' and (calc_intersec_side(intersection_road_left) > 0):
                dist = calc_dist(intersection_road_left)
                append_dist(idx, dist)
            elif intersection_road_left[0] == 'e': append_dist(idx, dist)

            intersection_road_right = calc_intersection(a_road, b_road_right)
            if intersection_road_right[0] != 'e' and (calc_intersec_side(intersection_road_right) > 0):
                dist = calc_dist(intersection_road_right)
                append_dist(idx, dist)
            elif intersection_road_left[0] == 'e': append_dist(idx, dist)

        return dist_holder

    def _execute_motion(self, v, delta_degree):
        """function to execute the motion of a single track model with an odeint"""
        if v == 0:
            return 0, 0, 0

        delta = math.radians(delta_degree)

        # side slip rigidity
        s_s_r_f = self._side_slip_rigidity_front * np.cos(delta)
        s_s_r_b = self._car_side_slip_rigidity_back

        # car sizing
        l_f = self._car_length_front
        l_b = self._car_length_back

        # inertia and mass
        c_i = self._car_inertia
        c_m = self._car_mass

        # Differential equation
        # psipp = ((((np.arctan((v * np.sin(beta) - l_f * psip) / (v * np.cos(beta)))) + delta) * s_s_r_f * l_f) -
        # ((np.arctan((v * np.sin(beta) - l_f * psip) / (v * np.cos(beta)))) * s_s_r_b * l_b)) / c_i
        # psip = yawrate
        # yawratep = ((((np.arctan((v * np.sin(beta) - l_f * yawrate) / (v * np.cos(beta)))) + delta) * s_s_r_f * l_f) -
        # ((np.arctan((v * np.sin(beta) - l_f * yawrate) / (v * np.cos(beta)))) * s_s_r_b * l_b)) / c_i

        # functions of derivatives
        def dy_fn_psi(y, t, v, delta, l_f, l_b, s_s_r_f, s_s_r_b, c_i, c_m):
            yawangle, yawrate, sideslipangle, dist = y

            alpha_front = (np.arctan((v * np.sin(sideslipangle) - l_f * yawrate) / (v * np.cos(sideslipangle)))) + delta
            alpha_back = np.arctan((v * np.sin(sideslipangle) + l_b * yawrate) / (v * np.cos(sideslipangle)))
            F_y_front = alpha_front * s_s_r_f
            F_y_back = alpha_back * s_s_r_b
            a_y = (F_y_front + F_y_back) / c_m

            dydt = [yawrate,
                    (F_y_front * self._car_length_front - F_y_back * self._car_length_back) / c_i,
                    yawrate - (a_y * np.cos(sideslipangle) / v),
                    v]

            return dydt

        # initial integration values
        t = np.linspace(0, self._stepsize, (self._stepsize * self._ode_iterations + 1))
        y0 = [0.0, 0.0, 0.0, 0.0]

        # call of the ode solver
        sol_psi = odeint(dy_fn_psi, y0, t, args=(v, delta, l_f, l_b, s_s_r_f, s_s_r_b, c_i, c_m))

        # sol_psi[:,0] = yaw angle
        # sol_psi[:,1] = yaw rate
        # sol_psi[:,2] = slip angle
        # sol_psi[:,3] = abs(dist)

        # process return variables
        psi = sol_psi[self._ode_variables, 0]
        dist = sol_psi[self._ode_variables, 3]
        delta_x = np.cos(psi) * dist
        delta_y = np.sin(psi) * dist

        return delta_x, delta_y, psi  # psi in radians

    def _pos_agent_initial_lane(self):
        """determine an initial lane"""
        lane_array = []
        position = (self._road_width_right / self._number_of_lanes_right) / 2
        for i in range(self._number_of_lanes_right):
            lane = position + i * (self._road_width_right / self._number_of_lanes_right)
            lane_array.append(lane)

        np_lane_array = np.array(lane_array)
        initial_lane_pos = np.random.choice(np_lane_array)
        return initial_lane_pos

    def _get_run_time(self):
        """get current time"""
        act_time_temp = time.clock() - self._t0
        return act_time_temp
