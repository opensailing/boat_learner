from boat_simulation import wrap_phase, speed_interp, angle_difference
from gymnasium import spaces, Env
import numpy as np
from scipy.special import erf
from collections import deque


class MultiMarkNoModelEnv(Env):
    def __init__(self, config, dt=15, seq_size=2, bounds=None, target_phase_steps=4, heading_phase_steps=4, radius_multipliers=[1], target_phase_probabilities=None, heading_phase_probabilities=None):
        if bounds is None:
            self.MIN_X = -250
            self.MAX_X = 250
            self.MIN_Y = 0
            self.MAX_Y = 250
        else:
            self.MIN_X = bounds[0]
            self.MAX_X = bounds[1]
            self.MIN_Y = bounds[2]
            self.MAX_Y = bounds[3]
        self.MAX_SPEED = 10
        self.MAX_MARKS = config['max_marks']
        self.MAX_REMAINING_SECONDS = config['max_seconds_per_leg']
        self.LEG_RADIUS = config['leg_radius']
        self.target_x = np.zeros((self.MAX_MARKS,))
        self.target_y= np.zeros((self.MAX_MARKS,))
        self.current_mark = 0
        self.heading_change = 0
        self.speed = 0

        self.vmg_history = []
        self.heading_change_history = []
        self.vmg = 0
        self.heading = 0
        self.twa = 0
        self.target_tolerance_multiplier = config['target_tolerance_multiplier']
        self.TARGET_PHASE_STEPS = target_phase_steps
        self.HEADING_PHASE_STEPS = heading_phase_steps
        self.radius_multipliers = radius_multipliers
        self.target_phase_probabilities = target_phase_probabilities
        self.heading_phase_probabilities = heading_phase_probabilities
        self.MAX_EFFECTIVE_TACKS = 3
        self.BOAT_LENGTH = 7

        self.actions = np.array(config['actions'])

        self.plot_fn = config['plot_fn']

        self.trajectory = []

        self.MAX_DISTANCE = np.sqrt((self.MAX_X - self.MIN_X) ** 2 + (self.MAX_Y - self.MIN_Y) ** 2)

        self.seq_size = seq_size
        self.obs_size = 11

        self.observation = np.zeros((self.seq_size, self.obs_size,))
        num_actions = self.actions.shape[0]
        # self.action_space = spaces.Box(low=-1, high=1) # spaces.Discrete(num_actions)
        self.action_space = spaces.Discrete(num_actions)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.seq_size, self.obs_size,))
        self.reward_range = spaces.Box(low=-10, high=10, shape=())
        self.dt = dt
        self.has_reached_mark = False
        self.has_entered_mark_zone = False
        self.prev_distance = 0
        # self.pid_controller = PIDController(2, 0.1, 0.001)

    def reset(self, seed = None, options = None, heading=None, speed=None, vmg=None):
        super().reset(seed=seed)
        self.observation = np.zeros((self.seq_size, self.obs_size,))

        self.speeds = deque(maxlen=2)

        # self.pid_controller.reset()
        # Initialization logic
        # Initialize state variables: x, y, speed, etc.
        # Return the initial observation

        # Always start at 0.0, and generate random target marks from there
        if self.trajectory != []:
            bounds = (self.MIN_X, self.MAX_X, self.MIN_Y, self.MAX_Y)
            marks = [None] * self.MAX_MARKS
            for i in range(self.MAX_MARKS):
                marks[i] = (self.target_x[i], self.target_y[i])
            self.plot_fn(self.trajectory, marks, bounds)

        self.x = 0.0
        self.y = 0.0

        current_x = 0
        current_y = 0

        self.current_radius_multiplier = np.zeros((self.MAX_MARKS,))
        for i in range(self.MAX_MARKS):
            radius_multiplier = np.random.choice(self.radius_multipliers)
            self.current_radius_multiplier[i] = radius_multiplier

            phase = self.random_phase(self.TARGET_PHASE_STEPS, offset=np.pi/2, probabilities=self.target_phase_probabilities)
            current_x = current_x + np.cos(phase) * self.LEG_RADIUS * radius_multiplier
            current_y = current_y + np.sin(phase) * self.LEG_RADIUS * radius_multiplier
            self.target_x[i] = current_x
            self.target_y[i] = current_y

        rand_heading = self.random_phase(self.HEADING_PHASE_STEPS)
        self.heading = rand_heading if heading is None else heading
        self.twa = 0
        self.current_mark = 0
        self.tack_count = 0
        self.prev_heading = self.heading
        self.angle_to_mark = 0
        self.speed = 0 if speed is None else speed
        self.vmg = 0 if vmg is None else vmg
        self.vmg_history = []
        self.has_tacked = False
        self.remaining_seconds = self.MAX_REMAINING_SECONDS
        self.delta_t = 0
        self.reward = 0
        self.heading_change = 0
        self.heading_change_history = []
        self.has_reached_mark = False

        self.target_heading = self.heading

        self.distance = np.sqrt((self.target_x[0] - self.x) ** 2 + self.target_y[0] ** 2)
        self.prev_distance = self.distance
        self.initial_distance = self.distance
        self.min_distance = self.distance

        self.has_missed_mark = False

        self.current_target_x = self.target_x[0]
        self.current_target_y = self.target_y[0]

        orientation_coeff = self.calculate_boat_relative_orientation()

        self.observation[:, 0] = self.distance / self.MAX_DISTANCE
        self.observation[:, 2] = self.vmg / self.MAX_SPEED
        self.observation[:, -3] = np.sign(orientation_coeff)
        self.observation[:, -2] = np.tanh(orientation_coeff)

        self.is_terminal = False
        self.is_truncated = False
        self.has_collided = False

        self.trajectory = []
        self.append_to_trajectory()
        self.has_entered_mark_zone = False

        return self.observation, {}

    def random_phase(self, phase_steps, offset=0, probabilities=None):
        phase_step = np.random.choice(np.arange(0, phase_steps), p=probabilities)
        return 2 * np.pi / (1.0 * phase_steps) * phase_step + offset

    def append_to_trajectory(self):
        target_heading_def = self.target_heading * 180 / np.pi
        heading_deg = self.heading * 180 / np.pi
        if heading_deg > 180:
            heading_deg = heading_deg - 360
        meta = {
            'current_mark': self.current_mark,
            'vmg': self.vmg,
            'heading': heading_deg,
            'reward': self.reward,
            'speed': self.speed,
            'distance': self.distance,
            'min_distance': self.min_distance,
            'has_missed_mark': self.has_missed_mark
        }
        self.trajectory.append({'x': self.x, 'y': self.y, 'meta': meta})
        return self

    def calculate_boat_relative_orientation(self):
        """
        Determine if the boat whether the boat is headed towards the target,
        or if the target is at starboard or port.
        """

        x_a = self.x
        y_a = self.y
        x_b = self.x + np.sin(self.heading)
        y_b = self.y + np.cos(self.heading)
        x = self.current_target_x
        y = self.current_target_y

        return (x_b - x_a) * (y - y_a) - (y_b - y_a) * (x - x_a)

    def step(self, action):
        min_reward = self.reward_range.low
        max_reward = self.reward_range.high

        orientation_coeff = self.calculate_boat_relative_orientation()
        twa_abs = np.where(self.twa > np.pi, 2 * np.pi - self.twa, self.twa)

        self.apply_action(action, self.dt)
        self.is_terminal_state().calculate_reward()
        self.observation[:-1, :] = self.observation[1:, :]
        self.observation[-1, :] = np.stack([
            self.distance / self.MAX_DISTANCE,
            self.vmg / self.MAX_SPEED,
            self.heading / np.pi,
            self.angle_to_mark / np.pi,
            twa_abs / np.pi,
            self.is_terminal,
            self.has_collided,
            (self.reward - min_reward) * 2 / (max_reward - min_reward) - 1,
            np.sign(orientation_coeff),
            np.tanh(orientation_coeff),
            self.tack_count / self.MAX_EFFECTIVE_TACKS
        ])

        self.append_to_trajectory()

        return self.observation, self.reward, self.is_terminal, self.is_truncated, {}

    def render(self, mode='human'):
        pass

    def get_speed_from_polar_chart(self, twa, dead_zone_angle):
        speed = speed_interp(twa)
        if twa < dead_zone_angle or twa > 2 * np.pi - dead_zone_angle:
            speed *= 0.7

        self.speeds.append(speed)

        return np.min(self.speeds)


    def apply_action(self, action, dt):
        heading_change = np.squeeze(self.actions[action] * np.pi)
        self.heading_change_history.append(heading_change)

        self.prev_heading = self.heading
        heading = wrap_phase(self.heading + heading_change)

        true_wind_direction = 0
        true_wind_angle = wrap_phase(angle_difference(true_wind_direction, heading))
        self.twa = true_wind_angle
        self.speed = self.get_speed_from_polar_chart(true_wind_angle, np.pi/6)

        self.x += self.speed * np.sin(heading) * dt
        self.y += self.speed * np.cos(heading) * dt

        self.heading_change = heading_change

        self.heading = heading
        self.has_tacked = (self.prev_heading < np.pi) != (heading < np.pi)

        self.current_target_x = self.target_x[self.current_mark]
        self.current_target_y = self.target_y[self.current_mark]
        dx = self.current_target_x - self.x
        dy = self.current_target_y - self.y

        self.angle_to_mark = wrap_phase(np.arctan2(dx, dy))

        target_unit = np.stack([np.cos(self.angle_to_mark), np.sin(self.angle_to_mark)])
        heading_unit = np.stack([np.cos(heading), np.sin(heading)])

        self.vmg = (target_unit @ heading_unit) * self.speed
        self.vmg_history.append(self.vmg)


        self.prev_distance = self.distance
        self.distance = np.sqrt(dx ** 2 + dy ** 2)
        self.min_distance = np.minimum(self.distance, self.min_distance)

        self.tack_count += 1 if self.has_tacked else 0
        self.tack_count = np.minimum(self.tack_count, self.MAX_EFFECTIVE_TACKS)
        self.remaining_seconds -= dt
        self.delta_t = dt

        return self

    def is_terminal_state(self):
        mark_zone_threshold = self.BOAT_LENGTH * 3
        self.has_entered_mark_zone = self.has_entered_mark_zone or (self.distance <= mark_zone_threshold)

        if self.has_entered_mark_zone and (self.distance > self.min_distance):
            print("Missed mark")
            self.is_truncated = True
            self.is_terminal = True
            self.has_missed_mark = True
            return self

        if self.distance < self.target_tolerance_multiplier * self.BOAT_LENGTH:
            self.current_mark += 1
            self.remaining_seconds = self.MAX_REMAINING_SECONDS
            if self.current_mark == self.MAX_MARKS:
                self.is_terminal = True
                self.is_truncated = False
                self.has_reached_mark = True
                return self

        has_collided = self.x < self.MIN_X or self.x > self.MAX_X or self.y < self.MIN_Y or self.y > self.MAX_Y

        self.has_collided = has_collided

        if has_collided or self.remaining_seconds < 1:
            self.is_terminal = True
            self.is_truncated = True
            return self

        self.is_terminal = False
        self.is_truncated = False
        return self

    def calculate_reward(self):
        if self.has_missed_mark:
            self.reward = -100
            return self
        # tack_scaling = (1 - 0.25 * self.tack_count / self.MAX_EFFECTIVE_TACKS)

        vmg_reward = self.vmg / self.MAX_SPEED
        # vmg_reward = np.where(vmg_reward < 0, 20 * vmg_reward, tack_scaling * vmg_reward)
        vmg_reward = 5 * vmg_reward

        smoothness_reward = 0
        terminal_bonus = 0
        if self.is_terminal and self.has_reached_mark:
            # vmg_smoothness = np.std(np.diff(self.vmg_history))
            # vmg_smoothness_reward = np.exp(-vmg_smoothness)

            smoothness_reward = np.exp(-np.sum(np.abs(self.heading_change_history)))
            terminal_bonus = 100


        distance_scaling = 1 / self.current_radius_multiplier[min(self.current_mark, self.MAX_MARKS - 1)]

        # self.reward = (vmg_reward + smoothness_reward * tack_scaling) * distance_scaling + terminal_bonus * smoothness_reward * self.remaining_seconds / self.MAX_REMAINING_SECONDS
        self.reward = (vmg_reward + smoothness_reward) * distance_scaling + terminal_bonus * smoothness_reward * self.remaining_seconds / self.MAX_REMAINING_SECONDS

        return self