from random import randint, random
from boat_simulation import Boat, wrap_phase
from gymnasium import spaces, Env
import numpy as np

class MultiMarkEnv(Env):
    def __init__(self, config, dt=15, seq_size=2, bounds=None, phase_steps=8):
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
        self.boat = Boat(mass=960+320, drag_coefficient=0.003, sail_area=67+25)
        self.MAX_MARKS = config['max_marks']
        self.MAX_REMAINING_SECONDS = config['max_seconds_per_leg']
        self.LEG_RADIUS = config['leg_radius']
        self.target_x = np.zeros((self.MAX_MARKS,))
        self.target_y= np.zeros((self.MAX_MARKS,))
        self.current_mark = 0
        self.heading_change = 0
        self.speed = 0
        self.vmg = 0
        self.heading = 0
        self.target_tolerance_multiplier = config['target_tolerance_multiplier']
        self.PHASE_STEPS = phase_steps

        self.actions = np.array(config['actions'])

        self.plot_fn = config['plot_fn']

        self.trajectory = []

        self.MAX_DISTANCE = np.sqrt((self.MAX_X - self.MIN_X) ** 2 + (self.MAX_Y - self.MIN_Y) ** 2)

        self.seq_size = seq_size
        self.obs_size = 9

        self.observation = np.zeros((self.seq_size, self.obs_size,))
        num_actions = self.actions.shape[0]
        self.action_space = spaces.Discrete(num_actions)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.seq_size, self.obs_size,))
        self.reward_range = spaces.Box(low=-100, high=100, shape=())
        self.dt = dt

    def reset(self, seed = None, options = None, heading=None, speed=None, vmg=None):
        super().reset(seed=seed)
        self.observation = np.zeros((self.seq_size, self.obs_size,))
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
        for i in range(self.MAX_MARKS):
            radius_multiplier = np.random.choice([1, 1.5, 2])
            phase = self.random_phase()
            current_x = current_x + np.cos(phase) * self.LEG_RADIUS * radius_multiplier
            current_y = current_y + np.sin(phase) * self.LEG_RADIUS * radius_multiplier
            self.target_x[i] = current_x
            self.target_y[i] = current_y

        self.current_mark = 0
        self.tack_count = 0
        self.heading = self.random_phase() if heading is None else heading
        self.prev_heading = self.heading
        self.angle_to_mark = 0
        self.speed = 0 if speed is None else speed
        self.vmg = 0 if vmg is None else vmg
        self.has_tacked = False
        self.remaining_seconds = self.MAX_REMAINING_SECONDS
        self.delta_t = 0
        self.reward = 0
        self.heading_change = 0

        self.boat.x = self.x
        self.boat.y = self.y
        self.boat.speed = self.speed
        self.boat.heading = self.heading

        self.distance = np.sqrt((self.target_x[0] - self.x) ** 2 + self.target_y[0] ** 2)
        self.initial_distance = self.distance

        self.current_target_x = self.target_x[0]
        self.current_target_y = self.target_y[0]

        orientation_coeff = self.calculate_boat_relative_orientation()

        self.observation[:, 0] = self.distance / self.MAX_DISTANCE
        self.observation[:, 2] = self.vmg / self.MAX_SPEED
        self.observation[:, -2] = np.sign(orientation_coeff)
        self.observation[:, -1] = np.tanh(orientation_coeff)

        self.is_terminal = False
        self.is_truncated = False
        self.has_collided = False

        self.trajectory = []
        self.append_to_trajectory()

        return self.observation, {}

    def random_phase(self):
        phase_step = randint(0, self.PHASE_STEPS)
        return 2 * np.pi / (1.0 * self.PHASE_STEPS) * phase_step

    def append_to_trajectory(self):
        heading_deg = self.boat.heading * 180 / np.pi
        if heading_deg > 180:
            heading_deg = heading_deg - 360
        rudder_angle = self.boat.rudder_angle * 180 / np.pi
        if rudder_angle > 180:
            rudder_angle = rudder_angle - 360
        meta = {
            'current_mark': self.current_mark,
            'vmg': self.vmg,
            'heading': heading_deg,
            'rudder_angle': rudder_angle,
            'reward': self.reward,
            'speed': self.boat.speed
        }
        self.trajectory.append({'x': self.boat.x, 'y': self.boat.y, 'meta': meta})
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

    def step(self, action_idx):
        min_reward = self.reward_range.low
        max_reward = self.reward_range.high
        action = self.actions[action_idx]

        orientation_coeff = self.calculate_boat_relative_orientation()


        self.apply_action(action, self.dt)
        self.is_terminal_state().calculate_reward()
        self.observation[:-1, :] = self.observation[1:, :]
        self.observation[-1, :] = np.stack([
            self.distance / self.MAX_DISTANCE,
            self.vmg / self.MAX_SPEED,
            self.heading / np.pi,
            self.angle_to_mark / np.pi,
            self.is_terminal,
            self.has_collided,
            (self.reward - min_reward) * 2 / (max_reward - min_reward) - 1,
            np.sign(orientation_coeff),
            np.tanh(orientation_coeff)
        ])

        self.append_to_trajectory()

        return self.observation, self.reward, self.is_terminal, self.is_truncated, {}

    def render(self, mode='human'):
        pass

    def apply_action(self, action, dt):
        rudder_angle = action * np.pi
        self.prev_heading = self.heading

        self.boat.step(6.17, 0, rudder_angle, dt)

        heading = self.boat.heading
        self.speed = self.boat.speed
        self.x = self.boat.x
        self.y = self.boat.y

        self.heading_change = heading - self.prev_heading

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

        self.distance = np.sqrt(dx ** 2 + dy ** 2)
        self.tack_count = 1 if self.has_tacked else 0
        self.remaining_seconds -= dt
        self.delta_t = dt

        return self

    def is_terminal_state(self):
        if self.distance < self.target_tolerance_multiplier * self.boat.length:
            self.current_mark += 1
            self.remaining_seconds = self.MAX_REMAINING_SECONDS
            if self.current_mark == self.MAX_MARKS:
                self.is_terminal = True
                self.is_truncated = False
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
        self.reward = 0.1 * self.vmg
        return self