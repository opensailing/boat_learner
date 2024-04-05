import numpy as np
from scipy.interpolate import interp1d
from collections import deque

def angle_difference(angle1, angle2):
    # Normalize angles to [0, 2*pi)
    angle1 = angle1 % (2 * np.pi)
    angle2 = angle2 % (2 * np.pi)

    # Compute the difference
    diff = angle1 - angle2

    # Adjust differences larger than pi to find the shortest path
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi

    return diff

class PIDController:
    def __init__(self, Kp, Kd, Ki, error_fun=angle_difference):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.prev_error = 0
        self.integral = 0
        self.integral_leakage = 0.8
        self.set_points = deque(maxlen=1)
        self.error_fun = error_fun

    def step(self, input, set_point, dt, output_min, output_max):
        # Ensure dt is non-zero to avoid division by zero
        if dt <= 0:
            dt = 1e-6  # Set to a small value to avoid division by zero

        self.set_points.append(set_point)

        set_point = np.mean(self.set_points)
        error = self.error_fun(set_point, input)

        # Apply integral leakage unconditionally, remove the error-based condition if not needed
        self.integral = self.integral * self.integral_leakage + error * dt

        # Calculate derivative
        derivative = (error - self.prev_error) / dt

        # Update previous error for the next cycle
        self.prev_error = error

        # Calculate PID output and clamp to output limits
        pid_output = self.Kp * error + self.Kd * derivative + self.Ki * self.integral
        pid_output_clamped = np.clip(pid_output, output_min, output_max)

        return pid_output_clamped

    def reset(self):
        self.prev_error = 0
        self.integral = 0

THETA = np.load('./data/theta.npy')
SPEED = np.load('./data/speed.npy')

speed_interp = interp1d(THETA, SPEED)

def wrap_phase(angles):
    return np.remainder(np.remainder(angles, 2 * np.pi) + 2 * np.pi, 2 * np.pi)

class Boat:
    def __init__(self, mass, length=7.3, beam=2.5, drag_coefficient=0.003, sail_area=67):
        self.mass = mass # mass in kg
        self.length = length # length in meters
        self.beam = beam # beam in meters
        self.moment_of_inertia = self.estimate_moment_of_inertia()
        self.drag_coefficient = drag_coefficient
        self.sail_area = sail_area # Sail area in m² -- assumes a single constant sail
        self.water_density = 1025  # Density of salt water in kg/m^3
        self.reference_area = 7.48 # Estimated wetted surface area in square meters
        self.heading = 0
        self.speed = 0
        self.x = 0
        self.y = 0
        self.rudder_angle = 0
        self.headings = deque(maxlen=5)

    def estimate_moment_of_inertia(self):
        return self.mass * (self.length ** 2 + 0 * self.beam ** 2) / 12

    def step(self, wind_speed, wind_angle, rudder_angle, dt):
        self.rudder_angle = wrap_phase(rudder_angle)
        self.rudder_angle = np.where(self.rudder_angle > np.pi, self.rudder_angle - 2 * np.pi, self.rudder_angle)
        self.rudder_angle = np.clip(rudder_angle, -np.pi/2, np.pi/2)

        self.update_heading(dt)
        self.update_speed(wind_speed, wind_angle, dt)
        self.update_position(dt)

    def update_speed(self, wind_speed, wind_angle, dt, dead_zone_angle=np.pi/6):
        apparent_wind_speed, apparent_wind_angle = self.calculate_apparent_wind(wind_speed, wind_angle, self.speed, self.heading)
        V_max = self.get_speed_from_polar_chart(apparent_wind_angle, dead_zone_angle)
        propulsive_force = self.calculate_propulsive_force(apparent_wind_speed)
        drag_force = self.calculate_drag_force(self.speed)
        net_force = propulsive_force - drag_force
        linear_acceleration = net_force / self.mass
        self.speed += linear_acceleration * dt
        self.speed = min(self.speed, V_max)

    def update_heading(self, dt):
        turning_torque = self.calculate_turning_torque(self.rudder_angle, self.speed)
        angular_acceleration = turning_torque / self.moment_of_inertia
        # max_angular_acceleration = 13 / 180 * np.pi
        # if angular_acceleration > max_angular_acceleration:
        #     angular_acceleration = max_angular_acceleration
        # elif angular_acceleration < -max_angular_acceleration:
        #     angular_acceleration = -max_angular_acceleration

        angular_velocity = angular_acceleration * dt
        next_heading = wrap_phase(self.heading + angular_velocity * dt)
        self.headings.append(next_heading)
        self.heading = self.average_heading()

    def average_heading(self):
        heading_vector_x = np.mean(np.array([np.sin(heading) for heading in self.headings]))
        heading_vector_y = np.mean(np.array([np.cos(heading) for heading in self.headings]))
        return np.arctan2(heading_vector_x, heading_vector_y)

    def update_position(self, dt):
        self.x += self.speed * np.sin(self.heading) * dt
        self.y += self.speed * np.cos(self.heading) * dt

    def get_speed_from_polar_chart(self, apparent_wind_angle, dead_zone_angle):
        if apparent_wind_angle < dead_zone_angle or apparent_wind_angle > 2 * np.pi - dead_zone_angle:
            return 0.1
        return speed_interp(apparent_wind_angle)

    def calculate_propulsive_force(self, apparent_wind_speed):
        return self.sail_area * apparent_wind_speed**2

    def calculate_drag_force(self, boat_speed):
        return 0.5 * self.water_density * self.drag_coefficient * self.reference_area * boat_speed**2

    def calculate_turning_torque(self, rudder_angle, speed):
        speed = np.maximum(speed, 1)
        return rudder_angle * speed * self.sail_area

    def calculate_apparent_wind(self, true_wind_speed, true_wind_direction, boat_speed, boat_heading):
        true_wind_angle = wrap_phase(true_wind_direction - boat_heading)
        wind_vector = true_wind_speed * np.array([np.sin(true_wind_angle), np.cos(true_wind_angle)])
        boat_vector = boat_speed * np.array([np.sin(boat_heading), np.cos(boat_heading)])
        apparent_wind_vector = wind_vector - boat_vector
        apparent_wind_speed = np.linalg.norm(apparent_wind_vector)
        apparent_wind_angle = np.arctan2(apparent_wind_vector[0], apparent_wind_vector[1])
        apparent_wind_angle = wrap_phase(apparent_wind_angle)
        return apparent_wind_speed, apparent_wind_angle
