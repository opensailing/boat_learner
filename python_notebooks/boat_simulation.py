import numpy as np
from scipy.interpolate import interp1d

THETA = np.load('./data/theta.npy')
SPEED = np.load('./data/speed.npy')

speed_interp = interp1d(THETA, SPEED)

def wrap_phase(angles):
    return np.remainder(np.remainder(angles, 2 * np.pi) + 2 * np.pi, 2 * np.pi)

def predict_speed(theta, dead_zone_angle=np.pi/6, wrap=True):
    theta = wrap_phase(theta) if wrap else theta
    speed = speed_interp(theta)
    return np.where(np.abs(theta - np.pi) < dead_zone_angle, 0, speed)

def angle_difference(angle1, angle2):
    # Normalize angles to [0, 2*pi)
    angle1 = angle1 % (2 * np.pi)
    angle2 = angle2 % (2 * np.pi)

    # Compute the difference
    diff = angle2 - angle1

    # Adjust differences larger than pi to find the shortest path
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi

    return diff

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

    def estimate_moment_of_inertia(self):
        return self.mass/8 * (self.length**2 + self.beam**2) / 12

    def step(self, wind_speed, wind_angle, rudder_angle, dt):
        self.update_heading(rudder_angle, dt)
        self.update_speed(wind_speed, wind_angle, dt)
        self.update_position(dt)

    def update_speed(self, wind_speed, wind_angle, dt):
        V_max = self.get_speed_from_polar_chart(wind_speed, wind_angle, self.heading)
        propulsive_force = self.calculate_propulsive_force(wind_speed)
        drag_force = self.calculate_drag_force(self.speed)
        net_force = propulsive_force - drag_force
        linear_acceleration = net_force / self.mass
        self.speed += linear_acceleration * dt
        self.speed = min(self.speed, V_max)

    def update_heading(self, rudder_angle, dt):
        turning_torque = self.calculate_turning_torque(rudder_angle, self.speed)
        angular_acceleration = turning_torque / self.moment_of_inertia
        self.heading += angular_acceleration * dt
        self.heading = wrap_phase(self.heading)

    def update_position(self, dt):
        self.x += self.speed * np.sin(self.heading) * dt
        self.y += self.speed * np.cos(self.heading) * dt

    def get_speed_from_polar_chart(self, wind_speed, wind_angle, boat_heading):
        apparent_wind_angle = np.abs(angle_difference(wind_angle, boat_heading))
        return speed_interp(apparent_wind_angle)

    def calculate_propulsive_force(self, wind_speed):
        return self.sail_area * wind_speed**2

    def calculate_drag_force(self, boat_speed):
        return 0.5 * self.water_density * self.drag_coefficient * self.reference_area * boat_speed**2

    def calculate_turning_torque(self, rudder_angle, speed):
        speed = np.maximum(speed, 1)
        return rudder_angle * speed * self.sail_area