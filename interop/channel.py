import numpy as np
from tqdm import tqdm

# ok
def angle_to_state(angle):
    return int(30 * ((angle + np.pi) / (2 * np.pi) % 1))  # Discretization of the angle space

# ok
def x_to_state(x):
    return int(40 * ((x + -10) / 20))  # Discretization of the x space

# ok
def vel(theta, theta_0=0, theta_dead=np.pi / 12):
    return 1 - np.exp(-(theta - theta_0) ** 2 / theta_dead)


def rew(theta, theta_0=0, theta_dead=np.pi / 12):
    return vel(theta, theta_0, theta_dead) * np.cos(theta)

random_ys = []
for episode in tqdm(range(100), desc='Running: Random agent on channel sea task'):  # run for 500 episodes
    x = np.random.uniform(-9.9, 9.9)
    angle = 0  # always start with angle 0
    y = 0
    for i in range(200):
        a = np.random.choice(range(2) ) # Sample the action from the softmax distribution

        out = [-0.1, 0.1][a]  # Get the change in angle as a result of the selected angle

        # Update the angle
        angle += out

        # Update the x position
        x += vel(angle + out) * np.sin(angle + out)

        y += vel(angle + out) * np.cos(angle + out)

        # Stop the episode if the pier was hit
        if np.abs(x) > 10:
            break

    random_ys.append(y)


Q = np.zeros((30, 40, 2))  # Initialization of the Q-values with zeros
# There are 30 angle states and 2 actions

rho = 0  # Initialize the average reward to 0
td_ys = []
for episode in tqdm(range(1000), desc='Running: Train agent on channel sea task'):  # run for 1000 episodes
    x = np.random.uniform(-9.9, 9.9)
    angle = 0  # always start with angle 0
    y = 0
    for i in range(200):
        state = angle_to_state(angle)
        x_state = x_to_state(x)

        p = np.exp(Q[state, x_state]) / np.sum(np.exp(Q[state, x_state]))  # Action selection using softmax
        a = np.random.choice(range(2), p=p)  # Sample the action from the softmax distribution

        out = [-0.1, 0.1][a]  # Get the change in angle as a result of the selected angle

        new_state = angle_to_state(angle + out)
        new_x_state = x_to_state(x + vel(angle + out) * np.sin(angle + out))

        # Calculate the prediction error
        if np.abs(x + vel(angle + out) * np.sin(angle + out)) > 10:
            delta = -10  # Terminal case
        else:
            delta = rew(angle + out) - rho + Q[new_state, new_x_state].max() - Q[state, x_state, a]

        # Update the average reward
        rho += 0.1 * (rew(angle + out) - rho)

        # Update the Q-value
        Q[state, x_state, a] += 0.1 * delta

        # Update the angle
        angle += out

        # Update the x position
        x += vel(angle + out) * np.sin(angle + out)

        y += vel(angle + out) * np.cos(angle + out)

        # Stop the episode if the pier was hit
        if np.abs(x) > 10:
            break

    td_ys.append(y)

random_mean = np.mean(random_ys[-100:])
random_std = np.std(random_ys[-100:])

td_mean = np.mean(td_ys[-100:])
td_stq = np.std(td_ys[-100:])

print('Results from last 100 episodes')
print('| ===== agent ===== | ===== mean ===== | ===== std ===== |')
print(f'{"| Random":<20}| {random_mean:<17.2f}| {random_std:<16.2f}|')
print(f'{"| Trained":<20}| {td_mean:<17.2f}| {td_stq:<16.2f}|')