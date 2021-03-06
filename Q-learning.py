import lunar_lander
import numpy as np
import math
from statistics import mean
import time
import os

# X Pos, Y Pos, X Vel, Y Vel, Angle, Angular Vel, Leg 1, Leg 2
MIN_VALS = [-2, -1, -2, -4, -math.pi / 2, -2, 0, 0] # TODO: determine realistic min/max values
MAX_VALS = [2, 2, 2, 2, math.pi / 2, 2, 1, 1]
NUM_BINS = [11, 11, 11, 11, 11, 11, 2, 2] # TODO: decide on how many bins for each value

def state_to_table_indices(s):
    indices = []
    for value, min_val, max_val, bins in zip(s, MIN_VALS, MAX_VALS, NUM_BINS):
        if value >= max_val:
            index = bins - 1
        elif value <= min_val:
            index = 0
        else:
            index = int(((value - min_val) * bins) // (max_val - min_val))
        indices.append(index)
    return indices

# Number of episodes to run before stopping
NUM_EPS = 100

# Which Lunar Lander environment to run
# Options are "LunarLander", "LunarLanderMoreRandStart", "LunarLanderMovingZone", "LunarLanderRandomZone", "LunarLanderLimitedFuel"
ENV_NAME = "LunarLander"

env = getattr(lunar_lander, ENV_NAME)()

# Initialize q-table with a value of 0 for each (state, action) pair
q_table = np.zeros(NUM_BINS + [env.action_space.n])

# Un-comment line below to load q-table from file instead
# q_table = np.load("Q-learning/{}/q_table.npy".format(ENV_NAME))

alpha = 0.1
gamma = 0.6
epsilon = 0.1

ep_rewards = []
avg_rewards = []
steps = []
success_percent = []
successes = []
final_pos = []

# Start timer
learn_start = time.time()

# Q-learning algorithm from RL book (6.5, page 131)
try:
    total = 0
    success_count = 0
    # Loop for each episode
    for i in range(NUM_EPS):
        # Initialize S
        state_i = state_to_table_indices(env.reset())
        
        total_reward = 0
        step_count = 0
        success = False
        done = False

        # Loop for each step of episode
        while not done:
            # env.render() # Un-comment to render lander graphics

            # Choose A from S using policy derived from Q
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[tuple(state_i)])

            # Take action A, observe R, S'
            next_state, reward, done, info = env.step(action)

            # Reward is exactly 100 when lander makes successful landing
            if reward == 100:
                success = True
                success_count += 1

            total_reward += reward
            next_state_i = state_to_table_indices(next_state)

            # Q(S, A)
            old_q_val = q_table[tuple(state_i)][action]
            # max_a(Q(S', a))
            next_max = np.max(q_table[tuple(next_state_i)])

            # Q(S, A) <- Q(S, A) + alpha[R + gamma * max_a(Q(S', a)) - Q(S, A)]
            q_table[tuple(state_i)][action] = old_q_val + alpha * (reward + gamma * next_max - old_q_val)
            # S <- S'
            state_i = next_state_i

            step_count += 1

        final_pos.append((next_state[0], next_state[1]))
        successes.append(success)
        steps.append(step_count)
        ep_rewards.append(total_reward)
        total += total_reward

        # Every 100 episodes, print stats for previous 100 episodes
        if i % 100 == 0:
            if i != 0:
                print("Episodes {}-{} | Avg. total reward: {:.2f} | Successes: {:.2f}%".format(i - 99, i, total / 100, success_count))
                avg_rewards.append(total / 100)
                success_percent.append(success_count)
                total = 0
                success_count = 0
except KeyboardInterrupt: # Press CTRL+C in the terminal to stop early
    pass

# Stop timer and calculate elapsed time
learn_stop = time.time()
learn_time = learn_stop - learn_start

# Create directory to store data if it doesn't already exist
if not os.path.exists("Q-learning/{}".format(ENV_NAME)):
    os.makedirs("Q-learning/{}".format(ENV_NAME))

# Save data for processing later
np.save("Q-learning/{}/q_table".format(ENV_NAME), q_table)
np.save("Q-learning/{}/num_steps".format(ENV_NAME), steps)
np.save("Q-learning/{}/rewards".format(ENV_NAME), ep_rewards)
np.save("Q-learning/{}/successes".format(ENV_NAME), successes)
np.save("Q-learning/{}/final_pos".format(ENV_NAME), final_pos)

# Print stats
print("Total episodes: {} | Total time: {:.2f} sec".format(i + 1, learn_time))
print("q_table entries != 0: {}%".format(100 * np.count_nonzero(q_table) / q_table.size))
print("Steps/episode (min, avg., max): {:.2f}, {:.2f}, {:.2f}".format(min(steps), mean(steps), max(steps)))
print("Avg. time: {:.2f} sec/step, {:.2f} sec/episode".format(learn_time / sum(steps), learn_time / (i + 1)))