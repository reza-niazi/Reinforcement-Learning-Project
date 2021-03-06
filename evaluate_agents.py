import lunar_lander
import numpy as np
import math
import os.path
from statistics import mean

# Note: this assumes that the q-tables for SARSA and Q-learning are all set up with the same settings as below
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

NUM_EPS = 500 # Number of episodes to evaluate each agent over

# List of methods and environments to evaluate
METHODS = ["SARSA", "Q-learning", "Heuristic"]
ENVS = ["LunarLander", "LunarLanderMoreRandStart", "LunarLanderRandomZone", "LunarLanderMovingZone"]

# Open file to save evaluations to
f = open("evaluation-{}.csv".format(NUM_EPS), "w")
f.write("Method,Environment,Avg. Reward,Success,Avg. Steps\n")

# Evaluate each combination of method and environment
for method in METHODS:
    for env_name in ENVS:
        print("{}, {}".format(method, env_name))

        env = getattr(lunar_lander, env_name)()

        # Load the saved q-tables for SARSA or Q-learning
        if method != "Heuristic":
            path = "{}/{}/q_table.npy".format(method, env_name)
            if os.path.exists(path):
                q_table = np.load(path)
            else:
                print("No q-table found")
                print("-"*30)
                continue

        rewards = []
        success_count = 0
        steps = []

        for i in range(NUM_EPS):
            total_reward = 0
            step_count = 0
            
            state = env.reset()
            done = False
            while not done:
                # Get next action from current method
                if method == "Heuristic":
                    action = lunar_lander.heuristic(env, state)
                else:
                    action = np.argmax(q_table[tuple(state_to_table_indices(state))])

                # Execute action and get next state and reward
                state, reward, done, info = env.step(action)

                total_reward += reward

                # Reward is exactly 100 when lander makes successful landing
                if reward == 100:
                    success_count += 1

                step_count += 1

            steps.append(step_count)
            rewards.append(total_reward)

        # Print results for current method and environment
        print("Avg. reward: {:.2f}", mean(rewards))
        print("Successes: {:.2f}%".format(100 * success_count / NUM_EPS))
        print("Steps/episode (min, avg, max): {:.2f}, {:.2f}, {:.2f}".format(min(steps), mean(steps), max(steps)))
        print("-"*30)

        # Write results to file
        f.write("{},{},{},{},{}\n".format(method, env_name, mean(rewards), success_count / NUM_EPS, mean(steps)))

f.close()