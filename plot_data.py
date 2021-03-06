import numpy as np
from statistics import mean
import matplotlib
import matplotlib.pyplot as plt
import os

# List of methods and environments to make figures fo
METHODS = ["SARSA", "Q-learning"]
ENVS = ["LunarLander", "LunarLanderMoreRandStart", "LunarLanderRandomZone", "LunarLanderMovingZone"]
# Amount of episodes to use in running average. Larger values generally result in smoother graphs.
AVG_SIZE = 250

running_avg_reward = dict.fromkeys(METHODS)
avg_successes = dict.fromkeys(METHODS)
avg_steps = dict.fromkeys(METHODS)

# Create folder to put figures in
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# Create figures for each environment and method
for env_name in ENVS:
    for method in METHODS:
        running_avg_reward[method] = []
        avg_successes[method] = []
        avg_steps[method] = []

        # Load data for each RL method
        path = "{}/{}".format(method, env_name)
        if os.path.exists(path):
            rewards = np.load("{}/rewards.npy".format(path))
            successes = np.load("{}/successes.npy".format(path))
            steps = np.load("{}/num_steps.npy".format(path))
        else:
            print("Warning: some figures will not be generated (could not find directory {})".format(path))
            continue

        # Calculate the running average of the total reward, success percentage, and steps for each episode
        for i in range(len(rewards)):
            running_avg_reward[method].append(mean(rewards[max(0, i - AVG_SIZE - 1):(i+1)]))
            avg_successes[method].append(100*np.count_nonzero(successes[max(0, i - AVG_SIZE - 1):(i+1)]) / min(AVG_SIZE, i + 1))
            avg_steps[method].append(mean(steps[max(0, i - AVG_SIZE - 1):(i+1)]))

    # Plot avg. reward per episode
    fig, axis = plt.subplots()
    for method in METHODS:
        if running_avg_reward[method]:
            axis.plot(running_avg_reward[method], label=method)
    if axis.lines:
        axis.set_xlabel("Episode")
        axis.set_ylabel("Avg. Total Reward")
        axis.set_title(env_name)
        plt.legend(loc="best")
        plt.savefig("Figures/{}_reward.png".format(env_name))
    plt.close(fig)

    # Plot avg. success percentage
    fig, axis = plt.subplots()
    for method in METHODS:
        if avg_successes[method]:
            axis.plot(avg_successes[method], label=method)
    if axis.lines:
        axis.set_xlabel("Episode")
        axis.set_ylabel("Avg. Successes (%)")
        axis.set_title(env_name)
        plt.legend(loc="best")
        plt.savefig("Figures/{}_success.png".format(env_name))
    plt.close(fig)

    # Plot avg. reward and avg. success percentage on same plot
    fig, axes = plt.subplots(nrows=2, ncols=1)
    for method in METHODS:
        if running_avg_reward[method]:
            axes[0].plot(running_avg_reward[method], label=method)
        if avg_successes[method]:
            axes[1].plot(avg_successes[method], label=method)
    if axes[0].lines and axes[1].lines:
        axes[0].set_xlim([0, 5000]) # TODO: will only work correctly for graphing exactly 5000 episodes
        axes[1].set_xlim([0, 5000]) # Should be changed to the length of the longest method
        axes[0].set_xlabel("Episode")
        axes[1].set_xlabel("Episode")
        axes[0].set_ylabel("Avg. Total Reward")
        axes[1].set_ylabel("Avg. Successes (%)")
        axes[0].set_title(env_name)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("Figures/{}_reward_success.png".format(env_name))
    plt.close(fig)

    # Plot avg. steps per episode
    fig, axis = plt.subplots()
    for method in METHODS:
        if avg_steps[method]:
            axis.plot(avg_steps[method], label=method)
    if axis.lines:
        axis.set_xlabel("Episode")
        axis.set_ylabel("Avg. Steps")
        axis.set_title(env_name)
        plt.legend(loc="best")
        plt.savefig("Figures/{}_steps.png".format(env_name))
    plt.close(fig)