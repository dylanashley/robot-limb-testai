# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import glob

kill_names = ["kill0", "kill1", "kill12"]
titles = ["Keep All Cameras On", "Kill 1 Camera", "Kill 1-2 Cameras"]

# Create a 3x1 subplot layout
fig, axs = plt.subplots(3, 1, figsize=(6.3, 7))  # Set up 3 rows, 1 column

for idx, kill_name in enumerate(kill_names):
    # Load data for AC and PPO runs
    ac_files = glob.glob(f"results/ac_step_counts_{kill_name}_run*.npy")
    ppo_files = glob.glob(f"results/ppo_step_counts_{kill_name}_run*.npy")
    
    ac_episode_lengths = [np.load(file) for file in ac_files]
    ac_episode_lengths = np.array(ac_episode_lengths)

    ppo_episode_lengths = [np.load(file) for file in ppo_files]
    ppo_episode_lengths = np.array(ppo_episode_lengths)

    # Load brownian motion baseline data
    brownian_multi_servo_step_counts = np.load(f"results/brownian_step_counts_{kill_name}_multi.npy")
    brownian_single_servo_step_counts = np.load(f"results/brownian_step_counts_{kill_name}_single.npy")

    # Calculate mean and standard error for AC and PPO
    mean_ac_episode_lengths = np.mean(ac_episode_lengths, axis=0)
    sem_ac_episode_lengths = st.sem(ac_episode_lengths, axis=0)
    
    mean_ppo_episode_lengths = np.mean(ppo_episode_lengths, axis=2)
    mean_mean_ppo_episode_lengths = np.mean(mean_ppo_episode_lengths, axis=0)
    sem_mean_ppo_episode_lengths = st.sem(mean_ppo_episode_lengths, axis=0)

    ax = axs[idx]  # Access the current subplot
    caps = []
    bars = []

    # Plot AC episode length with error bars
    _, c, b = ax.errorbar(
        range(mean_ac_episode_lengths.shape[0]),
        mean_ac_episode_lengths,
        yerr=sem_ac_episode_lengths,
        label="AC",
    )
    caps += c
    bars += b

    # Plot PPO episode length with error bars
    _, c, b = ax.errorbar(
        range(mean_mean_ppo_episode_lengths.shape[0]),
        mean_mean_ppo_episode_lengths,
        yerr=sem_mean_ppo_episode_lengths,
        label="PPO",
    )
    caps += c
    bars += b

    # Plot brownian motion baseline
    line = np.arange(max(mean_ac_episode_lengths.shape[0], mean_mean_ppo_episode_lengths.shape[0]))
    ax.errorbar(
        line,
        line * 0 + np.mean(brownian_single_servo_step_counts),
        yerr=np.concatenate(
            [
                line[0:1],
                [
                    np.std(brownian_single_servo_step_counts) / np.sqrt(len(brownian_single_servo_step_counts))
                ],
                line[2:] * 0,
            ]
        ),
        label="BMSS",
        linestyle=":"
    )
    ax.errorbar(
        line,
        line * 0 + np.mean(brownian_multi_servo_step_counts),
        yerr=np.concatenate(
            [
                line[0:1],
                [
                    np.std(brownian_multi_servo_step_counts) / np.sqrt(len(brownian_multi_servo_step_counts))
                ],
                line[2:] * 0,
            ]
        ),
        label="BMMS",
        linestyle=":"
    )

    # Set transparency for error bars
    for bar in bars:
        bar.set_alpha(0.3)
    for cap in caps:
        cap.set_alpha(0.3)

    # Set plot titles, labels, and remove grid
    fontsize = 12
    ax.set_title(titles[idx], fontsize=fontsize)
    ax.set_xlabel("Iterations", labelpad=5, fontsize=fontsize)  # Move label closer to the axis
    ax.set_ylabel("Mean Episode Length", labelpad=10, fontsize=fontsize)
    ax.set_yscale("log")
    ax.legend(loc="upper right", ncol=4, frameon=False)  # Set legend with 4 columns in one row
    ax.grid(False)  # Remove grid

# Adjust layout to avoid overlapping
plt.tight_layout()

# Save the figure, do not display
plt.savefig("episode_lengths_3x1_with_transparency.pdf", bbox_inches="tight")
plt.show()
