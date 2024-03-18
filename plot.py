# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

# load all the data
ac_episode_lengths = list()
for i in range(30):
    ac_episode_lengths.append(np.load("results/{}_ac_step_counts.npy".format(i)))
ac_episode_lengths = np.array(ac_episode_lengths)

brownian_multi_servo_step_counts = np.load(
    "results/brownian_multi_servo_step_counts.npy"
)
brownian_single_servo_step_counts = np.load(
    "results/brownian_single_servo_step_counts.npy"
)

ppo_episode_lengths = list()
for i in range(10):
    ppo_episode_lengths.append(np.load("results/{}_ppo_step_counts.npy".format(i)))
ppo_episode_lengths = np.array(ppo_episode_lengths)

# calculate the length of episodes
mean_ac_episode_lengths = np.mean(ac_episode_lengths, axis=0)
sem_ac_episode_lengths = st.sem(ac_episode_lengths, axis=0)

mean_ppo_episode_lengths = np.mean(ppo_episode_lengths, axis=2)
mean_mean_ppo_episode_lengths = np.mean(mean_ppo_episode_lengths, axis=0)
sem_mean_ppo_episode_lengths = st.sem(mean_ppo_episode_lengths, axis=0)

# draw the plot showing episode lengths
fig, ax = plt.subplots(1, 1)
caps = list()
bars = list()
_, c, b = ax.errorbar(
    range(mean_ac_episode_lengths.shape[0]),
    mean_ac_episode_lengths,
    yerr=sem_ac_episode_lengths,
    label="AC",
)
caps += c
bars += b
_, c, b = ax.errorbar(
    range(mean_mean_ppo_episode_lengths.shape[0]),
    mean_mean_ppo_episode_lengths,
    yerr=sem_mean_ppo_episode_lengths,
    label="PPO",
)
caps += c
bars += b
line = np.arange(
    max(mean_ac_episode_lengths.shape[0], mean_mean_ppo_episode_lengths.shape[0])
)

# draw brownian motion baselines
ax.errorbar(
    line,
    line * 0 + np.mean(brownian_single_servo_step_counts),
    yerr=np.concatenate(
        [
            line[0:1],
            [
                np.std(brownian_single_servo_step_counts)
                / np.sqrt(len(brownian_single_servo_step_counts))
            ],
            line[2:] * 0,
        ]
    ),
    label="BMSS",
    linestyle=":",
)
ax.errorbar(
    line,
    line * 0 + np.mean(brownian_multi_servo_step_counts),
    yerr=np.concatenate(
        [
            line[0:1],
            [
                np.std(brownian_multi_servo_step_counts)
                / np.sqrt(len(brownian_multi_servo_step_counts))
            ],
            line[2:] * 0,
        ]
    ),
    label="BMMS",
    linestyle=":",
)

# make all the errorbars transparent
for bar in bars:
    bar.set_alpha(0.3)
for cap in caps:
    cap.set_alpha(0.3)

# clean up the plot
ax.legend(loc="best", ncol=2, frameon=False)
ax.set_xlabel("Iterations", labelpad=10)
ax.set_xlim(-1, 101)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_ylabel("Mean Episode Length", labelpad=10)
ax.set_yscale("log")
fig.set_figheight(2)
fig.set_figwidth(6)
sns.set_theme(
    context="paper",
    style="ticks",
    palette="colorblind",
)

# save the figure
plt.savefig("episode_lengths.pdf", bbox_inches="tight")
