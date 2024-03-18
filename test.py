# -*- coding: utf-8 -*-

from statsmodels.stats import multitest
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

# run statistical tests
t, p1 = st.ttest_ind(brownian_single_servo_step_counts, ac_episode_lengths[:, -1])
assert t > 0
p1 /= 2

t, p2 = st.ttest_ind(
    brownian_multi_servo_step_counts, ppo_episode_lengths[:, -1, :].flatten()
)
assert t > 0
p2 /= 2

assert all(multitest.multipletests([p1, p2], alpha=0.01, method="holm")[0])
