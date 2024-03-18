# -*- coding: utf-8 -*-

from brain import str2mac, mac2str
from brain.brain_buffer import BrainBuffer
from brain.brain_interface import BrainInterface
from collections import OrderedDict
from PIL import Image, UnidentifiedImageError
from PyFixedReps import TileCoder, TileCoderConfig
from queue import Empty
import argparse
import git
import io
import logging
import numpy as np
import os
import pickle
import pyapriltags
import time
import torch
import wandb
import warnings
import yaml


class ACAgent:
    """Basic continuous state space and discrete action space actor
    critic.
    """

    def __init__(
        self,
        n: int,
        num_actions: int,
    ):
        assert n > 0
        assert num_actions > 0
        self.num_actions = num_actions
        self.reward_bar = 0
        self.e_v = np.zeros(n, dtype=float)
        self.e_u = np.zeros((n, num_actions), dtype=float)
        self.w_v = np.zeros(n, dtype=float)
        self.w_u = np.zeros((n, num_actions), dtype=float)
        self.last_action = 0
        self.last_prediction = 0

    def softmax(self, x: list[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rv = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            rv[action] = np.dot(x, (self.w_u[:, action]).flatten())
        rv = np.exp(rv)
        return rv / sum(rv)

    def predict(self, x: list[float] | np.ndarray):
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def start(self, initial_x: list[float] | np.ndarray):
        initial_x = np.asarray(initial_x, dtype=float)
        pi = self.softmax(initial_x)
        action = np.random.choice(self.num_actions, p=pi)
        self.e_v += initial_x
        self.e_u[:, action] += initial_x * (1 - sum(pi))
        self.last_action = action
        return action

    def step(
        self,
        reward: float,
        gamma: float,
        x: list[float] | np.ndarray,
        eta: float,
        alpha_v: float,
        alpha_u: float,
        lambda_v: float,
        lambda_u: float,
    ) -> tuple[int, float]:
        assert 0 <= gamma <= 1
        assert eta > 0
        assert alpha_v > 0
        assert alpha_u > 0
        assert 0 <= lambda_v <= 1
        assert 0 <= lambda_u <= 1
        x = np.asarray(x, dtype=float)
        prediction = np.dot(self.w_v, x)
        delta = reward - self.reward_bar + gamma * prediction - self.last_prediction
        self.w_v += alpha_v * delta * self.e_v
        self.w_u += alpha_u * delta * self.e_u
        pi = self.softmax(x)
        action = np.random.choice(self.num_actions, p=pi)
        self.reward_bar += eta * delta
        self.e_v *= lambda_v * gamma
        self.e_v += x
        self.e_u *= lambda_u * gamma
        self.e_u[:, action] += x
        for other in range(self.num_actions):
            self.e_u[:, other] -= x * pi[other]
        self.last_action = action
        self.last_prediction = prediction
        return action, float(delta)


class Brain:
    def __init__(self):
        # initialize buffer
        self.brain_buffer = BrainBuffer()
        self.brain_buffer.queue_size = config["queue_size"]  # max queue size
        self.brain_buffer.external_new_mac_callback = (
            self.new_mac
        )  # to know when new MACs are added
        self.macs = list()
        self.queues = list()
        self.current_images = dict()

        # initialize interface
        self.brain_interface = BrainInterface()
        self.brain_interface.config["queue_size"] = config["queue_size"]
        self.brain_interface.new_mac_callback = self.brain_buffer.new_mac_callback
        self.brain_interface.initialize(config=config)

        # initialize detector
        self.at_detector = pyapriltags.Detector(families="tag16h5", quad_decimate=1.0)

        # wait for all the mac addresses to register
        while len(self.macs) != len(config["limits"]):
            time.sleep(0.01)
        time.sleep(config["sleep"])  # give it a moment for everything to normalize

        # send the servos to their start position
        if config["random_start_location"]:
            self.randomize_servos()
        else:
            self.zero_servos()

    def close(self):
        """Cleanly terminates the interface with the robot."""
        self.brain_interface.stop()

    def new_mac(self, mac, queue) -> None:
        """Hook for when a new mac address is detected."""
        mac = mac2str(mac)
        logging.info("Found new MAC: {}".format(mac))
        self.macs.append(mac)
        if config["set_slew_rate"]:
            self.brain_interface.configure(mac, slew_rate=config["slew_rate"])

    def randomize_servos(self, non_final=True) -> None:
        """Sends all the servos to a random location."""
        self.pos = OrderedDict({k: 0 for k in self.macs})
        for mac in self.macs:
            assert mac in config["limits"]
            self.pos[mac] = scale(
                np.random.rand(),
                0,
                1,
                config["limits"][mac]["min"],
                config["limits"][mac]["max"],
            )
            if not config["dummy_drive"]:
                self.brain_interface.drive(mac, self.pos[mac])
        time.sleep(
            config["sleep"]
        )  # give extra time for all the servos to get to the start position
        if non_final:
            # make sure we haven't hit a final state and if we have try again
            self.update_images()
            tags = self.at_detector.detect(
                np.asarray(self.stich_images()),
                estimate_tag_pose=False,
                camera_params=None,
                tag_size=None,
            )
            if 29 in [tag.tag_id for tag in tags]:
                logging.info("Found target while trying to randomize servos")
                self.randomize_servos(non_final=True)

    def run_ac(self):
        """Learns an optimal policy to move the given servo motors to
        find the tag in one of the given cameras. This version uses a
        discrete actor critic algorithm. Stops after a runtime minutes
        have passed and returns the number of steps taken in each
        iteration.
        """
        # build the tile coder
        tc = TileCoder(
            TileCoderConfig(
                dims=len(self.pos),
                input_ranges=[(v["min"], v["max"]) for v in config["limits"].values()],
                scale_output=False,
                tiles=4,
                tilings=4,
            )
        )
        initial_state = tc.encode(list(self.pos.values()))

        # build the learner
        learner = ACAgent(len(initial_state), 2 * len(config["servos"]))
        action = learner.start(initial_state)

        # start the run
        global_step = 0
        step = 0
        step_counts = list()
        while len(step_counts) < config["num_episodes"]:
            # execute the next action
            mac = config["servos"][action // 2]
            shift = (2 * (action % 2) - 1) * 0.25
            self.pos[mac] = clip(self.pos[mac] + shift, config["limits"][mac])
            if not config["dummy_drive"]:
                self.brain_interface.drive(mac, self.pos[mac])
            global_step += 1
            step += 1
            if config["wandb"]:
                wandb.log(
                    {mac: self.pos[mac] for mac in self.pos.keys()}, step=global_step
                )

            time.sleep(config["sleep"])  # give time for the servo to finish moving

            # get the new images from the cameras
            self.update_images()

            # calculate the reward
            tags = self.at_detector.detect(
                np.asarray(self.stich_images()),
                estimate_tag_pose=False,
                camera_params=None,
                tag_size=None,
            )
            if 29 in [
                tag.tag_id for tag in tags
            ]:  # tag detected; note it and start a new episode
                logging.info("Found target in {} step(s)".format(step))
                step_counts.append(step)
                if config["wandb"]:
                    wandb.log({"episode_length": step}, step=global_step)
                step = 0
                if config["random_start_location"]:
                    self.randomize_servos()
                else:
                    self.zero_servos()
                reward = 1
                gamma = 0
            else:
                reward = 0
                gamma = config["hyperparameters"]["gamma"]

            # update the learner and select the next action to take
            action, delta = learner.step(
                reward,
                gamma,
                tc.encode(list(self.pos.values())),
                config["hyperparameters"]["eta"],
                config["hyperparameters"]["alpha_v"],
                config["hyperparameters"]["alpha_u"],
                config["hyperparameters"]["lambda_v"],
                config["hyperparameters"]["lambda_u"],
            )
            if config["wandb"]:
                wandb.log(
                    {
                        "delta": delta,
                        "e_u norm": np.sum(np.square(learner.e_u)),
                        "e_v norm": np.sum(np.square(learner.e_v)),
                        "reward": reward,
                        "step": step,
                        "w_u norm": np.sum(np.square(learner.w_u)),
                        "w_v norm": np.sum(np.square(learner.w_v)),
                    },
                    step=global_step,
                )

        # save results
        if config["save_model"]:
            with open(config["model_outfile"], "wb") as outfile:
                pickle.dump(learner, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        if config["save_step_counts"]:
            np.save(config["step_counts_outfile"], step_counts)

    def run_brownian(self):
        """Moves the given servo motors repeatedly using Brownian motion
        until the tag is detected in one of the given cameras. Then
        resets all the servos and starts again. Stops after runtime
        minutes have passed and returns the number of steps taken in
        each iteration.
        """
        global_step = 0
        step = 0
        step_counts = list()
        start_time = time.time()
        while time.time() - start_time < 60 * config["runtime"]:
            # get the new images from the cameras
            self.update_images()

            # check for tags and move the servos
            tags = self.at_detector.detect(
                np.asarray(self.stich_images()),
                estimate_tag_pose=False,
                camera_params=None,
                tag_size=None,
            )
            if 29 in [
                tag.tag_id for tag in tags
            ]:  # tag detected; note it and reset the servo positions
                logging.info("Found target in {} step(s)".format(step))
                step_counts.append(step)
                if config["wandb"]:
                    wandb.log({"episode_length": step}, step=global_step)
                global_step += 1
                step = 0
                if config["random_start_location"]:
                    self.randomize_servos()
                else:
                    self.zero_servos()
            else:  # no tag detected; move each servo randomly
                global_step += 1
                step += 1
                if config["single_servo"]:
                    servos = [
                        config["servos"][np.random.randint(len(config["servos"]))]
                    ]
                else:
                    servos = config["servos"]
                for mac in servos:
                    self.pos[mac] = clip(
                        self.pos[mac]
                        + np.random.normal(scale=0.5)
                        * (config["limits"][mac]["max"] - config["limits"][mac]["min"]),
                        config["limits"][mac],
                    )  # Brownian noise
                    if not config["dummy_drive"]:
                        self.brain_interface.drive(mac, self.pos[mac])
                if config["wandb"]:
                    wandb.log({mac: self.pos[mac] for mac in self.pos.keys()})

            time.sleep(config["sleep"])  # give time for the servo(s) to finish moving

        # save results
        if config["save_step_counts"]:
            np.save(config["step_counts_outfile"], step_counts)

    def run_ppo(self):
        """Learns an optimal policy to move the given servo motors to
        find the tag in one of the given cameras. This version uses the
        proximal policy optimization algorithm. Stops after a certain
        number of steps have been taken.

        This implementation is based on
        https://github.com/vwxyzjn/cleanrl
        """
        # use cuda if requested
        if config["hyperparameters"]["cuda"]:
            assert torch.cuda.is_available()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # setup buffers
        obs = torch.zeros(
            (
                config["hyperparameters"]["max_steps"],
                config["hyperparameters"]["num_envs"],
                len(config["limits"]),
            )
        ).to(device)
        actions = torch.zeros(
            (
                config["hyperparameters"]["max_steps"],
                config["hyperparameters"]["num_envs"],
                len(config["servos"]),
            )
        ).to(device)
        logprobs = torch.zeros(
            (
                config["hyperparameters"]["max_steps"],
                config["hyperparameters"]["num_envs"],
            )
        ).to(device)
        rewards = torch.zeros(
            (
                config["hyperparameters"]["max_steps"],
                config["hyperparameters"]["num_envs"],
            )
        ).to(device)
        dones = torch.zeros(
            (
                config["hyperparameters"]["max_steps"],
                config["hyperparameters"]["num_envs"],
            )
        ).to(device)
        values = torch.zeros(
            (
                config["hyperparameters"]["max_steps"],
                config["hyperparameters"]["num_envs"],
            )
        ).to(device)
        next_obs = torch.zeros(
            (config["hyperparameters"]["num_envs"], len(self.pos))
        ).to(device)
        next_done = torch.zeros(config["hyperparameters"]["num_envs"]).to(device)
        all_returns = list()
        all_step_counts = list()

        # build the agent
        agent = PPOAgent().to(device)
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=config["hyperparameters"]["learning_rate"], eps=1e-5
        )

        # start the run
        global_step = 0
        start_time = time.time()
        for iteration in range(1, config["hyperparameters"]["num_iterations"] + 1):
            if config["hyperparameters"]["anneal_lr"]:
                # anneal the learning rate
                frac = (
                    1.0
                    - (iteration - 1.0) / config["hyperparameters"]["num_iterations"]
                )
                lrnow = frac * config["hyperparameters"]["learning_rate"]
                optimizer.param_groups[0]["lr"] = lrnow

            # clear out buffers
            iteration_returns = list()
            iteration_step_counts = list()
            next_done.fill_(0)

            # run each episode
            for run in range(0, config["hyperparameters"]["num_envs"]):
                if config["random_start_location"]:
                    self.randomize_servos()
                else:
                    self.zero_servos()
                next_obs[run, :] = torch.Tensor(list(self.pos.values())).to(device)
                for step in range(0, config["hyperparameters"]["max_steps"]):
                    global_step += 1
                    obs[step, run, :] = next_obs[run, :]
                    dones[step, run] = next_done[run]

                    # select action
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(
                            next_obs[run, :].unsqueeze(0),
                            print_parameters=(run + step) == 0,
                        )
                        values[step, run] = value.flatten()
                    actions[step, run, :] = action
                    logprobs[step, run] = logprob

                    # execute the next action
                    if not next_done[
                        run
                    ]:  # but skip actually running it if we're already done
                        for mac, shift in zip(
                            config["servos"], action.cpu().numpy()[0]
                        ):
                            self.pos[mac] = clip(
                                self.pos[mac] + max(min(shift, 0.4), -0.4),
                                config["limits"][mac],
                            )
                            if not config["dummy_drive"]:
                                self.brain_interface.drive(mac, self.pos[mac])

                        time.sleep(
                            config["sleep"]
                        )  # give time for the servo(s) to finish moving

                    # get the new images from the cameras
                    self.update_images()

                    # update variables
                    next_obs[run, :] = torch.Tensor(list(self.pos.values())).to(device)
                    tags = self.at_detector.detect(
                        np.asarray(self.stich_images()),
                        estimate_tag_pose=False,
                        camera_params=None,
                        tag_size=None,
                    )
                    if len(tags) > 0 and not next_done[run]:
                        logging.info(
                            "Found tag(s) {}".format(
                                ", ".join([str(tag.tag_id) for tag in tags])
                            )
                        )

                    if next_done[run]:
                        reward = 0
                    elif 29 in [tag.tag_id for tag in tags]:  # tag detected; note it
                        iteration_step_counts.append(step + 1)
                        if config["wandb"]:
                            wandb.log({"episode_length": step + 1}, step=global_step)
                        reward = 1
                        terminated = True
                    else:
                        reward = 0
                        terminated = False
                    rewards[step, run] = torch.tensor(reward).to(device)
                    truncated = (step + 1) == config["hyperparameters"]["max_steps"]
                    next_done[run] = 1 if terminated or truncated else 0

                # save the return and step count
                return_ = 0
                for i in reversed(range(config["hyperparameters"]["max_steps"])):
                    return_ *= config["hyperparameters"]["gamma"]
                    return_ += rewards[i, run].cpu()
                iteration_returns.append(return_)
                if not terminated:
                    iteration_step_counts.append(step + 1)
                    if config["wandb"]:
                        wandb.log({"episode_length": step + 1}, step=global_step)

            # log the returns for the iteration
            logging.info("Mean Return = {0:.4f}".format(np.mean(iteration_returns)))
            all_returns.append(iteration_returns)
            all_step_counts.append(iteration_step_counts)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(config["hyperparameters"]["max_steps"])):
                    if t == config["hyperparameters"]["max_steps"] - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + config["hyperparameters"]["gamma"]
                        * nextvalues
                        * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + config["hyperparameters"]["gamma"]
                        * config["hyperparameters"]["gae_lambda"]
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1, len(self.pos)))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, len(config["servos"])))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # optimize the policy and value network
            b_inds = np.arange(config["hyperparameters"]["batch_size"])
            clipfracs = []
            for epoch in range(config["hyperparameters"]["update_epochs"]):
                np.random.shuffle(b_inds)
                for start in range(
                    0,
                    config["hyperparameters"]["batch_size"],
                    config["hyperparameters"]["minibatch_size"],
                ):
                    end = start + config["hyperparameters"]["minibatch_size"]
                    mb_inds = b_inds[start:end]

                    (
                        _,
                        newlogprob,
                        entropy,
                        newvalue,
                    ) = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            (
                                (ratio - 1.0).abs()
                                > config["hyperparameters"]["clip_coef"]
                            )
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if config["hyperparameters"]["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - config["hyperparameters"]["clip_coef"],
                        1 + config["hyperparameters"]["clip_coef"],
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # value loss
                    newvalue = newvalue.view(-1)
                    if config["hyperparameters"]["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -config["hyperparameters"]["clip_coef"],
                            config["hyperparameters"]["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - config["hyperparameters"]["ent_coef"] * entropy_loss
                        + v_loss * config["hyperparameters"]["vf_coef"]
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.parameters(), config["hyperparameters"]["max_grad_norm"]
                    )
                    optimizer.step()

                if (
                    config["hyperparameters"]["target_kl"] is not None
                    and approx_kl > config["hyperparameters"]["target_kl"]
                ):
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            if config["wandb"]:
                # send some values to wandb
                wandb.log(
                    {
                        "approx_kl": approx_kl.item(),
                        "clipfrac": np.mean(clipfracs),
                        "entropy": entropy_loss.item(),
                        "explained_variance": explained_var,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "old_approx_kl": old_approx_kl.item(),
                        "policy_loss": pg_loss.item(),
                        "SPS": int(global_step / (time.time() - start_time)),
                        "value_loss": v_loss.item(),
                        "weight_norm": np.linalg.norm(
                            np.concatenate(
                                [
                                    param.detach().cpu().flatten().numpy()
                                    for param in agent.parameters()
                                ]
                            )
                        ),
                        "mean_return": np.mean(iteration_returns),
                    },
                    step=global_step,
                )

        # save results
        if config["save_episode_returns"]:
            np.save(config["episode_returns_outfile"], all_returns)
        if config["save_model"]:
            torch.save(agent.state_dict(), config["model_outfile"])
        if config["save_step_counts"]:
            np.save(config["step_counts_outfile"], all_step_counts)

    def stich_images(self) -> Image.Image:
        """Stitches current images for the given cameras together into a
        big grey scale one suitable for tag detection.
        """
        macs = list(config["limits"].keys())
        stiched = np.zeros((96 * 2, 96 * 2, 3), dtype=np.uint8)
        if macs[0] in config["cameras"]:
            try:
                np.copyto(stiched[:96, :96, :], self.current_images[macs[0]])
            except (KeyError, OSError):
                pass
        if macs[1] in config["cameras"]:
            try:
                np.copyto(stiched[96:, :96, :], self.current_images[macs[1]])
            except (KeyError, OSError):
                pass
        if macs[2] in config["cameras"]:
            try:
                np.copyto(stiched[:96, 96:, :], self.current_images[macs[2]])
            except (KeyError, OSError):
                pass
        if macs[3] in config["cameras"]:
            try:
                np.copyto(stiched[96:, 96:, :], self.current_images[macs[3]])
            except (KeyError, OSError):
                pass
        return Image.fromarray(stiched, "RGB").convert("L")

    def update_images(self):
        """Empties all the buffers and store the most recent image from
        each camera.
        """
        for mac in list(self.brain_buffer.received_data.keys()):
            buffer = self.brain_buffer.received_data[mac]
            while buffer.qsize() > 0:  # empty the buffer
                try:
                    item = buffer.get(block=False)
                    try:
                        image = Image.open(io.BytesIO(item["data"]))
                        self.current_images[mac2str(mac)] = image
                    except (TypeError, UnidentifiedImageError):
                        pass
                except Empty:
                    pass
        while self.brain_interface.tx_result_queue.qsize() > 0:
            result = self.brain_interface.tx_result_queue.get(block=False)

    def zero_servos(self) -> None:
        """Returns all the servos to their start position."""
        self.pos = OrderedDict({k: 0 for k in self.macs})
        for mac in self.macs:
            assert mac in config["limits"]
            if not config["dummy_drive"]:
                self.brain_interface.drive(mac, self.pos[mac])
        time.sleep(
            config["sleep"]
        )  # give extra time for all the servos to get to the start position


class PPOAgent(torch.nn.Module):
    """Continuous state and action space Proximal Policy Optimization.

    This implementation is based on https://github.com/vwxyzjn/cleanrl
    """

    def __init__(self):
        super().__init__()
        self.critic = torch.nn.Sequential(
            self._layer_init(
                torch.nn.Linear(np.array(len(config["limits"])).prod(), 1), std=1.0
            )
        )
        self.actor_mean = torch.nn.Sequential(
            self._layer_init(
                torch.nn.Linear(
                    np.array(len(config["limits"])).prod(), len(config["servos"])
                ),
                std=0.01,
            )
        )
        self.actor_logstd = torch.nn.Parameter(
            torch.ones(1, len(config["servos"])) * -1
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, print_parameters=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        if print_parameters:
            logging.info(
                "action_mean: {}, action_std: {}".format(
                    list(action_mean.detach().cpu().numpy()[0]),
                    list(action_std.detach().cpu().numpy()[0]),
                )
            )
        probs = torch.distributions.normal.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def clip(v: float, limits: dict[str, float]) -> float:
    """Constrains a servo motor position to be within the limits."""
    return max(min(v, limits["max"]), limits["min"])


def parse_args() -> dict[str, str]:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = vars(parser.parse_args())
    assert os.path.isfile(args["config"])
    return args


def scale(
    value: float, start_min: float, start_max: float, end_min: float, end_max: float
) -> float:
    """Returns the result of scaling value from the range
    [start_min, start_max] to [end_min, end_max].
    """
    return end_min + (end_max - end_min) * (value - start_min) / (start_max - start_min)


if __name__ == "__main__":
    # hide annoying warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # read command line arguments and parse config file
    args = parse_args()
    with open(args["config"], "r") as infile:
        config = yaml.safe_load(infile)
    config.update(args)

    # setup logging
    if config["log_level"] == "DEBUG":
        logging.root.setLevel(logging.DEBUG)
    elif config["log_level"] == "INFO":
        logging.root.setLevel(logging.INFO)
    elif config["log_level"] == "WARNING":
        logging.root.setLevel(logging.WARNING)
    elif config["log_level"] == "ERROR":
        logging.root.setLevel(logging.ERROR)
    elif config["log_level"] == "CRITICAL":
        logging.root.setLevel(logging.CRITICAL)

    # augment config
    config.update({"git_repo_head_hexsha": git.Repo().head.object.hexsha})
    if config["algorithm"] == "ppo":
        config["hyperparameters"]["batch_size"] = (
            config["hyperparameters"]["max_steps"]
            * config["hyperparameters"]["num_envs"]
        )
        config["hyperparameters"]["minibatch_size"] = (
            config["hyperparameters"]["batch_size"]
            // config["hyperparameters"]["num_minibatches"]
        )
        config["hyperparameters"]["num_iterations"] = (
            config["hyperparameters"]["total_timesteps"]
            // config["hyperparameters"]["batch_size"]
        )

    if config["wandb"]:
        # init wandb
        wandb.login()
        wandb.init(config=config, project="testai", save_code=True)

    # print settings
    print(yaml.dump(config))

    # make any needed directories
    for filename in [
        "episode_returns_outfile",
        "model_outfile",
        "step_counts_outfile",
    ]:
        if filename in config:
            if os.path.exists(config[filename]):
                logging.warning("Overwriting {}".format(config[filename]))
            if "/" in config[filename]:
                try:
                    os.makedirs("/".join(config[filename].split("/")[:-1]))
                except OSError:
                    pass

    # init brain
    brain = Brain()

    # start experiment
    try:
        if config["algorithm"] == "ac":
            brain.run_ac()
        elif config["algorithm"] == "brownian":
            brain.run_brownian()
        else:
            assert config["algorithm"] == "ppo"
            brain.run_ppo()
    finally:
        brain.close()
