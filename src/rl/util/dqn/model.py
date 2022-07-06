import os
import random
import shutil
from collections import deque
from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.rl.shetris.env.reporter.combi import ActionToObs
from src.rl.shetris.env.shenv import ShetrisEnv
from src.rl.util.dqn.network import DeepQNetwork
from src.rl.util.dqn.util import Agent


class ModelDqn:
    def __init__(self, env: ShetrisEnv):
        self.model = DeepQNetwork()
        self.env = env
        self.action_to_obs = ActionToObs(env.engine, env.provider.obs_factory)
        self.agent = Agent(self.model, self.env)

        # TODO:
        #   change to 3000, 200
        self.n_episodes, self.save_interval = 3000, 200
        self.episode_num = 0
        self.step_num = 0
        self.n_pieces, self.n_lines = 0, 0

        script_path = os.path.dirname(os.path.abspath(__file__)) + "/result/shetris"
        self.save_progress_path = script_path + "/saveload/progress/"
        self.save_latest_path = script_path + "/saveload/"
        self.log_path = script_path + "/tensorboard"
        self.writer = SummaryWriter(self.log_path)

        self.replay_memory_size = 30000
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        # TODO:
        #   change to /10
        self.replay_memory_size_pre_fill = self.replay_memory_size / 100

        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.HuberLoss()
        self.criterion = torch.nn.MSELoss()

        self.batch_size = 512
        self.gamma = 0.99

        # stop decreasing epsilon after some episodes
        self.epsilon_init = 1
        self.epsilon_n_decay_episodes = 2000
        self.epsilon_final = 1e-3

        self.setup()

    def setup(self):
        torch.manual_seed(123)
        if os.path.isdir(self.log_path):
            shutil.rmtree(self.log_path)
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_progress_path).mkdir(parents=True, exist_ok=True)

    def do_eval_no_grad(self, functional: Callable) -> Any:
        """
        Perform the task in
        1.  eval-mode
        and
        2.  no_grad-mode

        :param functional:
        :return:
        """

        self.model.eval()
        with torch.no_grad():
            val = functional()
        self.model.train()

        return val

    def train(self):
        obs = self.env.reset()

        replay_counter = 0
        while self.episode_num < self.n_episodes:
            # print(self.episode_num, self.step_num)

            if self.use_random():
                action, obs_next = self.agent.act_random()
            else:
                action, obs_next = self.do_eval_no_grad(self.agent.act_best)
            __, reward, done, __ = self.env.step(action)
            self.step_num += 1
            self.replay_memory.append([obs, reward, obs_next, done])

            if done:
                self.n_pieces, self.n_lines = self.env.n_pieces, self.env.n_lines
                obs = self.env.reset()
            else:
                # perform another step: episode is not over!
                obs = obs_next
                continue

            # pre-build replay memory:
            #   in the pre-build phase:
            #       1.  just fill in
            #       2.  do NOT count toward a training episode
            #       3.  start from beginning
            #   when pre-build is done: (filled past threshold)
            #   ->  start the real training phase
            if len(self.replay_memory) < self.replay_memory_size_pre_fill:
                replay_counter += 1
                print(
                    "[REPLAY] pre-built with {0} episodes: length now {1}".format(
                        replay_counter, len(self.replay_memory)
                    )
                )
                # pre-building buffer
                continue
            self.episode_num += 1

            batch = random.sample(
                self.replay_memory, min(len(self.replay_memory), self.batch_size)
            )
            obs_batch, reward_batch, obs_next_batch, done_batch = zip(*batch)
            obs_batch = torch.stack(tuple(obs for obs in obs_batch))
            reward_batch = torch.from_numpy(
                np.array(reward_batch, dtype=np.float32)[:, None]
            )
            obs_next_batch = torch.stack(tuple(state for state in obs_next_batch))

            q_values = self.model(obs_batch)
            self.model.eval()
            with torch.no_grad():
                next_prediction_batch = self.model(obs_next_batch)
            self.model.train()

            y_batch = torch.cat(
                tuple(
                    reward if done else reward + self.gamma * prediction
                    for reward, done, prediction in zip(
                        reward_batch, done_batch, next_prediction_batch
                    )
                )
            )[:, None]

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, y_batch)
            loss.backward()
            self.optimizer.step()

            print(
                "Episode-Nr.{0} of {1} | n_pieces {2} @ n_lines {3}".format(
                    self.episode_num,
                    self.n_episodes,
                    self.n_pieces,
                    self.n_lines,
                )
            )
            self.write_log()
            self.save_progress()

        self.save_final()

    def save_progress(self):
        if self.episode_num > 0 and self.episode_num % self.save_interval == 0:
            torch.save(
                self.model,
                "{0}/{1}".format(self.save_progress_path, self.episode_num),
            )

    def save_final(self):
        torch.save(self.model, "{0}/latest".format(self.save_latest_path))

    def use_random(self) -> bool:
        """
        Decide if a random action should be used:
        1.  if True: exploration of new actions
        2.  if False: exploitation of existing actions

        :return:
        """

        epsilon_curr = self.epsilon_final + (
            max(self.epsilon_n_decay_episodes - self.episode_num, 0)
            * (self.epsilon_init - self.epsilon_final)
            / self.epsilon_n_decay_episodes
        )
        return random.random() <= epsilon_curr

    def write_log(self) -> None:
        """
        Writing things to the log

        :return:
        """

        episode_num_log = self.episode_num - 1
        step_num_log = self.step_num - 1

        self.writer.add_scalar(
            "episodes/Number of Pieces", self.n_pieces, episode_num_log
        )
        self.writer.add_scalar(
            "episodes/Number of Lines", self.n_lines, episode_num_log
        )

        self.writer.add_scalar("steps/Number of Pieces", self.n_pieces, step_num_log)
        self.writer.add_scalar("steps/Number of Lines", self.n_lines, step_num_log)


def run_train():
    env = ShetrisEnv()
    model = ModelDqn(env)

    model.train()


if __name__ == "__main__":
    run_train()
