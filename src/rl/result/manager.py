# The Reinforcement-Learning Module of the Shetris-Project
#
# Copyright (C) 2022 Shengdi 'shc' Chen (me@shengdichen.xyz)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


import os.path
from pathlib import Path
from typing import Type

from src.rl.result.info import InfoProvider
from src.rl.util.sb3.model import ModelFactory
from src.rl.util.sb3.runner import Runner
from src.rl.util.sb3.saveload import SaveLoad
from src.rl.util.sb3.training import Trainer


class RlManager:
    """
    A collection of common tasks for great training:
    1.  training and saving
    2.  comparing models:
        1.  untrained
        2.  from saved zip

    """

    def __init__(self, info_provider: Type[InfoProvider]):
        self._env_gym, self._env = info_provider.get_envs()
        self._alg, self._policy = info_provider.get_algpol()

        script_path = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(script_path, "./result/")
        self.tensorboard_dir, self.saveload_dir = info_provider.get_abs_dirs(
            self.base_dir
        )

        self._model_raw = ModelFactory.create_model(
            self.env, self.alg, self.policy, self.tensorboard_dir
        )
        self.model = None
        self.saveload = SaveLoad(self.saveload_dir)

        self.init()

    @property
    def env_gym(self):
        return self._env_gym

    @property
    def env(self):
        return self._env

    @property
    def alg(self):
        return self._alg

    @property
    def policy(self):
        return self._policy

    @property
    def model_raw(self):
        return self._model_raw

    def _create_dirs(self) -> None:
        """
        Create all the paths required

        :return:
        """

        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self.saveload_dir).mkdir(parents=True, exist_ok=True)

    def init(self):
        """
        Some initial setups:
        1.  create all dirs
        2.  load the model
        ->  explicitly specify the env:
        ->  though not required for just viewing the model, necessary for
        continuing training

        :return:
        """

        self._create_dirs()

        if self.saveload.init():
            print("CONTINUE!")
            self.model = self.saveload.load_latest(self.alg, self.env)
        else:
            print("from scratch")
            self.model = ModelFactory.create_model(
                self.env, self.alg, self.policy, self.tensorboard_dir
            )

    def _train_save_one_cycle(self, n_steps: int) -> None:
        """
        Perform one train-save cycle:
        1.  train the model
        2.  save the in-progress model

        :return:
        """

        Trainer.train(self.model, n_steps)
        self.saveload.save_numbered(self.model)

    def train_save(self, n_cycles: int, n_steps: int) -> None:
        """
        Perform multiple train-save cycles:
        1.  train and save each cycle
        2.  save the final model

        :param n_cycles:
        :param n_steps:
        :return:
        """

        for __ in range(n_cycles):
            self._train_save_one_cycle(n_steps)

        self.saveload.save_latest(self.model)

    def compare_against_untrained(self):
        """
        1.  Load the latest saved model
        2.  eval it:
            1.  sb3
            2.  visual inspection

        :return:
        """

        print("[MODEL]-raw:")
        Runner.eval_sb3(self.model_raw, self.env)

        print()

        print("[MODEL]-trained:")
        Runner.eval_sb3(self.model, self.env)

    def run_visual(self):
        """
        1.  Load the latest saved model
        2.  eval it:
            1.  sb3
            2.  visual inspection

        :return:
        """

        print("[MODEL]-trained - visual inspection")
        Runner.run_episodes(self.env_gym, self.model, 5)

    def from_scratch(self, n_cycles: int, n_steps: int):
        """
        Force start training from scratch
        1.  without an existing save-load, must first create one
        2.  afterwards, the usual train-save routine

        :param n_cycles:
        :param n_steps:
        :return:
        """

        self.model = ModelFactory.create_model(
            self.env, self.alg, self.policy, self.tensorboard_dir
        )

        self.train_save(n_cycles, n_steps)

    def from_existing(self, n_cycles: int, n_steps: int):
        """
        Force training

        1.  directly load the existing save-load
        2.  keep training on it

        :param n_cycles:
        :param n_steps:
        :return:
        """

        self.model = self.saveload.load_latest(self.alg)

        self.train_save(n_cycles, n_steps)


def setup_test():
    # from src.rl.result.info import CartpoleInfo
    from src.rl.result.info import ShetrisInfo

    return RlManager(ShetrisInfo)


def auto_train():
    rt = setup_test()
    rt.train_save(500, 10000)


def load_eval_visual_test():
    rt = setup_test()
    rt.run_visual()


if __name__ == "__main__":
    # pass
    auto_train()
    # load_eval_visual_test()
