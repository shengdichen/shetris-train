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


import os
from typing import Tuple, Type, Optional

import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from src.rl.util.sb3.model import EnvFactorySb3, AlgPolFactory


class InfoProvider:
    """
    Raw information of a SB3-info:
    1.  static info: does not depend on training

    """

    tensorboard_subdir = "tensorboard/"
    saveload_subdir = "saveload/"

    @staticmethod
    def get_envs() -> Tuple[gym.Env, VecEnv]:
        """
        Provide:
        1.  the gym-env
        2.  the sb3-env

        :return:
        """

        pass

    @staticmethod
    def get_algpol() -> Tuple[Type[BaseAlgorithm], Type[BasePolicy]]:
        """
        Obtain the (alg, policy)-pair

        :return:

        """
        pass

    @staticmethod
    def get_rel_dirs() -> Tuple:
        """
        Set the relative directories for:
        1.  tensor-board logging
        2.  saving during training

        NOTE:
        1.  return relative-path such that the call-site's can still define
        the main location for the save-load

        :return:
        """

        pass

    @staticmethod
    def get_abs_dirs(base_abs: Optional[str] = None) -> Tuple:
        """
        Prepend the absolute-base

        :param base_abs:
        :return:
        """

        pass


class ShetrisInfo(InfoProvider):
    """
    For the Shetris Env

    """

    @staticmethod
    def get_envs() -> Tuple[gym.Env, VecEnv]:
        from src.rl.shetris.env.shenv import ShetrisEnv

        env_gym_type = ShetrisEnv
        env_gym = env_gym_type()
        env = EnvFactorySb3.get_env_dummy(env_gym_type)

        return env_gym, env

    @staticmethod
    def get_algpol() -> Tuple[Type[BaseAlgorithm], Type[BasePolicy]]:
        # return AlgPolFactory.get_ppo()
        # return AlgPolFactory.get_a2c()
        return AlgPolFactory.get_dqn()

    @staticmethod
    def get_rel_dirs() -> Tuple[str, str]:
        base = "./shetris/"
        tensorboard = os.path.join(base, InfoProvider.tensorboard_subdir)
        saveload = os.path.join(base, InfoProvider.saveload_subdir)

        return tensorboard, saveload

    @staticmethod
    def get_abs_dirs(base_abs: Optional[str] = None):
        """
        Prepend the absolute-base

        :param base_abs:
        :return:
        """

        tensorboard, saveload = ShetrisInfo.get_rel_dirs()
        return os.path.join(base_abs, tensorboard), os.path.join(base_abs, saveload)


class CartpoleInfo(InfoProvider):
    """
    For the Cartpole-Env

    """

    @staticmethod
    def get_envs() -> Tuple[gym.Env, VecEnv]:
        from gym.envs.classic_control.cartpole import CartPoleEnv

        env_gym_type = CartPoleEnv
        env_gym = env_gym_type()
        env = EnvFactorySb3.get_env_dummy(env_gym_type)

        return env_gym, env

    @staticmethod
    def get_algpol() -> Tuple[Type[BaseAlgorithm], Type[BasePolicy]]:
        return AlgPolFactory.get_ppo()

    @staticmethod
    def get_rel_dirs() -> Tuple[str, str]:
        base = "./cartpole/"
        tensorboard = os.path.join(base, InfoProvider.tensorboard_subdir)
        saveload = os.path.join(base, InfoProvider.saveload_subdir)

        return tensorboard, saveload

    @staticmethod
    def get_abs_dirs(base_abs: Optional[str] = None):
        """
        Prepend the absolute-base

        :param base_abs:
        :return:
        """

        tensorboard, saveload = CartpoleInfo.get_rel_dirs()
        return os.path.join(base_abs, tensorboard), os.path.join(base_abs, saveload)


if __name__ == "__main__":
    pass
