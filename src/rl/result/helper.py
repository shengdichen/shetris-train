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
from typing import Type, Optional

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv


class IoHelper:
    @staticmethod
    def get_script_abs_dir() -> str:
        """
        Retrieve the absolute path of this script.

        :return: absolute path of this script
        """

        real_path = os.path.realpath(__file__)
        return os.path.dirname(real_path)

    @staticmethod
    def get_filename_abs(filename_rel: str):
        """
        Retrieve the absolute filename from filename relative to script-dir

        :param filename_rel:
        :return:
        """

        scriptdir_abs = IoHelper.get_script_abs_dir()
        filename_abs = os.path.join(scriptdir_abs, filename_rel)

        return filename_abs

    @staticmethod
    def save_model(model: BaseAlgorithm, filename_rel: str) -> None:
        """
        Save the model under (relative) filename

        :param model:
        :param filename_rel:
        :return:
        """

        filename_abs = IoHelper.get_filename_abs(filename_rel)

        print("Saving model to {0}".format(filename_abs))
        model.save(filename_abs)

    @staticmethod
    def load_model(
        alg: Type[BaseAlgorithm], filename_rel: str, env: Optional[VecEnv] = None
    ) -> BaseAlgorithm:
        """
        1.  Use the algorithm
        2.  load from (calculated) absolute-path
        3.  use the provided env

        :param alg:
        :param filename_rel:
        :param env:
        :return:
        """

        filename_abs = IoHelper.get_filename_abs(filename_rel)
        return alg.load(path=filename_abs, env=env, verbose=1)


def abs_dir_test():
    print(IoHelper.get_script_abs_dir())
    print(IoHelper.get_filename_abs("test_name"))


if __name__ == "__main__":
    pass
    abs_dir_test()
