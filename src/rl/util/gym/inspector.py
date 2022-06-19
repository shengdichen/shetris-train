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


from typing import Any

import gym
import numpy as np


class SpaceInspector:
    """
    Inspect spaces

    """

    @staticmethod
    def inspect_entries(space: gym.spaces.Space, n_samples: int = 5) -> None:
        """
        View some randomly generated samples from the space

        :param space:
        :param n_samples:
        :return:
        """

        print("Randomly sampling entries from space", space)
        for __ in range(n_samples):
            print(space.sample())

    @staticmethod
    def inspect_entry_type(space: gym.spaces.Space) -> None:
        """
        View the type of entries of the space

        :param space:
        :return:
        """

        print(
            "[SPACE] {0} contains entries of type: {1}".format(
                space, type(space.sample())
            )
        )

    @staticmethod
    def check_in_space(space: gym.spaces.Space, entry: Any) -> None:
        """
        Check if any entry is in a space

        :param space:
        :param entry:
        :return:
        """

        print(
            "Checking entry {0} in space {1}: {2}".format(entry, space, entry in space)
        )

    @staticmethod
    def inspect_discrete() -> None:
        """
        Inspect the Discrete() space
            ->  entry is int

        :return:
        """

        space = gym.spaces.Discrete(7)
        SpaceInspector.inspect_entries(space)

        entry_1, entry_2 = 6, 7
        SpaceInspector.check_in_space(space, entry_1)
        SpaceInspector.check_in_space(space, entry_2)

    @staticmethod
    def inspect_multidiscrete() -> None:
        """
        Inspect the MultiDiscrete() space
            ->  entry is np.array
            ->  raw-list is also acceptable as entry's type

        :return:
        """

        space = gym.spaces.MultiDiscrete([10, 10, 7])
        SpaceInspector.inspect_entries(space)

        entry_1, entry_2 = np.append(np.array([1, 2]), 6), np.array([1, 2, 7])
        SpaceInspector.check_in_space(space, entry_1)
        SpaceInspector.check_in_space(space, entry_2)

    @staticmethod
    def inspect_tuple() -> None:
        """
        Inspect the Tuple() space:
            (entry-of-space1, entry-of-space2, ...)
            ->  which is a tuple

        :return:
        """

        space = gym.spaces.Tuple(
            (gym.spaces.Discrete(7), gym.spaces.MultiDiscrete([10, 10]))
        )
        SpaceInspector.inspect_entries(space)

        entry_1, entry_2 = (5, np.array((1, 9))), (5, np.array((1, 10)))
        SpaceInspector.check_in_space(space, entry_1)
        SpaceInspector.check_in_space(space, entry_2)

    @staticmethod
    def view_spaces(env: gym.Env) -> None:
        """
        View the action- and obs-spaces of a gym's env

        :return:
        """

        act_space = env.action_space

        print("Action")
        SpaceInspector.inspect_entry_type(act_space)

        obs_space = env.observation_space
        print("Observation")
        SpaceInspector.inspect_entry_type(obs_space)


def spaces_test():
    ins = SpaceInspector
    ins.inspect_discrete()
    ins.inspect_multidiscrete()
    ins.inspect_tuple()


if __name__ == "__main__":
    pass
