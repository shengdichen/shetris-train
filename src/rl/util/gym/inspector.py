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


if __name__ == "__main__":
    pass
