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


from typing import Any, Dict


class Agent:
    def get_action(self) -> Any:
        pass


class BestEffort(Agent):
    @staticmethod
    def get_best_action(actions_to_rewards: Dict[Any, float]) -> Any:
        """
        1.  sort the (action, reward)-pair by reward
        2.  find the best action:
            ->  the one that maximizes reward

        :return:
        """

        # print("all (action, reward) pairs", actions_to_rewards)

        # try to not sort the whole thing

        sorted_rewards = [
            k
            for k, __ in sorted(
                actions_to_rewards.items(), key=lambda item: item[1], reverse=True
            )
            # for k, v in max(rewards.items(), key=lambda item: item[1])
        ]
        # print("actions, sorted", sorted_rewards)

        best_action = sorted_rewards[0]
        return best_action
