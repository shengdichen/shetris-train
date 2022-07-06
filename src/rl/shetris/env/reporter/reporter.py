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

import numpy as np

from src.engine.engine import Engine
from src.rl.shetris.env.reporter.obs.obs import ObsStandard
from src.rl.shetris.env.reporter.reward.reward import RewardStandard


class Reporter:
    """
    Provide all gym-specific information
    1.  observation
    2.  reward
    3.  info

    """

    def __init__(self, engine: Engine):
        self._engine = engine
        self.obs_factory = ObsStandard(
            self.engine,
            use_compact_field=True,
            use_pid=False,
            # set to False only for manual DQN-training
            use_np=False,
        )
        self.rew_factory = RewardStandard(self.engine)

    @property
    def engine(self):
        return self._engine

    def reset(self) -> Any:
        """
        What to return when reset() is called

        :return:
        """

        return self.obs_factory.get_obs(line_chunks=[])

    def step_game_over(self) -> tuple[Any, float, dict]:
        """
        What to return if game is over

        :return:
        """

        obs = self.obs_factory.get_obs_game_over()
        reward = self.rew_factory.get_reward_gameover()
        info = {}

        return obs, reward, info

    def step_game_on(
        self, corrected: bool, line_chunks: list[np.ndarray]
    ) -> tuple[Any, float, dict]:
        """
        What to return if game is not over:
        1.  perform MOVE and FREEZE

        :param corrected:
        :param line_chunks:
        :return:
        """

        obs = self.obs_factory.get_obs(line_chunks=line_chunks)
        reward = self.rew_factory.get_reward(line_chunks, corrected=corrected)
        info = {}

        return obs, reward, info


if __name__ == "__main__":
    pass
