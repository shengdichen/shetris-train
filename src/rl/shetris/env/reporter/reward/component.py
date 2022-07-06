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


class _RewardComponent:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_reward(self, **kwargs) -> Any:
        pass


class RewardLineClear(_RewardComponent):
    def __init__(self, engine: Engine):
        self._engine = engine

        super().__init__()

    @staticmethod
    def _get_reward_chunk(chunk: np.ndarray) -> int:
        """
        Deploy the official guideline's scoring mechanism for a chunk:
            https://tetris.wiki/Scoring

        :param chunk:
        :return:
        """

        n_lines = chunk.size
        if n_lines == 1:
            return 1
        elif n_lines == 2:
            return 3
        elif n_lines == 3:
            return 5
        else:
            return 8

    def get_reward(self, line_chunks: list[np.ndarray]) -> int:
        """
        1.  if given an I-piece:
            Punish all line-clears that are not a Tetris-clear

        :param line_chunks:
        :return:
        """

        scores = [RewardLineClear._get_reward_chunk(chunk) for chunk in line_chunks]
        return 10 * self._engine.field.size[1] * sum(scores)


class RewardGameover(_RewardComponent):
    def __init__(self, engine: Engine):
        self._engine = engine

        super().__init__()

    def get_reward(self) -> int:
        """
        1.  1 if game-over
        2.  0 otherwise

        :return:
        """

        if self._engine.is_game_over:
            return -10
        else:
            return 1
