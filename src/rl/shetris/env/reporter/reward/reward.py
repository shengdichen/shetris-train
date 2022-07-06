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


import numpy as np

from src.engine.engine import Engine
from src.rl.shetris.env.reporter.reward.component import RewardLineClear, RewardGameover


class RewardFactory:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_reward(self, **kwargs) -> float:
        pass

    def get_reward_gameover(self, **kwargs) -> float:
        pass


class RewardStandard(RewardFactory):
    def __init__(self, engine: Engine):
        self._engine = engine

        self._lineclear = RewardLineClear(self._engine)
        self._gameover = RewardGameover(self._engine)

        super().__init__()

    def get_reward(self, line_chunks: list[np.ndarray], **kwargs) -> float:
        lineclear = self._lineclear.get_reward(line_chunks)
        gameover = self._gameover.get_reward()

        return float(lineclear + gameover)

    def get_reward_gameover(self) -> float:
        return 0.0


if __name__ == "__main__":
    pass
