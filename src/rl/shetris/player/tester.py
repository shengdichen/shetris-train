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


from typing import Tuple

from src.engine.engine import Engine
from src.rl.shetris.player.base import Agent


class AgentExtreme(Agent):
    def __init__(self, engine: Engine, go_left: bool):
        self._engine = engine
        self._go_left = go_left

    def _pre_left(self) -> Tuple[int, int]:
        rot = 0

        valid_range_shifted = self._engine.mover.analyzer.get_shifted_range1(
            self._engine.pid, rot
        )
        pos1 = valid_range_shifted[0]

        return rot, pos1

    def _pre_right(self) -> Tuple[int, int]:
        rot = 0

        print("PID", self._engine.pid)
        valid_range_shifted = self._engine.mover.analyzer.get_shifted_range1(
            self._engine.pid, rot
        )
        pos1 = valid_range_shifted[1]

        action = rot, pos1
        return action

    def get_action(self) -> Tuple[int, int]:
        if self._go_left:
            return self._pre_left()
        else:
            return self._pre_right()


def run_extremes():
    from src.rl.util.gym.runner import RunnerGym
    from src.rl.shetris.env.shenv import ShetrisEnv

    env = ShetrisEnv()
    r_ins = RunnerGym(env)
    agent = AgentExtreme(env.engine, go_left=False)
    r_ins.run_episodes(lambda: agent.get_action(), 5)


if __name__ == "__main__":
    pass
    # logging.basicConfig(level=logging.INFO)
    run_extremes()
