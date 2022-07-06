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


from src.engine.engine import Engine
from src.rl.shetris.env.displayer.base import Displayer


class DisplayerText(Displayer):
    def __init__(self, engine: Engine):
        super().__init__(engine)

    @staticmethod
    def _display_scores(n_pieces: int, n_lines: int):
        print("TOTAL PIECES {0} @ {1} LINES".format(n_pieces, n_lines))

    def display(self, n_pieces: int, n_lines: int):
        # self._engine.field.print_field()
        # print("[NEXT PIECE]\n", self._engine.piece)
        # print("PID-Reservoir", self._engine.generator.reservoir.data)

        DisplayerText._display_scores(n_pieces, n_lines)
