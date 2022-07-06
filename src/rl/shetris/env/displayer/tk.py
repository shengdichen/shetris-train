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


import tkinter

import numpy as np

from src.engine.engine import Engine
from src.front.fronttk import FieldTK
from src.rl.shetris.env.displayer.base import Displayer

from src.rl.shetris.env.shenv import ShetrisEnv


class EntryTk:
    root = tkinter.Tk()

    def __init__(self, size: tuple[int, int] = (20, 10)):
        self.env = ShetrisEnv(size)
        self._setups_tk()

    def _setups_tk(self) -> None:
        """
        All the setups necessary for Tk:
        1.  Perform all the key-binds
        2.  Set the correct default focus (on the game-frontend)
        3.  Start the main-loop of Tk

        :return:
        """

        EntryTk.root.bind("<space>", self._init)
        EntryTk.root.bind("<Return>", self._play_agent)

        EntryTk.root.title("Shetris")
        EntryTk.root.mainloop()

    def _init(self, __):
        self.env.render()

    def _play_agent(self, __):
        from src.rl.util.gym.runner import RunnerGym

        r_ins = RunnerGym(self.env)
        r_ins.run_reset()
        r_ins.run_episodes_random(2)


class DisplayerTk(Displayer):
    """
    Put all the GUI-components here:
    1.  the field
    2.  the score
    3.  the preview
    4.  the PRE-area

    """

    def __init__(self, engine: Engine, root):
        super().__init__(engine)

        self._field = FieldTK(root, self._engine.size)

    def display_field(self, field: np.ndarray):
        self._field.set_from_matrix(field)

    def display(self, field: np.ndarray):
        self.display_field(field)
