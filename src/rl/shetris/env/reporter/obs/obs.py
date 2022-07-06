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


from typing import Any, Optional

import gym
import numpy
import numpy as np
import torch

from src.engine.engine import Engine
from src.engine.placement.field import Field
from src.rl.shetris.env.reporter.obs.component import (
    _ObsLineClear,
    _ObsPid,
    _ObsFieldCompact,
    _ObsFieldFull,
)


class ObsFac:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_space(self) -> gym.spaces.Space:
        """
        Produce the final space

        :return:
        """

        pass

    def get_obs(self, **kwargs) -> Any:
        pass

    def get_obs_game_over(self, **kwargs) -> Any:
        pass


class ObsStandard(ObsFac):
    def __init__(
        self,
        engine: Engine,
        use_compact_field: bool = True,
        use_pid: bool = True,
        use_np: bool = True,
    ):
        self._engine = engine

        self._field_type = _ObsFieldCompact if use_compact_field else _ObsFieldFull
        self._lineclear = _ObsLineClear()
        self._use_pid = use_pid
        if self._use_pid:
            self._pid = _ObsPid(self._engine)

        self.space_list = self.set_space_list()

        self._use_np = use_np

        super().__init__()

    def set_space_list(self):
        space_list = []

        space_list += self._field_type(self._engine.field).get_space()
        space_list.append(self._lineclear.get_space())
        if self._use_pid:
            space_list.append(self._pid.get_space())

        return space_list

    def get_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete(self.space_list)

    def get_obs(
        self, line_chunks: list[np.ndarray], field_tmp: Optional[Field] = None, **kwargs
    ) -> np.ndarray | torch.Tensor:
        if field_tmp is None:
            field_observer = self._field_type(self._engine.field)
        else:
            field_observer = self._field_type(field_tmp)
        field = field_observer.get_obs()

        line = self._lineclear.get_obs(line_chunks)

        if self._use_pid:
            pid = self._pid.get_obs()
            obs_np = np.concatenate(
                (
                    field,
                    np.array((line, pid)),
                )
            )
        else:
            obs_np = np.concatenate(
                (
                    field,
                    np.array((line,)),
                )
            )

        if self._use_np:
            return obs_np
        else:
            return torch.from_numpy(numpy.array(obs_np, dtype=np.float32))

    def get_obs_game_over(self) -> np.ndarray | torch.Tensor:
        field_observer = self._field_type(self._engine.field)
        field = field_observer.get_obs_game_over()

        line = self._lineclear.get_obs_game_over()

        if self._use_pid:
            pid = self._pid.get_obs_game_over()
            obs_np = np.concatenate(
                (
                    field,
                    np.array((line, pid)),
                )
            )
        else:
            obs_np = np.concatenate(
                (
                    field,
                    np.array((line,)),
                )
            )

        if self._use_np:
            return obs_np
        else:
            return torch.from_numpy(numpy.array(obs_np, dtype=np.float32))


if __name__ == "__main__":
    test_dict = {"a": 1, "b": 2}
    print(test_dict)

    def p(**kwargs):
        print(type(kwargs), kwargs)

    p(a=1, b=3)
    p(**test_dict)
    p(**{"a": 1})
    print(test_dict.items())
    pass
