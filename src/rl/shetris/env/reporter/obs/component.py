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


from typing import Any, List

import numpy as np

from src.engine.engine import Engine
from src.engine.placement.field import Field
from src.rl.shetris.analyzer.field import (
    HeightAnalyzer,
    ElevationAnalyzer,
    HoleAnalyzer,
)


class _ObsComponent:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_space(self) -> int | List[int]:
        """
        Get the component to be used later with other components to
        construct the final gym-space

        NOTE:
        explicitly require return type to NOT be a gym-space

        :return:
        """

        pass

    def get_obs(self, **kwargs) -> Any:
        pass

    def get_obs_game_over(self, **kwargs) -> Any:
        pass


class _ObsLineClear(_ObsComponent):
    def __init__(self):
        self._max_lines_to_clear = 4
        super().__init__()

    def get_space(self) -> int:
        """
        Valid range: [0, 4], i.e., 5 different values

        :return:
        """

        return self._max_lines_to_clear + 1

    def get_obs(self, line_chunks: list[np.ndarray]) -> int:
        return sum([chunk.size for chunk in line_chunks])

    def get_obs_game_over(self) -> int:
        return 0


class _ObsFieldBased(_ObsComponent):
    def __init__(self, field: Field):
        super().__init__()

        self._field = field.field
        self._height, self._width = field.size


class _ObsFieldCompact(_ObsFieldBased):
    def __init__(self, field: Field):
        super().__init__(field)

    def get_space(self) -> List[int]:
        """
        For the standard 20*10 field:
        1.  height:
            ->  every entry (column) is [0; 20]; 10 of them
            ->  final range is [0, 200]
        2.  elevation:
            ->  every entry (between two columns) is [0; 20]; 9 of them
            ->  final range is [0, 180]
        3.  hole:
            ->  every entry is (column) [0; 19]; 10 of them
            ->  final range is [0, 190]

        Thus, final output is:
        1.  a list of 3 ints

        :return:
        """

        height = self._height * self._width + 1
        elevation = self._height * (self._width - 1) + 1
        hole = (self._height - 1) * self._width + 1

        return [height, elevation, hole]

    def get_obs(self) -> np.ndarray:
        height = HeightAnalyzer.get_height_abs_sum(self._field)
        elevation = ElevationAnalyzer.get_elevation_abs_sum(self._field)
        hole = HoleAnalyzer.get_n_holes_field(self._field)

        return np.array((height, elevation, hole))

    def get_obs_game_over(self) -> np.ndarray:
        return np.array((0, 0, 0))


class _ObsFieldFull(_ObsFieldBased):
    def __init__(self, field: Field):
        super().__init__(field)

    def get_space(self) -> List[int]:
        """
        For the standard 20*10 field:
        1.  height:
            ->  every entry (column) is [0; 20]; 10 of them
        2.  elevation:
            ->  every entry (between two columns) is [0; 20]; 9 of them
        3.  hole:
            ->  every entry is (column) [0; 19]; 10 of them

        Thus, final output is:
        1.  a list of 29 ints

        :return:
        """

        height_list = [self._height + 1] * self._width
        elevation_list = [self._height + 1] * (self._width - 1)
        hole_list = [self._height] * self._width

        space_list = []
        space_list += height_list
        space_list += elevation_list
        space_list += hole_list

        return space_list

    def get_obs(self, **kwargs) -> np.ndarray:
        height = HeightAnalyzer.get_height_abs(self._field)
        elevation = ElevationAnalyzer.get_elevation_abs(self._field)
        hole = HoleAnalyzer.get_n_holes_cols(self._field)

        return np.concatenate((height, elevation, hole))

    def get_obs_game_over(self, **kwargs) -> np.ndarray:
        height = [0] * self._width
        elevation = [0] * (self._width - 1)
        hole = [0] * self._width

        return np.concatenate((height, elevation, hole))


class _ObsEngineBased(_ObsComponent):
    def __init__(self, engine: Engine):
        self._engine = engine

        super().__init__()

    @property
    def engine(self):
        return self._engine


class _ObsPid(_ObsEngineBased):
    def __init__(self, engine: Engine):
        super().__init__(engine)

    def get_space(self) -> int:
        # return self.engine.generator.bag.size
        return 7

    def get_obs(self) -> int:
        return self.engine.pid

    def get_obs_game_over(self) -> int:
        return 0


if __name__ == "__main__":
    pass
