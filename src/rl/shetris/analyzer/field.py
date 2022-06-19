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


from typing import Optional

import numpy as np


class HeightAnalyzer:
    """
    Analyze the height of a field

    """

    @staticmethod
    def _get_heights_from_top(field: np.ndarray) -> np.ndarray:
        """
        Find the max-heights of every col from the TOP
        ->  the opposite of getting the max "height" of the field

        :param field:
        :return:
        """

        def _find_min_non_zero(col: np.ndarray) -> int:
            return np.min(np.nonzero(col))

        return np.apply_along_axis(_find_min_non_zero, 0, field)

    @staticmethod
    def get_heights_absolute(field: np.ndarray) -> np.ndarray:
        """
        Find the max-heights:
        1.  absolute
        2.  from the bottom, with python's range-style indexing

        :param field:
        :return:
        """

        def _find_min_non_zero(col: np.ndarray, height: int) -> int:
            """
            1.  Find smallest index of non-zero entry of the column
                ->  the "highest"-index from human's perspective
            2.  subtract from the height (of the field)

            NOTE:
            1.  special treatment must be provided for an all-zero column
            2.  output uses 0-indexing

            :param col: one column
            :param height:
            :return:
            """
            if np.count_nonzero(col) == 0:
                return 0
            return height - np.min(np.nonzero(col))

        return np.apply_along_axis(_find_min_non_zero, 0, field, field.shape[0] - 1)

    @staticmethod
    def get_height_max(field: np.ndarray) -> int:
        """
        Get the max height of the entire field

        :param field:
        :return:
        """

        return np.amax(HeightAnalyzer.get_heights_absolute(field))

    @staticmethod
    def get_height_min(field: np.ndarray) -> int:
        """
        Get the min height of the entire field

        :param field:
        :return:
        """

        return np.amin(HeightAnalyzer.get_heights_absolute(field))

    @staticmethod
    def get_height_sum(field: np.ndarray) -> int:
        """
        Get the sum of heights of all columns of the entire field

        :param field:
        :return:
        """

        return HeightAnalyzer.get_heights_absolute(field).sum()

    @staticmethod
    def get_heights_relative(
        field: np.ndarray, clip_at: Optional[int] = None
    ) -> np.ndarray:
        """
        Find the max-heights:
        1.  relative: subtract all by lowest height
        2.  from the bottom (range-style indexing)
        3.  cap out at some max-value if provided

        :param field:
        :param clip_at:
        :return:
        """

        heights_absolute = HeightAnalyzer.get_heights_absolute(field)

        lowest_height = np.min(heights_absolute)
        heights_relative = heights_absolute - lowest_height

        if clip_at is not None:
            return np.clip(heights_relative, a_min=None, a_max=clip_at)

        return heights_relative


def height_test():
    from src.util.fieldfac import FieldReader, FieldFactory

    col = np.array((0, 0, 0))
    print(np.count_nonzero(col))

    print("Sample field")
    f = FieldReader.read_from_file()
    print(HeightAnalyzer.get_heights_absolute(f))
    print(HeightAnalyzer.get_heights_relative(f, 1000))
    print(HeightAnalyzer.get_height_max(f))
    print(HeightAnalyzer.get_height_min(f))
    print(HeightAnalyzer.get_height_sum(f))

    print("Empty field")
    f = FieldFactory.get_all_zeros((20, 10))
    print(HeightAnalyzer.get_heights_absolute(f))
    print(HeightAnalyzer.get_heights_relative(f, 1000))

    print("Full field")
    f = FieldFactory.get_all_ones((20, 10))
    print(HeightAnalyzer.get_heights_absolute(f))
    print(HeightAnalyzer.get_heights_relative(f, 1000))
    print(HeightAnalyzer.get_height_sum(f))


if __name__ == "__main__":
    pass
