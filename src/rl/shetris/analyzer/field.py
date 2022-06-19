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


class ElevationAnalyzer:
    """
    Probably the best judge of stacking quality

    Convention:
    Elevation is defined as:
        Left-Col-Height - Right-Col-Height
    i.e., elevation is by definition DIFF in height of neighboring columns

    Thus,
    1.  a field of constantly decreasing height from left to right
    ->  all positive elevation values
    2.  the returned np.ndarray has length of (n_cols - 1)

    NOTE:
    extension documentation at:
        https://tetris.wiki/Stack_shape_terminology
        https://tetris.wiki/Stacking_for_Tetrises

    """

    @staticmethod
    def get_elevations(field: np.ndarray) -> np.ndarray:
        """
        Get all elevations

        :param field:
        :return:
        """

        heights_rel = HeightAnalyzer.get_heights_relative(field)

        return heights_rel[:-1] - heights_rel[1:]

    @staticmethod
    def get_n_level(field: np.ndarray, level: int) -> int:
        """
        Get the number of:
        1.  specific elevation levels

        :param field:
        :param level:
        :return:
        """

        elevations = ElevationAnalyzer.get_elevations(field)
        return np.count_nonzero(elevations == level)

    @staticmethod
    def get_n_floor_2(field: np.ndarray) -> int:
        """
        Get the number of 1-length floor:
        1.  1 consecutive elevation of 0

        Usage:
        1.  encourage: necessary for an O-piece

        :param field:
        :return:
        """

        return ElevationAnalyzer.get_n_level(field, 0)

    @staticmethod
    def get_n_level_greater(field: np.ndarray, limit: int) -> int:
        """
        Get the number of significant elevations

        Usage:
        1.  punish: I-piece dependency

        :param field:
        :param limit:
        :return:
        """

        elevations = ElevationAnalyzer.get_elevations(field)
        return np.count_nonzero(elevations > limit)

    @staticmethod
    def get_n_level_less(field: np.ndarray, limit: int) -> int:
        """
        Get the number of significant elevations

        Usage:
        1.  punish: I-piece dependency

        :param field:
        :param limit:
        :return:
        """

        elevations = ElevationAnalyzer.get_elevations(field)
        return np.count_nonzero(elevations < limit)

    @staticmethod
    def get_n_level_abs_greater(field: np.ndarray, limit: int) -> int:
        """
        Get the number of significant elevations:
        1.  elevation > +limit
        2.  elevation < -limit

        NOTE:
        1.  it is up to the caller to guarantee that limit is of sensible
        value, i.e. at least positive

        Usage:
        1.  pass in limit as +2 to punish I-piece dependency

        :param field:
        :param limit:
        :return:
        """

        elevations = ElevationAnalyzer.get_elevations(field)
        return np.count_nonzero((elevations > limit) | (elevations < -limit))


def elevation_test():
    from src.util.fieldfac import FieldReader, FieldFactory

    col = np.array((0, 0, 0))
    print(np.count_nonzero(col))

    print("Sample field")
    f = FieldReader.read_from_file("checks/check_elevation")
    print(ElevationAnalyzer.get_elevations(f))
    print(ElevationAnalyzer.get_n_level(f, 1))
    print(ElevationAnalyzer.get_n_floor_2(f))

    print()
    print(ElevationAnalyzer.get_n_level_greater(f, 2))
    print(ElevationAnalyzer.get_n_level_less(f, -2))
    print(ElevationAnalyzer.get_n_level_abs_greater(f, 2))

    f = FieldFactory.get_all_zeros((20, 10))
    print(ElevationAnalyzer.get_elevations(f))
    print(ElevationAnalyzer.get_n_level(f, 1))
    print(ElevationAnalyzer.get_n_floor_2(f))

    print()
    print(ElevationAnalyzer.get_n_level_greater(f, 2))
    print(ElevationAnalyzer.get_n_level_less(f, -2))
    print(ElevationAnalyzer.get_n_level_abs_greater(f, 2))


class HoleAnalyzer:
    """
    Get the holes of a field

    """

    @staticmethod
    def get_holes_col(col: np.ndarray) -> np.ndarray:
        """
        Find (numpy-)indexes of holes of a column:
        1.  find the highest point
        2.  find all entries (strictly) under the highest point
        3.  find all indexes of zero entries of these

        NOTE:
        return type is np.ndarray, where length is apparently varying
        ->  the results must be stored in a python's list (not np.ndarray!)

        :return:
        """

        if np.count_nonzero(col) == 0:
            return np.empty((0,), dtype=int)

        idx_to_start = np.min(np.nonzero(col)) + 1
        col_under = col[idx_to_start:]
        col_inverted_under = np.logical_not(col_under)

        idx_holes_under = np.nonzero(col_inverted_under)[0]
        return idx_holes_under + idx_to_start

    @staticmethod
    def get_n_holes_col(col: np.ndarray) -> int:
        """
        1.  find the number of holes of a column

        NOTE:
        1.  slight hack using np's internal bool-type

        :return:
        """

        if np.count_nonzero(col) == 0:
            return 0

        idx_to_start = np.min(np.nonzero(col)) + 1
        col_under = col[idx_to_start:]
        return np.count_nonzero(col_under == np.False_)

    @staticmethod
    def get_n_holes_cols(field: np.ndarray) -> np.ndarray:
        """
        1.  find the num-of-cols of all columns
            ->  returns a np.ndarray

        :param field:
        :return:
        """

        return np.apply_along_axis(HoleAnalyzer.get_n_holes_col, 0, field)


if __name__ == "__main__":
    pass
