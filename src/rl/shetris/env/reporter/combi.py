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


from typing import Dict, Tuple, Optional, Any, Type, Callable

import numpy as np
import torch

from src.engine.engine import Engine
from src.engine.placement.field import Field
from src.engine.placement.piece import CoordFactory
from src.rl.shetris.env.reporter.obs.obs import ObsStandard


class _Helper:
    """
    Report:
    1.  (action, obs)

    """

    # if pid == 0:
    #     pairs = self._get_action_obs_pairs(1)
    # elif pid == 1 or pid == 2 or pid == 3:
    #     pairs = self._get_action_obs_pairs(2)
    # else:
    #     pairs = self._get_action_obs_pairs(4)

    _pid = list(range(7))
    _n_rot = [1] + [2] * 3 + [4] * 3
    pid_to_n_rot = dict(zip(_pid, _n_rot))

    def __init__(self, engine: Engine, observer: ObsStandard):
        self._engine = engine
        self._observer = observer

    def do_for_all_actions(self, get_job_func: Callable) -> None:
        n_rot = _Helper.pid_to_n_rot[self._engine.pid]
        for rot in range(n_rot):
            range_pos1 = (
                self._engine.mover.analyzer.get_shifted_range1(self._engine.pid, rot)[1]
                + 1
            )
            for pos1 in range(range_pos1):
                action = rot, pos1

                get_job_func(action)

    def get_obs_tmp(
        self, action: Tuple[int, int], return_none_on_fail: bool = False
    ) -> Optional[Any]:
        """
        Find the obs-vector of the final position of a piece, but do not write
        back to the field

        1.  first action leads to check-gameover:
        2.  if not game-over:
            ->  fall, find num of line-clears
            ->  find the obs
        3.  if game-over:
            ->  return None or the game-over observation by the obs-factory

        NOTE:
        action is guaranteed to be within range

        :param action:
        :param return_none_on_fail:
        :return:
        """

        pre_rot, pre_pos1 = action
        # logging.info("testing with", pre_rot, pre_pos1)
        result = self._engine.mover.attempt_pre(self._engine.piece, pre_rot, pre_pos1)

        if result is not None:
            # logging.debug("PRE-Phase SUCCESSFUL:", result)
            result = self._engine.mover.attempt_drop(result)

            field_tmp = Field(np.copy(self._engine.field.field))
            field_tmp.set_many(result.coord, True)

            vertical_range = CoordFactory.get_range(
                self._engine.pid, self._engine.piece.config, True
            )
            line_chunks = field_tmp.lineclear(vertical_range)
            obs = self._observer.get_obs(line_chunks, field_tmp)
            # logging.info("OBS:", obs)

            # logging.debug("TMP-FIELD", field_tmp.field)
            # logging.debug("TMP-real", engine.field.field)
        else:
            # logging.warning("PRE-Phase FAILED, GAMEOVER!")
            if return_none_on_fail:
                obs = None
            else:
                obs = self._observer.get_obs_game_over()

        return obs

    def get_action_obs_pairs(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        1.  get (action-obs) pairs
        2.  output has obs as Tensor!

        :return:
        """

        pid = self._engine.pid

        if pid == 0:
            pairs = self._get_action_obs_pairs(1)
        elif pid == 1 or pid == 2 or pid == 3:
            pairs = self._get_action_obs_pairs(2)
        else:
            pairs = self._get_action_obs_pairs(4)

        return pairs

    def _get_action_obs_pairs(self, n_rots: int) -> Dict[Tuple[int, int], Any]:
        """
        1.  get (action-obs) pairs

        NOTE:
        unlike the agent, we unconditionally return obs:
        1.  if an action leads to PRE-fail, return all 0
            ->  game-over is for the agent to handle

        :param n_rots:
        :return:
        """

        action_to_obs = {}

        for rot in range(n_rots):
            range_pos1 = (
                self._engine.mover.analyzer.get_shifted_range1(self._engine.pid, rot)[1]
                + 1
            )
            for pos1 in range(range_pos1):
                action = rot, pos1

                action_to_obs[action] = self.get_obs_tmp(
                    action, return_none_on_fail=False
                )

        return action_to_obs

    def single_action_obs(self, action: Tuple[int, int], action_obs_pairs):
        obs = self.get_obs_tmp(action)

        # actually failed PRE
        if obs is None:
            obs = torch.zeros(4, dtype=torch.float)
        else:
            obs = torch.tensor(
                torch.from_numpy(obs),
                dtype=torch.float,
            )
        action_obs_pairs[action] = obs


class ActionToObs:
    def __init__(self, engine: Engine, observer: Optional[ObsStandard] = None):
        self._engine = engine
        if observer is None:
            self._observer = (
                ObsStandard(
                    self._engine, use_compact_field=True, use_pid=True, use_np=True
                ),
            )
        else:
            self._observer = observer
        self._helper = _Helper(self._engine, self._observer)

    def get_action_to_obs(self) -> Dict[Any, Any]:
        """
        Find all:
            (action, obs)-pairs
        packed in a dict: {action1: obs1, oction2: obs2...}

        :return:
        """

        action_to_obs = {}

        def job_func(action):
            action_to_obs[action] = self._helper.get_obs_tmp(
                action, return_none_on_fail=False
            )

        self._helper.do_for_all_actions(job_func)

        return action_to_obs

    def get_action_obs_unpacked(self) -> Tuple[Any, Any]:
        """
        Find all:
            (action, obs)-pairs
        unpacked as two tuples:
            (action1, action2...), (obs1, obs2...)

        :return:
        """

        action_obs_pairs = self.get_action_to_obs()
        action_all, obs_all = zip(*action_obs_pairs.items())

        return action_all, obs_all


class ObsToReward:
    def __init__(self, obs: Any, **kwargs):
        super().__init__(**kwargs)

        self._obs = obs

    def convert(self) -> float:
        """
        Produce the reward

        :return:
        """

        pass


class ActionToReward:
    def __init__(
        self,
        engine: Engine,
        type_converter_obs_to_reward: Type[ObsToReward],
    ):
        self._engine = engine
        self._observer = ObsStandard(
            self._engine,
            use_compact_field=True,
            use_pid=False,
            use_np=True,
        )
        self._helper = _Helper(self._engine, self._observer)

        self._converter_type = type_converter_obs_to_reward

    def get_action_to_reward(self):
        """
        1.  get all (acton-reward) pairs
        2.  adapt for different pieces' varying amount of rotations
            ->  not really required for calculation, but saves some work

        :return:
        """

        action_to_reward = {}

        self._helper.do_for_all_actions(self._get_add_reward_to_dict(action_to_reward))
        return action_to_reward

    def _get_add_reward_to_dict(self, action_to_reward: Dict):
        def add_reward_to_dict(action):
            obs = self._helper.get_obs_tmp(action)
            reward = self._converter_type(obs).convert()
            action_to_reward[action] = reward

        return add_reward_to_dict


if __name__ == "__main__":
    pass
