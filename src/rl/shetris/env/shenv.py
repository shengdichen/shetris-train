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


from typing import Tuple, Any, Optional, List

import gym
import numpy as np

from src.engine.engine import Engine
from src.entry.fetcher.gym import FetcherGym
from src.entry.stepper.freeze import FreezePhaseGym
from src.entry.stepper.move import MovePhase
from src.entry.stepper.pre import PrePhaseGym
from src.rl.shetris.env.displayer.base import Displayer
from src.rl.shetris.env.displayer.text import DisplayerText
from src.rl.shetris.env.reporter.reporter import Reporter


class ShetrisEnv(gym.Env):
    """
    Entry for Gym, using the Minimal setup:
    0.  init the entry
    1.  allow conversation with gym:
        1.  convert engine's internal to gym's-observation (and info)
        2.  convert gym-action to engine's input

    1.  PRE-phase
        1.  corrected
    2.  MOVE-phase:
        1.  ignored
    3.  FREEZE-phase:
        1.  include the drop

    """

    def __init__(
        self,
        size: tuple[int, int] = (20, 10),
        displayer: Optional[List[Displayer]] = None,
    ):
        super().__init__()

        self._engine = Engine(size)
        self._fetcher = FetcherGym
        self._provider = Reporter(self.engine)
        if displayer is None:
            self._displayers = [DisplayerText(self.engine)]
        else:
            self._displayers = displayer

        self.observation_space, self.action_space = None, None
        # set to True only if using DQN
        self._flatten_action = False
        self._set_spaces()

        self.n_pieces, self.n_lines = 0, 0

    @property
    def engine(self):
        return self._engine

    @property
    def fetcher(self):
        return self._fetcher

    @property
    def provider(self):
        return self._provider

    def _set_spaces(self):
        """
        Set the spaces of the shetris

        :return:
        """

        self.observation_space = self._provider.obs_factory.get_space()
        print(self.observation_space)

        if self._flatten_action:
            self.action_space = gym.spaces.Discrete(40)
        else:
            self.action_space = gym.spaces.MultiDiscrete((4, 10))

    def _misc_init(self):
        """
        1.  set-up RNG
        2.  set-up front-end (if necessary)

        This is called:
        2.  in the constructor only!
            ->  i.e., once (even if running multiple episodes)

        :return:
        """

        self._set_seed()

    def _set_seed(self, seed=None):
        """
        Use gym's internal seeding mechanism to perform seeding, conforms to
        the recommendation of gym's documentation:
            https://www.gymlibrary.ml/content/api/#resetting
            https://www.gymlibrary.ml/content/environment_creation/#reset

        from gym.utils.seeding import np_random

        self.np_random, seed = np_random(seed)
        return [seed]

        NOTE:
        since
        1.  gym's np_random uses the legacy RNG
        2.  our engine's only random-source, the bag-generator, applies its own
        RNG, nothing is performed here

        :param seed:
        :return:
        """

        return

    def reset(self):
        """
        Produce the gym-obs just after a reset

        :return: observation (after a reset)
        """

        # print("RESET")
        self.engine.reset()
        self.n_pieces, self.n_lines = 0, 0
        return self.provider.reset()

    def _pre_phase(self, action: np.ndarray) -> bool:
        """
        Apply pre_phase:
        1.  use correction
        2.  return the correction-result:
            1.  True if correction was applied
            2.  False otherwise

        :return:
        """

        def action_generator():
            return self.fetcher.get_pre_corrected(self.engine, action)

        return PrePhaseGym.correction_aware(self.engine, action_generator)

    @staticmethod
    def _move_phase():
        """
        1.  Use the minimal Move-Phase

        :return:
        """

        MovePhase.minimal()

    def _freeze_phase(self) -> list[np.ndarray]:
        """
        1.  Include a drop

        :return:
        """

        return FreezePhaseGym.with_drop(self.engine)

    def step(self, action: int | np.ndarray) -> Tuple[Any, float, bool, dict]:
        if self._flatten_action:
            return self._step_flattened(action)
        else:
            return self._step(action)

    def _step_flattened(self, action: int) -> Tuple[Any, float, bool, dict]:
        """
        Used if action-space is MULTI():
            ->  e.g. if using dqn of SB3

        :param action:
        :return:
        """

        action = self._action_int_to_np(action)

        return self._step(action)

    def _action_int_to_np(self, action: int) -> np.ndarray:
        """
        Convert an action of Multi() to an array for the engine

        Usage:
        if action-space is Multi (necessary for DQN)

        :param action:
        :return:
        """

        width = self.engine.field.size[1]
        return np.array((action // width, action % width))

    def _step(self, action: np.ndarray) -> Tuple[Any, float, bool, dict]:
        """
        1.  Convert gym-action to entry-action
        2.  Perform 1-step by the entry
        3.  Fetch the result after the step

        :param action:
        :return:
        """

        # print("STEP")
        corrected = self._pre_phase(action)
        if self.engine.is_game_over:
            return self.step_game_over()
        else:
            return self.step_game_on(corrected)

    def step_game_over(self) -> Tuple[Any, float, bool, dict]:
        done = True
        obs, reward, info = self.provider.step_game_over()

        return obs, reward, done, info

    def step_game_on(self, corrected: bool) -> Tuple[Any, float, bool, dict]:
        ShetrisEnv._move_phase()
        line_chunks = self._freeze_phase()
        self.n_pieces += 1
        self.n_lines += sum([chunk.size for chunk in line_chunks])

        done = False
        obs, reward, info = self.provider.step_game_on(corrected, line_chunks)

        return obs, reward, done, info

    def render(self, mode: str = "human"):
        """
        1.  Do NOT put this in step()
        2.  let the enjoyer decide if they would like to render

        :param mode:
        :return:
        """

        for displayer in self._displayers:
            displayer.display(n_pieces=self.n_pieces, n_lines=self.n_lines)

    def close(self):
        pass


def run_sb3_check():
    from src.rl.util.sb3.model import EnvFactorySb3

    EnvFactorySb3.env_check(ShetrisEnv())


def run_gym_check():
    from src.rl.util.gym.inspector import MiscInspector

    MiscInspector.run_api_check(ShetrisEnv())


def episode_run():
    from src.rl.util.gym.runner import RunnerGym

    env = ShetrisEnv()
    r_ins = RunnerGym(env)
    # r_ins.run_reset()
    r_ins.run_episodes_random(2)


if __name__ == "__main__":
    # pass
    # EntryTk()
    # run_sb3_check()
    run_gym_check()
    # episode_run()
