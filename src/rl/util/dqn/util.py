import os
import random
from pathlib import Path
from typing import Tuple, Any

import torch

from src.rl.shetris.env.reporter.combi import ActionToObs
from src.rl.shetris.env.shenv import ShetrisEnv


class Loader:
    def __init__(self):
        script_path = os.path.dirname(os.path.abspath(__file__)) + "/result/shetris01"
        self.save_latest_path = Path(script_path + "/saveload/")

    def load_model(self):
        """
        1.  load the model from the path
        2.  set the model in eval() mode

        :return:
        """

        torch.manual_seed(123)

        model = torch.load(
            "{0}/latest".format(self.save_latest_path),
            map_location=lambda storage, loc: storage,
        )
        model.eval()

        return model


class Chooser:
    @staticmethod
    def get_action_obs_random(action_all, obs_all):
        idx = random.randint(0, len(action_all) - 1)
        return Chooser.get_action_obs_chosen(action_all, obs_all, idx)

    @staticmethod
    def get_idx_best(q_val_all: torch.Tensor):
        """
        find the index that produces the highest q_val

        :param q_val_all:
        :return:
        """

        best_idx = torch.argmax(q_val_all).item()
        return best_idx

    @staticmethod
    def get_action_obs_best(q_val_all, action_all, obs_all):
        idx = Chooser.get_idx_best(q_val_all)
        return Chooser.get_action_obs_chosen(action_all, obs_all, idx)

    @staticmethod
    def get_action_obs_chosen(action_all, obs_all, idx):
        return action_all[idx], obs_all[idx, :]


class Agent:
    def __init__(self, model: torch.nn.Module, env: ShetrisEnv):
        self._model = model
        self._env = env
        self._action_to_obs = ActionToObs(
            self._env.engine, self._env.provider.obs_factory
        )

    def _get_action_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1.  find all (action, obs)-pairs, unpacked in tuples:
            (action1, action2...), (obs1, obs2...)
        2.  group the tuple of many obs into one matrix:
            1.  each row is an obs

        3.  Thus, returns:
            1.  (action1, action2...), (matrix-of-obs)
        :return:
        """

        action_all, obs_all = self._action_to_obs.get_action_obs_unpacked()
        obs_all = torch.stack(obs_all)
        # print("ACT", action_all)
        # print("OBS_ALL", obs_all)

        return action_all, obs_all

    def _get_q_val_all(self, obs_all: torch.Tensor) -> torch.Tensor:
        """
        Find every q-val of every obs

        2.  torch.nn.Module(x) outputs y-val as a matrix: each row is a y-val
        ->  since in our case, every y-val is the 1D q-val:
        ->  can flatten this into one a VECTOR

        :param obs_all:
        :return:
        """

        q_val_matrix = self._model(obs_all)
        q_val_vector = q_val_matrix[:, 0]

        return q_val_vector

    def act_best(self) -> Tuple[Any, Any]:
        """
        1.  Act in the best-sense of q-val
        2.  return the chosen (action, obs)

        NOTE:
        1.  this follows the style of sb3's predict()

        :return:
        """

        action_all, obs_all = self._get_action_obs()
        q_val_all = self._get_q_val_all(obs_all)

        return Chooser.get_action_obs_best(q_val_all, action_all, obs_all)

    def act_random(self) -> Tuple[Any, Any]:
        action_all, obs_all = self._get_action_obs()

        return Chooser.get_action_obs_random(action_all, obs_all)
