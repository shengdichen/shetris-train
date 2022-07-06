import logging

import numpy as np

from src.engine.engine import Engine
from src.rl.shetris.env.reporter.combi import ObsToReward, ActionToReward
from src.rl.shetris.player.base import Agent, BestEffort


class ObsToRewardGenetic(ObsToReward):
    """
    Reward function based on the Genetic-Algorithm proposed by:
       https://github.com/LeeYiyuan/tetrisai

    """

    reward_coeff = np.array((-0.51, -0.18, -0.35, 0.76))

    def __init__(self, obs: np.ndarray):
        super().__init__(obs)

    def convert(self):
        """
        Dot-product the reward-vector with coeff to find the final reward

        :return:
        """

        if self._obs is not None:
            logging.info(
                "weighted separates", ObsToRewardGenetic.reward_coeff * self._obs
            )
            reward = np.dot(ObsToRewardGenetic.reward_coeff, self._obs)
        else:
            reward = None
        return reward


class AgentGenetic(Agent):
    def __init__(self, engine: Engine):
        super().__init__()

        self._engine = engine

        self._reporter_combi = ActionToReward(self._engine, ObsToRewardGenetic)

    def get_action(self) -> np.ndarray:
        actions_to_rewards = self._reporter_combi.get_action_to_reward()
        return np.array(BestEffort.get_best_action(actions_to_rewards))


def run_genetic():
    from src.rl.util.gym.runner import RunnerGym
    from src.rl.shetris.env.shenv import ShetrisEnv

    env = ShetrisEnv()
    r_ins = RunnerGym(env)
    agent = AgentGenetic(env.engine)
    r_ins.run_episodes(lambda: agent.get_action(), 5)


if __name__ == "__main__":
    pass
    # logging.basicConfig(level=logging.INFO)
    run_genetic()
