import time
from typing import Any, Callable

import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from src.rl.util.gym.runner import RunnerGym


class Runner:
    """
    All about evaluating SB3's model:

    1.  with sb3's internal mean/std-dev
    2.  native running as a gym's env

    """

    @staticmethod
    def eval_sb3(model: BaseAlgorithm, env: VecEnv, n_eval_episodes: int = 5):
        """
        Use sb3's internal evaluation:
        1.  mean-reward
        2.  standard-variation reward

        Usage:
        1.  use this on an
            1.  untrained
            2.  trained
            model for quick inspection of training gain

        :return:
        """

        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes
        )
        print("[REWARD] (mean {0}) += (std {1})".format(mean_reward, std_reward))

    @staticmethod
    def get_action_generator_model(model: BaseAlgorithm, obs: Any) -> Callable:
        """
        The function that provides the model's predicted action

        :param model:
        :param obs:
        :return:
        """

        def get_model_prediction():
            # explicitly discard the second return value
            return model.predict(obs)[0]

        return get_model_prediction

    @staticmethod
    def _run_episode(env_gym: gym.Env, model: BaseAlgorithm) -> None:
        """
        Perform one episode:
        1.  use the action from the model

        :param env_gym:
        :param model:
        :return:
        """

        ins = RunnerGym(env_gym)
        obs = ins.run_reset()

        episode_num, done = 0, False
        while not done:
            obs, __, done, __ = ins.run_step(
                Runner.get_action_generator_model(model, obs)
            )
            episode_num += 1

            time.sleep(0.1)

        print("Episode finished: {0} steps in total\n\n".format(episode_num))
        time.sleep(2)

    @staticmethod
    def run_episodes(env_gym: gym.Env, model: BaseAlgorithm, n_episodes: int):
        """
        Visually eval a model for many episodes

        :param env_gym:
        :param model:
        :param n_episodes:
        :return:
        """

        for t in range(n_episodes):
            print("Episode Nr. {0}".format(t))
            Runner._run_episode(env_gym, model)


def eval_model_test():
    pass


if __name__ == "__main__":
    pass
    eval_model_test()
