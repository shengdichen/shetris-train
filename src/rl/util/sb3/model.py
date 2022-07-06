from typing import Type, Callable, Any

import gym
import stable_baselines3
import stable_baselines3.common
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


class EnvFactorySb3:
    """
    Given:
    1.  gym-env
    Returns
    1.  a sb3-ready env
        ->  wrapped with monitor
        ->  wrapped with vec

    Usage:
    0.  for gym-env:
        1.  API-check
    2.  create sb3-env:
        ->  wrap with vector-wrappers

    """

    @staticmethod
    def env_check(env_gym: gym.Env) -> None:
        """
        Check the raw gym-env for API-compatibility with sb3

        :return:
        """

        import stable_baselines3.common.env_checker

        print("api-checks by SB3")
        env_gym.reset()
        stable_baselines3.common.env_checker.check_env(env_gym)

    @staticmethod
    def get_maker_monitored_env(env_gym_type: Type[gym.Env]) -> Callable[[], gym.Env]:
        """
        Produces a function to create a SB3-monitored env:
        1.  the Monitor is a just gym-wrapper,
        ->  i.e., the result is still a gym.Env

        :param env_gym_type:
        :return:
        """

        def make_one_env() -> gym.Env:
            from stable_baselines3.common.monitor import Monitor

            return Monitor(env_gym_type())

        return make_one_env

    @staticmethod
    def get_env_dummy(env_gym_type: Type[gym.Env]) -> DummyVecEnv:
        """
        1.  Create a sb3-env, wrapped with:
            1.  sb3's Monitor-wrapper
            2.  some sb3's vector-wrapper
        2.  vector-wrappers:
            ->  DummyVec (just one env, faking the required vectorization)
            ->  Subproc (real vectorization)
            https://stable-baselines.readthedocs.io/en/master/guide/
            examples.html#
            multiprocessing-unleashing-the-power-of-vectorized-environments
        3.  necessary for A2C and PPO

        :return:
        """

        maker_monitored_env = EnvFactorySb3.get_maker_monitored_env(env_gym_type)
        return DummyVecEnv([maker_monitored_env])

    @staticmethod
    def get_env_subproc(env_gym_type: Type[gym.Env]) -> SubprocVecEnv:
        pass


class AlgPolFactory:
    """
    Create various pairs of (alg, pol):
    1.  PPO
    2.  A2C

    """

    @staticmethod
    def get_dqn() -> tuple[Type[BaseAlgorithm], Type[BasePolicy]]:
        """
        Create the pair for using A2C

        :return:
        """

        alg = stable_baselines3.DQN
        policy = stable_baselines3.dqn.policies.DQNPolicy

        return alg, policy

    @staticmethod
    def get_ppo() -> tuple[Type[stable_baselines3.ppo.PPO], Type[BasePolicy]]:
        """
        Create the pair for using PPO

        :return:
        """

        alg = stable_baselines3.PPO
        policy = stable_baselines3.ppo.policies.MlpPolicy

        return alg, policy

    @staticmethod
    def get_a2c() -> tuple[Type[stable_baselines3.a2c.A2C], Type[BasePolicy]]:
        """
        Create the pair for using A2C

        :return:
        """

        alg = stable_baselines3.A2C
        policy = stable_baselines3.a2c.policies.MlpPolicy

        return alg, policy


class ModelFactory:
    """
    Create a sb3-model:
    1.  a sb3-env
    2.  a (alg, polj)-pair

    """

    @staticmethod
    def create_model(
        env_sb3: VecEnv, alg: Any, policy: Type[BasePolicy], tb_dir: str
    ) -> BaseAlgorithm:
        """
        Create the model from scratch

        NOTE:
        1.  the env must be sb3-model

        :return:
        """

        return alg(policy, env_sb3, verbose=1, tensorboard_log=tb_dir, seed=147)

    @staticmethod
    def inspect_on_policy_specifics(model: OnPolicyAlgorithm):
        """
        Not available from a BasePolicy

        :param model:
        :return:
        """

        print(
            "On-Policy Model:\n"
            "[GAMMA]: {0} | [N-STEPS]: {1}".format(model.gamma, model.n_steps)
        )

    @staticmethod
    def inspect_model(model: OnPolicyAlgorithm):
        """
        1.  view the model
        2.  view the underlying sb3-env
            ->  find wrappers

        :return:
        """

        print("MODEL: ", model)
        print("ENV-SB3: ", model.env)
        print("[ENV] unwrapped: ", model.env.unwrapped)


def setup_test():
    from gym.envs.classic_control.cartpole import CartPoleEnv

    return CartPoleEnv


def check_api_test():
    from gym.envs.classic_control.cartpole import CartPoleEnv

    EnvFactorySb3.env_check(CartPoleEnv())


if __name__ == "__main__":
    pass
    check_api_test()
