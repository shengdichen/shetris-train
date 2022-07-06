from stable_baselines3.common.base_class import BaseAlgorithm


class Trainer:
    """
    Perform all the training to a sb3-model
    """

    @staticmethod
    def train(model: BaseAlgorithm, n_steps: int = 10000) -> None:
        """
        1.  train: YES
        2.  save: NO

        NOTE:
        not resetting the time-steps is critical for:
        1.  verifying continuous-training
        2.  tensorboard-logging

        :param model:
        :param n_steps:
        :return:
        """

        model.learn(total_timesteps=n_steps, reset_num_timesteps=False)

    @staticmethod
    def train_and_eval(model: BaseAlgorithm, n_steps: int = 10000) -> None:
        """
        1.  Train
            ->  Evaluate the model every 1000 steps on 5 test episodes
            ->  save the evaluation to the "logs/" folder
        2.  Eval after every couple of steps

        Source:
            https://stable-baselines3.readthedocs.io/en/master/guide/
            examples.html#id3

        :param model:
        :param n_steps:
        :return:
        """

        model.learn(n_steps, eval_freq=1000, n_eval_episodes=5, eval_log_path="./logs/")


def quick_train_test():
    rt = setup_test()

    print("[EVAL] before")
    rt.eval_sb3()

    rt.train_quick()

    print("[EVAL] after training")
    rt.eval_sb3()


if __name__ == "__main__":
    pass
