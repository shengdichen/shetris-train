import os
from pathlib import Path
from typing import Type, Optional

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv


class SaveLoad:
    """
    Handle the saving and loading of models
    1.  numbered save-load to document the process
    2.  final save-load for the latest result

    Structure:
    <...>/base_dir
        |__ ./progress/
        |   |__ 0.zip
        |   |__ 1.zip
        |   |__ ...
        |__ latest.zip

    """

    def __init__(
        self,
        base_dir: str,
        progress_name: str = "./progress/",
        latest_name: str = "./latest",
    ):
        self.progress_dir = base_dir + progress_name
        self.latest_name = base_dir + latest_name

        self._curr_cycle_n: int = 0

    @property
    def curr_cycle_n(self):
        return self._curr_cycle_n

    @curr_cycle_n.setter
    def curr_cycle_n(self, value: int):
        self._curr_cycle_n = value

    def save_numbered(self, model: BaseAlgorithm) -> None:
        """
        1.  save to "<location>-<curr_number>"

        :param model:
        :return:
        """

        filename = os.path.join(self.progress_dir, str(self.curr_cycle_n))
        model.save(filename)

        self.curr_cycle_n += 1

    def save_latest(self, model: BaseAlgorithm) -> None:
        """
        Final save

        Usage:
        1.  After a long training cycle, save:
            1.  the final result
            2.  to the final-location

        :param model:
        :return:
        """

        print("Saving model {0} to {1}".format(model, self.latest_name))
        model.save(self.latest_name)
        print("DONE")

    def load_latest(
        self, alg: Type[BaseAlgorithm], env: Optional[VecEnv] = None
    ) -> BaseAlgorithm:
        """
        1.  load the final model
        2.  for the argument env:
            1.  if to continue training:
            ->  must specify
            2.  if to evaluate:
            ->  purely optional: use another sb3-env
            ->  really should not be doing this:
                why use a model on some env that it is not trained on?

        :param alg:
        :param env:
        :return:
        """

        print("Loading model from {0}".format(self.latest_name))
        model = alg.load(path=self.latest_name, env=env, verbose=1)
        print("Loaded model {0}".format(model))

        return model

    def could_continue(self) -> bool:
        """
        check if can continue training:
        ->  check that the "latest"-save exists
        :return:
        """

        latest_zip_name = self.latest_name + ".zip"

        if os.path.isfile(latest_zip_name):
            return True
        else:
            return False

    def get_max_progress(self) -> int:
        """
        1.  find the max file-stem in the progress-dir

        NOTE:
        1.  it is implicitly assumed that all progress saveloads are
        ->  <some-number>.zip

        :return:

        """

        file_stems = [
            int(Path(file).stem)
            for file in os.listdir(self.progress_dir)
            if os.path.isfile(os.path.join(self.progress_dir, file))
        ]
        if file_stems:
            return max(file_stems)
        else:
            return -1

    def init(self) -> bool:
        """
        1.  create the progress-dir
        ->  actually not required, as sb3 will create it itself, but will
        produce a warning
        2.  Check that training could continue
            ->  if yes: set the internal counter to the next stem-number
            ->  else: directly return false

        :return:
        """

        if not os.path.isdir(self.progress_dir):
            Path(self.progress_dir).mkdir(parents=True, exist_ok=True)

        if self.could_continue():
            self.curr_cycle_n = self.get_max_progress() + 1
            return True
        else:
            return False


if __name__ == "__main__":
    pass
