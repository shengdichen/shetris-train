import numpy as np

from src.rl.shetris.env.shenv import ShetrisEnv
from src.rl.util.dqn.util import Loader, Agent
from src.rl.util.gym.runner import RunnerGym


def test_run():
    env = ShetrisEnv()
    model = Loader().load_model()
    agent = Agent(model, env)

    r_ins = RunnerGym(env)
    r_ins.run_episodes(lambda: agent.act_best()[0], 5)


if __name__ == "__main__":
    # 3, dqn
    # print(np.average([152, 572, 1127, 708, 143, 402, 263, 520]))
    # ALL, dqn
    # print(np.average([211, 123, 222, 518, 195, 311, 164, 180, 41, 188]))

    # 3, hardcoded
    # print(np.average([145, 289, 385, 232, 346, 46, 178, 190, 165, 212]))
    # ALL, Hard-coded
    # print(np.average([311, 397, 434, 1400, 691, 380, 1523, 2343, 1700]))
    print(5 + 3 + 4 + 1 + 0 + 3 + 2 + 5 + 6)
    # test_run()
