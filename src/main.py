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


def libraries_test():
    """
    Test that we have all the required external packages:
    1.  gym
    2.  stable-baseline3

    :return:
    """

    def gym_test():
        import gym

        gym.make("CartPole-v1")
        print("gym_test passed!")

    def sb3_test():
        from stable_baselines3 import PPO

        PPO("MlpPolicy", "CartPole-v1").learn(1000)
        print("sb3_test passed!")

    gym_test()
    sb3_test()


def main():
    print("Hello, world")
    libraries_test()


if __name__ == "__main__":
    pass
