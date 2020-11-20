import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from src.sim.cartpole import CartPoleEnv
import numpy as np


class CartpoleRL(object):

    def __init__(self, algo="PPO2", tensorboard_logs="../../../logs/cartpole/",
                 env_params=['length', 'masspole'], verbose=0):

        self.algo = algo
        self.verbose = verbose
        self.env_params = env_params
        self.tensorboard_logs = tensorboard_logs

        env = gym.make("CartPole-v1")
        env = DummyVecEnv([lambda: env])

        if self.algo == "PPO2":
            self.model = PPO2(MlpPolicy, env, verbose=self.verbose, tensorboard_log=self.tensorboard_logs)

    def train(self, total_timesteps=5000, save=False,
              save_path="sac_cartpole", params_proposal=None,
              reset_num_timesteps=False,
              tb_log_name="PPO2",
              prior_lb=[0.1, 0.1],
              prior_ub=[2.0, 2.0]):

        length = 1.0
        masspole = 1.0

        if params_proposal is not None:
            sampling = True
            params = None

            while sampling:
                params = params_proposal().ravel()
                params_out_of_range = 0

                for idx, p in enumerate(self.env_params):
                    if params[idx] < prior_lb[idx] or params[idx] > prior_ub[idx]:
                        params_out_of_range += 1

                if params_out_of_range == 0:
                    sampling = False

            for idx, p in enumerate(self.env_params):
                cur_value = params[idx]

                if p == "length":
                    length = cur_value
                if p == "masspole":
                    masspole = cur_value

        for wrapped_env in self.model.env.envs:
            wrapped_env.env.masspole = masspole
            wrapped_env.env.length = length

        self.model.learn(total_timesteps=total_timesteps, log_interval=10, reset_num_timesteps=reset_num_timesteps,
                         tb_log_name=tb_log_name)

        if save:
            self.model.save(save_path)


if __name__ == "__main__":
    cartpole_rl = CartpoleRL(verbose=1)
    #
    # for i in range(50):
    #     random = np.random.uniform(0.1, 2.0, 2)
    #
    #     mass = random[0]
    #     length = random[1]
    #
    #     print("Starting Iteration: {:.2f} mass value is {:.2f} and length value is {:.2f}".format(i, mass, length))
    #
    #     params = {"mass": mass, "length": length}
    #     pendulum_rl.train(save=False, params=params, reset_timesteps=True)


    # env = gym.make('Pendulum-v2')
    # env.mass = 0.5
    # env.length = 0.5
    # env = DummyVecEnv([lambda: env])

    for i in range(20):
        cartpole_rl.train(total_timesteps=1, reset_num_timesteps=False, tb_log_name="PPO2", params_proposal=param_proposal)

    env = cartpole_rl.model.env.envs[0].env

    while True:
        obs = env.reset()
        print("Mass: {}, Length: {}".format(env.masspole, env.length))
        for _ in range(200):
            action, _states = cartpole_rl.model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()


    # for i in range(20):
    #     pendulum_rl.train(total_timesteps=1000, reset_num_timesteps=False, tb_log_name="SAC", params_proposal=param_proposal)

    # obs = env.reset()
    #
    # while True:
    #     for _ in range(200):
    #         action, _states = pendulum_rl.model.predict(obs)
    #         obs, rewards, dones, info = env.step(action)
    #         env.render()



