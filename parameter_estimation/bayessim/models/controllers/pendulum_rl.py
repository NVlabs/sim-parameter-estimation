import gym
from stable_baselines import SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac.policies import MlpPolicy
from src.sim.pendulum import PendulumEnv
import numpy as np


class PendulumRL(object):

    def __init__(self, algo="SAC", tensorboard_logs="../../../logs/pendulum/",
                 env_params=['length', 'mass'], verbose=0):

        self.algo = algo
        self.verbose = verbose
        self.env_params = env_params
        self.tensorboard_logs = tensorboard_logs

        env = gym.make("Pendulum-v2")
        env = DummyVecEnv([lambda: env])

        if self.algo == "SAC":
            self.model = SAC(MlpPolicy, env, verbose=self.verbose, tensorboard_log=self.tensorboard_logs)

    def dump_model(self):
        data = {
            "learning_rate": self.model.learning_rate,
            "buffer_size": self.model.buffer_size,
            "learning_starts": self.model.learning_starts,
            "train_freq": self.model.train_freq,
            "batch_size": self.model.batch_size,
            "tau": self.model.tau,
            "ent_coef": self.model.ent_coef if isinstance(self.model.ent_coef, float) else 'auto',
            "target_entropy": self.model.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.model.gamma,
            "verbose": self.model.verbose,
            "observation_space": self.model.observation_space,
            "action_space": self.model.action_space,
            "policy": self.model.policy,
            "n_envs": self.model.n_envs,
            "_vectorize_action": self.model._vectorize_action,
            "policy_kwargs": self.model.policy_kwargs
        }

        params = self.model.sess.run(self.model.params)
        target_params = self.model.sess.run(self.model.target_params)

        return (data, params + target_params)

    def load(self, model_cls, model_tuple, env=None, **kwargs):
        data, params = model_tuple

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = model_cls(policy=data["policy"], env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params + model.target_params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model

    def train(self, total_timesteps=5000, save=False,
              save_path="sac_pendulum", params_proposal=None,
              reset_num_timesteps=False,
              tb_log_name="SAC",
              prior_lb=[0.1, 0.1],
              prior_ub=[2.0, 2.0]):

        length = 1.0
        mass = 1.0

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
                if p == "mass":
                    mass = cur_value

        for wrapped_env in self.model.env.envs:
            wrapped_env.env.set_dynamics(mass=mass, length=length)

        self.model.learn(total_timesteps=total_timesteps, log_interval=10, reset_num_timesteps=reset_num_timesteps,
                         tb_log_name=tb_log_name)

        if save:
            self.model.save(save_path)


if __name__ == "__main__":
    pendulum_rl = PendulumRL(verbose=1)
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

    def param_proposal():
        return np.array([0.5, 0.5])

    # env = gym.make('Pendulum-v2')
    # env.mass = 0.5
    # env.length = 0.5
    # env = DummyVecEnv([lambda: env])

    for i in range(20):
        pendulum_rl.train(total_timesteps=200, reset_num_timesteps=False, tb_log_name="SAC", params_proposal=param_proposal)

    env = pendulum_rl.model.env.envs[0].env

    while True:
        obs = env.reset()
        print("Mass: {}, Length: {}".format(env.mass, env.length))
        for _ in range(200):
            action, _states = pendulum_rl.model.predict(obs)
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



