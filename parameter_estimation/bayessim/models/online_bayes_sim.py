import time

from tqdm import tqdm

from src.utils.param_inference import *


class OnlineBayesSim(object):
    def __init__(self,
                 mog=None,
                 generator=None,
                 seed=None,
                 verbose=True,
                 controller=None):

        self.generator = generator  # generates the data
        self.seed = seed
        self.verbose = verbose
        self.round = 0
        self.proposal = None
        self.controller = controller

        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

        self.mog = mog

    def gen(self, n_samples, current_policy=None, param_proposal=None, prior_lb=None, prior_ub=None, env_params=None):
        """Generate from generator and z-transform

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        n_reps : int
            Number of repeats per parameter
        verbose : None or bool or str
            If None is passed, will default to self.verbose
        """

        all_data = []
        all_params = []

        for _ in tqdm(range(n_samples)):
            sampling = True
            cur_params = None

            while sampling:
                cur_params = param_proposal().ravel()
                params_out_of_range = 0

                for idx, p in enumerate(env_params):
                    if cur_params[idx] < prior_lb[idx] or cur_params[idx] > prior_ub[idx]:
                        params_out_of_range += 1

                if params_out_of_range == 0:
                    sampling = False

            all_data.append(self.generator.gen_single(cur_params)["data"])
            all_params.append(cur_params)

        # z-transform params and stats
        # params = (params - self.params_mean) / self.params_std
        # stats = (stats - self.stats_mean) / self.stats_std
        return all_params, all_data

    def run(self, init_training_samples=500,
            online_training_samples=100,
            init_batch_size=50,
            online_batch_size=10,
            init_controller_steps=10000,
            online_controller_steps=3000,
            init_mog_epochs=200,
            online_mog_epochs=100,
            total_epochs=20,
            init_policy=None,
            prior_iterations=50,
            param_prior=None,
            true_obs=[0.5, 0.1],
            prior_lb=None,
            prior_ub=None,
            env_params=None,
            save_every=None,
            save_progress=None):
        """
        Run online Bayes Sim training both policy and posterior over dynamics

        :param init_training_samples: int
            Number of samples collected by the forward model to train initial policy
        :param online_training_samples: int
            Number of samples collected by the forward model to train online policy
        :param init_batch_size: int
            Batch size used on the initial policy training
        :param online_batch_size: int
            Batch size used on the online policy training
        :param init_controller_steps: int
            Number of steps/episodes to train the initial controller
        :param online_controller_steps: int
            Number of steps/episodes to train the online controller
        :param init_mog_epochs: int
            Number of epochs to train the initial Mixture of Gaussians model
        :param online_mog_epochs: int
            Number of epochs to train the online Mixture of Gaussians
        :param total_epochs: int
            Total number of epochs to perform online training
        :param init_policy: policy that follows stable-baselines interface
            Provide a pre-trained controller to intialize the algorithm
        :param prior_iterations: int
            How many parameters to sample from the prior to train the initial controller
        :param param_prior: function that returns array with the size of num of parameters
            A Prior function that returns simulations parameters when called
        :param true_obs:
            The true observation to evaluate the prior with
        :param prior_lb:
            Prior lower bound
        :param prior_ub:
            Prior Upper bound
        :param env_params:
            The name of the environment params
        :param save_every:
            Save the progress every "x" step
        :return:
        """
        logs = []
        trn_datasets = []

        self.rewards_arr = []

        cur_time = time.strftime("%Y%m%d-%H%M%S")
        print("\n\nTraining Initial Policy..")

        if init_policy is None:
            for _ in tqdm(range(prior_iterations)):
                self.controller.train(save=False, params_proposal=param_prior,
                                      reset_num_timesteps=False, tb_log_name="INIT_SAC",
                                      total_timesteps=init_controller_steps)
        else:
            self.controller.model = init_policy

        print("\n\nGenerating data from initial policy...")

        y_data, x_data = self.gen(init_training_samples, current_policy=self.controller.model,
                                  param_proposal=param_prior,
                                  prior_lb=prior_lb, prior_ub=prior_ub, env_params=env_params)

        _, log = self.mog.train(x_data=x_data,
                                y_data=y_data,
                                nepoch=init_mog_epochs,
                                batch_size=init_batch_size)

        print("\n\nRunning online learning for {} epochs".format(total_epochs))

        if save_every is not None:
            if save_progress is not None:
                save_progress(idx=0, cur_time=cur_time, posterior=None,
                              param_samples=y_data, true_obs=true_obs,
                              samples_title="Uniform Prior Samples",
                              controller=self.controller,
                              rewards_arr=self.rewards_arr)

            # This is needed as deep copy doesn't work on keras models, need to decouple them
            self.mog.save(save_config=True,
                          save_model_weights=False,
                          model_config_fname=os.path.join(str(cur_time), "config", "mog_config.pkl"))

        for idx in tqdm(range(total_epochs)):

            posterior = get_posterior_from_true_obs(generator=self.generator, inf=self,
                                                    true_obs=np.array(true_obs)[np.newaxis, :])
            print("\n\nTraining new controller...")
            self.controller.train(save=False, params_proposal=posterior.gen, reset_num_timesteps=False,
                                  total_timesteps=online_controller_steps)

            new_y, new_x = self.gen(online_training_samples, current_policy=self.controller.model,
                                    param_proposal=posterior.gen,
                                    prior_lb=prior_lb, prior_ub=prior_ub, env_params=env_params)

            y_data = np.concatenate((y_data, new_y))
            x_data = np.concatenate((x_data, new_x))

            print("\nBuffer size: {}".format(len(y_data)))

            random_idx = np.random.randint(0, len(y_data), size=online_training_samples)

            if save_every is not None:
                if save_progress is not None:
                    save_progress(idx=idx + 1, cur_time=cur_time, posterior=posterior,
                                  param_samples=new_y, true_obs=true_obs,
                                  rewards_arr=self.rewards_arr,
                                  controller=self.controller)

                self.mog.save(save_config=False,
                              save_model_weights=True,
                              weights_name=os.path.join(str(cur_time), "checkpoints",
                                                        "{0:0=2d}".format(idx) + "_mog_weights.h5"))

                self.controller.model.save(os.path.join(str(cur_time), "checkpoints",
                                                        "{0:0=2d}".format(idx) + "_controller.pkl"))

            print("\n\nMean sampled params: {}".format(np.mean(y_data, axis=0)))

            _, log = self.mog.train(x_data=x_data[random_idx],
                                    y_data=y_data[random_idx],
                                    nepoch=online_mog_epochs,
                                    batch_size=online_batch_size)

        return logs, trn_datasets

    def predict(self, x):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        return self.mog.predict_mog(x)[0]  # via super

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2 ** 31)
