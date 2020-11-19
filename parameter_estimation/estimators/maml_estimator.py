import jax.numpy as np
from jax.experimental import optimizers

from parameter_estimation.estimators import BaseEstimator
from parameter_estimation.maml.data import sample_tasks
from parameter_estimation.maml.network import generate_network
# from parameter_estimation.maml.train_utils import loss, inner_update, batch_maml_loss
from parameter_estimation.maml.data import sample_tasks

from jax import vmap, random, grad
from functools import partial

from parameter_estimation.utils.rollout_evaluation import evaluate_actions

from jax import vmap # for auto-vectorizing functions
from functools import partial # for use with vmap
from jax import jit # for compiling functions for speedup

import jax.numpy as np
from jax import grad

from jax.experimental import optimizers
from jax.tree_util import tree_multimap 


class MAMLEstimator(BaseEstimator):
    def __init__(self,
                 reference_env,
                 randomized_env,
                 seed,
                 **kwargs):

        self.reference_env = reference_env
        self.randomized_env = randomized_env
        self.seed = seed

        self.randomization_dim = randomized_env.randomization_space.shape[0]
        self.nenvs = randomized_env.nenvs

        input_dim = randomized_env.action_space.shape[0] + randomized_env.observation_space.shape[0] * 2
        print(input_dim)
        in_shape = (-1, input_dim)

        self.net_init, self.net_apply = generate_network(self.randomization_dim)

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=1e-3)

        rng = random.PRNGKey(0)
        self.out_shape, self.net_params = self.net_init(rng, in_shape)
        self.opt_state = self.opt_init(self.net_params)

        self.i = 0

    def load_trajectory(self, reference_env, reference_action_fp):
        self.reference_actions = np.load(reference_action_fp)
        self.reference_trajectory = evaluate_actions(reference_env, self.reference_actions)
        self.flattened_reference = np.squeeze(self.reference_trajectory)    
    
    def get_parameter_estimate(self, randomized_env):
        predictions_array = []

        net_params_inference = self.get_params(self.opt_state)
        predictions = np.mean(vmap(partial(self.net_apply, net_params_inference))(self.flattened_reference), axis=0)

        predictions_array.append(predictions)

        for _ in range(2):
            _, _, xval, yval = sample_tasks(randomized_env, self.reference_actions) 
            net_params_inference = self._inner_update(net_params_inference, xval, yval)
            predictions = np.mean(vmap(partial(self.net_apply, net_params_inference))(self.flattened_reference), axis=0)
            predictions_array.append(predictions)

        return predictions_array
    
    def update_parameter_estimate(self, randomized_env):
        xtrain, ytrain, xval, yval = sample_tasks(randomized_env, self.reference_actions) 
        self.opt_state, l = self._step(xtrain, ytrain, xval, yval)
        
        self.i += 1

    # TODO: needs to be jit-ed
    def _step(self, x1, y1, x2, y2):
        p = self.get_params(self.opt_state)

        g = grad(self._batch_maml_loss)(p, x1, y1, x2, y2)
        l = self._batch_maml_loss(p, x1, y1, x2, y2)
        return self.opt_update(self.i, g, self.opt_state), l

    def _loss(self, params, inputs, targets):
        # Computes average loss for the batch
        predictions = self.net_apply(params, inputs)
        return np.mean((targets - predictions)**2)

    def _inner_update(self, p, x1, y1, alpha=.1):
        grads = grad(self._loss)(p, x1, y1)
        inner_sgd_fn = lambda g, state: (state - alpha*g)
        return tree_multimap(inner_sgd_fn, grads, p)

    def _maml_loss(self, p, x1, y1, x2, y2):
        p2 = self._inner_update(p, x1, y1)
        return self._loss(p2, x2, y2)

    # vmapped version of maml loss.
    # returns scalar for all tasks.
    def _batch_maml_loss(self, p, x1_b, y1_b, x2_b, y2_b):
        task_losses = vmap(partial(self._maml_loss, p))(x1_b, y1_b, x2_b, y2_b)
        return np.mean(task_losses)

            