import jax.numpy as np
import numpy as onp

from parameter_estimation.utils.rollout_evaluation import evaluate_actions


def _get_batch(randomized_env, actions):
    # Select amplitude and phase for the task
    nenvs = randomized_env.nenvs

    random_params = onp.random.uniform(size=(nenvs, randomized_env.randomization_space.shape[0]))
    randomized_env.randomize(randomized_values=random_params)
    randomized_trajectory = evaluate_actions(randomized_env, actions)

    random_params = onp.repeat(random_params[:, :, np.newaxis], len(actions), axis=2)
    random_params = onp.transpose(random_params, (0, 2, 1))
    flattened_params = [random_params[i] for i in range(nenvs)]
    flattened_params = np.concatenate(flattened_params)

    flattened_randomized = [randomized_trajectory[i] for i in range(nenvs)]
    flattened_randomized = np.concatenate(flattened_randomized)

    return flattened_randomized, flattened_params

def sample_tasks(randomized_env, actions):
    xtrain, ytrain = _get_batch(randomized_env, actions)
    xval, yval = _get_batch(randomized_env, actions)
    
    return xtrain, ytrain, xval, yval