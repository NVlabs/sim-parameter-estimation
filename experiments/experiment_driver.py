import matplotlib
matplotlib.use('Agg')

import random
import logging

import numpy as np
import torch
import gym
import argparse
import os

from parameter_estimation.utils.logging import reshow_hyperparameters
from parameter_estimation.envs.randomized_vecenv import make_vec_envs

from experiments.args import get_args
from experiments.estimator_helper import get_estimator

def run_experiment(args):
    reshow_hyperparameters(args, paths={})

    if os.path.exists('evaluations{}/{}-{}-{}-evals.npy'.format(args.suffix, args.estimator_class, args.reference_env_id, args.learned_reward)):
        return

    if args.estimator_class in ['maml', 'regression', 'bayesopt'] and args.learned_reward:
        return

    reference_env = make_vec_envs(args.reference_env_id, args.seed, 1)
    randomized_env = make_vec_envs(args.randomized_env_id, args.seed, args.nagents)

    parameter_estimator = get_estimator(reference_env, randomized_env, args)

    logging.root.setLevel(logging.INFO)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    parameter_estimator.load_trajectory(reference_env, 'trajectories/{}-actions.npy'.format(args.reference_env_id))
    
    logging.info('Loaded Trajectories')
    evaluations = []
    for iteration in range(500):
        parameter_estimator.update_parameter_estimate(randomized_env)
        if iteration % 5 == 0:
            print(
                args.estimator_class,
                args.reference_env_id,
                args.learned_reward,
                iteration, 
                parameter_estimator.get_parameter_estimate(randomized_env)                
            )
        
        evaluations.append(parameter_estimator.get_parameter_estimate(randomized_env))
        np.save('evaluations{}/{}-{}-{}-evals'.format(args.suffix, args.estimator_class, args.reference_env_id, args.learned_reward), evaluations)
        
    reshow_hyperparameters(args, paths={})

if __name__ == '__main__':
    args = get_args()
    run_experiment(args)