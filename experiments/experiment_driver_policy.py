# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


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

from policy.ddpg import DDPG

def run_experiment(args):
    reshow_hyperparameters(args, paths={})

    reference_env = make_vec_envs(args.reference_env_id, args.seed, args.nagents)
    randomized_env = make_vec_envs(args.randomized_env_id, args.seed, args.nagents)

    parameter_estimator = get_estimator(reference_env, randomized_env, args)

    logging.root.setLevel(logging.INFO)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    parameter_estimator.load_trajectory(reference_env, f'trajectories/{args.reference_env_id}.npy')
    logging.info('Loaded Trajectories')

    t = 0

    state_dim = randomized_env.observation_space.shape[0]
    action_dim = randomized_env.action_space.shape[0] 
    max_action = 1.0

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }

    policy = DDPG(**kwargs)
    evaluations = []

    for iteration in range(args.num_iterations):
        parameter_estimator.update_parameter_estimate(randomized_env, policy, reference_env)

        if iteration % args.log_interval == 0:
            logging.info(
                args.jobid,
                args.estimator_class,
                args.reference_env_id,
                args.learned_reward,
                iteration, 
                parameter_estimator.get_parameter_estimate(randomized_env)                
            )
        
        # TODO: Should be held out, test data
        evaluations.append(parameter_estimator.get_parameter_estimate(randomized_env))
        np.save('evaluations/{}-{}-{}-{}-policy-evals'.format(
            args.estimator_class, args.reference_env_id, args.learned_reward, args.suffix), evaluations)

    reshow_hyperparameters(args, paths={})

if __name__ == '__main__':
    args = get_args()
    run_experiment(args)