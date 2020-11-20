# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import numpy as np

from parameter_estimation.estimators import ADREstimator, SimOptEstimator, BayesianOptEstimator, RegressionEstimator, MAMLEstimator, BayesSimEstimator

def get_estimator(reference_env, randomized_env, args):
    if args.estimator_class == 'adr':
        return ADREstimator(
            reference_env=reference_env, 
            randomized_env=randomized_env,
            seed=args.seed,
            nagents=args.nagents,
            nparams=args.nparams,
            temperature=args.temperature, 
            svpg_rollout_length=args.svpg_rollout_length,
            svpg_horizon=args.svpg_horizon,
            max_step_length=args.max_step_length,
            initial_svpg_steps=args.initial_svpg_steps,
            learned_reward=args.learned_reward,
            discriminator_batchsz=320,
        )
    if args.estimator_class == 'simopt':
        e = SimOptEstimator(
            reference_env=reference_env, 
            randomized_env=randomized_env,
            seed=args.seed,
            nagents=args.nagents,
            mean_init=args.mean_init,
            cov_init=args.cov_init,
            reps_updates=args.reps_updates,
            random_init=args.random_init,
            learned_reward=args.learned_reward,
            discriminator_batchsz=320,
        )

        return e 

    elif args.estimator_class == 'bayesopt':
        return BayesianOptEstimator(
            reference_env=reference_env,
            randomized_env=randomized_env,
            seed=args.seed
        )
    elif args.estimator_class == 'regression':
        return RegressionEstimator(
            reference_env=reference_env,
            randomized_env=randomized_env,
            seed=args.seed
        )
    elif args.estimator_class == 'maml':
        e = MAMLEstimator(
            reference_env=reference_env,
            randomized_env=randomized_env,
            seed=args.seed,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate
        )

        return e
    elif args.estimator_class == 'bayessim':
        return BayesSimEstimator(
            reference_env=reference_env,
            randomized_env=randomized_env,
            seed=args.seed,
            prior_mean=None,
            prior_std=None,
            model_name='mdn_torch'
        )