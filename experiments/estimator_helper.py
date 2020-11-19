import numpy as np

from parameter_estimation.estimators import ADREstimator, SimOptEstimator, BayesianOptEstimator, RegressionEstimator, MAMLEstimator

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
        return SimOptEstimator(
            reference_env=reference_env, 
            randomized_env=randomized_env,
            seed=args.seed,
            nagents=args.nagents,
            mean_init=0.5,
            cov_init=0.1,
            reps_updates=args.reps_updates,
            learned_reward=args.learned_reward,
            discriminator_batchsz=320,
        )
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
        return MAMLEstimator(
            reference_env=reference_env,
            randomized_env=randomized_env,
            seed=args.seed
        )