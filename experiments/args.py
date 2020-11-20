import argparse
import logging

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Domain Randomization Driver')

    subparsers = parser.add_subparsers(help='sub-command help', dest='estimator_class')
    
    simopt_subparser = subparsers.add_parser('simopt', help='SimOpt subparser')
    simopt_subparser.add_argument("--reps-updates", default=4, type=int, help="Number of randomization parameters")
    simopt_subparser.add_argument("--nagents", default=4, type=int, 
            help="Number of threads")

    bayesopt_subparser = subparsers.add_parser('bayesopt', help='bayesopt subparser')
    bayesopt_subparser.add_argument("--nagents", default=1, type=int, 
            help="Number of threads")

    regression_subparser = subparsers.add_parser('regression', help='regression subparser')
    regression_subparser.add_argument("--nagents", default=1, type=int, 
            help="Number of threads")

    maml_subparser = subparsers.add_parser('maml', help='maml subparser')
    maml_subparser.add_argument("--nagents", default=4, type=int, 
            help="Number of threads")

    adr_subparser = subparsers.add_parser('adr', help='ADR subparser')
    adr_subparser.add_argument("--nparams", default=2, type=int, help="Number of randomization parameters")
    adr_subparser.add_argument("--temperature", default=10.0, type=float, 
        help="SVPG temperature")
    adr_subparser.add_argument("--svpg-rollout-length", default=5, type=int, 
        help="length of one svpg particle rollout")
    adr_subparser.add_argument("--svpg-horizon", default=25, type=int, 
        help="how often to fully reset svpg particles")
    adr_subparser.add_argument("--max-step-length", default=0.05, 
        type=float, help="step length / delta in parameters; If discrete, this is fixed, If continuous, this is max.")
    adr_subparser.add_argument("--initial-svpg-steps", default=0, type=float, 
        help="number of svpg steps to take before updates")
    adr_subparser.add_argument("--nagents", default=4, type=int, 
            help="Number of threads")

    for subparser in [adr_subparser, simopt_subparser, bayesopt_subparser, regression_subparser, maml_subparser]:
        subparser.add_argument("--randomized-env-id", default="DoorRandomized-v0",
        type=str, help="Name of the reference environment")
        subparser.add_argument("--reference-env-id", default="DoorDefault-v0", 
        type=str, help="Name of the randomized environment")
        subparser.add_argument("--max-agent-timesteps", default=1e6, type=float, 
            help="max iterations, counted in terms of AGENT env steps")
        subparser.add_argument("--seed", default=123, type=int)
        subparser.add_argument("--learned-reward", action='store_true', help="Use learned reward or not") 
        subparser.add_argument("--suffix", type=str, default="")

    return parser.parse_args()