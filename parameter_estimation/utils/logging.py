import os
import time
import logging
import numpy as np


def reshow_hyperparameters(args, paths):
    logging.info("---------------------------------------")
    logging.info("Arguments for Finished Experiment:")
    for arg in vars(args):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    logging.info("Relevant Paths for Finished Experiment:")
    for key, value in paths.items():
        logging.info("{}: {}".format(key, value))
    logging.info("---------------------------------------\n")