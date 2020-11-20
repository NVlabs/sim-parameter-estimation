# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from abc import ABC, abstractmethod

class BaseEstimator(object):
    @abstractmethod
    def __init__(self,
                 reference_env,
                 randomized_env,
                 seed,
                 **kwargs):

        self.reference_env = reference_env
        self.randomized_env = randomized_env
        self.seed = seed
    
    @abstractmethod
    def load_trajectory(self, reference_action_fp):
        pass
    
    @abstractmethod
    def get_parameter_estimate(self, randomized_env):
        pass
    
    @abstractmethod
    def update_parameter_estimate(self, randomized_env):
        pass
