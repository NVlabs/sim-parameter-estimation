# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from gym import utils
from gym.envs.robotics import fetch_env

import numpy as np
import mujoco_py

import xml.etree.ElementTree as et

from gym.spaces import Box

class FetchPushRandomizedEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', **kwargs):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }

        # Ensure we get the path separator correct on windows
        MODEL_XML_PATH = os.path.join('fetch', 'push.xml')

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

        self.config_file = kwargs.get('config')
        self.dimensions = []
        self.reference_xml = et.parse(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', MODEL_XML_PATH))
        self._locate_randomization_parameters()

    def _locate_randomization_parameters(self):
        self.root = self.reference_xml.getroot()
        self.object_joints = self.root.findall(".//body[@name='object0']/joint")
        xml = self._create_xml(randomize=False)
        self._re_init(xml)
        
    def _update_randomized_params(self):
        xml = self._create_xml()
        self._re_init(xml)

    def _re_init(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(4))

        assert not done
        if self.viewer:
            self.viewer.update_sim(self.sim)

    def _create_xml(self, randomize=True):
        if randomize:
            self._randomize_friction()
            self._randomize_damping()

        return et.tostring(self.root, encoding='unicode', method='xml')

    def _randomize_friction(self):
        frictionloss = self.dimensions[0].current_value

        for joint in self.object_joints:
            joint.set('frictionloss', '{:3f}'.format(frictionloss))

    def _randomize_damping(self):
        damping = self.dimensions[1].current_value
        for joint in self.object_joints:
            joint.set('damping', '{:3f}'.format(damping))

class FetchPickAndPlaceRandomizedEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', **kwargs):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }

        MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
            
        utils.EzPickle.__init__(self)

        self.config_file = kwargs.get('config')
        self.dimensions = []
        self.reference_xml = et.parse(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', MODEL_XML_PATH))
        self._locate_randomization_parameters()

    def _locate_randomization_parameters(self):
        self.root = self.reference_xml.getroot()
        self.object_joints = self.root.findall(".//body[@name='object0']/joint")
        xml = self._create_xml(randomize=False)
        self._re_init(xml)
        
    def _update_randomized_params(self):
        xml = self._create_xml()
        self._re_init(xml)

    def _re_init(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(4))

        assert not done
        if self.viewer:
            self.viewer.update_sim(self.sim)

    def _create_xml(self, randomize=True):
        if randomize:
            self._randomize_friction()
            self._randomize_damping()

        return et.tostring(self.root, encoding='unicode', method='xml')

    def _randomize_friction(self):
        frictionloss = self.dimensions[0].current_value

        for joint in self.object_joints:
            joint.set('frictionloss', '{:3f}'.format(frictionloss))

    def _randomize_damping(self):
        damping = self.dimensions[1].current_value
        for joint in self.object_joints:
            joint.set('damping', '{:3f}'.format(damping))

class FetchSlideRandomizedEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', **kwargs):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }

        MODEL_XML_PATH = os.path.join('fetch', 'slide.xml')
        
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
            
        utils.EzPickle.__init__(self)

        self.config_file = kwargs.get('config')
        self.dimensions = []
        self.reference_xml = et.parse(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', MODEL_XML_PATH))
        self._locate_randomization_parameters()

    def _locate_randomization_parameters(self):
        self.root = self.reference_xml.getroot()
        self.object_joints = self.root.findall(".//body[@name='object0']/joint")
        self.object_geoms = self.root.findall(".//body[@name='object0']/geom")
        xml = self._create_xml(randomize=False)
        self._re_init(xml)
        
    def _update_randomized_params(self):
        xml = self._create_xml()
        self._re_init(xml)

    def _re_init(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(4))

        assert not done
        if self.viewer:
            self.viewer.update_sim(self.sim)

    def _create_xml(self, randomize=True):
        if randomize:
            self._randomize_friction()
            self._randomize_damping()

        return et.tostring(self.root, encoding='unicode', method='xml')

    def _randomize_friction(self):
        friction = self.dimensions[0].current_value

        for geom in self.object_geoms:
            geom.set('friction', '{:3f} 0.005 0.0001'.format(friction))

    def _randomize_damping(self):
        damping = self.dimensions[1].current_value
        for joint in self.object_joints:
            joint.set('damping', '{:3f}'.format(damping))