import os.path as osp
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
import os
import xml.etree.ElementTree as et

from gym import spaces

ADD_BONUS_REWARDS = True

class DoorRandomizedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        self.reference_path = osp.join(osp.dirname(os.path.abspath(__file__)), 'assets', 'hms', 'DAPG_door.xml')

        mujoco_env.MujocoEnv.__init__(self, self.reference_path, frame_skip=5)
        utils.EzPickle.__init__(self)
        
        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.door_hinge_did = self.model.jnt_dofadr[self.model.joint_name2id('door_hinge')]
        self.grasp_sid = self.model.site_name2id('S_grasp')
        self.handle_sid = self.model.site_name2id('S_handle')
        self.door_bid = self.model.body_name2id('frame')

        self.reference_xml = et.parse(self.reference_path)
        self.config_file = kwargs.get('config')
        self.dimensions = []

        # self.action_space = spaces.Box(-np.inf, np.inf, (self.action_space.shape[0] - 6,))

        self._locate_randomization_parameters()

    def _locate_randomization_parameters(self):
        self.root = self.reference_xml.getroot()

        self.frame_inertial = self.root.find(".//body[@name='frame']/inertial")
        self.door_inertial = self.root.find(".//body[@name='door']/inertial")
        self.latch_inertial = self.root.find(".//body[@name='latch']/inertial")

        self.door_joint = self.root.find(".//body[@name='door']/joint")
        self.latch_joint = self.root.find(".//body[@name='latch']/joint")

        self.compiler = self.root.find(".//compiler")

        xml = self._create_xml(randomize=False)
        self._re_init(xml)

    def _randomize_mass(self):
        frame_mass = str(self.dimensions[0].current_value)
        door_mass = str(self.dimensions[1].current_value)
        latch_mass = str(self.dimensions[2].current_value)
        
        self.frame_inertial.set('mass', frame_mass)
        self.door_inertial.set('mass', door_mass)
        self.latch_inertial.set('mass', latch_mass)

    def _randomize_frictionloss(self):
        door_fl = str(self.dimensions[3].current_value)
        latch_fl = str(self.dimensions[4].current_value)

        self.door_joint.set('frictionloss', door_fl)
        self.latch_joint.set('frictionloss', latch_fl)

    def _randomize_damping(self):
        door_fl = str(self.dimensions[5].current_value)
        latch_fl = str(self.dimensions[6].current_value)

        self.door_joint.set('damping', door_fl)
        self.latch_joint.set('damping', latch_fl)

    def _create_xml(self, randomize=True):
        if randomize:
            self._randomize_mass()
            self._randomize_frictionloss()
            self._randomize_damping()

        self.compiler.set('meshdir', os.path.join(os.path.dirname(__file__), "assets", "Adroit", "resources", "meshes"))
        self.compiler.set('texturedir', os.path.join(os.path.dirname(__file__), "assets", "Adroit", "resources", "textures"))

        return et.tostring(self.root, encoding='unicode', method='xml')

    def _re_init(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)

        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        if self.viewer:
            self.viewer.update_sim(self.sim)

    def _update_randomized_params(self):
        xml = self._create_xml()
        self._re_init(xml)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        # get to handle
        reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
        # open door
        reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        reward += -1e-5*np.sum(self.data.qvel**2)

        if ADD_BONUS_REWARDS:
            # Bonus
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10

        goal_achieved = True if door_pos >= 1.35 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_did]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qp[-1]
        return np.concatenate([qp[1:-2], [latch_pos], door_pos, palm_pos, handle_pos, palm_pos-handle_pos, [door_open]])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)
        self.sim.forward()

        self.set_env_state(self._get_init_state_dict())

        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if door open for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage

    def _get_init_state_dict(self):
        array = np.array
        return {'door_body_pos': array([-0.195038,  0.362202,  0.302945]), 'qpos': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'qvel': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}

