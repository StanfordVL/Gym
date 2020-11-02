import os
from .... import utils
from .. import fetch_env
import numpy as np

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b)

class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

        # TODO move magic numbers
        self.contact_reward = 1.
        self.contact_shaped_reward = 0.5

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense':
            reward = 0
            if self.has_object:
                eef_position = self.sim.data.get_site_xpos('robot0:grip')
                obj_position = self.sim.data.get_site_xpos('object0')
                obj_dist = np.linalg.norm(eef_position - obj_position)
                if obj_dist < self.distance_threshold and abs(eef_position[2] - obj_position[2]) < 0.02:
                    reward += self.contact_reward
                else:
                    reward += (self.contact_shaped_reward * (1 - np.tanh(3 * obj_dist))).squeeze()
            return reward - d
        else:
            return -d
