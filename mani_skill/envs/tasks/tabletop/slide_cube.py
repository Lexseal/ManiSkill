from typing import Any, Dict, Union

import numpy as np
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import PandaPizza
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("SlideCube-v1", max_episode_steps=50)
class SlideCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_pizza",]
    agent: PandaPizza
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda_pizza", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 1.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube"
        )

    def _generate_cube_init_pos(self, env_idx):
        """ get a batch of cube poses """
        # poses = self.agent.tcp.pose[env_idx]
        base_pose = self.agent.base_link.pose[env_idx]
        # use the base since it seems robot is reset after all other objects
        # from base-0.6150,  0.0000,  0.0000) to tcp(-0.1,  0.01,  0.78), considering thickness of
        p = base_pose.p + torch.tensor([0.515, 0.01, 0.81], device=base_pose.device) 
        b = len(env_idx)
        # then find the x and y axis of the pizza peel by reading their quaternion
        # tf = base_pose.to_transformation_matrix()
        # x_axis = tf[..., :3, 0]
        # y_axis = tf[..., :3, 1]
        # z_axis = tf[..., :3, 2]
        # # randomly shift p along x and y axis a little bit
        # p += x_axis * (torch.rand(b, 3) - 0.5) * 0.2
        # p += y_axis * (torch.rand(b, 3) - 0.5) * 0.2
        # # set z to be just above the pizza peel by the height of the cube
        # p += z_axis * (self.cube_half_size + 0.02)

        # randomly shift p along x and y axis a little bit
        p[..., 0] += (torch.rand(b, ) - 0.5) * 0.25
        p[..., 1] += (torch.rand(b, ) - 0.5) * 0.25
        # set z to be just above the pizza peel by the height of the cube
        p[..., 2] += torch.rand(b, ) * 0.01 + (self.cube_half_size + 0.02)
        return p

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = self._generate_cube_init_pos(env_idx)
            qs = randomization.random_quaternions(b) # lock_x=True, lock_y=True
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            
    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                # obj_pose=self.cube.pose.raw_pose,
                # obj_state=self.cube.get_state()[..., -6:],
                # cube_q=self.cube.pose.q,
                cube_angular_velocity=self.cube.get_angular_velocity(),
                cube_linear_velocity=self.cube.get_linear_velocity(),
                tcp_linear_velocity=self.agent.tcp.get_linear_velocity(),
                tcp_angular_velocity=self.agent.tcp.get_angular_velocity(),
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,  # how far it is to the hole
            )
        return obs

    def _signed_distance_cube_to_pizza_peel(self):
        tf = self.agent.tcp.pose.to_transformation_matrix()
        z_axis = tf[..., :3, 2]
        return torch.sum((self.cube.pose.p - self.agent.tcp.pose.p) * z_axis, dim=1)

    def _distance_cube_to_pizza_peel_hole(self):
        return torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )

    def evaluate(self):
        # success is if distance is less than the cube size and cube has signed distance less than 0
        success = self._distance_cube_to_pizza_peel_hole() <= self.cube_half_size * 2
        # print(self._distance_cube_to_pizza_peel_hole(), self._signed_distance_cube_to_pizza_peel())
        success = success & (0 <= (self.cube.pose.p[..., 2] - self.agent.tcp.pose.p[..., 2])) & (self._distance_cube_to_pizza_peel_hole() < self.cube_half_size + 0.01)
        return {
            "success": success,
            "is_robot_static": self.agent.is_static(0.2),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        tf = self.agent.tcp.pose.to_transformation_matrix()
        z_axis = tf[..., :3, 2]
        # take the last column of the z axis as reward
        reward += 0.5 * z_axis[..., 2]
        dropped = (self.cube.pose.p[..., 2] - self.agent.tcp.pose.p[..., 2]) < 0
        reward[dropped] = -5


        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
