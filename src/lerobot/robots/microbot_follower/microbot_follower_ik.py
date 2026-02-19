# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Any

import numpy as np
import os
from lerobot.cameras import make_cameras_from_configs
# from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from . import MicrobotFollower
from .config_microbot_follower import MicrobotFollowerIKConfig

from typing import Dict, Union

def get_action_type(data: Dict[str, float]) -> str:
    """
    Differentiates between two types of dictionaries based on their keys.

    Args:
        data: The dictionary to classify.

    Returns:
        A string indicating the type of dictionary:
        "speed_control" if it contains speed-related keys.
        "joint_position" if it contains joint position-related keys.
        "unknown" if it doesn't match either type.
    """
    speed_keys = {"speed_x", "speed_y", "speed_z", "speed_gripper"}
    joint_pos_keys = {"shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"}

    # Convert dictionary keys to a set for efficient comparison
    data_keys = set(data.keys())

    if data_keys == speed_keys:
        return "speed_control"
    elif data_keys == joint_pos_keys:
        return "joint_position"
    else:
        return "unknown"

logger = logging.getLogger(__name__)


class MicrobotFollowerIK(MicrobotFollower):
    """
    MicrobotFollower robot with end-effector space control.

    This robot inherits from MicrobotFollower but transforms actions from
    end-effector space to joint space before sending them to the motors.
    """

    config_class = MicrobotFollowerIKConfig
    name = "microbot_follower_ik"

    def __init__(self, config: MicrobotFollowerIKConfig):
        super().__init__(config)
        # self.cameras = make_cameras_from_configs(config.cameras)
        self.joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos"
        ]
        self.ee_joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
        ]
        self.abort_action = False
        self.config = config
        # self.config.urdf_path = os.path.abspath("src/lerobot/robots/microbot_follower/microbot.urdf")
        self.config.urdf_path = os.path.abspath("robots/microbot_follower")
        self.config.target_frame_name = "wrist"
        self.config.end_effector_bounds = {
            "max": [0.15, 0.1, 0.30],
            "min": [-0.15, -0.1, 0.15]
        }
        self.config.end_effector_step_sizes = {
            "x": 0.01,
            "y": 0.01,
            "z": 0.01,
        }
        # Initialize the kinematics module for the microbot robot
        if self.config.urdf_path is None:
            raise ValueError(
                "urdf_path must be provided in the configuration for end-effector control. "
                "Please set urdf_path in your MicrobotFollowerIKConfig."
            )

        self.kinematics = RobotKinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
        )

        # Store the bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        self.current_ee_pos = None
        self.current_joint_pos = None

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        # return {
        #     "dtype": "float32",
        #     "shape": (4,),
        #     "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        # }
        # return {
        #             "delta_x": float,
        #             "delta_y": float,
        #             "delta_z": float,
        #             "gripper": float,
        #         }
        return {
                    "shoulder_pan.pos": float,
                    "shoulder_lift.pos": float,
                    "elbow_flex.pos": float,
                    "wrist_flex.pos": float,
                    "wrist_roll.pos": float,
                    "gripper.pos": float,
                }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.abort_action:
            return
        action_type = get_action_type(action)
        if action_type == "joint_position":
            super().send_action(action)
            return action
            
        """
        Transform action from end-effector space to joint space and send to motors.

        Args:
            action: Dictionary with keys 'delta_x', 'delta_y', 'delta_z' for end-effector control
                   or a numpy array with [delta_x, delta_y, delta_z]

        Returns:
            The joint-space action that was sent to the motors
        """
        # original_action = action.copy()

        # Convert action to numpy array if not already
        if isinstance(action, dict):
            if all(k in action for k in ["delta_x", "delta_y", "delta_z"]):
                delta_ee = np.array(
                    [
                        action["delta_x"] * self.config.end_effector_step_sizes["x"],
                        action["delta_y"] * self.config.end_effector_step_sizes["y"],
                        action["delta_z"] * self.config.end_effector_step_sizes["z"],
                    ],
                    dtype=np.float32,
                )
                if "gripper" not in action:
                    action["gripper"] = [1.0]
                action = np.append(delta_ee, action["gripper"])
            else:
                logger.warning(
                    f"Expected action keys 'delta_x', 'delta_y', 'delta_z', got {list(action.keys())}"
                )
                action = np.zeros(4, dtype=np.float32)

        if self.current_joint_pos is None:
            # Read current joint positions
            #waly read joints
            current_joint_pos = super().get_robot_angles()
            # current_joint_pos = self.bus.sync_read("Present_Position")
            self.current_joint_pos = np.array([current_joint_pos[self.ee_joint_names[i]] for i in range(len(self.ee_joint_names))])
            self.current_gripper_pos = current_joint_pos["gripper.pos"]

        # Calculate current end-effector position using forward kinematics
        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos)

        # Set desired end-effector position by adding delta
        desired_ee_pos = np.eye(4)
        desired_ee_pos[:3, :3] = self.current_ee_pos[:3, :3]  # Keep orientation

        # Add delta to position and clip to bounds
        desired_ee_pos[:3, 3] = self.current_ee_pos[:3, 3] + action[:3]
        if self.end_effector_bounds is not None:
            desired_ee_pos[:3, 3] = np.clip(
                desired_ee_pos[:3, 3],
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )

        # Compute inverse kinematics to get joint positions
        # print(desired_ee_pos)
        target_joint_values_in_degrees = self.kinematics.inverse_kinematics(
            self.current_joint_pos, desired_ee_pos
        )

        # joint_action needs to be calculated properly
        target_gripper_action = np.clip(
            self.current_gripper_pos + (action[-1] - 1.0) * -2.0,
            120.0,
            # self.config.max_gripper_pos,
            180.0
        )

        # Create joint space action dictionary
        joint_action = {
            f"{self.ee_joint_names[i]}": target_joint_values_in_degrees[i] for i in range(len(self.ee_joint_names))
        }

        joint_action["gripper.pos"] = target_gripper_action
        # Handle gripper separately if included in action
        # Gripper delta action is in the range 0 - 2,
        # We need to shift the action to the range -1, 1 so that we can expand it to -Max_gripper_pos, Max_gripper_pos
            
        self.current_ee_pos = desired_ee_pos.copy()
        self.current_joint_pos = target_joint_values_in_degrees.copy()
        self.current_gripper_pos = target_gripper_action

        # Send joint space action to parent class
        copied_action = joint_action.copy()
        super().send_action(copied_action)
        return joint_action

    # def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    #     super().send_action(action)
    #     return action

    def get_observation(self) -> dict[str, Any]:
        return super().get_observation()

    def reset(self):
        self.abort_action = True
        time.sleep(1.0) # Small delay to make sure GUI is responsive
        self.current_ee_pos = None
        self.current_joint_pos = None
        joint_action = {"shoulder_pan.pos":0.0,
                        "shoulder_lift.pos":0.0,
                        "elbow_flex.pos":-55,
                        "wrist_flex.pos":-105,
                        "wrist_roll.pos":0.0,
                        "gripper.pos":123.0}
        # joint_action = {"shoulder_pan.pos":90.0,
        #                 "shoulder_lift.pos":30.0,
        #                 "elbow_flex.pos":85,
        #                 "wrist_flex.pos":135,
        #                 "wrist_roll.pos":90.0,
        #                 "gripper.pos":123.0}
        super().send_action(joint_action)
        time.sleep(6.0) # Small delay to make sure GUI is responsive
        print("finished reset")
        self.abort_action = False

    # def reset(self):
    #     super().reset()
