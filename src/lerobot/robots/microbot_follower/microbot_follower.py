#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property
from typing import Any
import cv2
import random

dx_adj = 0.0
dy_adj = 0.0

from lerobot.cameras.utils import make_cameras_from_configs
# from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_microbot_follower import MicrobotFollowerConfig

import serial

logger = logging.getLogger(__name__)
ARDUINO_PORT = '/dev/ttyACM0'  # <--- CHANGE THIS TO YOUR ARDUINO'S PORT
BAUD_RATE = 115200       # <--- CHANGE THIS TO YOUR ARDUINO'S BAUD RATE

def setup_serial_port(port, baudrate, timeout=1):
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"Successfully connected to serial port {port} at {baudrate} baud.")
        time.sleep(2) # Give some time for the Arduino to reset after serial connection
        return ser
    except Exception as e:
        print(f"Error: Could not open serial port {port}. Please check if the port is correct and not in use.")
        print(f"Details: {e}")
        raise

class MicrobotFollower(Robot):
    """
    [SO-100 Follower Arm](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = MicrobotFollowerConfig
    name = "microbot_follower"

    def __init__(self, config: MicrobotFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.arduino_serial = setup_serial_port(ARDUINO_PORT, BAUD_RATE)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
                    "shoulder_pan.pos": float,
                    "shoulder_lift.pos": float,
                    "elbow_flex.pos": float,
                    "wrist_flex.pos": float,
                    "wrist_roll.pos": float,
                    "gripper.pos": float,
                }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass
    
    def get_robot_angles(self) -> dict[str, float]:
        keepReceiving = True
        while keepReceiving:
            s = f"x\n".encode('utf-8')
            self.arduino_serial.write(s)  # send arm action to arduino
            line = self.arduino_serial.readline()
            try:
                dr = [random.uniform(-1.5, 1.5) for _ in range(6)]
                line = line.decode('utf-8').strip()
                arr = line.split('\t')
                arr = [float(s) for s in arr]
                obs_dict = {
                                "shoulder_pan.pos": arr[5]-90.0,
                                "shoulder_lift.pos": (arr[4]-30.0)*-1.0,
                                "elbow_flex.pos": (arr[3]-30.0)*-1.0,
                                "wrist_flex.pos": (arr[2]-30.0)*-1.0,
                                "wrist_roll.pos": (arr[1]-90.0),
                                "gripper.pos": arr[0],
                            }
                keepReceiving = False
            except Exception as e:
                print(f"Error {e}")
                time.sleep(0.005)
        return obs_dict

    def get_observation(self) -> dict[str, Any]:
        # Read arm position
        start = time.perf_counter()
        obs_dict = self.get_robot_angles()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            # obs_dict[cam_key] = cv2.rotate(obs_dict[cam_key], cv2.ROTATE_180)

            h, w, _ = obs_dict[cam_key].shape
    
            # Left boundary mask
            cv2.rectangle(obs_dict[cam_key], (0, 0), (60, h), (0, 0, 0), -1) 
            # # Top boundary mask
            # cv2.rectangle(obs_dict[cam_key], (0, 0), (w, 50), (0, 0, 0), -1) 
            # Bottom boundary mask
            cv2.rectangle(obs_dict[cam_key], (0, h-40), (w, h), (0, 0, 0), -1)
            # Right boundary mask
            cv2.rectangle(obs_dict[cam_key], (w-20, 0), (w, h), (0, 0, 0), -1)
            
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        values = action.values()
        action["shoulder_pan.pos"] = action["shoulder_pan.pos"] + 90.0
        action["shoulder_lift.pos"] = -1.0*action["shoulder_lift.pos"] + 30.0
        action["elbow_flex.pos"] = -1.0*action["elbow_flex.pos"] + 30.0
        action["wrist_flex.pos"] = -1.0*action["wrist_flex.pos"] + 30.0
        action["wrist_roll.pos"] = action["wrist_roll.pos"] + 90.0

        string_values = [str(int(value)) for value in values]
        string_values.reverse()
        tab_delimited_string = "\t".join(string_values)
        tab_delimited_string = tab_delimited_string + '\n'
        self.arduino_serial.write(tab_delimited_string.encode('utf-8'))
        # print(f"Sent action: {tab_delimited_string}")
        # Send goal position to the arm
        return action

    # def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    #     if isinstance(action, dict):
    #         if all(k in action for k in ["action"]):
    #             try:
    #                 my_val = action["action"]
    #                 my_char = 's'
    #                 if my_val== 1:
    #                     my_char = 'f'
    #                 elif my_val == 2:
    #                     my_char = 'b'
    #                 elif my_val == 3:
    #                     my_char = 'l'
    #                 elif my_val == 4:
    #                     my_char = 'r'
    #                 elif my_val == 5:
    #                     my_char = 'u'
    #                 elif my_val == 6:
    #                     my_char = 'd'
    #                 elif my_val == 7:
    #                     my_char = 'h'
    #                 elif my_val == 8:
    #                     my_char = 'g'
    #                 print(my_char)
    #                 self.arduino_serial.write(f"{my_char}\n".encode('utf-8'))
    #             except serial.SerialException as e:
    #                 print(f"Error sending data to serial port: {e}")
    #         else:
    #             logger.warning(f"Expected action keys 'action', got {list(action.keys())}")
    #     return action

    def get_rand_dir(self) -> str:
        key_to_send = 's'
        random_number = random.uniform(0.0, 100.0)
        if random_number > 65.0:
            random_number2 = random.uniform(0.0, 100.0)
            if random_number2 > 50:
                key_to_send = 'b'
            else:
                key_to_send = 'f'
        return key_to_send
    
    def send_ee_action(self, action: dict[str, Any]) -> dict[str, Any]:
        key_to_send = 's'  # 's' for stop or stationary
        global dx_adj, dy_adj

        if isinstance(action, dict):
            if all(k in action for k in ["speed_x", "speed_y", "speed_z", "speed_gripper"]):
                action["speed_x"] = action["speed_x"] * 1.0
                key_with_abs_max_value = max(action, key=lambda k: abs(action[k]))
                # print(action)
                # print(key_with_abs_max_value)

                if abs(action[key_with_abs_max_value]) > 0.01:
                    if key_with_abs_max_value == "speed_x":
                        if action[key_with_abs_max_value] > 0.0:
                            key_to_send = 'r'
                        else:
                            key_to_send = 'l'
                    elif key_with_abs_max_value == "speed_y":
                        if action[key_with_abs_max_value] > 0.0:
                            key_to_send = 'f'
                        else:
                            key_to_send = 'b'
                    elif key_with_abs_max_value == "speed_z":
                        if action[key_with_abs_max_value] > 0.0:
                            key_to_send = 'u'
                        else:
                            key_to_send = 'd'
                    elif key_with_abs_max_value == "speed_gripper":
                        if action[key_with_abs_max_value] > 0.0:
                            key_to_send = 'h'
                        else:
                            key_to_send = 'g'
            else:
                logger.warning(f"Expected action keys 'speed_x', 'speed_y', 'speed_z', got {list(action.keys())}")
            
        try:
            normal_mode = True
            if key_to_send == 's':
                if dx_adj==0.0: #and dy_adj==0.0:
                    key_to_send = self.get_rand_dir()
                    if key_to_send == 'b':
                        dx_adj = 1.0
                        normal_mode = False
                    elif key_to_send == 'f':
                        dx_adj = -1.0
                        normal_mode = False
                else:
                    normal_mode = False
                    if dx_adj>0.0:
                        key_to_send = 'f'
                        dx_adj = 0.0
                    elif dx_adj<0.0:
                        key_to_send = 'b'
                        dx_adj = 0.0

            print(key_to_send)
            if normal_mode:
                self.arduino_serial.write(f"{key_to_send}\n".encode('utf-8'))
            else:
                self.arduino_serial.write(f"{key_to_send}\n".encode('utf-8'))
                time.sleep(0.07)
                self.arduino_serial.write(f"s\n".encode('utf-8'))

        except serial.SerialException as e:
            print(f"Error sending data to serial port: {e}")
        
    def disconnect(self):
        for cam in self.cameras.values():
            cam.disconnect()

    def reset(self):
        try:
            self.arduino_serial.write(f"t\n".encode('utf-8'))
        except serial.SerialException as e:
            print(f"Error sending data to serial port: {e}")
