import argparse
import os
import sys
import time

import joblib
import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil

sys.path.append(os.getcwd())
from tqdm import tqdm

from utils import torch_utils

# Initialize Gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Asset and Environment Information")

# create simulation context
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
plane_params.distance = 0.0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = "robots/adam_standard"
asset_file = "urdf/adam_standard_foot_contact.urdf"
robot_asset = gym.load_asset(sim, asset_root, asset_file)
rigid_body_names = gym.get_asset_rigid_body_names(robot_asset)
dof_names = gym.get_asset_dof_names(robot_asset)

left_ankle_idx = rigid_body_names.index("toeLeft")
right_ankle_idx = rigid_body_names.index("toeRight")
left_foot_indices = [i for i, name in enumerate(rigid_body_names) if "leftFoot" in name]
right_foot_indices = [
    i for i, name in enumerate(rigid_body_names) if "rightFoot" in name
]

useful_body_indices = [
    i
    for i, name in enumerate(rigid_body_names)
    if "leftFoot" not in name and "rightFoot" not in name
]

real_dof_idx = []
for i, dof_name in enumerate(dof_names):
    if "wrist" not in dof_name and "gripper" not in dof_name:
        real_dof_idx.append(i)

spacing = 2.0
lower = gymapi.Vec3(-spacing, -spacing, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
# pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

actor_handle = gym.create_actor(env, robot_asset, pose, "adam_actor", 0, 1)


def find_flat_segments(arr, threshold=0.002, min_length=10):
    segments = []
    start = 0

    for i in range(1, len(arr)):
        if abs(arr[i] - arr[i - 1]) >= threshold:
            if i - start >= min_length:
                segments.append((start, i - 1))
            start = i

    # Check the last segment
    if len(arr) - start >= min_length:
        segments.append((start, len(arr) - 1))

    flat_positions = []
    for flat_segment in segments:
        start_idx, end_idx = flat_segment
        flat_positions.append(np.arange(start_idx, end_idx))

    flat_positions = np.concatenate(flat_positions)

    return flat_positions


def label_contact(left_ankle_poses, right_ankle_poses, threshold=0.004):

    left_z = left_ankle_poses[:, 2]
    right_z = right_ankle_poses[:, 2]

    left_flat_positions = find_flat_segments(left_z)
    right_flat_positions = find_flat_segments(right_z)

    mean_left_z = np.mean(left_z[left_flat_positions])
    corrected_left_z = left_z - mean_left_z

    mean_right_z = np.mean(right_z[right_flat_positions])
    corrected_right_z = right_z - mean_right_z

    left_foot_contact = np.zeros_like(left_z)
    right_foot_contact = np.zeros_like(right_z)

    diff = corrected_left_z - corrected_right_z
    left_foot_contact[np.abs(diff) < threshold] = 1
    right_foot_contact[np.abs(diff) < threshold] = 1

    left_foot_contact[diff >= threshold] = 0
    right_foot_contact[diff >= threshold] = 1

    left_foot_contact[diff <= -threshold] = 1
    right_foot_contact[diff <= -threshold] = 0

    return left_foot_contact, right_foot_contact


def adam_to_isaac(adam_pose, name):

    root_pos = adam_pose["root_pos"]
    root_rot = adam_pose["root_rot"]
    joint_poses = adam_pose["joint_poses"]
    joint_names = adam_pose["joint_names"]
    useful_list = [joint_names.index(name) for name in dof_names]
    joint_poses = joint_poses[:, useful_list]

    frame_num = len(root_pos)

    if frame_num <= 2:
        return None

    root_states = torch.cat(
        [
            torch.from_numpy(root_pos),
            torch.from_numpy(root_rot),
            torch.zeros(frame_num, 6),
        ],
        dim=1,
    ).type(torch.float32)
    dof_states = torch.zeros(frame_num, len(dof_names), 2).type(torch.float32)
    dof_states[:, real_dof_idx] = torch.stack(
        [torch.from_numpy(joint_poses), torch.zeros(frame_num, len(real_dof_idx))],
        axis=2,
    ).type(torch.float32)
    rigid_body_states = []

    # Simulate
    new_root_poses = []
    left_ankle_poses = []
    right_ankle_poses = []

    for i in range(frame_num):

        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states[i]))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states[i]))
        gym.refresh_rigid_body_state_tensor(sim)
        rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        left_foot_poses = rigid_body_state[left_foot_indices, 0:3]
        right_foot_poses = rigid_body_state[right_foot_indices, 0:3]
        left_ankle_pose = rigid_body_state[left_ankle_idx, 0:3]
        right_ankle_pose = rigid_body_state[right_ankle_idx, 0:3]
        new_root_pose = root_states[i, 0:3]
        new_root_pose[2] = root_states[i, 2] - torch.min(
            torch.cat([left_foot_poses, right_foot_poses])[:, 2]
        )
        new_root_poses.append(new_root_pose.clone())
        left_ankle_poses.append(left_ankle_pose.clone())
        right_ankle_poses.append(right_ankle_pose.clone())

        new_rigid_body_state = rigid_body_state[useful_body_indices].clone()
        new_rigid_body_state[:, 2] = new_rigid_body_state[:, 2] - torch.min(
            torch.cat([left_foot_poses, right_foot_poses])[:, 2]
        )

        if i >= 1:  # skip the first frame
            rigid_body_states.append(new_rigid_body_state)

    try:
        left_foot_contact, right_foot_contact = label_contact(
            torch.stack(left_ankle_poses).numpy(),
            torch.stack(right_ankle_poses).numpy(),
        )
    except ValueError:
        print(name)
        return None

    dt = 1 / adam_pose["real_frame_rate"]
    rigid_body_states = torch.stack(rigid_body_states, dim=0)
    current_body_pos = rigid_body_states[:-1, :, 0:3]
    next_body_pos = rigid_body_states[1:, :, 0:3]
    current_body_rot = rigid_body_states[:-1, :, 3:7]
    next_body_rot = rigid_body_states[1:, :, 3:7]

    body_vel = next_body_pos - current_body_pos
    body_vel = body_vel / dt

    dof_states = dof_states.numpy()
    dof_vel = dof_states[1:, :, 0] - dof_states[:-1, :, 0]
    dof_vel = dof_vel / dt

    diff_global_body_rot = torch_utils.quat_mul(
        next_body_rot, torch_utils.quat_conjugate(current_body_rot)
    )
    diff_global_body_angle, diff_global_body_axis = torch_utils.quat_to_angle_axis(
        diff_global_body_rot
    )
    body_angular_vel = diff_global_body_angle[:, :, None] * diff_global_body_axis / dt

    result = {
        "body_pos": current_body_pos.numpy(),
        "root_pos": current_body_pos[:, 0, :].numpy(),
        "dof_pos": dof_states[1:-1, :, 0],
        "body_rot": current_body_rot.numpy(),
        "root_rot": root_rot[1:-1],
        "body_vel": body_vel.numpy(),
        "root_vel": body_vel[:, 0, :].numpy(),
        "body_angular_vel": body_angular_vel.numpy(),
        "root_angular_vel": body_angular_vel[:, 0, :].numpy(),
        "dof_vel": dof_vel[1:],
        "dof_vel_sign": np.sign(dof_vel[1:]),
        "left_foot_contact": left_foot_contact[1:-1],
        "right_foot_contact": right_foot_contact[1:-1],
        "dt": dt,
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="dataset directory",
        default="/home/jianrenw/mocap/data",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="output directory",
        default="/home/jianrenw/MAP/data",
    )

    args = parser.parse_args()

    # load motion data
    adam_poses = joblib.load(args.data_path + "/adam_lite_data.pt")

    adam_walks = {}
    keys = adam_poses.keys()
    walks = []
    for k in keys:
        if (
            "walk" in k
            and "hop" not in k
            and "jump" not in k
            and "leap" not in k
            and "run" not in k
            and "skip" not in k
        ):
            walks.append(k)
    for w in walks:
        adam_pose = adam_poses[w]
        adam_pose = adam_to_isaac(adam_pose, w)
        if adam_pose is not None:
            adam_walks[w] = adam_pose
    joblib.dump(adam_walks, args.out_dir + "/isaac_adam_standard_walk.pt")
