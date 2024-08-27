import os
import numpy as np
from isaacgym import gymapi,gymtorch,gymutil
import joblib
import torch
import time
import argparse
import sys
sys.path.append(os.getcwd())
from utils import torch_utils
from tqdm import tqdm

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

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0.0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

# add cartpole urdf asset
# asset_root = "adam"
# asset_file = "urdf/adam.urdf"
# asset_root = "adam_lite_v2"
# asset_file = "urdf/adam_lite_v2_wrist_yaw.urdf"
asset_root = "robots/adam_standard"
asset_file = "urdf/adam_standard_foot_contact.urdf"
robot_asset = gym.load_asset(sim, asset_root, asset_file)
rigid_body_names = gym.get_asset_rigid_body_names(robot_asset)
dof_names = gym.get_asset_dof_names(robot_asset)


left_foot_indices = [i for i, name in enumerate(rigid_body_names) if 'leftFoot' in name]
right_foot_indices = [i for i, name in enumerate(rigid_body_names) if 'rightFoot' in name]

spacing = 2.0
lower = gymapi.Vec3(-spacing, -spacing, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 2.0)

actor_handle = gym.create_actor(env, robot_asset, pose, "adam_actor", 0, 1)


def adam_to_isaac(adam_pose):

    root_pos = adam_pose["root_pos"]
    root_rot = adam_pose["root_rot"]
    root_vel = adam_pose["root_vel"]
    root_angular_vel = adam_pose["root_angular_vel"]
    dof_pos = adam_pose["dof_pos"]
    dof_vel = adam_pose["dof_vel"]
    
    frame_num = len(root_pos)

    root_states = torch.cat([torch.from_numpy(root_pos), torch.from_numpy(root_rot), torch.from_numpy(root_vel), torch.from_numpy(root_angular_vel)], dim=1).type(torch.float32)
    dof_states = torch.stack([torch.from_numpy(dof_pos), torch.from_numpy(dof_vel)],axis=2).type(torch.float32)
    rigid_body_states = []

    # Simulate
    new_root_poses = []
    min_foot_poses = 100
    for i in range(frame_num):

        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states[i]))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states[i]))
        gym.refresh_rigid_body_state_tensor(sim)
        rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        left_foot_poses = rigid_body_state[left_foot_indices, 0:3]
        right_foot_poses = rigid_body_state[right_foot_indices, 0:3]
        new_root_poses.append(torch.copy(root_states[i, 0:3] - torch.min(left_foot_poses, right_foot_poses)))

    adam_pose["root_pos"] = new_root_poses

    return adam_pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="dataset directory", default="/home/jianrenw/MAP/data"
    )
    parser.add_argument(
        "--out_dir", type=str, help="output directory", default="/home/jianrenw/mocap/data/videos"
    )

    args = parser.parse_args()

    adam_poses = joblib.load(args.data_path + "/isaac_adam_standard_add_sign.pt")






    