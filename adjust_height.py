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

asset_root = "robots/adam_standard"
asset_file = "urdf/adam_standard_foot_contact.urdf"
robot_asset = gym.load_asset(sim, asset_root, asset_file)
rigid_body_names = gym.get_asset_rigid_body_names(robot_asset)
dof_names = gym.get_asset_dof_names(robot_asset)

left_ankle_idx = rigid_body_names.index("toeLeft")
right_ankle_idx = rigid_body_names.index("toeRight")
left_foot_indices = [i for i, name in enumerate(rigid_body_names) if 'leftFoot' in name]
right_foot_indices = [i for i, name in enumerate(rigid_body_names) if 'rightFoot' in name]

spacing = 2.0
lower = gymapi.Vec3(-spacing, -spacing, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 2.0)

actor_handle = gym.create_actor(env, robot_asset, pose, "adam_actor", 0, 1)


def find_flat_segments(arr, threshold=0.002, min_length=10):
    segments = []
    start = 0
    
    for i in range(1, len(arr)):
        if abs(arr[i] - arr[i-1]) >= threshold:
            if i - start >= min_length:
                segments.append((start, i-1))
            start = i
    
    # Check the last segment
    if len(arr) - start >= min_length:
        segments.append((start, len(arr)-1))
    
    flat_positions = []
    for flat_segment in segments:
        start_idx, end_idx = flat_segment
        flat_positions.append(np.arange(start_idx, end_idx))

    flat_positions = np.concatenate(flat_positions)
        
    return flat_positions

def label_contact(left_ankle_poses, right_ankle_poses, threshold = 0.004):
    
    left_z = left_ankle_poses[:,2]
    right_z = right_ankle_poses[:,2]

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


def adam_to_isaac(adam_pose, w):

    root_pos = adam_pose["root_pos"]
    root_rot = adam_pose["root_rot"]
    root_vel = adam_pose["root_vel"]
    root_angular_vel = adam_pose["root_angular_vel"]
    dof_pos = adam_pose["dof_pos"]
    dof_vel = adam_pose["dof_vel"]
    
    frame_num = len(root_pos)

    root_states = torch.cat([torch.from_numpy(root_pos), torch.from_numpy(root_rot), torch.from_numpy(root_vel), torch.from_numpy(root_angular_vel)], dim=1).type(torch.float32)
    dof_states = torch.stack([torch.from_numpy(dof_pos), torch.from_numpy(dof_vel)],axis=2).type(torch.float32)

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
        new_root_pose[2] = root_states[i, 2] - torch.min(torch.cat([left_foot_poses, right_foot_poses])[:,2])
        new_root_poses.append(new_root_pose.clone())
        left_ankle_poses.append(left_ankle_pose.clone())
        right_ankle_poses.append(right_ankle_pose.clone())

    new_root_poses = torch.stack(new_root_poses)
    adam_pose["root_pos"] = new_root_poses.numpy()
    try:
        left_foot_contact, right_foot_contact = label_contact(torch.stack(left_ankle_poses).numpy(), torch.stack(right_ankle_poses).numpy())
    except ValueError:
        print(w)
        return None
    adam_pose["left_foot_contact"] = left_foot_contact
    adam_pose["right_foot_contact"] = right_foot_contact

    return adam_pose

def label_smoother():
    


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
    adam_walks = {}
    keys = adam_poses.keys()
    walks = []
    for k in keys:
        if 'walk' in k and 'hop' not in k and 'jump' not in k and 'leap' not in k and 'run' not in k and 'skip' not in k:
            walks.append(k)
    for w in walks:
        adam_pose = adam_poses[w]
        adam_pose = adam_to_isaac(adam_pose, w)
        if adam_pose is not None:
            adam_walks[w] = adam_pose
    joblib.dump(adam_walks, args.data_path + "/isaac_adam_standard_walk.pt")





    