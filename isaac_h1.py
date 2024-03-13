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
asset_root = "h1"
asset_file = "h1.urdf"
robot_asset = gym.load_asset(sim, asset_root, asset_file)

spacing = 2.0
lower = gymapi.Vec3(-spacing, -spacing, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
# pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

actor_handle = gym.create_actor(env, robot_asset, pose, "H1_actor", 0, 1)


def h1_to_isaac(h1_pose):

    root_pos = h1_pose["root_pos"]
    root_rot = h1_pose["root_rot"]
    joint_poses = h1_pose["joint_poses"]
    joint_names = h1_pose["joint_names"]
    frame_num = len(root_pos)

    if frame_num <= 2:
        return None

    root_states = torch.cat([torch.from_numpy(root_pos), torch.from_numpy(root_rot), torch.zeros(frame_num, 6)], dim=1).type(torch.float32)
    dof_states = torch.stack([torch.from_numpy(joint_poses[:,:19]), torch.zeros(frame_num, 19)],axis=2).type(torch.float32)
    rigid_body_states = []

    # Simulate
    for i in range(frame_num):

        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states[i]))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states[i]))
        gym.simulate(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        rigid_body_states.append(rigid_body_state.clone())

    dt = 1 / h1_pose['real_frame_rate'] # 
    rigid_body_states = torch.stack(rigid_body_states, dim=0)
    current_body_pos = rigid_body_states[:-1, :, 0:3]
    next_body_pos = rigid_body_states[1:, :, 0:3]
    current_body_rot = rigid_body_states[:-1, :, 3:7]
    next_body_rot = rigid_body_states[1:, :, 3:7]

    body_vel = next_body_pos - current_body_pos
    body_vel = body_vel / dt

    dof_vel = joint_poses[1:] - joint_poses[:-1]
    dof_vel = dof_vel / dt

    diff_global_body_rot = torch_utils.quat_mul(next_body_rot, torch_utils.quat_conjugate(current_body_rot))
    diff_global_body_angle, diff_global_body_axis = torch_utils.quat_to_angle_axis(diff_global_body_rot)   
    body_angular_vel = diff_global_body_angle[:,:,None] * diff_global_body_axis / dt

    result = {
        'body_pos': current_body_pos.numpy(), # [frame_num-1, 23, 3]
        'root_pos': root_pos[:-1], # [frame_num-1, 3]
        'dof_pos': joint_poses[:-1, :19], # [frame_num-1, 23]
        'body_rot': current_body_rot.numpy(), # [frame_num-1, 23, 4]
        'root_rot': root_rot[:-1], # [frame_num-1, 4]
        'body_vel': body_vel.numpy(), # [frame_num-1, 23, 3]
        'root_vel': body_vel[:,0,:].numpy(), # [frame_num-1, 3]
        'body_angular_vel': body_angular_vel.numpy(), # [frame_num-1, 23, 3]
        'root_angular_vel': body_angular_vel[:,0,:].numpy(), # [frame_num-1, 3]
        'dof_vel': dof_vel[:,:19], # [frame_num-1, 23]
        'dt': dt, # scalar
    }

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="dataset directory", default="/home/jianrenw/mocap/data/out"
    )
    parser.add_argument(
        "--out_dir", type=str, help="output directory", default="/home/jianrenw/mocap/data/out"
    )

    args = parser.parse_args()

    # load motion data
    h1_poses = joblib.load(args.data_path + "/h1_data.pt")

    isaac_data = {}

    for key in tqdm(h1_poses.keys()):
        adam_pose = h1_poses[key]
        result = h1_to_isaac(adam_pose)
        if result is not None:
            isaac_data[key] = result

    joblib.dump(isaac_data, args.out_dir + "/isaac_h1.pt")

    # save one
    # key = list(adam_poses.keys())[10]
    # adam_pose = adam_poses[key]
    # result = adam_to_isaac(adam_pose)
    # joblib.dump(result, args.out_dir + "/{}.pt".format(key))
    

    # data_path = "/home/jianrenw/Research/foundation_locomotion/data/h1_isaac.pt"
    # isaac_data = joblib.load(data_path)
    # keys = list(isaac_data.keys())
    # for key in keys:
    #     result = isaac_data[key]
    #     print(result['root_pos'].shape)
    #     print(result['root_rot'].shape)
    #     print(result['dof_pos'].shape)
    #     print(result['body_pos'].shape)
    #     print(result['body_rot'].shape)
    #     print(result['body_vel'].shape)
    #     print(result['root_vel'].shape)
    #     print(result['body_angular_vel'].shape)
    #     print(result['root_angular_vel'].shape)
    #     print(result['dof_vel'].shape)
    #     break





