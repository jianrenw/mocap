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

# add cartpole urdf asset
# asset_root = "adam"
# asset_file = "urdf/adam.urdf"
# asset_root = "adam_lite_v2"
# asset_file = "urdf/adam_lite_v2_wrist_yaw.urdf"
asset_root = "robots/adam_standard"
asset_file = "urdf/adam_standard.urdf"
robot_asset = gym.load_asset(sim, asset_root, asset_file)
rigid_body_names = gym.get_asset_rigid_body_names(robot_asset)
dof_names = gym.get_asset_dof_names(robot_asset)
print(rigid_body_names)
print(dof_names)

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


def adam_to_isaac(adam_pose):

    root_pos = adam_pose["root_pos"]
    root_rot = adam_pose["root_rot"]
    joint_poses = adam_pose["joint_poses"]
    joint_names = adam_pose["joint_names"]
    useful_list = [joint_names.index(name) for name in dof_names]
    joint_poses = joint_poses[:, useful_list]

    frame_num = len(root_pos)

    if frame_num <= 2:
        return None

    # # fix trajectory case by case
    # euler = torch_utils.euler_from_quat(torch.from_numpy(root_rot))
    # euler = euler.numpy()
    # euler[372:,1] = 0.1
    # euler = torch.from_numpy(euler)
    # roll = torch.zeros(euler.shape[0])
    # pitch = torch.zeros(euler.shape[0])
    # yaw = torch.zeros_like(roll) + np.pi / 2
    # root_rot_new = torch_utils.quat_from_euler_xyz(roll, pitch, yaw)

    # root_rot = root_rot_new.numpy()
    # root_pos[:,0] = 0.

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
    for i in range(frame_num):

        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states[i]))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states[i]))
        gym.refresh_rigid_body_state_tensor(sim)
        rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        if i >= 1:  # skip the first frame
            rigid_body_states.append(rigid_body_state.clone())

    # dt = 1 / adam_pose['real_frame_rate'] #
    dt = 1 / 60
    rigid_body_states = torch.stack(rigid_body_states, dim=0)
    current_body_pos = rigid_body_states[:-1, :, 0:3]
    next_body_pos = rigid_body_states[1:, :, 0:3]
    current_body_rot = rigid_body_states[:-1, :, 3:7]
    next_body_rot = rigid_body_states[1:, :, 3:7]

    body_vel = next_body_pos - current_body_pos
    body_vel = body_vel / dt

    # dof_vel = joint_poses[1:] - joint_poses[:-1]
    dof_states = dof_states.numpy()
    dof_vel = dof_states[1:, :, 0] - dof_states[:-1, :, 0]
    dof_vel = dof_vel / dt

    # diff_global_body_rot = torch_utils.quat_mul(next_body_rot, torch_utils.quat_conjugate(current_body_rot))
    # diff_global_body_angle, diff_global_body_axis = torch_utils.quat_to_angle_axis(diff_global_body_rot)
    # body_angular_vel = diff_global_body_angle[:,:,None] * diff_global_body_axis / dt

    diff_global_body_rot = torch_utils.quat_mul(
        next_body_rot, torch_utils.quat_conjugate(current_body_rot)
    )
    len_traj = diff_global_body_rot.shape[0]
    euler_from_quat = (
        torch_utils.euler_from_quat(diff_global_body_rot.reshape(-1, 4)).reshape(
            len_traj, -1, 3
        )
        / dt
    )

    # result = {
    #     'body_pos': current_body_pos.numpy(),
    #     'root_pos': root_pos[1:-1],
    #     # 'dof_pos': joint_poses[1:-1],
    #     'dof_pos': dof_states[1:-1,:,0],
    #     'body_rot': current_body_rot.numpy(),
    #     'root_rot': root_rot[1:-1],
    #     'body_vel': body_vel.numpy(),
    #     'root_vel': body_vel[:,0,:].numpy(),
    #     # 'body_angular_vel': body_angular_vel.numpy(),
    #     # 'root_angular_vel': body_angular_vel[:,0,:].numpy(),
    #     'body_angular_vel': euler_from_quat.numpy(),
    #     'root_angular_vel': euler_from_quat[:,0,:].numpy(),
    #     'dof_vel': dof_vel[1:],
    #     'dt': dt,
    # }

    result = {
        "body_pos": current_body_pos.numpy()[8:],
        "root_pos": root_pos[1:-1][8:],
        # 'dof_pos': joint_poses[1:-1],
        "dof_pos": dof_states[1:-1, :, 0][8:],
        "body_rot": current_body_rot.numpy()[8:],
        "root_rot": root_rot[1:-1][8:],
        "body_vel": body_vel.numpy()[8:],
        "root_vel": body_vel[:, 0, :].numpy()[8:],
        # 'body_angular_vel': body_angular_vel.numpy(),
        # 'root_angular_vel': body_angular_vel[:,0,:].numpy(),
        "body_angular_vel": euler_from_quat.numpy()[8:],
        "root_angular_vel": euler_from_quat[:, 0, :].numpy()[8:],
        "dof_vel": dof_vel[1:][8:],
        "dt": dt,
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="dataset directory",
        default="/home/jianrenw/mocap/data/videos",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="output directory",
        default="/home/jianrenw/mocap/data/videos",
    )

    args = parser.parse_args()

    # # load motion data
    # adam_poses = joblib.load(args.data_path + "/adam_lite_data.pt")

    # isaac_data = {}

    # for key in tqdm(adam_poses.keys()):
    #     adam_pose = adam_poses[key]
    #     result = adam_to_isaac(adam_pose)
    #     if result is not None:
    #         isaac_data[key] = result

    # joblib.dump(isaac_data, args.out_dir + "/isaac_adam_standard.pt")

    # # save one
    # key = list(adam_poses.keys())[10]
    # adam_pose = adam_poses[key]
    # result = adam_to_isaac(adam_pose)
    # joblib.dump(result, args.out_dir + "/{}.pt".format(key))

    # motions = os.listdir(args.data_path)
    # for motion in motions:
    #     print(args.data_path + "/{}".format(motion))
    #     adam_pose = joblib.load(args.data_path + "/{}".format(motion))
    #     result = adam_to_isaac(adam_pose)
    #     joblib.dump(result, args.out_dir + "/{}.pt".format(motion[:-3]))

    motions = os.path.join(args.data_path, "processed")
    motions = os.listdir(motions)
    for motion in motions:
        adam_pose = joblib.load(
            os.path.join(args.data_path, "processed", motion, "adam_output.pkl")
        )
        result = adam_to_isaac(adam_pose)
        joblib.dump(result, os.path.join(args.out_dir, "{}_fixed.pkl".format(motion)))
