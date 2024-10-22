import os
import time

import joblib
import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil

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
plane_params.distance = 0
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

actor_handle = gym.create_actor(env, robot_asset, pose, "h1_actor", 0, 1)

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError("*** Failed to create viewer")

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# load motion data
data_path = "data/out/isaac_h1.pt"
h1_poses = joblib.load(data_path)
keys = list(h1_poses.keys())
key = keys[10]
h1_pose = h1_poses[key]


root_pos = h1_pose["root_pos"]
root_rot = h1_pose["root_rot"]
dof_pos = h1_pose["dof_pos"]
frame_num = len(root_pos)

# root_pos[:,2] += 0.05

root_states = torch.cat(
    [torch.from_numpy(root_pos), torch.from_numpy(root_rot), torch.zeros(frame_num, 6)],
    dim=1,
).type(torch.float32)
# joint_poses = torch.stack([torch.from_numpy(dof_pos), torch.zeros(frame_num, 19)],axis=2).type(torch.float32)
joint_poses = torch.stack(
    [torch.from_numpy(dof_pos), torch.zeros(frame_num, 19)], axis=2
).type(torch.float32)
# print(root_states)


def draw_reference(gym, viewer, env_handle, body_pos, radius=0.01):
    track_link = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20, 21]
    updated_body_pos = body_pos[track_link]
    for ubp in updated_body_pos:
        sphere_pose = gymapi.Transform(gymapi.Vec3(ubp[0], ubp[1], ubp[2]), r=None)
        sphere_geom = gymutil.WireframeSphereGeometry(
            radius, 4, 4, None, color=(1, 0, 0)
        )
        gymutil.draw_lines(sphere_geom, gym, viewer, env_handle, sphere_pose)


# Simulate
for i in range(frame_num):

    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states[i]))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(joint_poses[i]))
    # draw_reference(gym, viewer, env, body_pos[i])

    # # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    time.sleep(0.05)
    gym.clear_lines(viewer)
