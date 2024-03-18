import os
import numpy as np
from isaacgym import gymapi,gymtorch,gymutil
import joblib
import torch
import time

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
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

# add cartpole urdf asset
asset_root = "adam"
asset_file = "urdf/adam.urdf"
robot_asset = gym.load_asset(sim, asset_root, asset_file)
rigid_body_names = gym.get_asset_rigid_body_names(robot_asset)
rigid_body_dict = gym.get_asset_rigid_body_dict(robot_asset)

print(rigid_body_names)
print(rigid_body_dict)

spacing = 2.0
lower = gymapi.Vec3(-spacing, -spacing, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
# pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

actor_handle = gym.create_actor(env, robot_asset, pose, "adam_actor", 0, 1)

props = gym.get_actor_dof_properties(env, actor_handle)
names = gym.get_actor_dof_names(env, actor_handle)
lower = props['lower']
upper = props['upper']
print(names)

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# Look at the first env
# cam_pos = gymapi.Vec3(8, 4, 1.5)
# cam_target = gymapi.Vec3(0, 2, 1.5)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# load motion data
key = 'ACCAD_Male2MartialArtsStances_c3d_D5 - ready to walk away_poses'
data_path = "{}.pt".format(key)
adam_pose = joblib.load(data_path)
# adam_pose = adam_poses[key]


root_pos = adam_pose["root_pos"]
root_rot = adam_pose["root_rot"]
dof_pos = adam_pose["dof_pos"]
body_pos = adam_pose["body_pos"]
frame_num = len(root_pos)

root_states = torch.cat([torch.from_numpy(root_pos), torch.from_numpy(root_rot), torch.zeros(frame_num, 6)], dim=1).type(torch.float32)
joint_poses = torch.stack([torch.from_numpy(dof_pos), torch.zeros(frame_num, 23)],axis=2).type(torch.float32)

for i, joint_pose in enumerate(dof_pos[0]):
    # import ipdb; ipdb.set_trace()
    if joint_pose<lower[i] or joint_pose>upper[i]:
        print(names[i], joint_pose, lower[i], upper[i])