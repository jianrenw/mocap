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
# asset_root = "adam"
# asset_file = "urdf/adam.urdf"
# asset_root = "adam_lite_v2"
# asset_file = "urdf/adam_lite_v2_wrist_yaw.urdf"
# asset_root = "robots/adam_lite"
# asset_file = "urdf/adam_lite.urdf"
asset_root = "robots/adam_standard"
asset_file = "urdf/adam_standard.urdf"
robot_asset = gym.load_asset(sim, asset_root, asset_file)
rigid_body_names = gym.get_asset_rigid_body_names(robot_asset)
rigid_body_dict = gym.get_asset_rigid_body_dict(robot_asset)

print(rigid_body_names)
print(len(rigid_body_names))
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
print(props['lower'])
print(props['upper'])


# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# load motion data
# data_path = "data/out/isaac_adam_standard.pt"
# key = 'ACCAD_Male2MartialArtsStances_c3d_D5 - ready to walk away_poses'
# key = 'KIT_3_walking_medium02_poses'
# key = 'CMU_88_88_05_poses'
# adam_poses = joblib.load(data_path)
# # keys = list(adam_poses.keys())
# # key = keys[301]
# adam_pose = adam_poses[key]
# joblib.dump(adam_pose, "data/out/walk_fixed.pt")
adam_pose = joblib.load("data/out/walk_fixed.pt")



root_pos = adam_pose["root_pos"]
root_rot = adam_pose["root_rot"]
root_vel = adam_pose["root_vel"]
root_angular_vel = adam_pose["root_angular_vel"]
dof_pos = adam_pose["dof_pos"]
dof_vel = adam_pose["dof_vel"]
body_pos = adam_pose["body_pos"]
frame_num = len(root_pos)

# root_states = torch.cat([torch.from_numpy(root_pos), torch.from_numpy(root_rot), torch.zeros(frame_num, 6)], dim=1).type(torch.float32)
# joint_poses = torch.stack([torch.from_numpy(dof_pos), torch.zeros(frame_num, dof_pos.shape[1])],axis=2).type(torch.float32)
root_states = torch.cat([torch.from_numpy(root_pos), torch.from_numpy(root_rot), torch.from_numpy(root_vel), torch.from_numpy(root_angular_vel)], dim=1).type(torch.float32)
joint_poses = torch.stack([torch.from_numpy(dof_pos), torch.from_numpy(dof_vel)],axis=2).type(torch.float32)
print(root_vel[0])

print(body_pos.shape)


def draw_reference(gym, viewer, env_handle, body_pos, radius=0.01):
    track_link = [rigid_body_names.index('anklePitchLeft'), rigid_body_names.index('anklePitchRight'), rigid_body_names.index('wristPitchLeft'), rigid_body_names.index('wristPitchRight')]
    updated_body_pos = body_pos[track_link]
    for ubp in updated_body_pos:
        sphere_pose = gymapi.Transform(gymapi.Vec3(ubp[0], ubp[1], ubp[2]), r=None)
        sphere_geom = gymutil.WireframeSphereGeometry(radius, 10, 10, None, color=(1, 0, 0))
        gymutil.draw_lines(sphere_geom, gym, viewer, env_handle, sphere_pose)


# Simulate
for i in range(frame_num):
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states[i]))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(joint_poses[i]))
    draw_reference(gym, viewer, env, body_pos[i])

    # # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    time.sleep(0.01)
    # time.sleep(5000)
    # gym.clear_lines(viewer)