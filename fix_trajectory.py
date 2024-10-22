import joblib
import matplotlib.pyplot as plt
import torch

from utils import torch_utils

adam_pose = joblib.load("data/videos/walk.pkl")

root_pos = adam_pose["root_pos"]
root_rot = adam_pose["root_rot"]
root_vel = adam_pose["root_vel"]
root_angular_vel = adam_pose["root_angular_vel"]
dof_pos = adam_pose["dof_pos"]
dof_vel = adam_pose["dof_vel"]
body_pos = adam_pose["body_pos"]
frame_num = len(root_pos)

root_rot = torch.from_numpy(root_rot)

euler = torch_utils.euler_from_quat(root_rot)
euler = euler.numpy()

# plot euler angles and root positions
euler_fig = plt.subplot(211)
root_fig = plt.subplot(212)
euler_fig.plot(euler)
root_fig.plot(root_pos)
legend = ["x", "y", "z"]
plt.legend(legend)
plt.show()

# fix euler angles
euler[373:, 1] = 0.1
euler = torch.from_numpy(euler)
roll = euler[:, 0]
pitch = euler[:, 1]
yaw = 0.8
root_rot_new = torch_utils.quat_from_euler_xyz(roll, pitch, yaw)

root_rot_new = root_rot_new.numpy()

root_pos[:, 0] = 0


adam_pose["root_rot"] = root_rot_new
adam_pose["root_pos"] = root_pos

joblib.dump(adam_pose, "data/videos/walk_fixed.pkl")
