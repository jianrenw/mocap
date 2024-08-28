import joblib
import ast
import numpy as np

adam_poses = joblib.load("/data/home/jianrenw/MAP/data/isaac_adam_standard_cleaned.pt")

keys = list(adam_poses.keys())
for key in keys:
    dof_vel = adam_poses[key]['dof_vel']
    dof_vel_sign = np.sign(dof_vel)
    adam_poses[key]['dof_vel_sign'] = dof_vel_sign

joblib.dump(adam_poses, "/data/home/jianrenw/MAP/data/isaac_adam_standard_add_sign.pt")

