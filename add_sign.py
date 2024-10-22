import ast

import joblib
import numpy as np

adam_poses = joblib.load("/data/home/jianrenw/MAP/data/isaac_adam_standard_walk.pt")

keys = list(adam_poses.keys())
for key in keys:
    dof_vel = adam_poses[key]["dof_vel"]
    dof_vel_sign = np.sign(dof_vel)
    adam_poses[key]["dof_vel_sign"] = dof_vel_sign
    adam_poses[key]["left_foot_contact"] = adam_poses[key]["left_foot_contact"][:-1]
    adam_poses[key]["right_foot_contact"] = adam_poses[key]["right_foot_contact"][:-1]

joblib.dump(adam_poses, "/data/home/jianrenw/MAP/data/isaac_adam_standard_walk.pt")
