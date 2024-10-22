import joblib

# load motion data
data_path = "data/out/isaac_adam_lite.pt"
adam_poses = joblib.load(data_path)
keys = list(adam_poses.keys())
# key = keys[10]
key = "ACCAD_Male2MartialArtsStances_c3d_D5 - ready to walk away_poses"
adam_pose = adam_poses[key]

# save motion data
save_path = "{}.pt".format(key)
joblib.dump(adam_pose, save_path)
