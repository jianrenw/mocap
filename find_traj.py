import joblib

data_path = "data/out/isaac_adam_standard.pt"
adam_pose = joblib.load(data_path)

occlusion_path = "data/out/amass_occlusion.pkl"
occlusion = joblib.load(occlusion_path)

# keys = list(occlusion.keys())
# for key in keys:
#     if 'flip' in key:
#         print(key)

keys = list(adam_pose.keys())
for key in keys:
    if "CMU_88" in key:
        print(key)
