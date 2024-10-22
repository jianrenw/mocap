import argparse
import os
import os.path as osp
import sys
import time
from contextlib import contextmanager
from multiprocessing import Pool

import joblib
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def whole_body_ik(urdf_path, amass_data):

    physicsClient = p.connect(p.DIRECT)  # non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    robot_start_pos = [0, 0, 0]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    with suppress_stdout():
        humanoid = p.loadURDF(urdf_path, robot_start_pos, robot_start_orientation)

    num_frames = amass_data["num_frames"]
    pelvis = amass_data["pelvis"]
    upper_rot = amass_data["upper_rot"]
    lower_rot = amass_data["lower_rot"]
    l_elbow_rot = amass_data["l_elbow_rot"]
    r_elbow_rot = amass_data["r_elbow_rot"]
    l_knee_rot = amass_data["l_knee_rot"]
    r_knee_rot = amass_data["r_knee_rot"]
    l_upperarm_dir = amass_data["l_upperarm_dir"]
    r_upperarm_dir = amass_data["r_upperarm_dir"]
    l_thigh_dir = amass_data["l_thigh_dir"]
    r_thigh_dir = amass_data["r_thigh_dir"]
    torso_angle = amass_data["torso_angle"]
    l_foot_angle = amass_data["l_foot_angle"]
    r_foot_angle = amass_data["r_foot_angle"]
    l_foot_dir = amass_data["l_foot_dir"]
    r_foot_dir = amass_data["r_foot_dir"]

    upperarm_len = 0.3328145877824913
    thigh_len = 0.40
    toe_len = 0.188
    heel_len = 0.096

    toe_R = R.from_euler("y", np.arctan(0.07 / 0.175)).as_matrix()
    heel_R = R.from_euler("y", np.pi / 2 + np.arctan(0.07 / 0.065)).as_matrix()

    num_joints = p.getNumJoints(humanoid)

    joint_names = []

    pelvis_copy = pelvis.copy()

    # Iterate through the joints and print their names
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(humanoid, joint_index)
        joint_name = joint_info[1].decode(
            "utf-8"
        )  # Decode the byte string to get the joint name
        joint_names.append(joint_name)

    joint_poses = []
    jointDamping = [0.1] * num_joints
    jointDamping[10] = 100
    restPoses = [0.0] * num_joints
    init_pose = [0.0] * num_joints

    min_foot_z = np.inf

    for i in range(num_frames):

        init_pose[10] = torso_angle[i, 0]
        restPoses[10] = torso_angle[i, 0]

        p.resetBasePositionAndOrientation(
            humanoid, (pelvis[i]).tolist(), upper_rot[i].as_quat().tolist()
        )
        for joint_index, angle in enumerate(init_pose):
            p.resetJointState(humanoid, joint_index, angle)

        l_shoulder_pos, _ = p.getLinkState(
            humanoid, joint_names.index("left_shoulder_roll_joint")
        )[4:6]
        r_shoulder_pos, _ = p.getLinkState(
            humanoid, joint_names.index("right_shoulder_roll_joint")
        )[4:6]
        l_hip_pos, _ = p.getLinkState(
            humanoid, joint_names.index("left_hip_pitch_joint")
        )[4:6]
        r_hip_pos, _ = p.getLinkState(
            humanoid, joint_names.index("right_hip_pitch_joint")
        )[4:6]

        # IK for l hand
        l_elbow_pos = np.array(l_shoulder_pos) + l_upperarm_dir[i] * upperarm_len

        # IK for r hand
        r_elbow_pos = np.array(r_shoulder_pos) + r_upperarm_dir[i] * upperarm_len

        # IK for l foot
        l_knee_pos = np.array(l_hip_pos) + l_thigh_dir[i] * thigh_len

        # IK for r foot
        r_knee_pos = np.array(r_hip_pos) + r_thigh_dir[i] * thigh_len

        # # Calculate the IK solution
        ik_solution_rh = p.calculateInverseKinematics(
            humanoid,
            joint_names.index("right_elbow_joint"),
            r_elbow_pos,
            targetOrientation=r_elbow_rot.as_quat()[i],
            jointDamping=jointDamping,  # Adjust damping as needed,
            restPoses=restPoses,  # Initial joint angles
        )

        ik_solution_lh = p.calculateInverseKinematics(
            humanoid,
            joint_names.index("left_elbow_joint"),
            l_elbow_pos,
            targetOrientation=l_elbow_rot.as_quat()[i],
            jointDamping=jointDamping,  # Adjust damping as needed
            restPoses=restPoses,  # Initial joint anglesss
        )

        ik_solution_rf = p.calculateInverseKinematics(
            humanoid,
            joint_names.index("right_knee_joint"),
            r_knee_pos,
            targetOrientation=r_knee_rot.as_quat()[i],
            jointDamping=jointDamping,  # Adjust damping as needed
            restPoses=restPoses,  # Initial joint angles
        )

        ik_solution_lf = p.calculateInverseKinematics(
            humanoid,
            joint_names.index("left_knee_joint"),
            l_knee_pos,
            targetOrientation=l_knee_rot.as_quat()[i],
            jointDamping=jointDamping,  # Adjust damping as needed
            restPoses=restPoses,  # Initial joint angles
        )

        # p.resetJointState(humanoid, joint_index, ik_solution)

        ik_solution = np.zeros(num_joints)
        ik_solution[:4] = ik_solution_lf[:4]
        ik_solution[4] = l_foot_angle[i]
        ik_solution[5:9] = ik_solution_rf[5:9]
        ik_solution[9] = r_foot_angle[i]
        ik_solution[10] = torso_angle[i, 0]
        ik_solution[11:15] = ik_solution_lh[11:15]
        ik_solution[15:19] = ik_solution_rh[15:19]

        restPoses = ik_solution
        init_pose = ik_solution

        for joint_index, angle in enumerate(init_pose):
            p.resetJointState(humanoid, joint_index, angle)

        l_ankle_pos, l_ankle_rot = p.getLinkState(
            humanoid, joint_names.index("left_ankle_joint")
        )[4:6]
        r_ankle_pos, r_ankle_rot = p.getLinkState(
            humanoid, joint_names.index("right_ankle_joint")
        )[4:6]

        l_rot_matrix = p.getMatrixFromQuaternion(l_ankle_rot)
        l_rot_matrix = np.array(l_rot_matrix).reshape(3, 3)
        l_toe_pose = np.dot(l_rot_matrix @ toe_R, [toe_len, 0, 0]) + l_ankle_pos
        l_heel_pose = np.dot(l_rot_matrix @ heel_R, [heel_len, 0, 0]) + l_ankle_pos

        r_rot_matrix = p.getMatrixFromQuaternion(r_ankle_rot)
        r_rot_matrix = np.array(r_rot_matrix).reshape(3, 3)
        r_toe_pose = np.dot(r_rot_matrix @ toe_R, [toe_len, 0, 0]) + r_ankle_pos
        r_heel_pose = np.dot(r_rot_matrix @ heel_R, [heel_len, 0, 0]) + r_ankle_pos

        if i < 30:
            current_min = np.min(
                [l_toe_pose[2], r_toe_pose[2], l_heel_pose[2], r_heel_pose[2]]
            )
            if min_foot_z > current_min:
                min_foot_z = current_min

        pelvis_copy[i, 2] = pelvis[i, 2]

        # Step the simulation
        joint_poses.append(ik_solution)

    p.disconnect()

    result = {
        "root_pos": pelvis_copy - min_foot_z,
        "root_rot": upper_rot.as_quat(),
        "joint_poses": np.array(joint_poses),
        "joint_names": joint_names,
    }

    return result


def amass2h1(skeleton):

    num_frames = skeleton.shape[0]

    pelvis = skeleton[:, 0, :]

    l_upperarm = skeleton[:, 16, :]
    l_forearm = skeleton[:, 18, :]
    l_hand = skeleton[:, 20, :]

    r_upperarm = skeleton[:, 17, :]
    r_forearm = skeleton[:, 19, :]
    r_hand = skeleton[:, 21, :]

    l_thigh = skeleton[:, 1, :]
    l_calf = skeleton[:, 4, :]
    l_foot = skeleton[:, 7, :]
    l_toe = skeleton[:, 10, :]

    r_thigh = skeleton[:, 2, :]
    r_calf = skeleton[:, 5, :]
    r_foot = skeleton[:, 8, :]
    r_toe = skeleton[:, 11, :]

    l_elbow_i = skeleton[:, 24, :]
    l_elbow_o = skeleton[:, 25, :]
    r_elbow_i = skeleton[:, 26, :]
    r_elbow_o = skeleton[:, 27, :]

    # upper body
    head = (l_upperarm + r_upperarm) / 2
    upper_z = (head - pelvis) / np.linalg.norm(head - pelvis, axis=1, keepdims=True)
    upper_x = np.cross(pelvis - l_upperarm, pelvis - r_upperarm)
    upper_x = upper_x / np.linalg.norm(upper_x, axis=1, keepdims=True)
    upper_y = np.cross(upper_z, upper_x)
    upper_rot = np.stack([upper_x, upper_y, upper_z], axis=2)

    # lower body
    crotch = (l_thigh + r_thigh) / 2
    lower_z = -(crotch - pelvis) / np.linalg.norm(
        crotch - pelvis, axis=1, keepdims=True
    )
    lower_x = np.cross(pelvis - r_thigh, pelvis - l_thigh)
    lower_x = lower_x / np.linalg.norm(lower_x, axis=1, keepdims=True)
    lower_y = np.cross(lower_z, lower_x)
    lower_rot = np.stack([lower_x, lower_y, lower_z], axis=2)

    # left arm
    l_elbow_rot_x = l_hand - l_forearm
    l_elbow_rot_x = l_elbow_rot_x / np.linalg.norm(l_elbow_rot_x, axis=1, keepdims=True)
    l_elbow_rot_y_1 = l_elbow_i - l_elbow_o
    l_elbow_rot_y_1 = l_elbow_rot_y_1 / np.linalg.norm(
        l_elbow_rot_y_1, axis=1, keepdims=True
    )
    l_elbow_rot_y_2 = l_upperarm - l_forearm
    l_elbow_rot_y_2 = l_elbow_rot_y_2 / np.linalg.norm(
        l_elbow_rot_y_2, axis=1, keepdims=True
    )
    l_elbow_rot_y_2 = np.cross(l_elbow_rot_y_2, l_elbow_rot_x)
    l_elbow_rot_y_2 = l_elbow_rot_y_2 / np.linalg.norm(
        l_elbow_rot_y_2, axis=1, keepdims=True
    )
    correction = np.einsum("ij,ij->i", l_elbow_rot_y_1, l_elbow_rot_y_2) < 0
    l_elbow_rot_y_2[correction] = -l_elbow_rot_y_2[correction]

    l_elbow_rot_z = np.cross(l_elbow_rot_x, l_elbow_rot_y_2)
    l_elbow_rot_z = l_elbow_rot_z / np.linalg.norm(l_elbow_rot_z, axis=1, keepdims=True)

    l_elbow_rot = np.stack([l_elbow_rot_x, l_elbow_rot_y_2, l_elbow_rot_z], axis=2)

    l_upperarm_dir = l_forearm - l_upperarm
    l_upperarm_dir = l_upperarm_dir / np.linalg.norm(
        l_upperarm_dir, axis=1, keepdims=True
    )

    # right arm
    r_elbow_rot_x = r_hand - r_forearm
    r_elbow_rot_x = r_elbow_rot_x / np.linalg.norm(r_elbow_rot_x, axis=1, keepdims=True)
    r_elbow_rot_y_1 = r_elbow_i - r_elbow_o
    r_elbow_rot_y_1 = r_elbow_rot_y_1 / np.linalg.norm(
        r_elbow_rot_y_1, axis=1, keepdims=True
    )
    r_elbow_rot_y_2 = r_upperarm - r_forearm
    r_elbow_rot_y_2 = r_elbow_rot_y_2 / np.linalg.norm(
        r_elbow_rot_y_2, axis=1, keepdims=True
    )
    r_elbow_rot_y_2 = np.cross(r_elbow_rot_y_2, r_elbow_rot_x)
    r_elbow_rot_y_2 = r_elbow_rot_y_2 / np.linalg.norm(
        r_elbow_rot_y_2, axis=1, keepdims=True
    )
    correction = np.einsum("ij,ij->i", r_elbow_rot_y_1, r_elbow_rot_y_2) < 0
    r_elbow_rot_y_2[correction] = -r_elbow_rot_y_2[correction]

    r_elbow_rot_z = np.cross(r_elbow_rot_x, r_elbow_rot_y_2)
    r_elbow_rot_z = r_elbow_rot_z / np.linalg.norm(r_elbow_rot_z, axis=1, keepdims=True)

    r_elbow_rot = np.stack([r_elbow_rot_x, r_elbow_rot_y_2, r_elbow_rot_z], axis=2)

    r_upperarm_dir = r_forearm - r_upperarm
    r_upperarm_dir = r_upperarm_dir / np.linalg.norm(
        r_upperarm_dir, axis=1, keepdims=True
    )

    # left knee
    l_calf_dir = l_calf - l_foot
    l_calf_dir = l_calf_dir / np.linalg.norm(l_calf_dir, axis=1, keepdims=True)
    l_foot_dir = l_toe - l_foot
    l_foot_dir = l_foot_dir / np.linalg.norm(l_foot_dir, axis=1, keepdims=True)
    l_knee_rot_y = np.cross(l_calf_dir, l_foot_dir)
    l_knee_rot_y = l_knee_rot_y / np.linalg.norm(l_knee_rot_y, axis=1, keepdims=True)
    l_knee_rot_x = np.cross(l_knee_rot_y, l_calf_dir)
    l_knee_rot_x = l_knee_rot_x / np.linalg.norm(l_knee_rot_x, axis=1, keepdims=True)
    l_knee_rot = np.stack([l_knee_rot_x, l_knee_rot_y, l_calf_dir], axis=2)

    l_thigh_dir = l_calf - l_thigh
    l_thigh_dir = l_thigh_dir / np.linalg.norm(l_thigh_dir, axis=1, keepdims=True)

    # left ankle angle
    cos = np.einsum("ij,ij->i", l_calf_dir, l_foot_dir) / (
        np.linalg.norm(l_calf_dir, axis=1) * np.linalg.norm(l_foot_dir, axis=1)
    )
    l_foot_angle = np.arccos(cos) - np.arctan(0.07 / 0.175) - np.pi / 2

    # right knee
    r_calf_dir = r_calf - r_foot
    r_calf_dir = r_calf_dir / np.linalg.norm(r_calf_dir, axis=1, keepdims=True)
    r_foot_dir = r_toe - r_foot
    r_foot_dir = r_foot_dir / np.linalg.norm(r_foot_dir, axis=1, keepdims=True)
    r_knee_rot_y = np.cross(r_calf_dir, r_foot_dir)
    r_knee_rot_y = r_knee_rot_y / np.linalg.norm(r_knee_rot_y, axis=1, keepdims=True)
    r_knee_rot_x = np.cross(r_knee_rot_y, r_calf_dir)
    r_knee_rot_x = r_knee_rot_x / np.linalg.norm(r_knee_rot_x, axis=1, keepdims=True)
    r_knee_rot = np.stack([r_knee_rot_x, r_knee_rot_y, r_calf_dir], axis=2)

    r_thigh_dir = r_calf - r_thigh
    r_thigh_dir = r_thigh_dir / np.linalg.norm(r_thigh_dir, axis=1, keepdims=True)

    # right ankle angle
    cos = np.einsum("ij,ij->i", r_calf_dir, r_foot_dir) / (
        np.linalg.norm(r_calf_dir, axis=1) * np.linalg.norm(r_foot_dir, axis=1)
    )
    r_foot_angle = np.arccos(cos) - np.arctan(0.07 / 0.175) - np.pi / 2

    # to pybullet
    upper_rot = R.from_matrix(upper_rot)
    lower_rot = R.from_matrix(lower_rot)
    l_knee_rot = R.from_matrix(l_knee_rot)
    r_knee_rot = R.from_matrix(r_knee_rot)
    l_elbow_rot = R.from_matrix(l_elbow_rot)
    r_elbow_rot = R.from_matrix(r_elbow_rot)

    # torso angle
    torso_angle = np.arccos(
        np.clip(np.sum(lower_y * upper_y, axis=1, keepdims=True), -1, 1)
    )
    torso_sign = np.sign(
        np.sum(np.cross(lower_y, upper_y) * upper_z, axis=1, keepdims=True)
    )
    torso_angle = torso_angle * torso_sign

    # waist angle
    R_u_l = np.einsum(
        "ijk,ikl->ijl",
        np.transpose(lower_rot.as_matrix(), (0, 2, 1)),
        upper_rot.as_matrix(),
    )
    R_u_l = R.from_matrix(R_u_l)
    waist_angles = R_u_l.as_euler("xyz", degrees=False)

    amass_data = {
        "num_frames": num_frames,
        "pelvis": pelvis,
        "upper_rot": upper_rot,
        "lower_rot": lower_rot,
        "l_elbow_rot": l_elbow_rot,
        "r_elbow_rot": r_elbow_rot,
        "l_knee_rot": l_knee_rot,
        "r_knee_rot": r_knee_rot,
        "l_upperarm_dir": l_upperarm_dir,
        "r_upperarm_dir": r_upperarm_dir,
        "l_thigh_dir": l_thigh_dir,
        "r_thigh_dir": r_thigh_dir,
        "torso_angle": torso_angle,
        "l_foot_angle": l_foot_angle,
        "r_foot_angle": r_foot_angle,
        "l_foot_dir": l_foot_dir,
        "r_foot_dir": r_foot_dir,
    }

    return amass_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="dataset directory",
        default="/home/jwang/mocap/data/out",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="output directory",
        default="/home/jwang/mocap/data/out",
    )

    args = parser.parse_args()

    target_fr = 50
    amass_skeleton = joblib.load(args.data_path + "/amass.pt")
    amass_occlusion = joblib.load(args.data_path + "/amass_occlusion.pkl")

    # load robot to pybullet
    home_dir = os.path.expanduser("~")
    urdf_path = "{}/mocap/h1/h1.urdf".format(home_dir)

    keys = list(amass_skeleton.keys())
    occlusion_keys = list(amass_occlusion.keys())
    occlusion_keys = [occlusion_key[2:] for occlusion_key in occlusion_keys]

    # def process(key):
    #     if key in occlusion_keys:
    #         print('occlusion', key)
    #         return
    #     useful_poses = amass_skeleton[key]['skeleton']
    #     framerate = amass_skeleton[key]['mocap_framerate']
    #     skip = int(framerate / target_fr)
    #     useful_poses = useful_poses[::skip]
    #     real_frame_rate = framerate / skip
    #     amass_data = amass2h1(useful_poses)
    #     result = whole_body_ik(urdf_path, amass_data)
    #     result['real_frame_rate'] = real_frame_rate
    #     joblib.dump(result, args.out_dir + "/temp/{}_h1.pt".format(key))

    # with Pool(15) as p:
    #     p.map(process, keys)

    h1_data = {}
    for key in tqdm(amass_skeleton.keys()):
        if key in occlusion_keys:
            print("occlusion", key)
            continue
        result = joblib.load(args.out_dir + "/temp/{}_h1.pt".format(key))
        h1_data[key] = result
    joblib.dump(h1_data, "h1_data.pt")
