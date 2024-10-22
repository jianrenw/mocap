import argparse
import json
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


def whole_body_ik(urdf_path, motion_data):

    physicsClient = p.connect(p.DIRECT)  # non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    robot_start_pos = [0, 0, 0]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    with suppress_stdout():
        humanoid = p.loadURDF(urdf_path, robot_start_pos, robot_start_orientation)

    num_frames = motion_data["num_frames"]
    pelvis = motion_data["pelvis"]
    upper_rot = motion_data["upper_rot"]
    lower_rot = motion_data["lower_rot"]
    l_elbow_rot = motion_data["l_elbow_rot"]
    r_elbow_rot = motion_data["r_elbow_rot"]
    l_knee_rot = motion_data["l_knee_rot"]
    r_knee_rot = motion_data["r_knee_rot"]
    l_upperarm_dir = motion_data["l_upperarm_dir"]
    r_upperarm_dir = motion_data["r_upperarm_dir"]
    l_thigh_dir = motion_data["l_thigh_dir"]
    r_thigh_dir = motion_data["r_thigh_dir"]
    waist_angles = motion_data["waist_angles"]
    l_foot_angle = motion_data["l_foot_angle"]
    r_foot_angle = motion_data["r_foot_angle"]
    l_foot_dir = motion_data["l_foot_dir"]
    r_foot_dir = motion_data["r_foot_dir"]

    upperarm_len = 0.2648365892539233
    thigh_len = 0.4252746432372357
    toe_len = 0.167
    heel_len = 0.079

    toe_R = R.from_euler("y", np.arctan(0.047 / 0.160)).as_matrix()
    heel_R = R.from_euler("y", np.pi / 2 + np.arctan(0.047 / 0.064)).as_matrix()

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
    jointDamping[12] = 100
    jointDamping[13] = 100
    jointDamping[14] = 100
    restPoses = [0.0] * num_joints
    init_pose = [0.0] * num_joints

    min_foot_z = np.inf

    for i in range(num_frames):

        init_pose[12] = waist_angles[i, 0]
        init_pose[13] = waist_angles[i, 1]
        init_pose[14] = waist_angles[i, 2]
        restPoses[12] = waist_angles[i, 0]
        restPoses[13] = waist_angles[i, 1]
        restPoses[14] = waist_angles[i, 2]

        p.resetBasePositionAndOrientation(
            humanoid, (pelvis[i]).tolist(), lower_rot[i].as_quat().tolist()
        )
        for joint_index, angle in enumerate(init_pose):
            p.resetJointState(humanoid, joint_index, angle)

        l_shoulder_pos, _ = p.getLinkState(
            humanoid, joint_names.index("shoulderRoll_Left")
        )[4:6]
        r_shoulder_pos, _ = p.getLinkState(
            humanoid, joint_names.index("shoulderRoll_Right")
        )[4:6]
        l_hip_pos, _ = p.getLinkState(humanoid, joint_names.index("hipRoll_Left"))[4:6]
        r_hip_pos, _ = p.getLinkState(humanoid, joint_names.index("hipRoll_Right"))[4:6]

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
            joint_names.index("elbow_Right"),
            r_elbow_pos,
            targetOrientation=r_elbow_rot.as_quat()[i],
            jointDamping=jointDamping,  # Adjust damping as needed,
            restPoses=restPoses,  # Initial joint angles
        )

        ik_solution_lh = p.calculateInverseKinematics(
            humanoid,
            joint_names.index("elbow_Left"),
            l_elbow_pos,
            targetOrientation=l_elbow_rot.as_quat()[i],
            jointDamping=jointDamping,  # Adjust damping as needed
            restPoses=restPoses,  # Initial joint anglesss
        )

        ik_solution_rf = p.calculateInverseKinematics(
            humanoid,
            joint_names.index("kneePitch_Right"),
            r_knee_pos,
            targetOrientation=r_knee_rot.as_quat()[i],
            jointDamping=jointDamping,  # Adjust damping as needed
            restPoses=restPoses,  # Initial joint angles
        )

        ik_solution_lf = p.calculateInverseKinematics(
            humanoid,
            joint_names.index("kneePitch_Left"),
            l_knee_pos,
            targetOrientation=l_knee_rot.as_quat()[i],
            jointDamping=jointDamping,  # Adjust damping as needed
            restPoses=restPoses,  # Initial joint angles
        )

        # p.resetJointState(humanoid, joint_index, ik_solution)

        # adam
        # ik_solution = np.zeros(num_joints)
        # ik_solution[:4] = ik_solution_lf[:4]
        # ik_solution[4] = l_foot_angle[i]
        # ik_solution[6:10] = ik_solution_rf[6:10]
        # ik_solution[10] = r_foot_angle[i]
        # ik_solution[12] = waist_angles[i,0]
        # ik_solution[13] = waist_angles[i,1]
        # ik_solution[14] = waist_angles[i,2]
        # ik_solution[16:20] = ik_solution_lh[15:19]
        # ik_solution[24:28] = ik_solution_rh[19:23]

        # adam lite with wrist
        # ik_solution = np.zeros(num_joints)
        # ik_solution[:4] = ik_solution_lf[:4]
        # ik_solution[4] = l_foot_angle[i]
        # ik_solution[6:10] = ik_solution_rf[6:10]
        # ik_solution[10] = r_foot_angle[i]
        # ik_solution[12] = waist_angles[i,0]
        # ik_solution[13] = waist_angles[i,1]
        # ik_solution[14] = waist_angles[i,2]
        # ik_solution[16:20] = ik_solution_lh[15:19]
        # ik_solution[22:26] = ik_solution_rh[20:24]

        # adam lite
        ik_solution = np.zeros(num_joints)
        ik_solution[:4] = ik_solution_lf[:4]
        ik_solution[4] = l_foot_angle[i]
        ik_solution[6:10] = ik_solution_rf[6:10]
        ik_solution[10] = r_foot_angle[i]
        ik_solution[12] = waist_angles[i, 0]
        ik_solution[13] = waist_angles[i, 1]
        ik_solution[14] = waist_angles[i, 2]
        ik_solution[16:20] = ik_solution_lh[15:19]
        ik_solution[22:26] = ik_solution_rh[19:23]

        restPoses = ik_solution
        init_pose = ik_solution

        for joint_index, angle in enumerate(init_pose):
            p.resetJointState(humanoid, joint_index, angle)

        l_ankle_pos, l_ankle_rot = p.getLinkState(
            humanoid, joint_names.index("anklePitch_Left")
        )[4:6]
        r_ankle_pos, r_ankle_rot = p.getLinkState(
            humanoid, joint_names.index("anklePitch_Right")
        )[4:6]

        l_rot_matrix = p.getMatrixFromQuaternion(l_ankle_rot)
        l_rot_matrix = np.array(l_rot_matrix).reshape(3, 3)
        l_toe_pose = np.dot(l_rot_matrix @ toe_R, [toe_len, 0, 0]) + l_ankle_pos
        l_heel_pose = np.dot(l_rot_matrix @ heel_R, [heel_len, 0, 0]) + l_ankle_pos

        r_rot_matrix = p.getMatrixFromQuaternion(r_ankle_rot)
        r_rot_matrix = np.array(r_rot_matrix).reshape(3, 3)
        r_toe_pose = np.dot(r_rot_matrix @ toe_R, [toe_len, 0, 0]) + r_ankle_pos
        r_heel_pose = np.dot(r_rot_matrix @ heel_R, [heel_len, 0, 0]) + r_ankle_pos

        if i < 20:
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
        "root_rot": lower_rot.as_quat(),
        "joint_poses": np.array(joint_poses),
        "joint_names": joint_names,
    }

    return result


def motion2adam(skeleton):

    num_frames = skeleton.shape[0]

    pelvis = skeleton[:, 0, :]

    l_upperarm = skeleton[:, 5, :]
    l_forearm = skeleton[:, 6, :]
    l_hand = skeleton[:, 7, :]

    r_upperarm = skeleton[:, 12, :]
    r_forearm = skeleton[:, 13, :]
    r_hand = skeleton[:, 14, :]

    l_thigh = skeleton[:, 1, :]
    l_calf = skeleton[:, 2, :]
    l_foot = skeleton[:, 3, :]
    l_toe = skeleton[:, 4, :]

    r_thigh = skeleton[:, 8, :]
    r_calf = skeleton[:, 9, :]
    r_foot = skeleton[:, 10, :]
    r_toe = skeleton[:, 11, :]

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
    l_elbow_rot_z = l_forearm - l_hand
    l_elbow_rot_z = l_elbow_rot_z / np.linalg.norm(l_elbow_rot_z, axis=1, keepdims=True)
    l_elbow_rot_y_2 = l_upperarm - l_forearm
    l_elbow_rot_y_2 = l_elbow_rot_y_2 / np.linalg.norm(
        l_elbow_rot_y_2, axis=1, keepdims=True
    )
    l_elbow_rot_y_2 = np.cross(l_elbow_rot_y_2, -l_elbow_rot_z)
    l_elbow_rot_y_2 = l_elbow_rot_y_2 / np.linalg.norm(
        l_elbow_rot_y_2, axis=1, keepdims=True
    )

    l_elbow_rot_x = np.cross(l_elbow_rot_y_2, l_elbow_rot_z)
    l_elbow_rot_x = l_elbow_rot_x / np.linalg.norm(l_elbow_rot_x, axis=1, keepdims=True)

    l_elbow_rot = np.stack([l_elbow_rot_x, l_elbow_rot_y_2, l_elbow_rot_z], axis=2)

    l_upperarm_dir = l_forearm - l_upperarm
    l_upperarm_dir = l_upperarm_dir / np.linalg.norm(
        l_upperarm_dir, axis=1, keepdims=True
    )

    # right arm
    r_elbow_rot_z = r_forearm - r_hand
    r_elbow_rot_z = r_elbow_rot_z / np.linalg.norm(r_elbow_rot_z, axis=1, keepdims=True)
    r_elbow_rot_y_2 = r_upperarm - r_forearm
    r_elbow_rot_y_2 = r_elbow_rot_y_2 / np.linalg.norm(
        r_elbow_rot_y_2, axis=1, keepdims=True
    )
    r_elbow_rot_y_2 = np.cross(r_elbow_rot_y_2, -r_elbow_rot_z)
    r_elbow_rot_y_2 = r_elbow_rot_y_2 / np.linalg.norm(
        r_elbow_rot_y_2, axis=1, keepdims=True
    )

    r_elbow_rot_x = np.cross(r_elbow_rot_y_2, r_elbow_rot_z)
    r_elbow_rot_x = r_elbow_rot_x / np.linalg.norm(r_elbow_rot_x, axis=1, keepdims=True)
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
    l_foot_angle = np.arccos(cos) - np.arctan(0.047 / 0.160) - np.pi / 2

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
    r_foot_angle = np.arccos(cos) - np.arctan(0.047 / 0.160) - np.pi / 2

    # to pybullet
    upper_rot = R.from_matrix(upper_rot)
    lower_rot = R.from_matrix(lower_rot)
    l_knee_rot = R.from_matrix(l_knee_rot)
    r_knee_rot = R.from_matrix(r_knee_rot)
    l_elbow_rot = R.from_matrix(l_elbow_rot)
    r_elbow_rot = R.from_matrix(r_elbow_rot)

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
        "waist_angles": waist_angles,
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
        default="/home/jianrenw/mocap/data/parkour/motions",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="output directory",
        default="/home/jianrenw/mocap/data/parkour/joints",
    )

    args = parser.parse_args()

    target_fr = 50

    # load robot to pybullet
    home_dir = os.path.expanduser("~")
    urdf_path = "{}/mocap/robots/adam_lite/urdf/adam_lite_pybullet.urdf".format(
        home_dir
    )

    motions = os.listdir(args.data_path)
    for motion in motions:
        print(args.data_path + "/{}".format(motion))
        skeleton = np.load(args.data_path + "/{}".format(motion))
        motion_data = motion2adam(skeleton)
        result = whole_body_ik(urdf_path, motion_data)
        result["real_frame_rate"] = 60
        joblib.dump(result, args.out_dir + "/{}.pt".format(motion[:-4]))
