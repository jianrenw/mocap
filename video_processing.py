import argparse
import glob
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import cv2
import joblib
import numpy as np
import pybullet as p
import pybullet_data
import torch
from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.models import build_body_model, build_network
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from loguru import logger
from progress.bar import Bar
from scipy.spatial.transform import Rotation as R


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


def setup_pybullet(urdf_path, gui=False):
    physicsClient = p.connect(p.GUI if gui else p.DIRECT)  # non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    robot_start_pos = [0, 0, 0]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    # with suppress_stdout():
    humanoid = p.loadURDF(urdf_path, robot_start_pos, robot_start_orientation)

    return physicsClient, humanoid


def amass2adam(skeleton):

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
    l_elbow_rot_z = l_forearm - l_hand
    l_elbow_rot_z = l_elbow_rot_z / np.linalg.norm(l_elbow_rot_z, axis=1, keepdims=True)
    l_elbow_rot_y_1 = l_elbow_i - l_elbow_o
    l_elbow_rot_y_1 = l_elbow_rot_y_1 / np.linalg.norm(
        l_elbow_rot_y_1, axis=1, keepdims=True
    )
    l_elbow_rot_y_2 = l_upperarm - l_forearm
    l_elbow_rot_y_2 = l_elbow_rot_y_2 / np.linalg.norm(
        l_elbow_rot_y_2, axis=1, keepdims=True
    )
    l_elbow_rot_y_2 = np.cross(l_elbow_rot_y_2, -l_elbow_rot_z)
    l_elbow_rot_y_2 = l_elbow_rot_y_2 / np.linalg.norm(
        l_elbow_rot_y_2, axis=1, keepdims=True
    )
    correction = np.einsum("ij,ij->i", l_elbow_rot_y_1, l_elbow_rot_y_2) < 0
    l_elbow_rot_y_2[correction] = -l_elbow_rot_y_2[correction]

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
    r_elbow_rot_y_1 = r_elbow_i - r_elbow_o
    r_elbow_rot_y_1 = r_elbow_rot_y_1 / np.linalg.norm(
        r_elbow_rot_y_1, axis=1, keepdims=True
    )
    r_elbow_rot_y_2 = r_upperarm - r_forearm
    r_elbow_rot_y_2 = r_elbow_rot_y_2 / np.linalg.norm(
        r_elbow_rot_y_2, axis=1, keepdims=True
    )
    r_elbow_rot_y_2 = np.cross(r_elbow_rot_y_2, -r_elbow_rot_z)
    r_elbow_rot_y_2 = r_elbow_rot_y_2 / np.linalg.norm(
        r_elbow_rot_y_2, axis=1, keepdims=True
    )
    correction = np.einsum("ij,ij->i", r_elbow_rot_y_1, r_elbow_rot_y_2) < 0
    r_elbow_rot_y_2[correction] = -r_elbow_rot_y_2[correction]

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


def whole_body_ik(humanoid, amass_data):

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
    waist_angles = amass_data["waist_angles"]
    l_foot_angle = amass_data["l_foot_angle"]
    r_foot_angle = amass_data["r_foot_angle"]
    l_foot_dir = amass_data["l_foot_dir"]
    r_foot_dir = amass_data["r_foot_dir"]

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

        # # adam lite
        # ik_solution = np.zeros(num_joints)
        # ik_solution[:4] = ik_solution_lf[:4]
        # ik_solution[4] = l_foot_angle[i]
        # ik_solution[6:10] = ik_solution_rf[6:10]
        # ik_solution[10] = r_foot_angle[i]
        # ik_solution[12] = waist_angles[i,0]
        # ik_solution[13] = waist_angles[i,1]
        # ik_solution[14] = waist_angles[i,2]
        # ik_solution[16:20] = ik_solution_lh[15:19]
        # ik_solution[22:26] = ik_solution_rh[19:23]

        # adam standard
        ik_solution = np.zeros(num_joints)
        ik_solution[:4] = ik_solution_lf[:4]
        ik_solution[4] = l_foot_angle[i]
        ik_solution[6:10] = ik_solution_rf[6:10]
        ik_solution[10] = r_foot_angle[i]
        ik_solution[12] = waist_angles[i, 0]
        ik_solution[13] = waist_angles[i, 1]
        ik_solution[14] = waist_angles[i, 2]
        ik_solution[15:19] = ik_solution_lh[15:19]
        ik_solution[23:27] = ik_solution_rh[19:23]

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

        # if i < 30:
        #     current_min = np.min([l_toe_pose[2], r_toe_pose[2], l_heel_pose[2], r_heel_pose[2]])
        #     if min_foot_z > current_min:
        #         min_foot_z = current_min

        min_foot_z = np.min(
            [l_toe_pose[2], r_toe_pose[2], l_heel_pose[2], r_heel_pose[2]]
        )

        pelvis_copy[i, 2] = pelvis[i, 2] - min_foot_z

        # Step the simulation
        joint_poses.append(ik_solution)

    p.disconnect()

    result = {
        # 'root_pos': pelvis_copy - min_foot_z,
        "root_pos": pelvis_copy,
        "root_rot": lower_rot.as_quat(),
        "joint_poses": np.array(joint_poses),
        "joint_names": joint_names,
    }

    return result


def conver_video_to_adam(output_pth):

    physicsClient, humanoid = setup_pybullet(
        "/home/jianrenw/mocap/robots/adam_standard/urdf/adam_standard_low.urdf",
        gui=False,
    )

    results = joblib.load(osp.join(output_pth, "wham_output.pkl"))

    n_frames = {k: len(results[k]["frame_ids"]) for k in results.keys()}
    sid = max(n_frames, key=n_frames.get)

    smpl = network.smpl

    sid = 0

    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    global_output = smpl.get_output(
        body_pose=tt(results[sid]["pose_world"][:, 3:]),
        global_orient=tt(results[sid]["pose_world"][:, :3]),
        betas=tt(results[sid]["betas"]),
        transl=tt(results[sid]["trans_world"]),
        return_full_pose=True,
    )

    joints = global_output.joints.cpu().numpy()
    vertices = global_output.vertices.cpu().numpy()

    joint_poses = joints[:, :24, :]
    elbow_vertices = vertices[:, [1696, 1647, 5035, 5169]]

    body_pos = results[sid]["trans_world"]
    body_rot = results[sid]["pose_world"][:, :3]

    skeleton = np.concatenate((joint_poses, elbow_vertices), axis=1)

    # Convert skeleton from z forward, x left, y up to x forward, y left, z up
    skeleton = np.concatenate(
        (skeleton[:, :, 0:1], -skeleton[:, :, 2:3], skeleton[:, :, 1:2]), axis=2
    )

    amass_data = amass2adam(skeleton)
    result = whole_body_ik(humanoid, amass_data)
    return result


try:
    from lib.models.preproc.slam import SLAMModel

    _run_global = True
except:
    logger.info("DPVO is not properly installed. Only estimate in local coordinates !")
    _run_global = False


def run(cfg, video, output_pth, network, calib=None, run_global=True, visualize=False):

    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f"Faild to load video file {video}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )

    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global

    os.makedirs(output_pth, exist_ok=True)

    # Preprocess
    with torch.no_grad():
        if not (
            osp.exists(osp.join(output_pth, "tracking_results.pth"))
            and osp.exists(osp.join(output_pth, "slam_results.pth"))
        ):

            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)

            if run_global:
                slam = SLAMModel(video, output_pth, width, height, calib)
            else:
                slam = None

            bar = Bar("Preprocess: 2D detection and SLAM", fill="#", max=length)
            while cap.isOpened():
                flag, img = cap.read()
                if not flag:
                    break

                # 2D detection and tracking
                detector.track(img, fps, length)

                # SLAM
                if slam is not None:
                    slam.track()

                bar.next()

            tracking_results = detector.process(fps)

            if slam is not None:
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0  # Unit quaternion

            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            tracking_results = extractor.run(video, tracking_results)
            logger.info("Complete Data preprocessing!")

            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_pth, "tracking_results.pth"))
            joblib.dump(slam_results, osp.join(output_pth, "slam_results.pth"))
            logger.info(f"Save processed data at {output_pth}")

        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, "tracking_results.pth"))
            slam_results = joblib.load(osp.join(output_pth, "slam_results.pth"))
            logger.info(
                f"Already processed data exists at {output_pth} ! Load the data ."
            )

    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)

    # run WHAM
    results = defaultdict(dict)

    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = flipped_batch
                flipped_pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

                # Forward pass with normal input
                batch = dataset.load_data(subj)
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = batch
                pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred["pose"].squeeze(
                    0
                ), flipped_pred["betas"].squeeze(0)
                pose, shape = pred["pose"].squeeze(0), pred["betas"].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(
                    -1, 24, 6
                )
                avg_pose, avg_shape = avg_preds(
                    pose, shape, flipped_pose, flipped_shape
                )
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (
                    flipped_pred["contact"][..., [2, 3, 0, 1]] + pred["contact"]
                ) / 2

                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

            else:
                # data
                batch = dataset.load_data(subj)
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = batch

                # inference
                pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

        # if False:
        if args.run_smplify:
            smplify = TemporalSMPLify(
                smpl, img_w=width, img_h=height, device=cfg.DEVICE
            )
            input_keypoints = dataset.tracking_results[_id]["keypoints"]
            pred = smplify.fit(pred, input_keypoints, **kwargs)

            with torch.no_grad():
                network.pred_pose = pred["pose"]
                network.pred_shape = pred["betas"]
                network.pred_cam = pred["cam"]
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

        # ========= Store results ========= #
        pred_body_pose = (
            matrix_to_axis_angle(pred["poses_body"]).cpu().numpy().reshape(-1, 69)
        )
        pred_root = (
            matrix_to_axis_angle(pred["poses_root_cam"]).cpu().numpy().reshape(-1, 3)
        )
        pred_root_world = (
            matrix_to_axis_angle(pred["poses_root_world"]).cpu().numpy().reshape(-1, 3)
        )
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred["trans_cam"] - network.output.offset).cpu().numpy()

        results[_id]["pose"] = pred_pose
        results[_id]["trans"] = pred_trans
        results[_id]["pose_world"] = pred_pose_world
        results[_id]["trans_world"] = pred["trans_world"].cpu().squeeze(0).numpy()
        results[_id]["betas"] = pred["betas"].cpu().squeeze(0).numpy()
        results[_id]["verts"] = (
            (pred["verts_cam"] + pred["trans_cam"].unsqueeze(1)).cpu().numpy()
        )
        results[_id]["frame_ids"] = frame_id

    joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))

    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo

        with torch.no_grad():
            run_vis_on_demo(
                cfg, video, results, output_pth, network.smpl, vis_global=run_global
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_file",
        type=str,
        help="dataset directory",
        default="/home/jianrenw/mocap/data/videos/raw",
    )
    parser.add_argument(
        "--output_pth",
        type=str,
        default="/home/jianrenw/mocap/data/videos/processed",
        help="output folder to write results",
    )

    parser.add_argument(
        "--calib", type=str, default=None, help="Camera calibration file path"
    )

    parser.add_argument(
        "--estimate_local_only",
        action="store_true",
        help="Only estimate motion in camera coordinate if True",
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the output mesh if True"
    )

    parser.add_argument(
        "--save_pkl", action="store_true", help="Save output as pkl file"
    )

    parser.add_argument(
        "--run_smplify",
        action="store_true",
        help="Run Temporal SMPLify for post processing",
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/yamls/demo.yaml")

    logger.info(f"GPU name -> {torch.cuda.get_device_name()}")
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()

    files = glob.glob(args.motion_file + "/*.MOV")
    print(files)

    for file in files:

        output_pth = args.output_pth + "/" + file.split("/")[-1].split(".")[0]

        run(
            cfg,
            file,
            output_pth,
            network,
            args.calib,
            run_global=not args.estimate_local_only,
            visualize=args.visualize,
        )

        result = conver_video_to_adam(output_pth)
        joblib.dump(result, output_pth + "/adam_output.pkl")
