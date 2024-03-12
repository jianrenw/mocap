import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import argparse
from human_body_prior.body_model.body_model import BodyModel
import torch
import joblib
import ipdb
from multiprocessing import Pool


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4

def compute_orth6d_from_rotation_matrix(rot_mats):
    rot_mats = rot_mats[:,:,:2].transpose(1, 2).reshape(-1, 6)
    return rot_mats

def convert_aa_to_orth6d(poses):
    if torch.is_tensor(poses):
        curr_pose = poses.to(poses.device).float().reshape(-1, 3)
    else:
        curr_pose = torch.from_numpy(poses).to(poses.device).float().reshape(-1, 3)
    rot_mats = angle_axis_to_rotation_matrix(curr_pose)
    rot_mats = rot_mats[:, :3, :]
    orth6d = compute_orth6d_from_rotation_matrix(rot_mats)
    orth6d = orth6d.view(poses.shape[0], -1, 6)
    return orth6d

def convert_aa_to_rotation_matrix(poses):
    if torch.is_tensor(poses):
        curr_pose = poses.to(poses.device).float().reshape(-1, 3)
    else:
        curr_pose = torch.from_numpy(poses).to(poses.device).float().reshape(-1, 3)
    rot_mats = angle_axis_to_rotation_matrix(curr_pose)
    rot_mats = rot_mats[:, :3, :3]
    return rot_mats


all_sequences = [
    "ACCAD",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HumanEva",
    "KIT",
    "MPI_HDM05",
    "MPI_Limits",
    "MPI_mosh",
    "SFU",
    "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
    "BMLhandball",
    "DanceDB"
]

smpl_path = "/home/jwang/data/body_models/smplh"
dmpl_path = "/home/jwang/data/body_models/dmpls"
folder = "/home/jwang/data/amass"


def read_data(seq_name):
    # sequences = [osp.join(folder, x) for x in sorted(os.listdir(folder)) if osp.isdir(osp.join(folder, x))]

    db_file = osp.join("/home/jwang/mocap/data/out", "{}.pt".format(seq_name))
    # if os.path.isfile(db_file):
    #     print(f"Skipping {seq_name} sequence...")
    #     return

    bms = {}
    for subject_gender in ['female', 'male', 'neutral']:
        bm_fname = '{}/{}/model.npz'.format(smpl_path,subject_gender)
        dmpl_fname = '{}/{}/model.npz'.format(dmpl_path, subject_gender)

        num_betas = 16 # number of body parameters
        num_dmpls = 8 # number of DMPL parameters

        bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
        bms[subject_gender] = bm


    print(f"Reading {seq_name} sequence...")
    seq_folder = osp.join(folder, seq_name)

    datas = read_single_sequence(seq_folder, seq_name, bms)
    
    joblib.dump(datas, db_file)


def read_single_sequence(folder, seq_name, bms):
    subjects = os.listdir(folder)

    datas = {}

    for subject in tqdm(subjects):
        if not osp.isdir(osp.join(folder, subject)):
            continue
        actions = [
            x for x in os.listdir(osp.join(folder, subject)) if x.endswith(".npz") 
        ]

        for action in actions:
            fname = osp.join(folder, subject, action)

            if fname.endswith("shape.npz"):
                continue

            skeleton, framerate = load_data(fname, bms)

            vid_name = f"{seq_name}_{subject}_{action[:-4]}"

            datas[vid_name] = {'skeleton': skeleton, 'mocap_framerate': framerate}
            

    return datas


def load_data(file_path, bms):

    bdata = np.load(file_path)

    # you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
    try:
        subject_gender = bdata['gender'].item().decode('UTF-8')
    except AttributeError:
        subject_gender = bdata['gender'].item()

    bm = bms[subject_gender]
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    time_length = len(bdata['trans'])

    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
    }

    body_trans_root = bm(**body_parms)
    # ipdb.set_trace()
    skeleton = body_trans_root.Jtr[:,:24,:].numpy()
    joint_poses = body_trans_root.Jtr[:,:24,:].numpy()
    elbow_vertices = body_trans_root.v.numpy()[:, [1696, 1647, 5035, 5169]]
    skeleton = np.concatenate((joint_poses, elbow_vertices), axis=1)

    framerate = bdata['mocap_framerate']

    return skeleton, framerate

comp_device = "cpu"

# for seq in all_sequences:
#     read_data(seq)

db = {}
for seq in all_sequences:
    db_file = osp.join("/home/jwang/mocap/data/out", "{}.pt".format(seq))
    datas = joblib.load(db_file)
    db.update(datas)

joblib.dump(db, "/home/jwang/mocap/data/out/amass.pt")