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
    if os.path.isfile(db_file):
        print(f"Skipping {seq_name} sequence...")
        return

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

            useful_poses, framerate = load_data(fname, bms)

            vid_name = f"{seq_name}_{subject}_{action[:-4]}"

            datas[vid_name] = {'pose': useful_poses, 'mocap_framerate': framerate}
            

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
    joint_poses = body_trans_root.Jtr[:,:24,:].numpy()
    elbow_vertices = body_trans_root.v.numpy()[:, [1696, 1647, 5035, 5169]]
    useful_poses = np.concatenate((joint_poses, elbow_vertices), axis=1)

    framerate = bdata['mocap_framerate']

    return useful_poses, framerate



comp_device = "cpu"
# with Pool(5) as p:
#     p.map(read_data, all_sequences)

# for seq in all_sequences:
# read_data("DanceDB")

db = {}
for seq in all_sequences:
    db_file = osp.join("/home/jianrenw/mocap/data/out", "{}.pt".format(seq))
    datas = joblib.load(db_file)
    db.update(datas)

joblib.dump(db, "/home/jianrenw/mocap/data/out/amass.pt")