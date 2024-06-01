"""
This script consists of two parts:
1. Load the precomputed 3D positions from .npy file
    Notes: .npy has 64 joints of Optitrack.
2. do the SMPLify(-X) optimization and get SMPL(-X) parameters
"""

import argparse
import sys
import time

import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import torch
from smplx import SMPLXLayer

from smplifyx.optimize import *
from utils.io import write_smplx
from utils.mapping import (OPTITRACK_TO_SMPLX, SELECTED_JOINTS,
                                     JointMapper)
from utils.torch_utils import *
from utils.visualize_smplx import visualize_smplx_model

def load_joint_positions(npy):
    joint_positions=np.load(npy) # (tf,64,3)
    # ignore the numb joints
    joint_positions=joint_positions[:,SELECTED_JOINTS] # (tf,tj,3)
    
    return joint_positions

def parse_shape_pkl(shape_pkl):
    # load pre-computed shape parameters
    pkl=joblib.load(shape_pkl)
    gender=pkl['gender']
    betas=pkl['betas']
    return gender,betas

def smplifyx(joint_positions):
    nf=joint_positions.shape[0]
    nj=joint_positions.shape[1]

    gender,betas=parse_shape_pkl(args.shape_pkl)
    betas=betas.repeat(nf,axis=0)
    # create body models
    body_models=SMPLXLayer(model_path='./body_models/smplx',num_betas=10,gender=gender,
                           joint_mapper=JointMapper(OPTITRACK_TO_SMPLX),flat_hand_mean=True).to('cuda')

    # create params to be optimized,

    params={
        'body_pose':np.zeros((nf,21,3)),
        'lhand_pose':np.zeros((nf,15,3)),
        'rhand_pose':np.zeros((nf,15,3)),
        'jaw_pose':np.zeros((nf,3)),
        'leye_pose':np.zeros((nf,3)),
        'reye_pose':np.zeros((nf,3)),
        'betas':betas,
        'expression':np.zeros((nf,10)),
        'global_orient':np.zeros((nf,3)),
        'transl':np.zeros((nf,3)),
    }
    params=init_params(params,body_models,nf)
    # add confidence 1.0
    joint_positions=np.concatenate([joint_positions,np.ones((nf,nj,1))],axis=-1)
    # however, make the confidence of NaN to 0.0
    nan_joints=np.isnan(joint_positions).any(axis=-1)
    joint_positions[nan_joints]=0.0
    # convert joint_positions to tensor
    joint_positions=torch.tensor(joint_positions,dtype=torch.float32,device='cuda')


    # SMPLify-X optimization
    start=time.time()
    params=multi_stage_optimize(params,body_models,joint_positions)
    end=time.time()
    print("------------------Fitting cost %d s!--------------------"%(end-start))
    # visualize the results
    if args.vis_smplx:
        visualize_smplx_model(params,gender,joint_positions,args.vis_kp3d)

    # save SMPL parameters
    params=tensor_to_numpy(params)
    write_smplx(params,args.save_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--npy',type=str,default='./test_data/P2.npy',
                        help='.npy file path that contains 64 joints of Optitrack')
    parser.add_argument('--shape_pkl',type=str,default='./test_data/P2.pkl',
                        help='Pre-computed shape parameters')
    parser.add_argument('--save_path',type=str,default='./test_data/P2_smplx.npz',
                        help='save the SMPLX parameters')
    parser.add_argument('--vis_smplx',action='store_true',
                        help='visualize the results')
    parser.add_argument('--vis_kp3d',action='store_true',
                        help='visualize the 3D joints positions')
    
    args=parser.parse_args()

    print('-----------------Parsing %s!-----------------'%(args.npy))
    # load 3D joints positions
    joint_positions=load_joint_positions(args.npy)

    print('-----------------Total %s frames!-----------------'%(joint_positions.shape[0]))
    print('-----------------SMPLify-X!-----------------')
    # do smplify
    smplifyx(joint_positions)