import numpy as np
import torch
from tqdm import tqdm

from smplifyx.config import *
from smplifyx.lbfgs import LBFGS
from smplifyx.loss import *
from utils.rotation_conversion import aa2rot_torch
from utils.limbs import OPTITRACK_LIMBS

def multi_stage_optimize(params,body_models,kp3ds):
    """
    kp3ds: nf,nj,4

    Multi-stage optimizing
    1. Shape
    2. Global orient and transl
    3. Poses
    """
    # Use pre-computed shape, otherwise `optimize_shape` can be used

    # optimize RT
    params=optimize_pose(params,body_models,kp3ds,
                         OPT_RT=True)

    # optimize body poses
    params=optimize_pose(params,body_models,kp3ds,
                         OPT_RT=True,OPT_POSE=True)
    
    # optimize hand poses
    params=optimize_pose(params,body_models,kp3ds,
                        OPT_RT=False,OPT_POSE=True,OPT_HAND=True)
    
    return params

def optimize_pose(params,body_models,kp3ds,
                  OPT_RT=False,OPT_POSE=False,
                  OPT_HAND=False,OPT_EXPR=False):
    nf=kp3ds.shape[0]
    loss_dict=[]
    opt_params=[]
    if OPT_RT:
        
        loss_dict+=[
            'k3d','reg_pose' ,'smooth_pose','smooth_body'
        ]
        opt_params+=[params['global_orient'],params['transl']]
        loss_weight=OPTIMIZE_RT
        desc='Optimizing RT...'
    if OPT_POSE:
        opt_params+=[params['body_pose']]
        loss_weight=OPTIMIZE_POSES
        desc='Optimizing Body pose...'
    if OPT_HAND:
        loss_dict+=[
            'k3d_hand','reg_hand','smooth_hand','k3d','reg_pose' ,'smooth_pose','smooth_body'
        ]
        opt_params+=[params['lhand_pose'],params['rhand_pose']] # add wrist to optimize
        loss_weight=OPTIMIZE_HAND
        desc='Optimizing Hand...'
    if OPT_EXPR:
        loss_dict+=[
            'k3d_face','reg_head','reg_expr','smooth_head'
        ]
        opt_params+=[params['jaw_pose'],params['leye_pose'],
                     params['reye_pose'],params['expression']]
        loss_weight=OPTIMIZE_EXPR
        desc='Optimizing Expression...'

    optimizer=LBFGS(opt_params,line_search_fn='strong_wolfe',max_iter=30)
    def closure(debug=False):
        optimizer.zero_grad()
        axis_angle=torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['leye_pose'][:,None,:],
                            params['reye_pose'][:,None,:],
                            params['jaw_pose'][:,None,:],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # nf*55,3
        rot_mat=aa2rot_torch(axis_angle).reshape((nf,-1,3,3)) # nf,55,3,3
        out_kp3d=body_models(
            betas=params['betas'],
            expression=params['expression'],
            global_orient=rot_mat[:,0,...],
            body_pose=rot_mat[:,1:22,...],
            leye_pose=rot_mat[:,22,...],
            reye_pose=rot_mat[:,23,...],
            jaw_pose=rot_mat[:,24,...],
            left_hand_pose=rot_mat[:,25:40,...],
            right_hand_pose=rot_mat[:,40:55,...],
            transl=params['transl']
        ).joints # nf,nj(JOINT_MAPPER),3
        final_loss_dict={loss_name:get_loss(loss_name,kp3ds,out_kp3d,params) 
                         for loss_name in loss_dict}
        loss=sum([final_loss_dict[key]*loss_weight[key]
                  for key in loss_dict])
        if not debug:
            loss.backward()
            return loss
        else:
            return final_loss_dict
    
    final_loss=run_fitting(optimizer,closure,opt_params,desc)
    final_loss_dict=closure(debug=True)
    for key in final_loss_dict.keys():
        print("%s : %f"%(key,final_loss_dict[key].item()))

    return params
        
def optimize_shape(params,body_models,kp3ds):
    nf=kp3ds.shape[0]

    start=torch.tensor(np.array(OPTITRACK_LIMBS)[:,0],device='cuda')
    end=torch.tensor(np.array(OPTITRACK_LIMBS)[:,1],device='cuda')
    start_kp3d=torch.index_select(kp3ds,1,start) # nf,nlimbs,4
    end_kp3d=torch.index_select(kp3ds,1,end)
    # nf,nlimbs,1
    limb_length=torch.norm(start_kp3d[...,:3]-end_kp3d[...,:3],dim=2,keepdim=True) 
    # nf,nlimbs,1
    limb_conf=torch.minimum(start_kp3d[...,3],end_kp3d[...,3])[...,None]

    opt_params=[params['betas']]
    optimizer=LBFGS(opt_params,line_search_fn='strong_wolfe',max_iter=30)

    def closure(debug=False):
        optimizer.zero_grad()
        axis_angle=torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['leye_pose'][:,None,:],
                            params['reye_pose'][:,None,:],
                            params['jaw_pose'][:,None,:],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # nf*55,3
        rot_mat=aa2rot_torch(axis_angle).reshape((nf,-1,3,3)) # nf,55,3,3
        out_kp3d=body_models(
            betas=params['betas'],
            expression=params['expression'],
            transl=params['transl'],
            global_orient=rot_mat[:,0,...],
            body_pose=rot_mat[:,1:22,...],
            leye_pose=rot_mat[:,22,...],
            reye_pose=rot_mat[:,23,...],
            jaw_pose=rot_mat[:,24,...],
            left_hand_pose=rot_mat[:,25:40,...],
            right_hand_pose=rot_mat[:,40:55,...]
        ).joints # nf,nj(JOINT_MAPPER),3
        out_start_kp3d=torch.index_select(out_kp3d,1,start)
        out_end_kp3d=torch.index_select(out_kp3d,1,end)
        out_dist=(out_start_kp3d[...,:3]-out_end_kp3d[...,:3]).detach()
        out_dist_norm=torch.norm(out_dist,dim=2,keepdim=True) 
        out_dist_normalized=out_dist/(out_dist_norm+1e-4)
        err=(out_start_kp3d[...,:3]-out_end_kp3d[...,:3])\
            -out_dist_normalized*limb_length
        loss_dict={
            'shape3d':Loss_shape3d(err,limb_conf,nf),
            'reg_shape':Loss_reg_shape(params['betas'])
        }
        loss_weight=OPTIMIZE_SHAPE
        loss=sum([loss_dict[key]*loss_weight[key] 
                  for key in loss_dict.keys()])
    
        if not debug:
            loss.backward()
            return loss
        else:
            return loss_dict
    final_loss=run_fitting(optimizer,closure,opt_params,"Optimizing shape...")
    loss_dict=closure(True)
    for key in loss_dict.keys():
        print("%s : %f"%(key,loss_dict[key].item()))
    return params

def run_fitting(optimizer,closure,opt_params,desc,maxiters=50,ftol=1e-9):
    prev_loss=None
    require_grad(opt_params,True)
    for iter in tqdm(range(maxiters),desc=desc):
        loss=optimizer.step(closure)
        if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

        if torch.isinf(loss).sum() > 0:
            print('Infinite loss value, stopping!')
            break

        if iter>0 and prev_loss is not None and ftol>0:
            loss_rel_change=rel_change(prev_loss,loss.item())
            if loss_rel_change<=ftol:
                break
        prev_loss=loss.item()
    require_grad(opt_params,False)
    return prev_loss

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])

def require_grad(opt_params,flag=False):
    for param in opt_params:
        param.requires_grad=flag