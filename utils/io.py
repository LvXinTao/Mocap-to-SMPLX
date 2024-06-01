import os.path as osp
import numpy as np

def write_smplx(smplx_params,out_path):
    """
    smplx_params: smplx params of all frames
    """
    nf=smplx_params['body_pose'].shape[0]
    smplx={
            'body_pose':smplx_params['body_pose'],
            'lhand_pose':smplx_params['lhand_pose'],
            'rhand_pose':smplx_params['rhand_pose'],
            'jaw_pose':smplx_params['jaw_pose'],
            'leye_pose':smplx_params['leye_pose'],
            'reye_pose':smplx_params['reye_pose'],
            'betas':smplx_params['betas'][0], # [10]
            'expression':smplx_params['expression'],
            'global_orient':smplx_params['global_orient'],
            'transl':smplx_params['transl']
    }
    np.savez(osp.join(out_path),**smplx)

def read_smplx(smplx_path):
    """
        smplx_path: path to smplx.npz file
        return:
                params: dict of np.array

    """
    params=np.load(smplx_path,allow_pickle=True)
    # restore to dict of np.array
    params={
        key:np.array(params[key].tolist()) for key in params.files
    }
    return params