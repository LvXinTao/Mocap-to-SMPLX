
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.viewer import Viewer

from utils.limbs import OPTITRACK_LIMBS
                                     
C.update_conf({'smplx_models':'./body_models'})

def visualize_smplx_model(params,gender,joint_positions=None,vis_kp3d=False):
    """
    params: smplx params of all frames,dict of tensors
    """
    # create viewer
    v=Viewer()
    v.playback_fps=120
    v.scene.fps=120

    nf=params['body_pose'].shape[0]

    betas=params['betas'][0]
    poses_root=params['global_orient']
    poses_body=params['body_pose'].reshape(nf,-1)
    poses_lhand=params['lhand_pose'].reshape(nf,-1)
    poses_rhand=params['rhand_pose'].reshape(nf,-1)
    transl=params['transl']

    # create body models
    smplx_layer=SMPLLayer(model_type='smplx',gender=gender,num_betas=10,device=C.device)

    # create smplx sequence
    smplx_seq=SMPLSequence(poses_body=poses_body,
                           smpl_layer=smplx_layer,
                           poses_root=poses_root,
                           betas=betas,
                           trans=transl,
                           poses_left_hand=poses_lhand,
                           poses_right_hand=poses_rhand,
                           device=C.device,
                           )
    
    v.scene.add(smplx_seq)

    if vis_kp3d and joint_positions is not None:

        # create keypoint sequence
        kp3d_seq=Skeletons(joint_positions=joint_positions[:,:,:3].detach().cpu().numpy(),
                           joint_connections=OPTITRACK_LIMBS,
                           radius=0.005,
                           color=(0.5,0.5,1,1))
        v.scene.add(kp3d_seq)
        
    v.run()
