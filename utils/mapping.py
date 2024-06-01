import numpy as np
import torch.nn as nn
import torch

OPTITRACK_SKEL=[
    'Hips',
    'RightUpLeg','RightLeg','RightFoot','RightToeBase',# 'RightToeBase_Nub',
    'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase',# 'LeftToeBase_Nub',
    'Spine','Spine_1',
    'RightShoulder','RightArm','RightForeArm','RightHand',
        'RightHandPinky1','RightHandPinky2','RightHandPinky3','RightHandPinky3_Nub',
        'RightHandRing1','RightHandRing2','RightHandRing3','RightHandRing3_Nub',
        'RightHandMiddle1','RightHandMiddle2','RightHandMiddle3','RightHandMiddle3_Nub',
        'RightHandIndex1','RightHandIndex2','RightHandIndex3','RightHandIndex3_Nub',
        'RightHandThumb1','RightHandThumb2','RightHandThumb3','RightHandThumb3_Nub',
    'LeftShoulder','LeftArm','LeftForeArm','LeftHand',
        'LeftHandPinky1','LeftHandPinky2','LeftHandPinky3','LeftHandPinky3_Nub',
        'LeftHandRing1','LeftHandRing2','LeftHandRing3','LeftHandRing3_Nub',
        'LeftHandMiddle1','LeftHandMiddle2','LeftHandMiddle3','LeftHandMiddle3_Nub',
        'LeftHandIndex1','LeftHandIndex2','LeftHandIndex3','LeftHandIndex3_Nub',
        'LeftHandThumb1','LeftHandThumb2','LeftHandThumb3','LeftHandThumb3_Nub',
    'Neck','Head',# 'Head_Nub'
] # 64 , 61 used,xxx_Nub is ignored hand_Nub can not be ignored

SELECTED_JOINTS=np.concatenate(
    [range(0,5),range(6,10),range(11,63)]
)

OPTITRACK_BODY=np.concatenate(
    [range(0,15),range(35,39),range(59,61)]
)
OPTITRACK_HAND=np.concatenate(
    [range(15,35),range(39,59)]
)

OPTITRACK_TO_SMPLX=np.array([
    0,
    2,5,8,11,
    1,4,7,10,
    3,9,
    14,17,19,21,
        46,47,48,75,
        49,50,51,74,
        43,44,45,73,
        40,41,42,72,
        52,53,54,71,
    13,16,18,20,
        31,32,33,70,
        34,35,36,69,
        28,29,30,68,
        25,26,27,67,
        37,38,39,66,
    12,15
])


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)
        
if __name__=='__main__':
    print(SELECTED_JOINTS)
    print(len(SELECTED_JOINTS))