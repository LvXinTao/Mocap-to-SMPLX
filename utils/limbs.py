"""
This file contains limbs settings of Optitrack Skeleton
"""

def _add_offset(limbs,offset):
    new_limbs=[]
    for limb in limbs:
        new_limbs.append([limb[0]+offset,limb[1]+offset])
    return new_limbs

OPTITRACK_LIMBS=[
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],
    [10,11],[11,12],[12,13],[13,14],
        [14,15],[15,16],[16,17],[17,18],
        [14,19],[19,20],[20,21],[21,22],
        [14,23],[23,24],[24,25],[25,26],
        [14,27],[27,28],[28,29],[29,30],
        [14,31],[31,32],[32,33],[33,34],
    [10,35],[35,36],[36,37],[37,38],
        [38,39],[39,40],[40,41],[41,42],
        [38,43],[43,44],[44,45],[45,46],
        [38,47],[47,48],[48,49],[49,50],
        [38,51],[51,52],[52,53],[53,54],
        [38,55],[55,56],[56,57],[57,58],
    [10,59],[59,60]
]

OPTITRACK_RIGHT_HAND=[
    [14,15],[15,16],[16,17],[17,18],
    [14,19],[19,20],[20,21],[21,22],
    [14,23],[23,24],[24,25],[25,26],
    [14,27],[27,28],[28,29],[29,30],
    [14,31],[31,32],[32,33],[33,34],
]
OPTITRACK_RIGHT_HAND=_add_offset(OPTITRACK_RIGHT_HAND,-14)

OPTITRACK_LEFT_HAND=[
    [38,39],[39,40],[40,41],[41,42],
    [38,43],[43,44],[44,45],[45,46],
    [38,47],[47,48],[48,49],[49,50],
    [38,51],[51,52],[52,53],[53,54],
    [38,55],[55,56],[56,57],[57,58],
]
OPTITRACK_LEFT_HAND=_add_offset(OPTITRACK_LEFT_HAND,-38)

OPTITRACK_BODY_LIMBS=[
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],
    [10,11],[11,12],[12,13],[13,14],
    [10,15],[15,16],[16,17],[17,18],
    [10,19],[19,20]
]