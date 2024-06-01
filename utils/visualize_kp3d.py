import matplotlib.pyplot as plt
import numpy as np

import os
import os.path as osp

import matplotlib as mpl
from matplotlib import animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from tqdm import tqdm

from utils.limbs import OPTITRACK_LIMBS

def visualize_joint_positions(joint_position,global_orients=None):
    nf=joint_position.shape[0]

    if global_orients is None:
        global_orients=np.zeros((nf,3))
    # since joint positions are Y-up,Z-front,X-right
    # we need to convert them to Z-up,X-front,Y-right

    joint_positions=joint_position[:,:,[2,0,1]]
    global_orient=global_orients[:,[2,0,1]]
    # joint_positions=joint_position
    # global_orient=global_orients

    # animate
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    radius=1.7
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    # ax.view_init(elev=15.,azim=-15,roll=0)
     #elev,azim,roll
    ax.dist=8.0 #default 10. fewer,closer

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Joint Positions')

    lines_x,lines_y,lines_z,lines=_generate_lines(ax,joint_positions,global_orient)
    def update(num,lines,lines_x,lines_y,lines_z):
        for i,line in enumerate(lines):
            line.set_data(lines_x[num][i],lines_y[num][i])
            line.set_3d_properties(lines_z[num][i],'z')    
    
    ani=animation.FuncAnimation(fig,update,nf,fargs=(lines,lines_x,lines_y,lines_z),
                                interval=1000/24)

    plt.show()

def _generate_lines(ax,joint_positions,global_orient):
    lines_x=[]
    lines_y=[]
    lines_z=[]
    first_lines=[]
    for frame,orient in zip(joint_positions,global_orient):
        lx,ly,lz=_generate_lines_by_part(frame,orient,limbs=OPTITRACK_LIMBS)

        lines_x.append(lx)
        lines_y.append(ly)
        lines_z.append(lz)

    for x0,y0,z0 in zip(lines_x[0],lines_y[0],lines_z[0]):
        line=ax.plot(x0,y0,z0,c='k',lw=0.5)[0]
        first_lines.append(line)
    
    return lines_x,lines_y,lines_z,first_lines

def _generate_lines_by_part(keypoints,global_orient,limbs):
    lines_x=[]
    lines_y=[]
    lines_z=[]
    for limb in limbs:
        start,end=limb
        lines_x.append([keypoints[start,0],keypoints[end,0]])
        lines_y.append([keypoints[start,1],keypoints[end,1]])
        lines_z.append([keypoints[start,2],keypoints[end,2]])
    # add global orient
    lines_x.append([keypoints[0,0],keypoints[0,0]+global_orient[0]])
    lines_y.append([keypoints[0,1],keypoints[0,1]+global_orient[1]])
    lines_z.append([keypoints[0,2],keypoints[0,2]+global_orient[2]])
        
    return lines_x,lines_y,lines_z