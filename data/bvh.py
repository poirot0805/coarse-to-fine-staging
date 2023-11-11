import re
import os
import json
import numpy as np

from motion_inbetween_normal.data import utils_np


class Animation(object):
    def __init__(self, rotations, positions, offsets, step_num, missing_list):
        self.rotations = rotations
        self.positions = positions
        self.offsets = offsets
        self.step_num = step_num  # step_num
        self.missing_list = missing_list


CHANNEL_MAP = {
    "Xrotation": "x",
    "Yrotation": "y",
    "Zrotation": "z",
}

def load_feature_npy(client_id,add_geo,remove_list=[]):
    if add_geo==False:
        return np.zeros((1,108),dtype=np.float32)
    ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] \
    + [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
    oid = {id: i for i, id in enumerate(ids)}   # dict{41:第0颗牙} 28颗牙
    data=np.empty((0,108),dtype=np.float32)
    zero_placeholder=np.zeros((1,108),dtype=np.float32)
    root_path=r"/media/backup/home-2023-08-25/mjy/motion-inbetween/feature"
    #r"/home/mjy/motion-inbetween/feature" 没有备份之前的pointr路径
    #"/home/mjy/PoinTr/experiments/Motion_Feature_snow"没有备份之前的snow路径#"/home/mjy/motion-inbetween/feature"# r"E:\PROJECTS\PoinTr\data\Motion\motionfeature_all"
    for i, id in enumerate(ids):
        npy_name=f"{client_id}-{id}.npy"
        npy_path=os.path.join(root_path,npy_name)
        if not os.path.exists(npy_path):
            data=np.concatenate([data,zero_placeholder],axis=0)
        else:
            temp=np.load(npy_path,mmap_mode='r')
            data=np.concatenate([data,temp],axis=0)
    
    assert data.shape[0]==28 and data.shape[1]==108
    # data[remove_list]=0 # FIXME:rot位置？
    # GEO: ctrl code
    seq_num=data.shape[1]//9
    b=data.reshape(28,seq_num,9).transpose((1,0,2))
    # b=data.reshape(28*108)
    return b

def load_tooth_json(json_path,order='M9D',start=None,remove_list=[]):
    """
    Load bvh file.

    Args:
        bvh_path (str): File path.
        order (str, optional): Quat or M9D

    Raises:
        Exception: [description]

    Returns:
        BvhAnimation
    """
    # pos:np.array(seq,joint,3)
    # rot:np.arrar(seq,joint,3,3)
    with open(json_path) as fh:
        data=json.load(fh)
        step_num=data["step_num"]
        missing_list = set()
        rotations = np.zeros((step_num, 28, 4),dtype=np.float64)
        positions = np.zeros((step_num, 28, 3),dtype=np.float64)
        for step in data["steps"]:
            step_id=step["step_id"]
            temp_pos=np.zeros((28, 3),dtype=np.float64)
            temp_rot=np.zeros((28, 4),dtype=np.float64)
            teeth=step["tooth_data"]
            for i in range(28):
                temp_pos[i,:]=teeth[str(i)][0:3]
                temp_rot[i,:]=teeth[str(i)][3:]
                
            positions[step_id-1]=temp_pos
            rotations[step_id-1]=temp_rot
        for i in range(28):
            if (positions[step_num-1,i,:]==[0.,0.,0.]).all():
                missing_list.add(i)
        
        #positions[:,remove_list,:]=0   要还原为原始位置，所以removed teeth不要覆盖掉
        #rotations[:,remove_list,:]=0
        #rotations[:,remove_list,-1]=1   
        
        rotations[:,:,[0,1,2,3]] = rotations[:,:,[3,0,1,2]] # NOTICE:[xyz,W]->[W,xyz] “qw, qx, qy, qz = quat[...,0:1],quat[...,1:2],quat[...,2:3],quat[...,3:4]”
        
        temp_remove_set=set(remove_list)
        missing_list=set.union(missing_list,temp_remove_set)
        missing_list=list(missing_list)        
        # rotations = utils_np.euler_to_matrix9D(rotations, order=order)
        if order=='M9D':
            rotations = utils_np.quat_to_matrix9D(rotations)
            assert len(rotations.shape)==4 and rotations.shape[-1]==3
            return Animation(rotations, positions, None, step_num, missing_list)
        else:
            assert len(rotations.shape)==3 and rotations.shape[-1]==4
            return Animation(rotations, positions, None, step_num, missing_list)
def load_tooth_json2(json_path,order='M9D',start=None,remove_list=[]):
    """
    Load bvh file.

    Args:
        bvh_path (str): File path.
        order (str, optional): Quat or M9D

    Raises:
        Exception: [description]

    Returns:
        BvhAnimation
    """
    # pos:np.array(seq,joint,3)
    # rot:np.arrar(seq,joint,3,3)
    with open(json_path) as fh:
        data=json.load(fh)
        step_num=data["step_num"]
        missing_list = set()
        rotations = np.zeros((step_num, 28, 9),dtype=np.float64)
        positions = np.zeros((step_num, 28, 3),dtype=np.float64)
        for step in data["steps"]:
            step_id=step["step_id"]
            temp_pos=np.zeros((28, 3),dtype=np.float64)
            temp_rot=np.zeros((28, 9),dtype=np.float64)
            teeth=step["tooth_data"]
            for i in range(28):
                temp_pos[i,:]=teeth[str(i)][0:3]
                temp_rot[i,:]=teeth[str(i)][3:]
                
            positions[step_id-1]=temp_pos
            rotations[step_id-1]=temp_rot
        for i in range(28):
            if (positions[step_num-1,i,:]==[0.,0.,0.]).all():
                missing_list.add(i)
        
        #positions[:,remove_list,:]=0
        #rotations[:,remove_list,:]=0
        #rotations[:,remove_list,-1]=1   
        

        
        temp_remove_set=set(remove_list)
        missing_list=set.union(missing_list,temp_remove_set)
        missing_list=list(missing_list)        
        # rotations = utils_np.euler_to_matrix9D(rotations, order=order)
        if order=='M9D':
            rotations = rotations.reshape(step_num, 28, 3, 3)
            assert len(rotations.shape)==4 and rotations.shape[-1]==3
            return Animation(rotations, positions, None, step_num, missing_list)
        else:
            assert len(rotations.shape)==3 and rotations.shape[-1]==4
            return Animation(rotations, positions, None, step_num, missing_list)


