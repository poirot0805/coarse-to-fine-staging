import os
import sys
import math
import json
import time
import argparse
import pandas as pd
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)


import torch

from motion_inbetween_space import benchmark, visualization
from motion_inbetween_space.model import STTransformer
from motion_inbetween_space.config import load_config_by_name
from motion_inbetween_space.train import rmi
from motion_inbetween_space.train import context_model,detail_model
from motion_inbetween_space.train import utils as train_utils
from motion_inbetween_space.data import utils_torch as data_utils
def process_rotation(r):
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        angles = np.arccos(np.clip([np.dot(x, [1, 0, 0]), 
                                    np.dot(y, [0, 1, 0]), 
                                    np.dot(z, [0, 0, 1])], -1, 1))
        return angles
def pos_error(self,p):
        assert len(p.shape)==1
        return math.sqrt(np.dot(p,p))
def rot_error(r_gt,r_est):
        # r (3,3)
        x1,y1,z1=r_gt[:,0],r_gt[:,1],r_gt[:,2]
        a1 = np.dot(x1,np.array([1,0,0]))
        b1 = np.dot(y1,np.array([0,1,0]))
        r1 = np.dot(z1,np.array([0,0,1]))
        a1 = min(max(a1,-1),1)
        b1 = min(max(b1,-1),1)
        r1 = min(max(r1,-1),1)
        a1,b1,r1 = math.acos(a1),math.acos(b1),math.acos(r1)
        x2,y2,z2=r_est[:,0],r_est[:,1],r_est[:,2]
        a2 = np.dot(x2,np.array([1,0,0]))
        b2 = np.dot(y2,np.array([0,1,0]))
        r2 = np.dot(z2,np.array([0,0,1]))
        a2 = min(max(a2,-1),1)
        b2 = min(max(b2,-1),1)
        r2 = min(max(r2,-1),1)
        a2,b2,r2 = math.acos(a2),math.acos(b2),math.acos(r2)
        theta=abs(a1-a2)+abs(b1-b2)+abs(r1-r2)
        return np.array([abs(a1-a2),abs(b1-b2),abs(r1-r2),theta])
def cal_n(self,pos1,pos2):
        # pos[0],pos[-1]
        def pos_len(x):
            return torch.sqrt(torch.dot(x,x))
        n_1 =[]
        n_2=[]
        n_3=[]
        n_4=[]
        for i in range(4,10):
            y = pos_len(pos1[i]-pos2[i])
            n_1.append(y)
        for i in range(0,4):
            y = pos_len(pos1[i]-pos2[i])
            n_2.append(y)
        for i in range(10,14):
            y = pos_len(pos1[i]-pos2[i])
            n_2.append(y)
        for i in range(18,24):
            y = pos_len(pos1[i]-pos2[i])
            n_3.append(y)
        for i in range(14,18):
            y = pos_len(pos1[i]-pos2[i])
            n_4.append(y)
        for i in range(24,28):
            y = pos_len(pos1[i]-pos2[i])
            n_4.append(y)
        return max(max(n_1)+max(n_2), max(n_3)+max(n_4))*5
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate context model. "
                                     "Post-processing is applied by default.")
    parser.add_argument("config", help="config name")

    parser.add_argument("-s", "--dataset",
                        help="dataset name (default=test)",
                        default="train")

    start_time = time.time()
    args = parser.parse_args()
    path_csv=r"/home/mjy/1126Teeth/scripts/infer_{}.csv".format(args.config)
    csv_filename=[]
    csv_realfn=[]
    csv_addlen=[]
    csv_totaln=[]
    csv_sp=[]
    csv_d=[]
    csv_gpos=[]
    csv_gquat=[]
    config = load_config_by_name(args.config)

    context_model.ATTENTION_MODE = config["train"]["attention_mode"]
    context_model.zscore_MODE = config["train"]["zscore_MODE"]
    context_model.INIT_INTERP = config["train"]["init_interp"]
    context_model.Data_Mask_MODE=config["train"]["data_mask_set"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, args.dataset, device=device, shuffle=False,inference_mode=True,add_geo=False)
    pos_res =[]
    rot_res =[]

    all_frames=0
    for i, data in enumerate(data_loader, 0):
        (positions, rotations, file_name, real_frame_num,trends,geo,remove_idx,data_idx) = data # FIXED:返回类型(1,seq,joint,3)
        pos_move = np.zeros((4,3))
        rot_move = np.zeros((4,3))
        file_name=file_name[0]
        print(f"file name:{file_name}")
        
        total_len=positions.shape[1]
        all_frames+=(total_len-2)
        print("total-len:{} real:{}".format(total_len,real_frame_num))
        # 相邻pos的差值
        adj_pos = torch.abs(positions[0,1:]-positions[0,:-1]).cpu().numpy()
        print(adj_pos.shape)    # (f-1,28,3)
        pos_move[:3,0]=np.min(adj_pos, axis=(0, 1))
        pos_move[:3,1]=np.max(adj_pos, axis=(0, 1))
        pos_move[:3,2]=np.mean(adj_pos, axis=(0, 1))
        inner_dot = np.sqrt(np.sum(adj_pos * adj_pos, axis=2, keepdims=True))
        pos_move[3,0]=np.min(inner_dot)
        pos_move[3,1]=np.max(inner_dot)
        pos_move[3,2]=np.mean(inner_dot)
        pos_res.append(pos_move[np.newaxis,:,:])
        # rot
        rotations = rotations.cpu().numpy()
        rot_3d = np.zeros((total_len,28,3))
        for m in range(rotations.shape[1]):
            for j in range(rotations.shape[2]):
                rot_3d[m,j]=process_rotation(rotations[0,m,j])
        angle_diff = np.abs(rot_3d[1:]-rot_3d[:-1])
        theta = np.sum(angle_diff,axis=-1)
        theta_new = theta[:,:,np.newaxis]
        adj_rot = np.concatenate((angle_diff,theta_new),axis=-1)
        rot_move[:,0] = np.min(adj_rot, axis=(0, 1))
        rot_move[:,1] = np.max(adj_rot, axis=(0, 1))
        rot_move[:,2] = np.mean(adj_rot, axis=(0, 1))
        rot_res.append(rot_move[np.newaxis,:,:])
        # rot的差值
    pos_data = np.concatenate(pos_res,axis=0)
    rot_data = np.concatenate(rot_res,axis=0)
    min_pos = [np.min(pos_data[:, i, 0]) for i in range(4)]
    max_pos = [np.max(pos_data[:, i, 1]) for i in range(4)]
    mean_pos = [np.mean(pos_data[:, i, 2]) for i in range(4)]
    
    min_rot = [np.min(rot_data[:, i, 0]) for i in range(4)]
    max_rot = [np.max(rot_data[:, i, 1]) for i in range(4)]
    mean_rot = [np.mean(rot_data[:, i, 2]) for i in range(4)]
    print("min-pos:",min_pos)
    print("max-pos:",max_pos)
    print("mean-pos:",mean_pos)
    
    print("min-r:",min_rot)
    print("max-r:",max_rot)
    print("mean-r:",mean_rot)
    
    
