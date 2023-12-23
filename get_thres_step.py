import os
import sys
import math
import json
import time
import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

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
                        default="test")

    start_time = time.time()
    args = parser.parse_args()
    path_csv=r"/home/mjy/1126Teeth/scripts/infer_{}.csv".format(args.config)
    csv_filename=[]
    csv_realn = []
    csv_trans = []
    csv_avg_pos=[]  # (samples,4)
    csv_avg_rot=[]  # (samples,4)
    csv_avg_acc_pos=[]  # (samples,4)
    csv_avg_acc_rot=[]  # (samples,4)
    config = load_config_by_name(args.config)

    context_model.ATTENTION_MODE = config["train"]["attention_mode"]
    context_model.zscore_MODE = config["train"]["zscore_MODE"]
    context_model.INIT_INTERP = config["train"]["init_interp"]
    context_model.Data_Mask_MODE=config["train"]["data_mask_set"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, args.dataset, device=device, shuffle=False,inference_mode=True,add_geo=False)

    all_frames=0
    
    side_tooth_idx = list(range(0,4))+list(range(10,14))+list(range(14,18))+list(range(24,28))
    center_tooth_idx = list(range(4,10))+list(range(18,24))
    for i, data in enumerate(data_loader, 0):
        (positions, rotations, file_name, real_frame_num,trends,geo,remove_idx,data_idx) = data # FIXED:返回类型(1,seq,joint,3)
        pos_move = np.zeros(4)
        rot_move = np.zeros(4)
        file_name=file_name[0]
        print(f"file name:{file_name} idx:{data_idx}")

        n_remove_idx=[ [] for j in range(remove_idx.shape[0])]
        static_ids = remove_idx.nonzero(as_tuple=False).cpu().numpy()
        for j in range(static_ids.shape[0]):
            x0=static_ids[j,0]
            y0=static_ids[j,1]
            n_remove_idx[x0].append(y0)
        remove_idx=n_remove_idx[0]
        # mask = np.ones(positions.shape[2], dtype=bool)
        # mask[remove_idx] = False

        # positions = positions[:,:, mask, :]
        # rotations = rotations[:,:, mask, :,:]

        total_len=positions.shape[1]
        joints = positions.shape[2]
        csv_realn.append(total_len)
        csv_trans.append(total_len-1)
        
        all_frames+=(total_len-2)
        print("total-len:{} real:{}".format(total_len,real_frame_num))
        # 相邻pos的差值
        adj_pos = torch.abs(positions[0,-1]-positions[0,0]).cpu().numpy()
        assert adj_pos.shape[0]==joints and adj_pos.shape[1]==3
        side_pos = adj_pos[side_tooth_idx]
        center_pos = adj_pos[center_tooth_idx]
        # adj_pos[adj_pos == 0] = np.nan
        pos_move[:3]=np.mean(adj_pos, axis=0)
        inner_dot = np.sqrt(np.sum(adj_pos * adj_pos, axis=1, keepdims=True))
        side_dot = inner_dot[side_tooth_idx]
        center_dot = inner_dot[center_tooth_idx]
        pos_move[-1]=np.mean(inner_dot)
        csv_avg_pos.append(pos_move.tolist())
        if pos_move.any() > 0.5*(total_len-1):
            print("***pos_move:",pos_move,"***file_name:",file_name)
        # pos的差值-sidetooth
        side_pos_move = np.zeros(4)
        side_pos_move[:3]=np.mean(side_pos, axis=0)
        side_pos_move[-1]=np.mean(side_dot)
        # pos的差值-centertooth
        center_pos_move = np.zeros(4)
        center_pos_move[:3]=np.mean(center_pos, axis=0)
        center_pos_move[-1]=np.mean(center_dot)
        # total-side+center
        total_pos_move = side_pos_move+center_pos_move
        csv_avg_acc_pos.append(total_pos_move.tolist())
        # rot
        rotations = rotations.cpu().numpy()
        rot_3d = np.zeros((2,joints,3))

        for j in range(joints):
            rot_3d[0,j]=process_rotation(rotations[0,-1,j])
            rot_3d[1,j]=process_rotation(rotations[0,0,j])
        angle_diff = np.abs(rot_3d[1]-rot_3d[0])
        theta = np.sum(angle_diff,axis=-1)
        theta_new = theta[:,np.newaxis]
        adj_rot = np.concatenate((angle_diff,theta_new),axis=-1)
        #adj_rot[adj_rot == 0] = np.nan
        
        rot_move = np.mean(adj_rot, axis=0)
        csv_avg_rot.append(rot_move.tolist())
        side_rot = adj_rot[side_tooth_idx]
        center_rot = adj_rot[center_tooth_idx]
        side_rot_move = np.mean(side_rot, axis=0)
        center_rot_move = np.mean(center_rot, axis=0)
        total_rot_move = side_rot_move+center_rot_move
        csv_avg_acc_rot.append(total_rot_move.tolist())
        # rot的差值
    
    # pos_data = np.concatenate(pos_res,axis=0)
    # rot_data = np.concatenate(rot_res,axis=0)
    # min_pos = [np.nanmin(pos_data[:, i, 0]) for i in range(4)]
    # max_pos = [np.nanmax(pos_data[:, i, 1]) for i in range(4)]
    # mean_pos = [np.nanmean(pos_data[:, i, 2]) for i in range(4)]
    
    # min_rot = [np.nanmin(rot_data[:, i, 0]) for i in range(4)]
    # max_rot = [np.nanmax(rot_data[:, i, 1]) for i in range(4)]
    # mean_rot = [np.nanmean(rot_data[:, i, 2]) for i in range(4)]
    # print("min-pos:",min_pos)
    # print("max-pos:",max_pos)
    # print("mean-pos:",mean_pos)
    
    # print("min-r:",min_rot)
    # print("max-r:",max_rot)
    # print("mean-r:",mean_rot)
    
    # pos-figure1
    a = np.array(csv_avg_pos)
    a1 = a[:,0]
    a2 = a[:,1]
    a3 = a[:,2]
    a4 = a[:,3]
    b = np.array(csv_trans)
    print("a-len:",len(a1))
    print(len(b))
    print(csv_avg_pos[:][0])
    print(csv_avg_pos[0])
    # Figure 1: Scatter diagram of a and b
    plt.figure(figsize=(20,4))
    plt.subplot(1, 4, 1)
    plt.scatter(a1, b, s=5)
    plt.title("Scatter Diagram")
    plt.xlabel("pos-x")
    plt.ylabel("number")
    
    plt.subplot(1, 4,2)
    plt.scatter(a2, b, s=5)
    plt.title("Scatter Diagram")
    plt.xlabel("pos-y")
    plt.ylabel("number")
    
    plt.subplot(1, 4,3)
    plt.scatter(a3, b, s=5)
    plt.title("Scatter Diagram")
    plt.xlabel("pos-z")
    plt.ylabel("number")
    
    plt.subplot(1, 4,4)
    plt.scatter(a4, b, s=5)
    plt.title("Scatter Diagram")    
    plt.xlabel("pos-len")
    plt.ylabel("number")
    plt.savefig('/home/mjy/pos_figure1.png')
    
    # pos-figure2
    # Figure 2: Frequency of a[i]/b[i]
    ratios1 = np.around(a1 / b,4)
    freq_ratios1 = Counter(ratios1)
    ratios2 = np.around(a2 / b,4)
    freq_ratios2 = Counter(ratios2)
    ratios3 = np.around(a3 / b,4)
    freq_ratios3 = Counter(ratios3)
    ratios4 = np.around(a4 / b,4)
    freq_ratios4 = Counter(ratios4)
    plt.figure(figsize=(20,4))
    plt.subplot(1, 4, 1)
    plt.scatter(list(freq_ratios1.keys()), list(freq_ratios1.values()), s=5)
    plt.title("Frequency of x/n")
    plt.xlabel("Ratio x/n")
    plt.ylabel("f")
    
    plt.subplot(1, 4, 2)
    plt.scatter(list(freq_ratios2.keys()), list(freq_ratios2.values()), s=5)
    plt.title("Frequency of y/n")
    plt.xlabel("Ratio y/n")
    plt.ylabel("f")
    
    plt.subplot(1, 4, 3)
    plt.scatter(list(freq_ratios3.keys()), list(freq_ratios3.values()), s=5)
    plt.title("Frequency of z/n")
    plt.xlabel("Ratio z/n")
    plt.ylabel("f")
    
    plt.subplot(1, 4, 4)
    plt.scatter(list(freq_ratios4.keys()), list(freq_ratios4.values()), s=5)
    plt.title("Frequency of len/n")
    plt.xlabel("Ratio len/n")
    plt.ylabel("f")
    plt.savefig('/home/mjy/pos_figure2.png')
    
    # Figure 3: Distribution of n
    freq_n = Counter(b)
    plt.figure(figsize=(10,20))
    plt.bar(freq_n.keys(), freq_n.values())
    plt.title("Distribution of n")
    plt.xlabel("n")
    plt.ylabel("Frequency")
    plt.savefig('/home/mjy/number.png')
    ##########################################
    ### rot-figure1
    a = np.array(csv_avg_rot)
    a1 = a[:,0]
    a2 = a[:,1]
    a3 = a[:,2]
    a4 = a[:,3]
    
    
    # Figure 1: Scatter diagram of a and b
    plt.figure(figsize=(20,4))
    plt.subplot(1, 4, 1)
    plt.scatter(a1, b, s=5)
    plt.title("Scatter Diagram-a")
    plt.xlabel("rot-x")
    plt.ylabel("number")
    
    plt.subplot(1, 4,2)
    plt.scatter(a2, b, s=5)
    plt.title("Scatter Diagram-b")
    plt.xlabel("rot-y")
    plt.ylabel("number")
    
    plt.subplot(1, 4,3)
    plt.scatter(a3, b, s=5)
    plt.title("Scatter Diagram-c")
    plt.xlabel("rot-z")
    plt.ylabel("number")
    
    plt.subplot(1, 4,4)
    plt.scatter(a4, b, s=5)
    plt.title("Scatter Diagram-r")    
    plt.xlabel("rot-len")
    plt.ylabel("number")
    plt.savefig('/home/mjy/rot_figure1.png')
    
    # pos-figure2
    # Figure 2: Frequency of a[i]/b[i]
    ratios1 = np.around(a1 / b,4)
    freq_ratios1 = Counter(ratios1)
    ratios2 = np.around(a2 / b,4)
    freq_ratios2 = Counter(ratios2)
    ratios3 = np.around(a3 / b,4)
    freq_ratios3 = Counter(ratios3)
    ratios4 = np.around(a4 / b,4)
    freq_ratios4 = Counter(ratios4)
    print(list(freq_ratios1.keys()), list(freq_ratios1.values()))
    plt.figure(figsize=(20,4))
    plt.subplot(1, 4, 1)
    plt.scatter(list(freq_ratios1.keys()), list(freq_ratios1.values()), s=5)
    plt.title("Frequency of a/n")
    plt.xlabel("Ratio a/n")
    plt.ylabel("f")
    
    plt.subplot(1, 4, 2)
    plt.scatter(list(freq_ratios2.keys()), list(freq_ratios2.values()), s=5)
    plt.title("Frequency of b/n")
    plt.xlabel("Ratio b/n")
    plt.ylabel("f")
    
    plt.subplot(1, 4, 3)
    plt.scatter(list(freq_ratios3.keys()), list(freq_ratios3.values()), s=5)
    plt.title("Frequency of c/n")
    plt.xlabel("Ratio c/n")
    plt.ylabel("f")
    
    plt.subplot(1, 4, 4)
    plt.scatter(list(freq_ratios4.keys()), list(freq_ratios4.values()), s=5)
    plt.title("Frequency of r/n")
    plt.xlabel("Ratio r/n")
    plt.ylabel("f")
    plt.savefig('/home/mjy/rot_figure2.png')