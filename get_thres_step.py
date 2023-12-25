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
                        default="train")

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
        config, "train", device=device, shuffle=False,inference_mode=True,add_geo=False)
    dataset2,data_loader2 = train_utils.init_bvh_dataset(
         config, "benchmark", device=device, shuffle=False,inference_mode=True,add_geo=True)
    dataset3,data_loader3 = train_utils.init_bvh_dataset(
         config, "test", device=device, shuffle=False,inference_mode=True,add_geo=True)
    all_frames=0
    data_loader_list=[data_loader,data_loader2,data_loader3]
    side_tooth_idx = list(range(0,4))+list(range(10,14))+list(range(14,18))+list(range(24,28))
    center_tooth_idx = list(range(4,10))+list(range(18,24))
    error_list=['/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002722632.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002722812.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002724937.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002726883.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002728672.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002737908.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002739797.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002739809.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002740294.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002742285.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002742814.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002743376.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002748270.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002752736.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002753894.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002757078.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002760218.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002760285.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002762513.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002764234.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002770466.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002772985.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002774123.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002774594.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002775269.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002784742.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002791706.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002792886.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002796891.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002800505.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002807805.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002809896.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002810292.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002811406.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002811855.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002812430.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002817413.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002818931.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002828437.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002828482.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002837246.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002838124.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002838337.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002840587.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002844772.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/complete/C01002849621.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002722823.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002725118.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002725736.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002727154.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002727817.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002728762.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002735973.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002736749.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002736806.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002737627.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002738954.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002742982.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002744298.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002744513.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002744715.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002745255.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002746492.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002746762.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002746784.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002747392.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002748258.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002750688.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002751746.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002752343.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002752398.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002756167.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002761703.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002763288.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002763514.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002764458.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002764650.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002767170.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002767967.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002770411.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002772389.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002772402.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002772660.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002775270.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002776709.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002778059.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002778116.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002781299.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002782256.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002782469.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002785002.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002787969.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002788634.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002791605.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002791650.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002792909.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002793517.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002795801.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002796969.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002799164.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002800279.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002801236.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002805533.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002808367.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002811237.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002811934.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002821159.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002823780.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002824747.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002830711.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002831149.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002834276.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002835435.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002836043.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002836706.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002840767.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002844996.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002846987.json', '/home/mjy/1126Teeth/datasets/teeth10k/train/incomplete/C01002847045.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/complete/C01002722788.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/complete/C01002747516.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/complete/C01002774628.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/complete/C01002785507.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/complete/C01002796914.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/complete/C01002803328.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/complete/C01002815310.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002735210.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002737403.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002756831.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002763198.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002763390.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002775708.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002789185.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002801630.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002814870.json', '/home/mjy/1126Teeth/datasets/teeth10k/val/incomplete/C01002826165.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002725466.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002726265.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002745749.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002757180.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002766258.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002767675.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002771849.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002780423.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002801146.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/complete/C01002847483.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/incomplete/C01002744748.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/incomplete/C01002776833.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/incomplete/C01002790266.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/incomplete/C01002796879.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/incomplete/C01002826705.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/incomplete/C01002827975.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/incomplete/C01002838720.json', '/home/mjy/1126Teeth/datasets/teeth10k/test/incomplete/C01002845920.json']

    for data_loader in data_loader_list:
        for i, data in enumerate(data_loader, 0):
            (positions, rotations, file_name, real_frame_num,trends,geo,remove_idx,data_idx) = data # FIXED:返回类型(1,seq,joint,3)
            pos_move = np.zeros(4)
            rot_move = np.zeros(4)
            file_name=file_name[0]
            print(f"file name:{file_name} idx:{data_idx}")
            if file_name in error_list:
                continue
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
            # if adj_pos.any() > 0.5:
            #     print("yes")
            #     indices = np.argwhere(adj_pos > 0.5)
            #     if indices.shape[0]>0:
            #         error_list.append(file_name)
            side_pos = adj_pos[side_tooth_idx]
            center_pos = adj_pos[center_tooth_idx]
            # adj_pos[adj_pos == 0] = np.nan
            pos_move[:3]=np.max(adj_pos, axis=0)
            inner_dot = np.sqrt(np.sum(adj_pos * adj_pos, axis=1, keepdims=True))
            side_dot = inner_dot[side_tooth_idx]
            center_dot = inner_dot[center_tooth_idx]
            pos_move[-1]=np.max(inner_dot)
            csv_avg_pos.append(pos_move.tolist())
            # if pos_move.any() > 0.5:
            #     print("***pos_move:",pos_move,"***file_name:",file_name)
            # pos的差值-sidetooth
            side_pos_move = np.zeros(4)
            side_pos_move[:3]=np.max(side_pos, axis=0)
            side_pos_move[-1]=np.max(side_dot)
            # pos的差值-centertooth
            center_pos_move = np.zeros(4)
            center_pos_move[:3]=np.max(center_pos, axis=0)
            center_pos_move[-1]=np.max(center_dot)
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
            
            rot_move = np.max(adj_rot, axis=0)
            csv_avg_rot.append(rot_move.tolist())
            side_rot = adj_rot[side_tooth_idx]
            center_rot = adj_rot[center_tooth_idx]
            side_rot_move = np.max(side_rot, axis=0)
            center_rot_move = np.max(center_rot, axis=0)
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
    def my_draw(x,y,size,tag1,tag2,path):
        plt.figure(figsize=(40,10))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.scatter(x[i], y,s=size)
            plt.title(f"{tag1}(src_{tag2}-tgt_{tag2})[{i}]->n")
            plt.xlabel(f"{tag2}-{i}")
            plt.ylabel("number")
        plt.savefig(path)
    def my_draw_frequency(x,y,size,tag1,tag2,tag3,path,digit):
        plt.figure(figsize=(40,10))
        for i in range(4):
            ratios = np.around(x[i],digit)
            freq_ratios = Counter(ratios)
            plt.subplot(1, 4, 1+i)
            plt.scatter(list(freq_ratios.keys()), list(freq_ratios.values()), s=size)
            plt.title(f"Frequency of {tag1}-{tag2}[i]{tag3}")
            plt.xlabel(f"Ratio {tag2}[i]{tag3}")
            plt.ylabel("f")
        plt.savefig(path)
    # pos-figure1
    a = np.array(csv_avg_pos)
    a1 = a[:,0]
    a2 = a[:,1]
    a3 = a[:,2]
    a4 = a[:,3]
    b = np.array(csv_trans)
    print("a-len:",len(a1))
    print(len(b))

    my_draw([a1,a2,a3,a4],b,3,"avg","pos",'/home/mjy/pos_figure1.png')
    my_draw_frequency([a1,a2,a3,a4],b,10,"avg","pos","",'/home/mjy/pos_figure1_f.png',digit=2)
    my_draw_frequency([a1/b,a2/b,a3/b,a4/b],b,10,"avg","pos","/n",'/home/mjy/pos_figure2_f.png',digit=3)

    a = np.array(csv_avg_acc_pos)
    a1 = a[:,0]
    a2 = a[:,1]
    a3 = a[:,2]
    a4 = a[:,3]
    b = np.array(csv_trans)
    my_draw([a1,a2,a3,a4],b,3,"sum","pos",'/home/mjy/side_center_pos_figure1.png')
    my_draw_frequency([a1,a2,a3,a4],b,10,"sum","pos","",'/home/mjy/side_center_pos_figure1_f.png',digit=2)
    my_draw_frequency([a1/b,a2/b,a3/b,a4/b],b,10,"sum","pos","/n",'/home/mjy/side_center_pos_figure2_f.png',digit=3)
    
    ##########################################
    ### rot-figure1
    a = np.array(csv_avg_rot)
    a1 = a[:,0]
    a2 = a[:,1]
    a3 = a[:,2]
    a4 = a[:,3]
    my_draw([a1,a2,a3,a4],b,3,"avg","rot",'/home/mjy/rot_figure1.png')
    my_draw_frequency([a1,a2,a3,a4],b,10,"avg","rot","",'/home/mjy/rot_figure1_f.png',digit=3)
    my_draw_frequency([a1/b,a2/b,a3/b,a4/b],b,10,"avg","rot","/n",'/home/mjy/rot_figure2_f.png',digit=4)
    
    a = np.array(csv_avg_acc_rot)
    a1 = a[:,0]
    a2 = a[:,1]
    a3 = a[:,2]
    a4 = a[:,3]
    my_draw([a1,a2,a3,a4],b,3,"sum","rot",'/home/mjy/side_center_rot_figure1.png')
    my_draw_frequency([a1,a2,a3,a4],b,10,"sum","rot","",'/home/mjy/side_center_rot_figure1_f.png',digit=3)
    my_draw_frequency([a1/b,a2/b,a3/b,a4/b],b,10,"sum","rot","/n",'/home/mjy/side_center_rot_figure2_f.png',digit=4)

    # Figure 3: Distribution of n
    freq_n = Counter(b)
    plt.figure(figsize=(10,20))
    plt.bar(freq_n.keys(), freq_n.values())
    plt.title("Distribution of n")
    plt.xlabel("n")
    plt.ylabel("Frequency")
    plt.savefig('/home/mjy/number.png')
    print(error_list)