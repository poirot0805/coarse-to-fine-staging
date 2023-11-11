import os
from typing_extensions import Self
import torch
from torch.utils.data import Dataset
import numpy as np
import json

from motion_inbetween_normal.data import bvh, utils_np


class BvhDataSet(Dataset):
    def __init__(self, bvh_folder, actors, window=50, offset=1,
                 start_frame=0, fill_mode="missing-zero",device="cpu", dtype=torch.float32,complete_flag=False, augment_flag=True,add_geo=True,inference_mode=False):
        """
        Bvh data set.

        Args:
            bvh_folder (str): Bvh folder path.
            actors (list of str): List of actors to be included in the dataset.
            window (int, optional): Length of window. Defaults to 50.
            offset (int, optional): Offset of window. Defaults to 1.
            start_frame (int, optional):
                Override the start frame of each bvh file. Defaults to 0.
            device (str, optional): Device. e.g. "cpu", "cuda:0".
                Defaults to "cpu".
            dtype: torch.float16, torch.float32, torch.float64 etc.
        """
        super(BvhDataSet, self).__init__()
        self.bvh_folder = bvh_folder
        self.actors = actors
        self.window = window
        self.offset = offset
        self.start_frame = start_frame
        self.device = device
        self.dtype = dtype
        self.complete_flag=complete_flag
        self.augment=augment_flag
        self.inference_mode=inference_mode
        self.add_geo=add_geo
        self.primes=[]
        self.fill_mode=fill_mode    # 0:missing-zero 1:missing-mean 2:vacant-mean
        print(f"complete:{complete_flag} augment:{augment_flag} add_geo:{add_geo} fill_mode:{self.fill_mode}")
        if inference_mode:
            for i in range(500):
                self.primes.append(1)
            for i in range(2,500):
                if self.primes[i]==1:
                    j=2*i
                    while j<500:
                        self.primes[j]=0
                        j+=i
        
        if self.complete_flag:
            print("--only complete teeth--")
            self.bvh_folder=[os.path.join(bvh_folder,"complete")]
        else:
            self.bvh_folder=[os.path.join(bvh_folder,"complete"),os.path.join(bvh_folder,"incomplete")]
        # self.load_bvh_files()
        json_root=r"/home/mjy/motion-inbetween"# r"E:\PROJECTS\tooth_spatial_temporal_transformer"
        mode=['test','train','val']
        self.remove_dict={}
        for subdir in mode:
            json_path=os.path.join(json_root,f"removeStatus3_{subdir}.json")
            with open(json_path) as fh:
                self.remove_dict.update(json.load(fh))
        self.load_json_files()

    def _to_tensor(self, array):
        return torch.tensor(array, dtype=self.dtype, device=self.device)
    
    def load_json_files(self):
        # tooth data使用了json格式作为数据源
        self.bvh_files = []
        self.positions = []
        self.rotations = []
        self.trends=[]  # trends for motion: 0 / 1
        self.geo=[]     # geometric info [28,108]
        self.geoids=[]  # ids to get geo
        self.frames = []
        self.extend_frames = []
        self.names=[]
        self.remove_list=[]
        current_id=0
        
        for dataset_path in self.bvh_folder:
            # load bvh files that match given actors
            for f in os.listdir(dataset_path):
                f = os.path.abspath(os.path.join(dataset_path, f))
                if f.endswith(".json"):
                    self.bvh_files.append(f)

        if not self.bvh_files:
            raise FileNotFoundError(
                "No bvh files found in {}. (Actors: {})".format(
                    self.bvh_folder, ", ".join(self.actors))
            )

        self.bvh_files.sort()
        total_len=len(self.bvh_files)
        for bvh_path in self.bvh_files:
            print("{}/{} Processing file {}".format(current_id,total_len,bvh_path))
            rootpath,basepath=os.path.split(bvh_path)
            basename,_=os.path.splitext(basepath)
            remove_list=self.remove_dict[basepath]['remove_ids']
            if "miss" in self.fill_mode:
                remove_list=[]
            geo=bvh.load_feature_npy(basename,self.add_geo,remove_list=remove_list)  # 【28，seq，9】

            anim = bvh.load_tooth_json(bvh_path, start=self.start_frame,remove_list=remove_list)

            self.names.append(bvh_path)
            self.geo.append(self._to_tensor(geo))
            current_id+=1

            self.positions.append(self._to_tensor(anim.positions))  # (seq,joint,3)
            self.rotations.append(self._to_tensor(anim.rotations))  # (seq,joint,3,3)

            self.remove_list.append(anim.missing_list)
            self.frames.append(anim.positions.shape[0]) # real frame number
            self.geoids.append(current_id-1)
            
            clip_cnt=anim.positions.shape[0] - self.window if anim.positions.shape[0] - self.window>0 else 0
            if self.inference_mode:
                clip_cnt=0
            
            if clip_cnt==0:
                self.extend_frames.append(self.window)
            # ADD:追加跨帧数据
            if clip_cnt>0:
                self.extend_frames.append(anim.step_num)
                if self.augment:
                    for cross_step in range(2,6):
                        if anim.step_num/cross_step<30: # NOTICE:原来是12，需要填补太多了
                            break
                        new_seq_id=slice(0,anim.step_num,cross_step)
                        npos=self._to_tensor(anim.positions)[new_seq_id]
                        nrot=self._to_tensor(anim.rotations)[new_seq_id]
                        self.positions.append(npos)  # (seq,joint,3)
                        self.rotations.append(nrot)  # (seq,joint,3,3)
                        self.frames.append(npos.shape[0]) # real frame number
                        self.remove_list.append(anim.missing_list)
                        self.geoids.append(current_id-1)
                        if npos.shape[0] - self.window>0:
                            self.extend_frames.append(npos.shape[0])
                        else:
                            self.extend_frames.append(self.window)
            

    def __len__(self):
        count = 0
        # for frame in self.frames:
        #     count += int(float(frame - self.window) / self.offset) + 1
        # return count
        if self.inference_mode:
            return len(self.frames)
        for frame in self.extend_frames:
            count += int(float(frame - self.window) / self.offset) + 1
        return count
    
    def getTransNumber(self,step):
        loss_refer_list_a= [
            0.20177436, 0.20177436, 0.20177436, 0.20177436, 0.20177436, 0.20177436,
            0.2285235, 0.25529256, 0.28209722, 0.3082302, 0.3358118, 0.3633846,
            0.39051858, 0.41746747, 0.44201124, 0.46724388, 0.49251252, 0.51677865,
            0.54076684, 0.5630635, 0.58442235, 0.60640174, 0.62820905, 0.649876,
            0.6686218, 0.6883557, 0.70835, 0.7287543, 0.7500327, 0.7677334, 0.7845982,
            0.79973966, 0.8134664, 0.82605225, 0.8359167, 0.8464236, 0.85664207,
            0.8667042, 0.8773259, 0.88731974, 0.89824706, 0.91049254, 0.9233855
        ]
        loss_refer_list_b = [0, 0.045345, 0.058997, 0.076935, 0.105967, 0.164109]
        add_len = 0
        dense = 0
        sp = step - 2
        total_loss = 100000
        # total_loss=sp*LOSS_A[sp]+(sp+1)*dense*LOSS_B[dense]
        # step+u=(sp+1)*(dense+1)+1
        # sp<=28,dense<=4
        for u in range(4):
            flag = False
            if self.primes[step + u - 1]:
                continue
            d=1
            if (step + u - 1) % (d + 1) == 0:
                    tmp_sp = (step + u - 1) // (d + 1) - 1
                    temp_loss = 0#tmp_sp * loss_refer_list_a[tmp_sp] + (tmp_sp + 1) * d * loss_refer_list_b[d]

                    total_loss = temp_loss
                    dense = d
                    sp = tmp_sp
                    flag = True

            if flag:
                add_len = u
                break
        #if step <= 44:
            #loss_a = loss_refer_list_a[step - 2]*(step-2)
            #if loss_a<=total_loss:
                #return step-2,0,0,loss_a
        return sp, dense, add_len, total_loss
    
    def __getitem__(self, idx):
        curr_idx = idx
        # FIXED:具体如何取是在这里改的
        if self.inference_mode:
            print("infer!")
            positions = self.positions[idx][0:]
            rotations = self.rotations[idx][0:]
            frame_num=self.frames[idx]
            add_len=0
            end_idx=frame_num

            sp,dense,add_len,_=self.getTransNumber(frame_num)
            
            add_pos=self.positions[idx][end_idx-1:end_idx]
            add_rot=self.rotations[idx][end_idx-1:end_idx]
            
            positions=torch.cat([positions,add_pos.expand([add_len,*add_pos.shape[1:]])],dim=0)
            rotations=torch.cat([rotations,add_rot.expand([add_len,*add_rot.shape[1:]])],dim=0)

            zeros_shape = [1,28]
            zeros = torch.zeros(*zeros_shape, dtype=self.dtype,device=self.device)
            b=(positions[1:,:,:]-positions[:-1,:,:]!=0)
            c=(rotations[1:,:,:]-rotations[:-1,:,:]!=0).flatten(start_dim=-2)
            x=torch.any(b,dim=-1)
            x2=torch.any(c,dim=-1)
            temp_trend=torch.logical_or(x,x2)
            trends=torch.cat([zeros,temp_trend],dim=0).type(self.dtype)

            # GEO:
            geo_id=self.geoids[idx]
            remove_idx=torch.zeros(28,dtype=torch.bool,device=self.device)  # 0: not remove / 1: remove
            for k in self.remove_list[idx]:
                remove_idx[k]=True
            return (
                positions,
                rotations,
                self.names[idx],
                frame_num,
                sp,
                dense,
                trends,
                self.geo[geo_id],
                remove_idx,
                idx
            )

        
        for i, frame in enumerate(self.extend_frames):
            tmp_idx = curr_idx - \
                int(float(frame - self.window) / self.offset) - 1

            if tmp_idx >= 0:
                curr_idx = tmp_idx
                continue
                         
            start_idx = curr_idx * self.offset
            end_idx = start_idx + self.window
            frame_num=self.window
            # print(idx, i, start_idx, end_idx)
            if end_idx<=self.frames[i]:
                positions = self.positions[i][start_idx: end_idx]
                rotations = self.rotations[i][start_idx: end_idx]

            else:
                extend_length=end_idx-self.frames[i]
                frame_num -= extend_length
                end_idx=self.frames[i]
                temp_positions = self.positions[i][start_idx: end_idx]
                temp_rotations = self.rotations[i][start_idx: end_idx]
                add_pos=self.positions[i][end_idx-1:end_idx]
                add_rot=self.rotations[i][end_idx-1:end_idx]
                positions=torch.cat([temp_positions,add_pos.expand([extend_length,*add_pos.shape[1:]])],dim=0)
                rotations=torch.cat([temp_rotations,add_rot.expand([extend_length,*add_rot.shape[1:]])],dim=0)
            # trends mask
            zeros_shape = [1,28]
            zeros = torch.zeros(*zeros_shape, dtype=self.dtype,device=self.device)
            b=(positions[1:,:,:]-positions[:-1,:,:]!=0)
            c=(rotations[1:,:,:]-rotations[:-1,:,:]!=0).flatten(start_dim=-2)
            x=torch.any(b,dim=-1)
            x2=torch.any(c,dim=-1)
            temp_trend=torch.logical_or(x,x2)
            trends=torch.cat([zeros,temp_trend],dim=0).type(self.dtype)
            # trends=trends.unsqueeze(2)
            # 1：move 0:static
            geo_id=self.geoids[i]
            remove_idx=torch.zeros(28,dtype=torch.bool,device=self.device)  # 0: not remove / 1: remove
            for k in self.remove_list[i]:
                remove_idx[k]=True
                
            assert positions.shape[0]==self.window and positions.shape[2]==3
            assert rotations.shape[0]==self.window and rotations.shape[2]==3
            assert trends.shape[0]==self.window 
            return (
                positions,
                rotations,
                self.names[geo_id],
                frame_num,
                trends,
                self.geo[geo_id],
                remove_idx,
                idx
            )
