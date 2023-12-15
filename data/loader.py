import os
from typing_extensions import Self
import torch
from torch.utils.data import Dataset
import numpy as np
import json

from motion_inbetween_pred.data import bvh, utils_np


class BvhDataSet(Dataset):
    def __init__(self, bvh_folder, actors, window=192, offset=1,
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
        self.window = window-13
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
        self.clip_num=0
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

            self.names.append(bvh_path) # 1
            self.geo.append(self._to_tensor(geo))
            current_id+=1

            self.positions.append(self._to_tensor(anim.positions))  # (seq,joint,3) #2
            self.rotations.append(self._to_tensor(anim.rotations))  # (seq,joint,3,3) #3

            self.remove_list.append(anim.missing_list)  #4
            self.frames.append(anim.positions.shape[0]) # real frame number:5
            self.geoids.append(current_id-1)#6
            
            clip_cnt=1 + anim.positions.shape[0]/5-4
            if self.inference_mode:
                clip_cnt=1
            self.clip_num+=clip_cnt
            # ADD:追加跨帧数据
            if clip_cnt>1:
                for i in range(25,anim.positions.shape[0],5):
                    self.names.append(bvh_path) # 1
                    self.positions.append(self._to_tensor(anim.positions[-i:]))  # (seq,joint,3) #2
                    self.rotations.append(self._to_tensor(anim.rotations[-i:]))  # (seq,joint,3,3) #3

                    self.remove_list.append(anim.missing_list)  #4
                    self.frames.append(i) # real frame number:5
                    self.geoids.append(current_id-1)#6

    def cal_n(self,pos1,pos2):
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
    def __len__(self):
        if self.inference_mode:
            return len(self.frames)

        return len(self.positions)

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

            if self.primes[frame_num-1]:
                flag=False
                factor=-1
                add_len=1
                for i in range(5,1,-1):
                    if (frame_num)%i==0 and (frame_num)//i<30:
                        flag=True
                    elif frame_num%i!=0:
                        factor=i
                if flag==False:
                    factor = factor if factor>0 else 2
                    add_len = (frame_num//factor+1)*factor-frame_num+1
            else:
                flag=False
                factor=-1
                for i in range(5,1,-1):
                    if (frame_num-1)%i==0 and (frame_num-1)//i<30:
                        flag=True
                    elif (frame_num-1)%i!=0:
                        factor=i
                if flag==False:
                    factor = factor if factor>0 else 2
                    add_len = ((frame_num-1)//factor+1)*factor-frame_num+1
            if frame_num<=30:
                add_len=0
            add_len=0
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
                trends,
                self.geo[geo_id],
                remove_idx,
                idx
            )

        tmp_positions = self.positions[idx]
        tmp_rotations = self.rotations[idx]
        frame_num = self.frames[idx]
        
        extend_length=self.window - frame_num

        add_pos=tmp_positions[-1:]
        add_rot=tmp_rotations[-1:]
        positions=torch.cat([tmp_positions,add_pos.expand([extend_length,*add_pos.shape[1:]])],dim=0)
        rotations=torch.cat([tmp_rotations,add_rot.expand([extend_length,*add_rot.shape[1:]])],dim=0)
            # # trends mask
            # zeros_shape = [1,28]
            # zeros = torch.zeros(*zeros_shape, dtype=self.dtype,device=self.device)
            # b=(positions[1:,:,:]-positions[:-1,:,:]!=0)
            # c=(rotations[1:,:,:]-rotations[:-1,:,:]!=0).flatten(start_dim=-2)
            # x=torch.any(b,dim=-1)
            # x2=torch.any(c,dim=-1)
            # temp_trend=torch.logical_or(x,x2)
            # trends=torch.cat([zeros,temp_trend],dim=0).type(self.dtype)
            # # trends=trends.unsqueeze(2)
            # # 1：move 0:static
        trends = positions.clone()
        geo_id=self.geoids[idx]
        remove_idx=torch.zeros(28,dtype=torch.bool,device=self.device)  # 0: not remove / 1: remove
        for k in self.remove_list[idx]:
                remove_idx[k]=True
        init_n = self.cal_n(positions[0],positions[-1])
        assert positions.shape[0]==self.window and positions.shape[2]==3
        assert rotations.shape[0]==self.window and rotations.shape[2]==3
        assert trends.shape[0]==self.window 
        return (
                positions,
                rotations,
                self.names[geo_id],
                self._to_tensor(frame_num),
                self._to_tensor(init_n),
                trends,
                self.geo[geo_id],
                remove_idx,
                idx
            )


class ValToothDataSet(Dataset):
    def __init__(self, bvh_folder,  window=192, offset=1,
                 start_frame=25, fill_mode="missing-zero",device="cpu", dtype=torch.float32,complete_flag=False, augment_flag=True,add_geo=True):
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
        super(ValToothDataSet, self).__init__()
        self.bvh_folder = bvh_folder
        self.window = window-13
        self.offset = offset
        self.start_frame = start_frame
        self.device = device
        self.dtype = dtype
        self.complete_flag=complete_flag
        self.augment=augment_flag

        self.add_geo=add_geo

        self.fill_mode=fill_mode    # 0:missing-zero 1:missing-mean 2:vacant-mean
        print(f"complete:{complete_flag} augment:{augment_flag} add_geo:{add_geo} fill_mode:{self.fill_mode}")
        
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
        self.clip_num=0
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
            if anim.step_num<self.start_frame-5 or anim.step_num>self.start_frame+5:
                continue
            self.names.append(bvh_path)
            self.geo.append(self._to_tensor(geo))
            current_id+=1

            self.positions.append(self._to_tensor(anim.positions))  # (seq,joint,3)
            self.rotations.append(self._to_tensor(anim.rotations))  # (seq,joint,3,3)

            self.remove_list.append(anim.missing_list)
            self.frames.append(anim.positions.shape[0]) # real frame number
            self.geoids.append(current_id-1)
            
            self.clip_num+=1
                
    def cal_n(self,pos1,pos2):
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
    def __len__(self):
        count = 0

        return len(self.positions)

    def __getitem__(self, idx):
        curr_idx = idx
        # FIXED:具体如何取是在这里改的
        tmp_positions = self.positions[idx]
        tmp_rotations = self.rotations[idx]
        frame_num = self.frames[idx]
        
        extend_length=self.window - frame_num

        add_pos=tmp_positions[-1:]
        add_rot=tmp_rotations[-1:]
        positions=torch.cat([tmp_positions,add_pos.expand([extend_length,*add_pos.shape[1:]])],dim=0)
        rotations=torch.cat([tmp_rotations,add_rot.expand([extend_length,*add_rot.shape[1:]])],dim=0)
            # # trends mask
            # zeros_shape = [1,28]
            # zeros = torch.zeros(*zeros_shape, dtype=self.dtype,device=self.device)
            # b=(positions[1:,:,:]-positions[:-1,:,:]!=0)
            # c=(rotations[1:,:,:]-rotations[:-1,:,:]!=0).flatten(start_dim=-2)
            # x=torch.any(b,dim=-1)
            # x2=torch.any(c,dim=-1)
            # temp_trend=torch.logical_or(x,x2)
            # trends=torch.cat([zeros,temp_trend],dim=0).type(self.dtype)
            # # trends=trends.unsqueeze(2)
            # # 1：move 0:static
        trends = positions.clone()
        geo_id=self.geoids[idx]
        remove_idx=torch.zeros(28,dtype=torch.bool,device=self.device)  # 0: not remove / 1: remove
        for k in self.remove_list[idx]:
                remove_idx[k]=True
        init_n = self.cal_n(positions[0],positions[-1])
        assert positions.shape[0]==self.window and positions.shape[2]==3
        assert rotations.shape[0]==self.window and rotations.shape[2]==3
        assert trends.shape[0]==self.window 
        return (
                positions,
                rotations,
                self.names[geo_id],
                self._to_tensor(frame_num),
                self._to_tensor(init_n),
                trends,
                self.geo[geo_id],
                remove_idx,
                idx
            )