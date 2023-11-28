import os
import sys
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
# python my_infer_onestage_sp.py ablationsp_context_modelgeo2dAlibiNoabsGeo0 newgeo_context_modelNOGEOdense
# python my_infer.py cmpgeo_context_modelVMeanNoMask newgeo_context_modelVMdense
# python my_infer.py cmpgeo_context_modelVMeanNoMaskNoG newgeo_context_modelNOGEOdense
# python my_infer.py cmpgeo_context_modelVacantMean cmpgeo_context_modelDenseVacant


def post_process(data_gt,rot_gt, key_id_list):
    #interp
    total_len=data_gt.shape[2]
    arr=range(total_len)
    pos_new=data_gt.clone().detach()
    rot_new=rot_gt.clone().detach()
    rot_6d=data_utils.matrix9D_to_6D_torch(rot_new)
    for j in range(len(key_id_list)-1):
                start_idx=key_id_list[j]
                end_idx=key_id_list[j+1]
                dense_trans=end_idx-start_idx-1
                dense_seq=slice(start_idx,end_idx+1)
                den_pos=pos_new[:,dense_seq]
                den_rot=rot_new[:,dense_seq]

                target_idx = 1 + dense_trans
                seq_slice = slice(1, target_idx)
                window_len = dense_trans+2
                dtype = dataset.dtype

                den_pos_new, den_rot_new = benchmark.get_interpolated_local_pos_rot(
            den_pos, den_rot, seq_slice)

                res_dense_slice=slice(seq_slice.start-1,seq_slice.stop+1)

                pos_new[:,dense_seq]=den_pos_new[:,res_dense_slice]
                rot_new[:,dense_seq]=den_rot_new[:,res_dense_slice]
                
    data=data_gt.clone().detach()
    data_rot=rot_6d.clone().detach()
    len_key=len(key_id_list)
    for i in range(1,len_key-2):
        start_idx=key_id_list[i]
        end_idx=key_id_list[i+1]

        delta_1 = (2 * data_gt[..., start_idx, :, :] - data_gt[..., start_idx - 1, :, :]- data_gt[..., start_idx +1, :, :])/2

        delta_2 = (2 * data_gt[..., end_idx, :, :] - data_gt[..., end_idx - 1, :, :]- data_gt[..., end_idx +1, :, :])/2
        
        delta_3 = (pos_new[..., start_idx+1, :, :]+pos_new[..., start_idx-1, :, :]-data_gt[..., start_idx+1, :, :]-data_gt[..., start_idx-1, :, :])/2
        delta_4 = (pos_new[..., end_idx+1, :, :]+pos_new[..., end_idx-1, :, :]-data_gt[..., end_idx+1, :, :]-data_gt[..., end_idx-1, :, :])/2
        delta = (delta_1+delta_2+delta_3+delta_4)/4
        seq_slice_delta=slice(start_idx+1,end_idx)
        data[..., seq_slice_delta, :, :] = (
            data_gt[..., seq_slice_delta, :, :] + delta[..., None, :, :])
        
        delta_5 = (2 * rot_6d[..., start_idx, :, :] - rot_6d[..., start_idx - 1, :, :]- rot_6d[..., start_idx +1, :, :])/2
        delta_6 = (2 * rot_6d[..., end_idx, :, :] - rot_6d[..., end_idx - 1, :, :]- rot_6d[..., end_idx +1, :, :])/2
        delta=(delta_5+delta_6)/2
        data_rot[..., seq_slice_delta, :, :] = (
            rot_6d[..., seq_slice_delta, :, :] + delta[..., None, :, :])
    # start
    start_idx=key_id_list[0]
    end_idx=key_id_list[1]
    
    delta = (pos_new[..., end_idx+1, :, :]+pos_new[..., end_idx-1, :, :]-data_gt[..., end_idx+1, :, :]-data_gt[..., end_idx-1, :, :])/2 #(2 * data_gt[..., end_idx, :, :] - data_gt[..., end_idx - 1, :, :]- data_gt[..., end_idx + 1, :, :])/2 # g(e+1)-g(e)-g(e)+f(e-1)+d=0-> d=2g(e)-g(e+1)-f(e-1)

    seq_slice_delta=slice(start_idx+1,end_idx)
    data[..., seq_slice_delta, :, :] = (
            data_gt[..., seq_slice_delta, :, :] + delta[..., None, :, :])

    # end
    start_idx=key_id_list[-2]
    end_idx=key_id_list[-1]
    delta= (pos_new[..., start_idx+1, :, :]+pos_new[..., start_idx-1, :, :]-data_gt[..., start_idx+1, :, :]-data_gt[..., start_idx-1, :, :])/2 #(2 * data_gt[..., start_idx, :, :] - data_gt[..., start_idx-1, :, :]- data_gt[..., start_idx +1, :, :])/2 # f(s+1)-2g(s)+g(s-1)+d=0 -> d=2g(s)-g(s-1)-f(s+1)
    seq_slice_delta=slice(start_idx+1,end_idx)
    data[..., seq_slice_delta, :, :] = (
            data_gt[..., seq_slice_delta, :, :] + delta[..., None, :, :])
    data_rot = data_utils.matrix6D_to_9D_torch(data_rot)
    return data,data_rot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate context model. "
                                     "Post-processing is applied by default.")
    parser.add_argument("config", help="config name")
    parser.add_argument("dconfig", help="dense config name")
    parser.add_argument("-s", "--dataset",
                        help="dataset name (default=test)",
                        default="test")
    parser.add_argument("-t", "--trans", type=int, default=23,
                        help="transition length (default=23)")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="debug mode")
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
    dense_config=load_config_by_name(args.dconfig)
    context_model.ATTENTION_MODE = config["train"]["attention_mode"]
    context_model.zscore_MODE = config["train"]["zscore_MODE"]
    context_model.INIT_INTERP = config["train"]["init_interp"]
    context_model.Data_Mask_MODE=config["train"]["data_mask_set"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, args.dataset, device=device, shuffle=False,inference_mode=True,add_geo=config["train"]["add_geo"])
    SEQNUM_GEO=12 if config["train"]["add_geo"] else 0
    add_geo_FLAG = config["train"]["add_geo"]
    # initialize model
    model = STTransformer(config["model"]).to(device)
    # dense_model=ContextTransformer(dense_config["model"]).to(device)
    # load checkpoint
    epoch, iteration = train_utils.load_checkpoint(config, model,suffix=".min")
    # train_utils.load_checkpoint(dense_config, dense_model,suffix=".min")

    indices = config["indices"]
    context_len = config["train"]["context_len"]+SEQNUM_GEO

    mean, std = context_model.get_train_stats_torch(
            config, dataset.dtype, device)
    mean_d,std_d=context_model.get_train_stats_torch(
            dense_config, dataset.dtype, device)
    # print("sparse: mean:{} / std:{}".format(mean,std))
    # print("dense: mean:{} / std:{}".format(mean_d,std_d))
    mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
            config, dataset.dtype, device)
    
    # FILL WITH MEAN VALUE
    fill_mode = config["datasets"]["train"]["fill_mode"]
    fill_value_dense = fill_value = torch.zeros(mean.shape,dtype=dataset.dtype,device=device)
    if fill_mode=="missing-mean" or fill_mode=="vacant-mean":
        fill_value=mean
        fill_value_dense=mean_d
    fill_value_p,fill_value_r6d=fill_value[:,6:],fill_value[:,:6]#context_model.from_flat_data_joint_data(fill_value)
    #fill_value_p_dense,fill_value_r6d_dense=context_model.from_flat_data_joint_data(fill_value_dense)
    
    gpos_loss_list=[]
    gquat_loss_list=[]
    npss_loss_list=[]
    npss_loss_pos_list=[]

    all_frames=0
    for i, data in enumerate(data_loader, 0):
        (positions, rotations, file_name, real_frame_num,trends,geo,remove_idx,data_idx) = data # FIXED:返回类型(1,seq,joint,3)
        tmp_pos=positions.clone()
        tmp_rot9d=rotations.clone()
        file_name=file_name[0]
        print(f"file name:{file_name}")
        
        total_len=positions.shape[1]
        all_frames+=(total_len-2)
        print("total-len:{} real:{}".format(total_len,real_frame_num))
        sparse_trans=30
        dense_trans=5
        add_len=total_len-real_frame_num
        assert add_len==0
        sparse_trans=total_len-2
        dense_trans=0

        print("dense trans:{} sparse trans:{}".format(dense_trans,sparse_trans))
        
        target_idx = context_len + sparse_trans
        seq_slice = slice(context_len, target_idx)
        window_len = sparse_trans+2+SEQNUM_GEO
        dtype = dataset.dtype

        midway=[]
        
        if context_model.INIT_INTERP!="POS-ONLY":
                inter_pos, inter_rot9d = context_model.get_interp_pos_rot(positions, rotations, seq_slice)
                inter_rot6d = data_utils.matrix9D_to_6D_torch(inter_rot9d)
        
        ##################33
        rot_6d = data_utils.matrix9D_to_6D_torch(rotations)
        if add_geo_FLAG==False:
            geo = None
        # switch torch.tensor to list of list
        n_remove_idx=[ [] for j in range(remove_idx.shape[0])]
        static_ids = remove_idx.nonzero(as_tuple=False).cpu().numpy()
        for j in range(static_ids.shape[0]):
            x0=static_ids[j,0]
            y0=static_ids[j,1]
            n_remove_idx[x0].append(y0)
        remove_idx=n_remove_idx
        # pre-process for missing/vacant teeth:
        remove_len=len(remove_idx)
        for j in range(remove_len):
            remove_list=remove_idx[j]
            positions[j,:,remove_list,:]=fill_value_p[remove_list,:]
            rot_6d[j,:,remove_list,:]=fill_value_r6d[remove_list,:]
            if add_geo_FLAG:
                geo[j,:,remove_list,6:]=fill_value_p[remove_list,:]
                geo[j,:,remove_list,:6]=fill_value_r6d[remove_list,:]
        if context_model.INIT_INTERP!="POS-ONLY":
                for j in range(remove_len):
                    remove_list=remove_idx[j]
                    inter_pos[j,:,remove_list,:]=fill_value_p[remove_list,:]
                    inter_rot6d[j,:,remove_list,:]=fill_value_r6d[remove_list,:]
                inter_x = context_model.get_model_input(inter_pos,inter_rot6d)
                inter_x_zs = (inter_x - mean) / std
        rotations = data_utils.matrix6D_to_9D_torch(rot_6d)
        ##################33
        # for sparse
        sparse_seq=slice(0,total_len,dense_trans+1)
        sp_pos=positions[:,sparse_seq]
        sp_rot=rot_6d[:,sparse_seq]     # rotations[:,sparse_seq]
        
        # attention mask
        atten_mask = context_model.get_attention_mask(
            window_len, context_len, target_idx, device)
        if (real_frame_num-1)%(dense_trans+1)==0:
            x=(real_frame_num-1)//(dense_trans+1)
            # print(type(x))
            midway.append(int(x))
            sp_pos_new, sp_rot_new = context_model.evaluate(
                model, sp_pos, sp_rot, seq_slice,
                indices, mean, std, atten_mask, post_process=False,midway_targets=midway,geo=geo,inter_x_zs = None)
        else:
            sp_pos_new, sp_rot_new = context_model.evaluate(
                model, sp_pos, sp_rot, seq_slice,
                indices, mean, std, atten_mask, post_process=False,geo=geo,inter_x_zs = None)
        parents=[]
        # NOTICE:evaluate的结果包含geo的长度,rot_9d
        assert sp_pos.shape[1]+SEQNUM_GEO==sp_pos_new.shape[1]
        
        # for dense
        res_slice=slice(seq_slice.start-1,seq_slice.stop+1)
        
        pos_new=sp_pos_new[:,res_slice]
        rot_new=sp_rot_new[:,res_slice]
        assert pos_new.shape[1]==sparse_trans+2
        print((positions==tmp_pos).all())
        # restore vacant teeth
        print(remove_len)
        for j in range(remove_len):
            remove_list=remove_idx[j]
            positions[j,:,remove_list,:]=tmp_pos[j,:,remove_list,:]
            pos_new[j,:,remove_list,:]=tmp_pos[j,:,remove_list,:]
            rotations[j,:,remove_list,:,:]=tmp_rot9d[j,:,remove_list,:,:]
            rot_new[j,:,remove_list,:,:]=tmp_rot9d[j,:,remove_list,:,:]
        
        target_idx=pos_new.shape[1]-1

        gpos_batch_loss, gquat_batch_loss, npss_batch_loss,w1,npss_batch_losspos,w2= \
            benchmark.get_rmi_style_batch_loss(
                positions, rotations, pos_new, rot_new, parents,
                1, target_idx, mean_rmi, std_rmi)
        w1 = w1 / np.sum(w1)
        w2 = w2 / np.sum(w2)
        npss1=np.sum(npss_batch_loss * w1, axis=-1)
        npss2=np.sum(npss_batch_losspos * w2, axis=-1)

        # 9d rot -> quat
        quat=data_utils.matrix9D_to_quat_torch(rotations)
        quat_new=data_utils.matrix9D_to_quat_torch(rot_new)

        print("{}, sparse-trans: {}, dense-trans: {}, gpos: {:.4f}, gquat: {:.4f}，npss1: {:.4f}, npss2: {:.4f}".format(
            args.config, sparse_trans, dense_trans,
            gpos_batch_loss[0], gquat_batch_loss[0],npss1[0],npss2[0]))

        csv_filename.append(file_name)
        csv_realfn.append(real_frame_num)
        csv_addlen.append(total_len-real_frame_num)
        csv_totaln.append(total_len)
        csv_sp.append(sparse_trans)
        csv_d.append(dense_trans)
        csv_gpos.append(gpos_batch_loss[0])
        csv_gquat.append(gquat_batch_loss[0])

        gpos_loss_list.append(gpos_batch_loss[0]*(total_len-2))
        gquat_loss_list.append(gquat_batch_loss[0]*(total_len-2))
        npss_loss_list.append(npss1[0])
        npss_loss_pos_list.append(npss2[0])

        datapath,name=os.path.split(file_name)
        basename,exp=os.path.splitext(name)
        tmp_removeidx=[int(kk) for kk in remove_idx[0]]
        json_path_gt = "./res1127_space/{}_{}_{}_sp{}_dense{}_gt.json".format(
            args.config, args.dataset, basename,sparse_trans,dense_trans)
        visualization.save_data_to_json_tooth(
            json_path_gt, positions[0], quat[0],gpos_batch_loss[0], gquat_batch_loss[0],sparse_trans,dense_trans,remove_idx=tmp_removeidx)

        json_path = "./res1127_space/{}_{}_{}_sp{}_dense{}.json".format(
            args.config, args.dataset, basename,sparse_trans,dense_trans)
        visualization.save_data_to_json_tooth(
            json_path, pos_new[0], quat_new[0],gpos_batch_loss[0], gquat_batch_loss[0],sparse_trans,dense_trans,remove_idx=tmp_removeidx)
        
    gpos_mean_loss=sum(gpos_loss_list)/all_frames
    gquat_mean_loss=sum(gquat_loss_list)/all_frames
    npss_mean_loss=sum(npss_loss_list)/1001
    npss_mean_loss_pos=sum(npss_loss_pos_list)/1001
    print("total | gpos:{:.6f} gquat:{:.6f}| all frames:{} | npss:{:.6f} | npss_pos:{:.6f}".format(gpos_mean_loss,gquat_mean_loss,all_frames,npss_mean_loss,npss_mean_loss_pos))
    end_time = time.time()
    print(f"total time:{end_time-start_time}")
    csv_data={"file":csv_filename,
              "realNumber":csv_realfn,
              "add_len":csv_addlen,
              "total_len":csv_totaln,
              "sparse_trans":csv_sp,
              "dense_trans":csv_d,
              "gpos":csv_gpos,
              "gquat":csv_gquat}
    df=pd.DataFrame(csv_data)
    df.to_csv(path_csv)