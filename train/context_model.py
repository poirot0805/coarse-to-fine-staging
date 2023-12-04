import os
import pickle
import random
from turtle import pos
import numpy as np
import math
import torch
from torch.optim import Adam

from motion_inbetween_space import benchmark
from motion_inbetween_space.model import STTransformer
from motion_inbetween_space.data import utils_torch as data_utils
from motion_inbetween_space.train import rmi
from motion_inbetween_space.train import utils as train_utils

SEQNUM_GEO=12
ATTENTION_MODE = "NOMASK"   # //"VANILLA"   //"NOMASK"  //"PRE" //"SQUARE"
INIT_INTERP = "POS-ONLY"
zscore_MODE = "seq"
Data_Mask_MODE = 1  # //0: no data mask //1: normal //2: geo mask =2
def get_model_input_geo_old(geo):
    # (batch,seq,joint,9)
    assert geo.shape[-1]==9
    data=geo.flatten(start_dim=-2)
    return data

def get_model_input_geo(geo):
    # (batch,seq,joint,9)
    assert geo.shape[-1]==9
    part1 = geo[..., :6].flatten(start_dim=-2)
    part2 = geo[..., 6:].flatten(start_dim=-2)
    data = torch.cat([part1, part2], dim=-1)
    return data
def from_flat_data_joint_data(x):
    # x:(b,seq,c)
    r=x[...,:168].reshape((*x.shape[:-1],28,6))
    p=x[...,168:].reshape((*x.shape[:-1],28,3))
    print(f"in from-flat-to-joint: r.shape->{r.shape}")
    return p,r
def get_model_input(positions, rotations):
    # positions: (batch, seq, joint, 3)
    # rotation: (batch, seq, joint, 3, 3)
    # return (batch, seq, joint*6+3)

    rot_6d = rotations
    rot = rot_6d.flatten(start_dim=-2)
    pos=positions.flatten(start_dim=-2)

    assert len(rot.shape)==len(pos.shape)

    # x = torch.cat([rot, positions[:, :, 0, :]], dim=-1)
    x = torch.cat([rot, pos], dim=-1) # NOTICE:已经将原来的root pos改为所有pos
    return x
def get_model_input_sp(positions, rotations):
    x = torch.cat([rotations,positions],dim=-1)
    return x
def get_model_output(model, state_zscore, keyframe_pos_idx, data_mask, atten_mask,
                     foot_contact, seq_slice, c_slice, rp_slice):
    data_mask = data_mask.expand(*state_zscore.shape[:-1], data_mask.shape[-1])
    # model_out = model(
    #     torch.cat([state_zscore, data_mask], dim=-1), mask=atten_mask)
    model_out= model(torch.cat([state_zscore, data_mask], dim=-1), keyframe_pos_idx, mask=atten_mask)
    # print("c_slice:{}".format(c_slice))
    c_out = foot_contact.clone().detach()
    c_out[..., seq_slice, :] = torch.sigmoid(
        model_out[..., seq_slice, c_slice])

    state_out = state_zscore.clone().detach()
    state_out[..., seq_slice, :] = model_out[..., seq_slice, rp_slice]

    return state_out,c_out

def get_train_stats(config, use_cache=True, stats_folder=None,
                    dataset_name="train_stats"):
    context_len = config["train"]["context_len"]
    if stats_folder is None:
        stats_folder = config["workspace"]
    stats_path = os.path.join(stats_folder, "train_stats_context.pkl")

    if use_cache and os.path.exists(stats_path):
        with open(stats_path, "rb") as fh:
            train_stats = pickle.load(fh)
        print("Train stats load from {}".format(stats_path))
    else:
        # calculate training stats
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset, data_loader = train_utils.init_bvh_dataset(
            config, dataset_name, device, shuffle=True, dtype=torch.float64)

        input_data = []
        for i, data in enumerate(data_loader, 0):
            (positions, rotations, names, frame_nums, trends, geo, remove_idx, data_idx) = data    # FIXED:返回需要改改
            # parents = parents[0]
            rotations = data_utils.matrix9D_to_6D_torch(rotations) # get-input需要的是6d
            # positions, rotations = data_utils.to_start_centered_data(positions, rotations, context_len)  # BUG:不需要这一句
            x = get_model_input_sp(positions, rotations)
            input_data.append(x.cpu().numpy())

        input_data = np.concatenate(input_data, axis=0)
        print("in <get_train_stats> input shape:{}".format(input_data.shape))
        mean = np.mean(input_data, axis=(0, 1))
        std = np.std(input_data, axis=(0, 1))

        train_stats = {
            "mean": mean,
            "std": std
        }

        with open(stats_path, "wb") as fh:
            pickle.dump(train_stats, fh)

        print("Train stats wrote to {}".format(stats_path))

    return train_stats["mean"], train_stats["std"]

def get_train_stats_torch(config, dtype, device,
                          use_cache=True, stats_folder=None,
                          dataset_name="train_stats"):
    mean, std = get_train_stats(config, use_cache, stats_folder, dataset_name)
    mean = torch.tensor(mean, dtype=dtype, device=device)
    std = torch.tensor(std, dtype=dtype, device=device)
    return mean, std


def get_midway_targets(seq_slice, midway_targets_amount, midway_targets_p):
    targets = set()
    trans_len = seq_slice.stop - seq_slice.start
    midway_targets_amount = int(midway_targets_amount * trans_len)
    for i in range(midway_targets_amount):
        if random.random() < midway_targets_p:
            targets.add(random.randrange(seq_slice.start, seq_slice.stop))
    return list(targets)


def get_attention_mask(window_len, context_len, target_idx, device,
                       midway_targets=()):
    global ATTENTION_MODE
    atten_mask = torch.ones(window_len, window_len,
                            device=device, dtype=torch.bool)
    # old-1:遮住未知项
    if ATTENTION_MODE=="VANILLA":
        atten_mask[:, target_idx] = False
        atten_mask[:, :context_len] = False
        atten_mask[:, midway_targets] = False

    # new:对角线下不被遮罩
    if ATTENTION_MODE=="PRE":
        # print(ATTENTION_MODE)
        atten_mask.triu_(diagonal=1)
        atten_mask[:, target_idx] = False
        atten_mask[:, :context_len] = False
        atten_mask[:, midway_targets] = False
    
    # old-2:全部不遮罩,常用模式
    if ATTENTION_MODE=="NOMASK":
        atten_mask[:, :target_idx + 1] = False
    if ATTENTION_MODE=="SQUARE":
        atten_mask[:target_idx + 1, :target_idx + 1] = False
    # assert context_len==13
    atten_mask = atten_mask.unsqueeze(0)

    # (1, seq, seq)
    return atten_mask

def get_spattention_mask(device,remove_idx=[]):
    atten_mask = torch.zeros(28,28,
                            device=device, dtype=torch.bool)
    
    atten_mask[:,remove_idx] = True
    atten_mask = atten_mask.unsqueeze(0)
    # (1, seq, seq)
    return atten_mask

def get_data_mask(window_len, d_mask, constrained_slices,
                  context_len, target_idx, device, dtype,
                  midway_targets=()):
    # 0 for unknown and 1 for known
    # print("get_data_mask_constraint:{}".format(constrained_slices))
    data_mask = torch.zeros((window_len, d_mask), device=device, dtype=dtype)
    data_mask[:context_len, :] = 1
    data_mask[target_idx, :] = 1
    # NOTICE: 可以设置config中的d_mask和constrained_slice来固定特定牙齿位置
    for s in constrained_slices:
        # print("get_data_mask:{}".format(s))
        data_mask[midway_targets, s] = 1

    # (seq, d_mask)
    return data_mask

def get_data_mask_sp(window_len, d_mask, constrained_slices,
                  context_len, target_idx, device, dtype,
                  midway_targets=()):
    # 0 for unknown and 1 for known
    # print("get_data_mask_constraint:{}".format(constrained_slices))
    data_mask = torch.zeros((window_len, 28,d_mask), device=device, dtype=dtype)
    data_mask[:context_len,:, :] = 1
    data_mask[target_idx, :,:] = 1
    # NOTICE: 可以设置config中的d_mask和constrained_slice来固定特定牙齿位置
    for s in constrained_slices:
        # print("get_data_mask:{}".format(s))
        data_mask[midway_targets,:, s] = 1

    # (seq, d_mask)
    return data_mask
def get_keyframe_pos_indices(window_len, seq_slice, dtype, device):
    # position index relative to context and target frame
    ctx_idx = torch.arange(window_len, dtype=dtype, device=device)
    ctx_idx = ctx_idx - (seq_slice.start - 1)
    ctx_idx = ctx_idx[..., None]

    tgt_idx = torch.arange(window_len, dtype=dtype, device=device)
    tgt_idx = -(tgt_idx - seq_slice.stop)
    tgt_idx = tgt_idx[..., None]

    # ctx_idx: (seq, 1), tgt_idx: (seq, 1)
    keyframe_pos_indices = torch.cat([ctx_idx, tgt_idx], dim=-1)

    # (1, seq, 2)
    return keyframe_pos_indices[None]


def set_placeholder_root_pos(x, seq_slice, midway_targets, p_slice,r_slice=None):
    # set root position of missing part to linear interpolation of
    # root position between constrained frames (i.e. last context frame,
    # midway target frames and target frame).
    global INIT_INTERP
    constrained_frames = [seq_slice.start - 1, seq_slice.stop]
    constrained_frames.extend(midway_targets)
    constrained_frames.sort()
    for i in range(len(constrained_frames) - 1):
        start_idx = constrained_frames[i]
        end_idx = constrained_frames[i + 1]
        start_slice = slice(start_idx, start_idx + 1)
        end_slice = slice(end_idx, end_idx + 1)
        inbetween_slice = slice(start_idx + 1, end_idx)

        x[..., inbetween_slice, p_slice] = \
            benchmark.get_linear_interpolation(
                x[..., start_slice, p_slice],
                x[..., end_slice, p_slice],
                end_idx - start_idx - 1
        )

    return x
def set_placeholder_root_pos_sp(x, seq_slice, midway_targets, p_slice):
    # set root position of missing part to linear interpolation of
    # root position between constrained frames (i.e. last context frame,
    # midway target frames and target frame).
    p_slice = slice(6,9)
    constrained_frames = [seq_slice.start - 1, seq_slice.stop]
    constrained_frames.extend(midway_targets)
    constrained_frames.sort()
    for i in range(len(constrained_frames) - 1):
        start_idx = constrained_frames[i]
        end_idx = constrained_frames[i + 1]
        start_slice = slice(start_idx, start_idx + 1)
        end_slice = slice(end_idx, end_idx + 1)
        inbetween_slice = slice(start_idx + 1, end_idx)

        x[..., inbetween_slice,:, p_slice] = \
            benchmark.get_linear_interpolation2(
                x[..., start_slice,:, p_slice],
                x[..., end_slice,:,p_slice],
                end_idx - start_idx - 1
        )
    return x
def get_interp_pos_rot(pos,rot9d,seq_slice, midway_targets=[]):
    global SEQNUM_GEO
    constrained_frames = [seq_slice.start - 1, seq_slice.stop]
    constrained_frames.extend(midway_targets)
    constrained_frames.sort()
    inter_pos = pos.clone()
    inter_rot9d = rot9d.clone()
    for i in range(len(constrained_frames) - 1):
        start_idx = constrained_frames[i]-SEQNUM_GEO
        end_idx = constrained_frames[i + 1]-SEQNUM_GEO
        start_slice = slice(start_idx, start_idx + 1)
        end_slice = slice(end_idx, end_idx + 1)
        inbetween_slice = slice(start_idx, end_idx+1)
        inter_slice =slice(1,end_idx-start_idx)
        inter_pos[...,inbetween_slice,:,:],inter_rot9d[...,inbetween_slice,:,:,:] = benchmark.get_interpolated_local_pos_rot(pos[...,inbetween_slice,:,:], rot9d[...,inbetween_slice,:,:,:], inter_slice)

    return inter_pos,inter_rot9d


def get_new_positions(positions, y, seq_slice=slice(None, None)):
    p_slice = slice(6,9)
    pos_tmp=y[..., seq_slice,:, p_slice]
    positions_new = positions.clone()
    positions_new[..., seq_slice, :, :] = pos_tmp    # FIXED:注意维度

    return positions_new


def get_new_rotations(y, rotations=None, seq_slice=slice(None, None)):
    r_slice = slice(0,6)

    if rotations is None:
        rot = y[..., r_slice]
        # rot = rot.reshape(*rot.shape[:-1], -1, 6)
        rot = data_utils.matrix6D_to_9D_torch(rot)
    else:
        rot_tmp = y[..., seq_slice,:, r_slice]
        # rot_tmp = rot_tmp.reshape(*rot_tmp.shape[:-1], -1, 6)
        rot_tmp = data_utils.matrix6D_to_9D_torch(rot_tmp)

        rot = rotations.clone()
        rot[..., seq_slice, :, :, :] = rot_tmp

    return rot

def get_new_rotations6D(y, rotations=None, seq_slice=slice(None, None)):
    r_slice = slice(0,6)

    if rotations is None:
        rot = y[..., r_slice]
    else:
        rot = y[..., seq_slice,:, r_slice]
    return rot
def train(config):
    global ATTENTION_MODE
    global INIT_INTERP
    global SEQNUM_GEO
    global zscore_MODE
    global Data_Mask_MODE
    ATTENTION_MODE = config["train"]["attention_mode"]
    INIT_INTERP = config["train"]["init_interp"]
    zscore_MODE = config["train"]["zscore_MODE"]    # normal/none/seq
    Data_Mask_MODE = config["train"]["data_mask_set"] 
    print(ATTENTION_MODE,INIT_INTERP,zscore_MODE)
    indices = config["indices"]
    info_interval = config["visdom"]["interval"]
    eval_interval = config["visdom"]["interval_eval"]
    eval_trans = config["visdom"]["eval_trans"]
    
    rp_slice = slice(indices["r_start_idx"], indices["p_end_idx"])
    r_slice = slice(indices["r_start_idx"], indices["r_end_idx"])
    p_slice = slice(indices["p_start_idx"], indices["p_end_idx"])
    c_slice = slice(indices["c_start_idx"], indices["c_end_idx"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, "train", device, shuffle=True,add_geo=config["train"]["add_geo"])  # FIXED:数据集载入
    dtype = dataset.dtype

    bench_dataset_names, _, bench_data_loaders = \
        train_utils.get_benchmark_datasets(config, device, shuffle=False,add_geo=config["train"]["add_geo"])   # FIXED:验证集载入

    _,val_dataloaders = train_utils.get_val_datasets(config,device,eval_trans,shuffle=False, dtype=dataset.dtype,add_geo=config["train"]["add_geo"])
    # visualization
    vis, info_idx = train_utils.init_visdom(config)

    # initialize model
    model = STTransformer(config["model"]).to(device)

    # initialize optimizer
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"])

    # learning rate scheduler
    scheduler = train_utils.get_noam_lr_scheduler(config, optimizer)

    # load checkpoint
    epoch, iteration = train_utils.load_checkpoint(
        config, model, optimizer, scheduler)

    # training stats
    mean, std = get_train_stats_torch(config, dtype, device)    
    print("mean shape:",mean.shape)
    add_geo_FLAG = config["train"]["add_geo"]
    if add_geo_FLAG:
        SEQNUM_GEO = 12
    else:
        SEQNUM_GEO = 0
    window_len = config["datasets"]["train"]["window"] + SEQNUM_GEO
    context_len = config["train"]["context_len"] + SEQNUM_GEO
    fill_mode = config["datasets"]["train"]["fill_mode"]
    fill_value = torch.zeros(mean.shape,dtype=dtype,device=device)
    if fill_mode=="missing-mean" or fill_mode=="vacant-mean":
        fill_value=mean
        print("YES-MEAN")
    fill_value_p,fill_value_r6d=fill_value[:,6:],fill_value[:,:6]#FIXME#from_flat_data_joint_data(fill_value) 
    min_trans = config["train"]["min_trans"]
    max_trans = config["train"]["max_trans"]
    midway_targets_amount = config["train"]["midway_targets_amount"]
    midway_targets_p = config["train"]["midway_targets_p"]

    loss_avg = 0
    p_loss_avg = 0
    r_loss_avg = 0
    smooth_loss_avg = 0
    c_loss_avg = 0
    f_loss_avg = 0

    min_benchmark_loss = float("inf")
    inter_pos,inter_rot9d,inter_rot6d = None,None,None
    while epoch < config["train"]["total_epoch"]:
        for i, data in enumerate(data_loader, 0):
            (positions, rotations, names, frame_nums, trends, geo, remove_idx, data_idx) = data 
            # trans
            min_trans = min(frame_nums)-2
            prob =random.uniform(0,1)
            trans_len = int(min_trans + math.sqrt(prob)*(max_trans-min_trans))
            trans_len = max_trans if trans_len>max_trans else trans_len
            target_idx = context_len + trans_len
            seq_slice = slice(context_len, target_idx)
            
            # get random midway target frames
            midway_targets = get_midway_targets(
                seq_slice, midway_targets_amount, midway_targets_p)
            if INIT_INTERP!="POS-ONLY":
                inter_pos, inter_rot9d = get_interp_pos_rot(positions, rotations, seq_slice, midway_targets)
                inter_rot6d = data_utils.matrix9D_to_6D_torch(inter_rot9d)
                assert (inter_pos==positions).all()==False
            rot_6d = data_utils.matrix9D_to_6D_torch(rotations) # get-input需要的是6d
            if add_geo_FLAG:
                trends=torch.cat([torch.zeros([trends.shape[0],SEQNUM_GEO,trends.shape[-1]],dtype=dtype,device=device),
                                trends],
                                dim=1
                                )
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
            if INIT_INTERP!="POS-ONLY":
                for j in range(remove_len):
                    remove_list=remove_idx[j]
                    inter_pos[j,:,remove_list,:]=fill_value_p[remove_list,:]
                    inter_rot6d[j,:,remove_list,:]=fill_value_r6d[remove_list,:]
                inter_x = get_model_input(inter_pos,inter_rot6d)
                inter_x_zs = (inter_x - mean) / std
            global_rotations = rotations
            global_positions = positions

            # randomize transition length
            # trans_len = random.randint(min_trans, max_trans)# random.choice([random.randint(min_trans, max_trans),min(frame_nums)-2])
            
            # max_trans_ = random.choice([max_trans,min(frame_nums)-2])
            # trans_len = max_trans_
            # if min_trans>= max_trans_:
            #     trans_len = max_trans_
            # else:
            #     trans_len = random.randint(min_trans, max_trans_)

            # attention mask
            atten_mask = get_attention_mask(
                window_len, context_len, target_idx, device,
                midway_targets=midway_targets)
            
            # data mask
            data_mask = get_data_mask_sp(
                window_len, model.d_mask, model.constrained_slices,
                context_len, target_idx, device, dtype, midway_targets)#FIXME

            # position index relative to context and target frame
            keyframe_pos_idx = get_keyframe_pos_indices(
                window_len, seq_slice, dtype, device)

            # prepare model input
            x_gt = get_model_input_sp(positions, rot_6d)#FIXME
            x_gt_zscore = (x_gt - mean) / std
            if add_geo_FLAG:
                geo_ctrl = geo#get_model_input_geo(geo)   # GEO: (BATCH,SEQ,JOINT*9)#FIXME
                if zscore_MODE=="seq":
                    x_gt_ = (x_gt - mean) / std
                    x_gt_zscore = torch.cat([geo_ctrl,x_gt_],dim=1)
                x_gt = torch.cat([geo_ctrl,x_gt],dim=1)    # GEO: (BATCH,SEQ*,DIMS)
            
            if zscore_MODE=="normal":
                x_gt_zscore = (x_gt - mean) / std
            elif zscore_MODE=="no":
                x_gt_zscore = x_gt
            x = None
            if Data_Mask_MODE==0:
                x = x_gt_zscore * data_mask[...,:1]
            else:
                x = torch.cat([
                    x_gt_zscore * data_mask[...,:1],
                    data_mask.expand(*x_gt_zscore.shape[:-1], data_mask.shape[-1])
                ], dim=-1)
            if Data_Mask_MODE==2:
                x[...,:12,-1]=2
            x = set_placeholder_root_pos_sp(x, seq_slice, midway_targets, p_slice)#FIXME
            if INIT_INTERP!="POS-ONLY":
                x[...,SEQNUM_GEO:,rp_slice]=inter_x_zs
            # calculate model output y
            optimizer.zero_grad()
            model.train()

            model_out = model(x, keyframe_pos_idx, mask=atten_mask)
            if config["train"]["trends_loss"]:
                c_out = trends.clone().detach()
                c_out[..., seq_slice, :] = torch.sigmoid(
                    model_out[..., seq_slice, c_slice]) 
            y = x_gt_zscore.clone().detach()    # BUG:x_gt是含有joint
            #tmp_out = model_out[..., seq_slice, :]
            y[..., seq_slice, :,:] = model_out[..., seq_slice, :,:]#tmp_out.reshape((*tmp_out.shape[:-1],28,9))


            if zscore_MODE!="no":#FIXME
                y = y * std + mean
            #y = get_model_input(y[...,6:],y[...,:6])

            if add_geo_FLAG:
                positions = torch.cat([torch.zeros([positions.shape[0],SEQNUM_GEO,*positions.shape[2:]],dtype=dtype,device=device),
                                        positions],
                                        dim=1
                                        ) # GEO
                global_positions = torch.cat([torch.zeros([global_positions.shape[0],SEQNUM_GEO,*global_positions.shape[2:]],dtype=dtype,device=device),
                                              global_positions],
                                             dim=1) # GEO
            pos_new = get_new_positions(positions, y)#train_utils.get_new_positions(positions, y, indices) # FIXED:似乎只考虑了root pos return (batch,seq,joint,3)
            #rot_new = train_utils.get_new_rotations(y, indices)
            
            rot_6d_new = get_new_rotations6D(y)#train_utils.get_new_rotations6D(y,indices)
            foot_state=torch.cat([pos_new,rot_6d_new],dim=-1)  
            
            
            r_loss = train_utils.cal_r_loss_sp(x_gt, y, seq_slice, indices)#FIXME
            # smooth_loss = train_utils.cal_smooth_loss(gpos_new, seq_slice)
            smooth_loss = train_utils.cal_smooth_loss(pos_new, seq_slice)   # FIXME:rot要不要加进去
            # p_loss = train_utils.cal_p_loss(global_positions, gpos_new, seq_slice)
            p_loss = train_utils.cal_p_loss(global_positions, pos_new, seq_slice)

            loss=None
            if config["train"]["trends_loss"]:
                c_loss = train_utils.cal_c_loss(trends, c_out, seq_slice)
                f_loss = train_utils.cal_f_loss(foot_state, c_out, seq_slice)
                loss = (
                    config["weights"]["rw"] * r_loss +
                    config["weights"]["pw"] * p_loss +
                    config["weights"]["cw"] * c_loss +
                    config["weights"]["fw"] * f_loss +
                    config["weights"]["sw"] * smooth_loss
                )
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                r_loss_avg += r_loss.item()
                p_loss_avg += p_loss.item()
                c_loss_avg += c_loss.item()
                f_loss_avg += f_loss.item()
                smooth_loss_avg += smooth_loss.item()
            else:
                # loss
                loss = (
                    config["weights"]["rw"] * r_loss +
                    config["weights"]["pw"] * p_loss +
                    config["weights"]["sw"] * smooth_loss
                )
                loss.backward()
                optimizer.step()
                scheduler.step()

                r_loss_avg += r_loss.item()
                p_loss_avg += p_loss.item()
                smooth_loss_avg += smooth_loss.item()

            loss_avg += loss.item()

            if iteration % config["train"]["checkpoint_interval"] == 0:
                train_utils.save_checkpoint(config, model, epoch, iteration,
                                            optimizer, scheduler)
                vis.save([config["visdom"]["env"]])

            if iteration % info_interval == 0:
                r_loss_avg /= info_interval
                p_loss_avg /= info_interval
                smooth_loss_avg /= info_interval
                c_loss_avg /= info_interval
                f_loss_avg /= info_interval
                loss_avg /= info_interval
                lr = optimizer.param_groups[0]["lr"]

                print("Epoch: {}, Iteration: {}, lr: {:.8f}, "
                      "loss: {:.6f}, r: {:.6f}, p: {:.6f}, "
                      "smooth: {:.6f}, c: {:.6f}, f: {:.6f}".format(
                          epoch, iteration, lr, loss_avg,
                          r_loss_avg, p_loss_avg, smooth_loss_avg,c_loss_avg, f_loss_avg))

                contents = [
                    ["loss", "r_loss", r_loss_avg],
                    ["loss", "p_loss", p_loss_avg],
                    ["loss", "smooth_loss", smooth_loss_avg],
                    ["loss weighted", "r_loss",
                        r_loss_avg * config["weights"]["rw"]],
                    ["loss weighted", "p_loss",
                        p_loss_avg * config["weights"]["pw"]],
                    ["loss weighted", "smooth_loss",
                        smooth_loss_avg * config["weights"]["sw"]],
                    ["loss weighted", "loss", loss_avg],
                    ["learning rate", "lr", lr],
                    ["epoch", "epoch", epoch],
                    ["iterations", "iterations", iteration],
                ]
                if config["train"]["trends_loss"]:
                    contents.extend([
                        ["loss", "c_loss", c_loss_avg],
                        ["loss", "f_loss", f_loss_avg],
                        ["loss weighted", "c_loss",
                            c_loss_avg * config["weights"]["cw"]],
                        ["loss weighted", "f_loss",
                            f_loss_avg * config["weights"]["fw"]],
                    ])

                if iteration % eval_interval == 0:
                    for trans in eval_trans:
                        print("trans: {}\n".format(trans))

                        for i in range(len(bench_data_loaders)):
                            ds_name = bench_dataset_names[i]
                            ds_loader = bench_data_loaders[i]

                            gpos_loss, gquat_loss, npss_loss, val_ploss,val_smoothloss = eval_on_dataset(
                                config, ds_loader, model, trans)

                            contents.extend([
                                [ds_name, "gpos_{}".format(trans),
                                 gpos_loss],
                                [ds_name, "gquat_{}".format(trans),
                                 gquat_loss],
                                [ds_name, "npss_{}".format(trans),
                                 npss_loss],
                                [ds_name, "val_p_{}".format(trans),
                                 val_ploss],
                                [ds_name, "val_s_{}".format(trans),
                                 val_smoothloss], 
                            ])
                            print("{}:\ngpos: {:6f}, gquat: {:6f}, "
                                  "npss: {:.6f}, p_loss: {:.6f}, smooth_loss: {:.6f}".format(ds_name, gpos_loss,
                                                        gquat_loss, npss_loss,val_ploss,val_smoothloss))

                            if ds_name == "benchmark":
                                # After iterations, benchmark_loss will be the
                                # sum of losses on dataset named "benchmark"
                                # with transition length equals to last value
                                # in eval_interval.
                                benchmark_loss = (gpos_loss + gquat_loss +
                                                  npss_loss)

                    if min_benchmark_loss > benchmark_loss:
                        min_benchmark_loss = benchmark_loss
                        # save min loss checkpoint
                        train_utils.save_checkpoint(
                            config, model, epoch, iteration,
                            optimizer, scheduler, suffix=f".{iteration}min")
                if iteration % (eval_interval*10)==0:
                    for i in range(len(val_dataloaders)):
                            ds_name = "val"
                            ds_loader = val_dataloaders[i]
                            trans = ds_loader.dataset.window-2
                            gpos_loss, gquat_loss, npss_loss, val_ploss,val_smoothloss = eval_on_dataset(
                                config, ds_loader, model, trans)

                            contents.extend([
                                [ds_name, "gpos_{}".format(trans),
                                 gpos_loss],
                                [ds_name, "gquat_{}".format(trans),
                                 gquat_loss],
                                [ds_name, "npss_{}".format(trans),
                                 npss_loss],
                                [ds_name, "val_p_{}".format(trans),
                                 val_ploss],
                                [ds_name, "val_s_{}".format(trans),
                                 val_smoothloss], 
                            ])
                            print("{}-{}:\ngpos: {:6f}, gquat: {:6f}, "
                                  "npss: {:.6f}, p_loss: {:.6f}, smooth_loss: {:.6f}".format(ds_name, trans, gpos_loss,
                                                        gquat_loss, npss_loss,val_ploss,val_smoothloss))
                        
                train_utils.to_visdom(vis, info_idx, contents)
                r_loss_avg = 0
                p_loss_avg = 0
                smooth_loss_avg = 0
                c_loss_avg = 0
                f_loss_avg = 0
                loss_avg = 0
                info_idx += 1

            iteration += 1
        
        epoch += 1
        if epoch==200:
            train_utils.save_checkpoint(
                            config, model, epoch, iteration,
                            optimizer, scheduler, suffix=".200emin")


def eval_on_dataset(config, data_loader, model, trans_len,
                    debug=False, post_process=False):
    global INIT_INTERP
    global SEQNUM_GEO
    device = data_loader.dataset.device
    dtype = data_loader.dataset.dtype
    window_len = data_loader.dataset.window + SEQNUM_GEO

    indices = config["indices"]
    context_len = config["train"]["context_len"] + SEQNUM_GEO
    target_idx = context_len + trans_len
    seq_slice = slice(context_len, target_idx)

    mean, std = get_train_stats_torch(config, dtype, device)
    mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
        config, dtype, device)  # FIXME:注意dataset，mean是用于使数据与模型匹配，mean_rmi是计算rmi_loss

    fill_mode = config["datasets"]["train"]["fill_mode"]
    fill_value = torch.zeros(mean.shape,dtype=dtype,device=device)
    if fill_mode=="missing-mean" or fill_mode=="vacant-mean":
        fill_value=mean
    fill_value_p,fill_value_r6d=fill_value[:,6:],fill_value[:,:6]#FIXME#from_flat_data_joint_data(fill_value) 
    
    # attention mask
    atten_mask = get_attention_mask(
        window_len, context_len, target_idx, device)

    data_indexes = []
    gpos_loss = []
    gquat_loss = []
    npss_loss = []
    npss_weights = []

    p_loss_avg = []
    smooth_loss_avg = []
    add_geo_FLAG = config["train"]["add_geo"]
    if add_geo_FLAG==False:
        assert SEQNUM_GEO==0
    for i, data in enumerate(data_loader, 0):
        (positions, rotations, names, frame_nums, trends, geo, remove_idx, data_idx) = data # FIXED:返回类型
        if INIT_INTERP!="POS-ONLY":
                inter_pos, inter_rot9d = get_interp_pos_rot(positions, rotations, seq_slice)
                inter_rot6d = data_utils.matrix9D_to_6D_torch(inter_rot9d)
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
        if INIT_INTERP!="POS-ONLY":
                for j in range(remove_len):
                    remove_list=remove_idx[j]
                    inter_pos[j,:,remove_list,:]=fill_value_p[remove_list,:]
                    inter_rot6d[j,:,remove_list,:]=fill_value_r6d[remove_list,:]
                inter_x = get_model_input(inter_pos,inter_rot6d)
                inter_x_zs = (inter_x - mean) / std
        # notice: rotations参与计算loss了
        rotations = data_utils.matrix6D_to_9D_torch(rot_6d)
        pos_new, rot_new = evaluate(
            model, positions, rot_6d, seq_slice,
            indices, mean, std, atten_mask, post_process,geo=geo,inter_x_zs = None)
        if add_geo_FLAG:
            positions=torch.cat([torch.zeros([positions.shape[0],SEQNUM_GEO,*positions.shape[2:]],dtype=dtype,device=device),positions],dim=1) # GEO
            rotations=torch.cat([torch.zeros([rotations.shape[0],SEQNUM_GEO,*rotations.shape[2:]],dtype=dtype,device=device),rotations],dim=1) # GEO
        
        (gpos_batch_loss, gquat_batch_loss,
         npss_batch_loss, npss_batch_weights,_,_) = \
            benchmark.get_rmi_style_batch_loss(
                positions, rotations, pos_new, rot_new, None,
                context_len, target_idx, mean_rmi, std_rmi
        )   # FIXED:rmi定义的损失计算
            

        smooth_loss = train_utils.cal_smooth_loss(pos_new, seq_slice)   # FIXME:rot要不要加进去
        p_loss = train_utils.cal_p_loss(positions, pos_new, seq_slice)

        p_loss_avg.append(p_loss.item())
        smooth_loss_avg.append(smooth_loss.item())
        
        gpos_loss.append(gpos_batch_loss)
        gquat_loss.append(gquat_batch_loss)
        npss_loss.append(npss_batch_loss)
        npss_weights.append(npss_batch_weights)
        data_indexes.extend(data_idx.tolist())

    gpos_loss = np.concatenate(gpos_loss, axis=0)
    gquat_loss = np.concatenate(gquat_loss, axis=0)

    npss_loss = np.concatenate(npss_loss, axis=0)           # (batch, dim)
    npss_weights = np.concatenate(npss_weights, axis=0)
    npss_weights = npss_weights / np.sum(npss_weights)      # (batch, dim)
    npss_loss = np.sum(npss_loss * npss_weights, axis=-1)   # (batch, )

    p_loss_avg = np.array(p_loss_avg)
    smooth_loss_avg = np.array(smooth_loss_avg)
    if debug:
        total_loss = gpos_loss + gquat_loss + npss_loss
        loss_data = list(zip(
            total_loss.tolist(),
            gpos_loss.tolist(),
            gquat_loss.tolist(),
            npss_loss.tolist(),
            data_indexes
        ))
        loss_data.sort()
        loss_data.reverse()

        return gpos_loss.mean(), gquat_loss.mean(), npss_loss.sum(), loss_data
    else:
        return gpos_loss.mean(), gquat_loss.mean(), npss_loss.sum(),p_loss_avg.mean(),smooth_loss_avg.mean()


def evaluate(model, positions, rotations, seq_slice, indices,
             mean, std, atten_mask, post_process=False,
             midway_targets=(),geo=None,inter_x_zs=None):
    """
    Generate transition animation.

    positions and rotation should already been preprocessed using
    motion_inbetween.data.utils.to_start_centered_data().

    positions, shape: (batch, seq, joint, 3) and
    rotations, shape: (batch, seq, joint, 3, 3)
    is a mixture of ground truth and placeholder data.

    There are two constraint mode:
    1) fully constrained: at constrained midway target frames, input values of
        all joints should be the same as output values.
    2) partially constrained: only selected joints' input value are kept in
        output (e.g. only constrain root joint).

    At context frames, target frame and midway target frames, ground truth
    data should be provided (in partially constrained mode, only provide ground
    truth value of constrained dimensions).

    The placeholder values:
    - for positions: set to zero
    - for rotations: set to identity matrix

    Args:
        model (nn.Module): context model
        positions (tensor): (batch, seq, joint, 3)
        rotations (tensor): (batch, seq, joint, 3, 3)
        seq_slice (slice): sequence slice where motion will be predicted
        indices (dict): config which defines the meaning of input's dimensions
        mean (tensor): mean of model input
        std (tensor): std of model input
        atten_mask (tensor): (1, seq, seq) model attention mask
        post_process (bool): Whether post processing is enabled or not.
            Defaults to True.
        midway_targets (list of int): list of midway targets (constrained
            frames indexes).

    Returns:
        tensor, tensor: new positions, new rotations with predicted animation
        with same shape as input.
    """
    global zscore_MODE
    global Data_Mask_MODE
    global INIT_INTERP
    global SEQNUM_GEO
    dtype = positions.dtype
    device = positions.device
    window_len = positions.shape[-3]+SEQNUM_GEO if geo is not None else positions.shape[-3]
    context_len = seq_slice.start
    target_idx = seq_slice.stop

    rp_slice = slice(indices["r_start_idx"], indices["p_end_idx"])
    # If current context model is not trained with constrants,
    # ignore midway_targets.
    if midway_targets and not model.constrained_slices:
        print(
            "WARNING: Context model is not trained with constraints, but "
            "midway_targets is provided with values while calling evaluate()! "
            "midway_targets is ignored!"
        )
        midway_targets = []

    with torch.no_grad():
        model.eval()

        if midway_targets:
            midway_targets.sort()
            atten_mask = atten_mask.clone().detach()
            atten_mask[0, :, midway_targets] = False

        # prepare model input          
        geo_ctrl=None
        x_orig = get_model_input_sp(positions, rotations)#FIXME
        x_zscore = (x_orig - mean) / std
        if geo is not None:
            geo_ctrl=geo #FIXME#get_model_input_geo(geo)   # GEO: (BATCH,SEQ,JOINT*9)
            if zscore_MODE=="seq":
                x_ = (x_orig - mean) / std
                x_zscore = torch.cat([geo_ctrl,x_],dim=1)
            x_orig=torch.cat([geo_ctrl,x_orig],dim=1)    # GEO: (BATCH,SEQ*,DIMS)
        # zscore
        if zscore_MODE=="normal":
            x_zscore = (x_orig - mean) / std
        elif zscore_MODE=="no":
            x_zscore = x_orig

        # data mask (seq, 1)
        data_mask = get_data_mask_sp(
            window_len, model.d_mask, model.constrained_slices, context_len,
            target_idx, device, dtype, midway_targets) #FIXME

        keyframe_pos_idx = get_keyframe_pos_indices(
            window_len, seq_slice, dtype, device)
        x = None
        if Data_Mask_MODE==0:
            x = x_zscore * data_mask[...,:1]
        else:
            x = torch.cat([
                x_zscore * data_mask[...,:1],
                data_mask.expand(*x_zscore.shape[:-1], data_mask.shape[-1])
            ], dim=-1)
        if Data_Mask_MODE==2:
            x[...,:SEQNUM_GEO,-1]=2

        p_slice = slice(indices["p_start_idx"], indices["p_end_idx"])
        r_slice = slice(indices["r_start_idx"], indices["r_end_idx"])
        x = set_placeholder_root_pos_sp(x, seq_slice, midway_targets, p_slice)#FIXME
        if INIT_INTERP!="POS-ONLY":
            x[...,SEQNUM_GEO:,rp_slice]=inter_x_zs
        # calculate model output y
        model_out = model(x, keyframe_pos_idx, mask=atten_mask)
        y = x_zscore.clone().detach()
        #y[..., seq_slice, :] = model_out[..., seq_slice, rp_slice]

        #tmp_out = model_out[..., seq_slice, :]
        y[..., seq_slice, :,:] = model_out[..., seq_slice, :,:]#tmp_out.reshape((*tmp_out.shape[:-1],28,9))

        if post_process:
            y = train_utils.anim_post_process(y, x_zscore, seq_slice)

        # reverse zscore
        if zscore_MODE!="no":#FIXME
            y = y * std + mean
        #y = get_model_input(y[...,6:],y[...,:6])
        # notice: 原来的版本中rotations一直是9D的，新版本输入evaluation的是6D的，因此转化一下
        rotations = data_utils.matrix6D_to_9D_torch(rotations)
        # GEO:
        if geo is not None:
            positions=torch.cat([torch.zeros([positions.shape[0],SEQNUM_GEO,*positions.shape[2:]],dtype=dtype,device=device),positions],dim=1) # GEO
            rotations=torch.cat([torch.zeros([rotations.shape[0],SEQNUM_GEO,*rotations.shape[2:]],dtype=dtype,device=device),rotations],dim=1) # GEO
        # new pos and rot
        pos_new = get_new_positions(positions,y,seq_slice)
        rot_new = get_new_rotations(y,rotations,seq_slice)
        # pos_new = train_utils.get_new_positions(
        #     positions, y, indices, seq_slice)
        # rot_new = train_utils.get_new_rotations(
        #     y, indices, rotations, seq_slice)

        return pos_new, rot_new