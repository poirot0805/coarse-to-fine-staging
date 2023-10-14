import os
import math
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from motion_inbetween.model import ContextTransformer, DetailTransformer
from motion_inbetween.data import utils_torch as data_utils
from motion_inbetween.train import rmi
from motion_inbetween.train import utils as train_utils
from motion_inbetween.train import context_model as ctx_mdl
from motion_inbetween.benchmark import get_rmi_style_batch_loss

SEQNUM_GEO=12

def from_flat_data_joint_data(x):
    # x:(b,seq,c)
    r=x[...,:168].reshape((*x.shape[:-1],28,6))
    p=x[...,168:].reshape((*x.shape[:-1],28,3))
    # print(f"in from-flat-to-joint: r.shape->{r.shape}")
    return p,r
def get_model_input_geo(geo):
    # (batch,seq,joint,9)
    assert geo.shape[-1]==9
    part1 = geo[..., :6].flatten(start_dim=-2)
    part2 = geo[..., 6:].flatten(start_dim=-2)
    data = torch.cat([part1, part2], dim=-1)
    return data
def get_pose_from_trends(state,trends,seq_slice):
    front_seq=slice(seq_slice.start-1,seq_slice.stop-1)

    state_new = (
        state[...,seq_slice,:,:] +
        (1-trends[..., seq_slice, :, :].detach()) * (state[...,front_seq,:,:]-state[...,seq_slice,:,:])        
    )
    
    return state_new

def get_model_input(positions, rotations):
    """
    Get detail model input. Including state, delta, offset.

    Args:
        positions (tensor): (batch, seq, joint, 3)
        rotations (tensor): (batch, seq, joint, 3, 3)

    Returns:
        tuple: state, delta
            state: (batch, seq, joint*3+3).
                Describe joint rotation and root position.
            delta: (batch, seq, joint*3+3), shape same as state.
                Delta between current frame's state and previous frame's state.
    """
    # state
    rot_6d = rotations# data_utils.matrix9D_to_6D_torch(rotations)
    rot = rot_6d.flatten(start_dim=-2)
    pos=positions.flatten(start_dim=-2)

    assert len(rot.shape)==len(pos.shape)

    state = torch.cat([rot, pos], dim=-1) # NOTICE:已经将原来的root pos改为所有pos
    # state = torch.cat([rot, positions[:, :, 0, :]], dim=-1)

    zeros_shape = list(state.shape)
    zeros_shape[-2] = 1
    zeros = torch.zeros(*zeros_shape, dtype=state.dtype, device=state.device)

    # delta
    delta = state[..., 1:, :] - state[..., :-1, :]
    # Pad zero on the first frame for shape consistency
    delta = torch.cat([zeros, delta], dim=-2)

    return state, delta


def get_model_output(detail_model, state_zscore, data_mask, atten_mask,
                     foot_contact, seq_slice, c_slice, rp_slice,flag=True):
    print(f"state shape:{state_zscore.shape} / datamask shape:{data_mask.shape}")
    data_mask = data_mask.expand(*state_zscore.shape[:-1], data_mask.shape[-1])
    model_out = detail_model(
        torch.cat([state_zscore, data_mask], dim=-1), mask=atten_mask)
    # print("c_slice:{}".format(c_slice))
    c_out = foot_contact.clone().detach()
    if flag:
        c_out[..., seq_slice, :] = torch.sigmoid(
            model_out[..., seq_slice, c_slice])

    state_out = state_zscore.clone().detach()
    state_out[..., seq_slice, :] = model_out[..., seq_slice, rp_slice]

    return state_out,c_out


def get_discriminator_input(state, foot_contact, p_slice):
    # change root position to root velocity
    root_vel = state[..., 1:, p_slice] - state[..., :-1, p_slice]

    # Pad zero on the first frame for shape consistency
    zero_shape = list(root_vel.shape)
    zero_shape[-2] = 1
    zeros = torch.zeros(*zero_shape, dtype=state.dtype, device=state.device)
    root_vel = torch.cat([zeros, root_vel], dim=-2)

    dis_state = state.clone()
    dis_state[..., :, p_slice] = root_vel

    dis_input = dis_state
    return dis_input


def get_train_stats(config, use_cache=True, stats_folder=None,
                    dataset_name="train_stats"):
    context_len = config["train"]["context_len"]
    if stats_folder is None:
        stats_folder = config["workspace"]
    stats_path = os.path.join(stats_folder, "train_stats_detail.pkl")

    if use_cache and os.path.exists(stats_path):
        with open(stats_path, "rb") as fh:
            train_stats = pickle.load(fh)
        print("Train stats load from {}".format(stats_path))
    else:
        # calculate training stats of state, delta
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        dataset, data_loader = train_utils.init_bvh_dataset(
            config, dataset_name, device, shuffle=True, dtype=torch.float64)

        state_data = []
        delta_data = []

        for i, data in enumerate(data_loader, 0):
            # (positions, rotations, global_positions, global_rotations,
            #  foot_contact, parents, data_idx) = data
            # parents = parents[0]
            (positions, rotations, names, frame_nums, trends, geo, remove_idx, data_idx) = data    # FIXED:返回需要改改

            # positions, rotations = data_utils.to_start_centered_data(
            #     positions, rotations, context_len)
            rotations = data_utils.matrix9D_to_6D_torch(rotations) # get-input需要的是6d
            state, delta = get_model_input(positions, rotations)
            state_data.append(state.cpu().numpy())
            # delta_data.append(delta.cpu().numpy())

        state_data = np.concatenate(state_data, axis=0)
        # delta_data = np.concatenate(delta_data, axis=0)

        train_stats = {
            "state": {
                "mean": np.mean(state_data, axis=(0, 1)),
                "std": np.std(state_data, axis=(0, 1)),
            },
            # "delta": {
            #     "mean": np.mean(delta_data, axis=(0, 1)),
            #     "std": np.std(delta_data, axis=(0, 1)),
            # }
        }

        with open(stats_path, "wb") as fh:
            pickle.dump(train_stats, fh)

        print("Train stats wrote to {}".format(stats_path))

    return (
        train_stats["state"]["mean"],
        train_stats["state"]["std"]
        # train_stats["delta"]["mean"],
        # train_stats["delta"]["std"],
    )


def get_train_stats_torch(config, dtype, device,
                          use_cache=True, stats_folder=None,
                          dataset_name="train_stats"):
    mean_state, std_state = get_train_stats(
        config, use_cache, stats_folder, dataset_name)

    mean_state = torch.tensor(mean_state, dtype=dtype, device=device)
    std_state = torch.tensor(std_state, dtype=dtype, device=device)
    # mean_delta = torch.tensor(mean_delta, dtype=dtype, device=device)
    # std_delta = torch.tensor(std_delta, dtype=dtype, device=device)

    return mean_state, std_state, None,None


def get_attention_mask(window_len, target_idx, device):
    atten_mask = torch.ones(window_len, window_len,
                            device=device, dtype=torch.bool)
    atten_mask[:, :target_idx + 1] = False
    atten_mask = atten_mask.unsqueeze(0)

    # (1, seq, seq)
    return atten_mask


def reset_constrained_values(state, state_gt, delta, delta_gt,
                             midway_targets, constrained_slices):
    # print("reset_constrain:{}".format(constrained_slices))
    for s in constrained_slices:
        state[..., midway_targets, s] = state_gt[..., midway_targets, s]
        # delta[..., midway_targets, s] = delta_gt[..., midway_targets, s]


def update_dropout_p(model, iteration, config):
    init_p = config["model"]["dropout"]
    max_iterations = config["model"]["dropout_iterations"]
    factor = 1 - max(max_iterations - iteration, 0) / max_iterations
    p = init_p * (1 + math.cos(factor * math.pi)) / 2

    model.dropout = p
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


def train(config, context_config):
    indices = config["indices"]
    info_interval = config["visdom"]["interval"]
    eval_interval = config["visdom"]["interval_eval"]
    eval_trans = config["visdom"]["eval_trans"]

    rp_slice = slice(indices["r_start_idx"], indices["p_end_idx"])
    p_slice = slice(indices["p_start_idx"], indices["p_end_idx"])
    c_slice = slice(indices["c_start_idx"], indices["c_end_idx"])

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, "train", device, shuffle=True,add_geo=config["train"]["add_geo"])
    dtype = dataset.dtype

    bench_dataset_names, _, bench_data_loaders = \
        train_utils.get_benchmark_datasets(config, device, shuffle=False,add_geo=config["train"]["add_geo"])

    # visualization
    vis, info_idx = train_utils.init_visdom(config)

    # initialize model
    detail_model = DetailTransformer(config["model"]).to(device)
    context_model = ContextTransformer(context_config["model"]).to(device)

    # initialize optimizer
    optimizer = Adam(detail_model.parameters(), lr=config["train"]["lr"])

    # learning rate scheduler
    scheduler = train_utils.get_noam_lr_scheduler(config, optimizer)

    # load checkpoint
    epoch, iteration = train_utils.load_checkpoint(
        config, detail_model, optimizer, scheduler)
    train_utils.load_checkpoint(context_config, context_model)

    # context model training stats
    mean_ctx, std_ctx = ctx_mdl.get_train_stats_torch(
        context_config, dtype, device)

    # detail model training stats
    mean_state, std_state, _, _ = get_train_stats_torch(
        config, dtype, device)

    add_geo_FLAG = config["train"]["add_geo"]
    if add_geo_FLAG:
        SEQNUM_GEO = 12
    else:
        SEQNUM_GEO = 0
    print(f"seqnum_geo:{SEQNUM_GEO}")
    window_len = config["datasets"]["train"]["window"] + SEQNUM_GEO
    context_len = config["train"]["context_len"] + SEQNUM_GEO
    
    fill_mode = config["datasets"]["train"]["fill_mode"]
    fill_value = fill_value=torch.zeros(mean_state.shape,dtype=dtype,device=device)
    if fill_mode=="missing-mean" or fill_mode=="vacant-mean":
        fill_value=mean_state
        print("mean YES")
    fill_value_p,fill_value_r6d=from_flat_data_joint_data(fill_value)
    
    min_trans = config["train"]["min_trans"]
    max_trans = config["train"]["max_trans"]
    midway_targets_amount = config["train"]["midway_targets_amount"]
    midway_targets_p = config["train"]["midway_targets_p"]

    loss_avg = 0
    p_loss_avg = 0
    r_loss_avg = 0
    c_loss_avg = 0
    f_loss_avg = 0

    min_benchmark_loss = float("inf")

    while epoch < config["train"]["total_epoch"]:
        for i, data in enumerate(data_loader, 0):
            # (positions, rotations, global_positions, global_rotations,
            #  foot_contact, parents, data_idx) = data
            # parents = parents[0]
            (positions, rotations, names, frame_nums, trends, geo, remove_idx, data_idx) = data
            rot_6d = data_utils.matrix9D_to_6D_torch(rotations) # get-input需要的是6d
            if add_geo_FLAG:
                trends=torch.cat([torch.zeros([trends.shape[0],SEQNUM_GEO,trends.shape[-1]],dtype=dtype,device=device),
                                trends],
                                dim=1
                                )
            else:
                geo=None
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
            # positions, rotations = data_utils.to_start_centered_data(
            #     positions, rotations, context_len)
            # global_rotations, global_positions = data_utils.fk_torch(
            #     rotations, positions, parents)
            global_rotations = rotations
            global_positions = positions

            # randomize transition length
            trans_len = random.randint(min_trans, max_trans)
            target_idx = context_len + trans_len
            seq_slice = slice(context_len, target_idx)

            # get random midway target frames
            midway_targets = ctx_mdl.get_midway_targets(
                seq_slice, midway_targets_amount, midway_targets_p)

            # attention mask for context model
            atten_mask_ctx = ctx_mdl.get_attention_mask(
                window_len, context_len, target_idx, device,
                midway_targets=midway_targets)

            # attention mask for detail model
            atten_mask = get_attention_mask(window_len, target_idx, device) 

            data_mask = ctx_mdl.get_data_mask(
                window_len, detail_model.d_mask,
                detail_model.constrained_slices, context_len, target_idx,
                device, dtype, midway_targets=midway_targets)

            # get context model output
            pos_ctx, rot_ctx = ctx_mdl.evaluate(
                context_model, positions, rot_6d, seq_slice,
                indices, mean_ctx, std_ctx, atten_mask_ctx,
                post_process=False, midway_targets=midway_targets,geo=geo) # fixed: post-process
            # rot_ctx(9d)->rot_ctx_6d
            rot_ctx_6d = data_utils.matrix9D_to_6D_torch(rot_ctx)
            
            # detail model inputs
            state_gt, delta_gt = get_model_input(positions, rot_6d) 
            state, delta = get_model_input(pos_ctx, rot_ctx_6d)
            if add_geo_FLAG:
                geo_ctrl = get_model_input_geo(geo)   # GEO: (BATCH,SEQ,JOINT*9)
                state_gt = torch.cat([geo_ctrl,state_gt],dim=1)    # GEO: (BATCH,SEQ*,DIMS)
            assert state_gt.shape==state.shape
            print(state_gt.shape,state.shape)
            reset_constrained_values(
                state, state_gt, delta, delta_gt,
                midway_targets, detail_model.constrained_slices)    

            state_zscore = (state - mean_state) / std_state

            # train details model --------------------------------
            optimizer.zero_grad()
            detail_model.train()

            # update model dropout p
            update_dropout_p(detail_model, iteration, config) 

            state_out,c_out = get_model_output(
                detail_model, state_zscore, data_mask, atten_mask,
                trends, seq_slice, c_slice, rp_slice,flag=config["train"]["trends_loss"]) # FIXME: 6

            state_out = state_out * std_state + mean_state
            
            if add_geo_FLAG:
                positions = torch.cat([torch.zeros([positions.shape[0],SEQNUM_GEO,*positions.shape[2:]],dtype=dtype,device=device),
                                        positions],
                                        dim=1
                                        ) # GEO
                global_positions = torch.cat([torch.zeros([global_positions.shape[0],SEQNUM_GEO,*global_positions.shape[2:]],dtype=dtype,device=device),
                                              global_positions],
                                             dim=1) # GEO
            pos_new = train_utils.get_new_positions(
                positions, state_out, indices)
            rot_new = train_utils.get_new_rotations(state_out, indices)
            rot_6d_new = train_utils.get_new_rotations6D(state_out,indices)
            foot_state=torch.cat([pos_new,rot_6d_new],dim=-1)
            
            # grot_new, gpos_new = data_utils.fk_torch(rot_new, pos_new, parents)

            r_loss = train_utils.cal_r_loss(
                state_gt, state_out, seq_slice, indices)
            p_loss = train_utils.cal_p_loss(
                global_positions, pos_new, seq_slice)
            if config["train"]["trends_loss"]:
                c_loss = train_utils.cal_c_loss(
                    trends, c_out, seq_slice)
                # state_new = get_pose_from_trends(state_out,c_out,seq_slice)
                # f_loss = train_utils.cal_f_loss(state_gt,state_new, seq_slice, indices)
                f_loss = train_utils.cal_f_loss(foot_state, c_out, seq_slice)

                # loss
                loss = (
                    config["weights"]["rw"] * r_loss +
                    config["weights"]["pw"] * p_loss +
                    config["weights"]["cw"] * c_loss +
                    config["weights"]["fw"] * f_loss
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

                r_loss_avg += r_loss.item()
                p_loss_avg += p_loss.item()
                c_loss_avg += c_loss.item()
                f_loss_avg += f_loss.item()
                loss_avg += loss.item()
            else:
                # loss
                loss = (
                    config["weights"]["rw"] * r_loss +
                    config["weights"]["pw"] * p_loss 
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

                r_loss_avg += r_loss.item()
                p_loss_avg += p_loss.item()
                loss_avg += loss.item()
                
            if iteration % config["train"]["checkpoint_interval"] == 0:
                train_utils.save_checkpoint(config, detail_model, epoch,
                                            iteration, optimizer, scheduler)
                vis.save([config["visdom"]["env"]])

            if iteration % info_interval == 0:
                r_loss_avg /= info_interval
                p_loss_avg /= info_interval
                c_loss_avg /= info_interval
                f_loss_avg /= info_interval
                loss_avg /= info_interval
                lr = optimizer.param_groups[0]["lr"]

                print("Epoch: {}, Iteration: {}, lr: {:.8f}, dropout: {:.6f}, "
                      "loss: {:.6f}, r: {:.6f}, p: {:.6f}, c: {:.6f}, "
                      "f: {:.6f}".format(
                          epoch, iteration, lr, detail_model.dropout, loss_avg,
                          r_loss_avg, p_loss_avg, c_loss_avg, f_loss_avg))

                contents = [
                    ["loss", "r_loss", r_loss_avg],
                    ["loss", "p_loss", p_loss_avg],
                    ["loss", "c_loss", c_loss_avg],
                    ["loss", "f_loss", f_loss_avg],
                    ["dropout", "p", detail_model.dropout],
                    ["loss weighted", "r_loss",
                        r_loss_avg * config["weights"]["rw"]],
                    ["loss weighted", "p_loss",
                        p_loss_avg * config["weights"]["pw"]],
                    ["loss weighted", "c_loss",
                        c_loss_avg * config["weights"]["cw"]],
                    ["loss weighted", "f_loss",
                        f_loss_avg * config["weights"]["fw"]],
                    ["loss weighted", "loss", loss_avg],
                    ["learning rate", "lr", lr],
                    ["epoch", "epoch", epoch],
                    ["iterations", "iterations", iteration],
                ]

                if iteration % eval_interval == 0:
                    for trans in eval_trans:
                        print("trans: {}".format(trans))

                        for i in range(len(bench_data_loaders)):
                            ds_name = bench_dataset_names[i]
                            ds_loader = bench_data_loaders[i]
                            gpos_loss, gquat_loss, npss_loss = eval_on_dataset(
                                config, ds_loader, detail_model,
                                context_model, trans, post_process=False)
                            contents.extend([
                                [ds_name, "gpos_{}".format(trans),
                                 gpos_loss],
                                [ds_name, "gquat_{}".format(trans),
                                 gquat_loss],
                                [ds_name, "npss_{}".format(trans),
                                 npss_loss],
                            ])
                            print("{}:\ngpos: {:6f}, gquat: {:6f}, "
                                  "npss: {:.6f}".format(ds_name, gpos_loss,
                                                        gquat_loss, npss_loss))

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
                            config, detail_model, epoch, iteration,
                            optimizer, scheduler, suffix=".min")

                train_utils.to_visdom(vis, info_idx, contents)
                r_loss_avg = 0
                p_loss_avg = 0
                c_loss_avg = 0
                f_loss_avg = 0
                loss_avg = 0
                info_idx += 1

            iteration += 1

        epoch += 1


def eval_on_dataset(config, data_loader, detail_model, context_model,
                    trans_len, debug=False, post_process=False):
    device = data_loader.dataset.device
    dtype = data_loader.dataset.dtype
    window_len = data_loader.dataset.window + SEQNUM_GEO

    indices = config["indices"]
    context_len = config["train"]["context_len"] + SEQNUM_GEO
    target_idx = context_len + trans_len
    seq_slice = slice(context_len, target_idx)

    mean_ctx, std_ctx = ctx_mdl.get_train_stats_torch(config, dtype, device)
    mean_state, std_state, _, _ = get_train_stats_torch(
        config, dtype, device)
    mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
        config, dtype, device)

    fill_mode = config["datasets"]["train"]["fill_mode"]
    fill_value = fill_value=torch.zeros(mean_state.shape,dtype=dtype,device=device)
    if fill_mode=="missing-mean" or fill_mode=="vacant-mean":
        fill_value= mean_state
    fill_value_p,fill_value_r6d=from_flat_data_joint_data(fill_value)
    
    # attention mask for context model
    atten_mask_ctx = ctx_mdl.get_attention_mask(
        window_len, context_len, target_idx, device)

    # attention mask for detail model
    atten_mask = get_attention_mask(window_len, target_idx, device)

    data_indexes = []
    gpos_loss = []
    gquat_loss = []
    npss_loss = []
    npss_weights = []

    add_geo_FLAG = config["train"]["add_geo"]
    if add_geo_FLAG==False:
        assert SEQNUM_GEO==0
        
    for i, data in enumerate(data_loader, 0):
        # (positions, rotations, global_positions, global_rotations,
        #     foot_contact, parents, data_idx) = data
        # parents = parents[0]
        (positions, rotations, names, frame_nums, trends, geo, remove_idx, data_idx) = data # FIXED:返回类型
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
        # notice: rotations参与计算loss了
        rotations = data_utils.matrix6D_to_9D_torch(rot_6d)
        # positions, rotations = data_utils.to_start_centered_data(
        #     positions, rotations, context_len)

        pos_new, rot_new, _ = evaluate(
            detail_model, context_model, positions, rot_6d, trends,
            seq_slice, indices, mean_ctx, std_ctx, mean_state, std_state,
            atten_mask, atten_mask_ctx, post_process,geo=geo,cf_flag=config["train"]["trends_loss"])
        if add_geo_FLAG:
            positions=torch.cat([torch.zeros([positions.shape[0],SEQNUM_GEO,*positions.shape[2:]],dtype=dtype,device=device),positions],dim=1) # GEO
            rotations=torch.cat([torch.zeros([rotations.shape[0],SEQNUM_GEO,*rotations.shape[2:]],dtype=dtype,device=device),rotations],dim=1) # GEO
        
        (gpos_batch_loss, gquat_batch_loss,
         npss_batch_loss, npss_batch_weights) = get_rmi_style_batch_loss(
            positions, rotations, pos_new, rot_new, None,
            context_len, target_idx, mean_rmi, std_rmi)

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
        return gpos_loss.mean(), gquat_loss.mean(), npss_loss.sum()


def evaluate(detail_model, context_model, positions, rotations, foot_contact,
             seq_slice, indices, mean_ctx, std_ctx, mean_state, std_state,
             atten_mask, atten_mask_ctx, post_process=True, midway_targets=(),geo=None,cf_flag=True):
    # NOTICE: rotations[6d]
    dtype = positions.dtype
    device = positions.device
    window_len = positions.shape[-3]+SEQNUM_GEO if geo is not None else positions.shape[-3]
    context_len = seq_slice.start
    target_idx = seq_slice.stop
    rp_slice = slice(indices["r_start_idx"], indices["p_end_idx"])
    c_slice = slice(indices["c_start_idx"], indices["c_end_idx"])

    if midway_targets:
        # If either context model or detail model is not trained with
        # constraints, ignore midway_targets.
        if not context_model.constrained_slices:
            print(
                "WARNING: Context model is not trained with constraints, but "
                "midway_targets is provided with values while calling evaluate()! "
                "midway_targets is ignored!"
            )
            midway_targets = []
        elif not detail_model.constrained_slices:
            print(
                "WARNING: Detail model is not trained with constraints, but "
                "midway_targets is provided with values while calling evaluate()! "
                "midway_targets is ignored!"
            )
            midway_targets = []

    with torch.no_grad():
        detail_model.eval()
        context_model.eval()

        # get context model output
        pos_ctx, rot_ctx = ctx_mdl.evaluate(
            context_model, positions, rotations, seq_slice,
            indices, mean_ctx, std_ctx, atten_mask_ctx,
            post_process=False, midway_targets=midway_targets,geo=geo)
        rot_ctx_6d = data_utils.matrix9D_to_6D_torch(rot_ctx)
        
        data_mask = ctx_mdl.get_data_mask(
            window_len, detail_model.d_mask,
            detail_model.constrained_slices, context_len, target_idx,
            device, dtype, midway_targets=midway_targets)

        state_in, delta_in = get_model_input(positions, rotations)
        state, delta = get_model_input(pos_ctx, rot_ctx_6d)
        geo_ctrl=None
        if geo is not None:
            geo_ctrl = get_model_input_geo(geo)   # GEO: (BATCH,SEQ,JOINT*9)
            state_in = torch.cat([geo_ctrl,state_in],dim=1) 
            foot_contact=torch.cat([torch.zeros([foot_contact.shape[0],SEQNUM_GEO,foot_contact.shape[-1]],dtype=dtype,device=device),
                                foot_contact],
                                dim=1
                                )
        reset_constrained_values(
            state, state_in, delta, delta_in,
            midway_targets, detail_model.constrained_slices)

        state_zscore = (state - mean_state) / std_state

        # get detail model output
        state_out,c_out = get_model_output(
            detail_model, state_zscore, data_mask, atten_mask, foot_contact,
            seq_slice, c_slice, rp_slice,flag=cf_flag)

        if post_process:
            state_out = train_utils.anim_post_process(
                state_out, state_zscore, seq_slice)

        state_out = state_out * std_state + mean_state

        # state_out = get_pose_from_trends(state_out,c_out,seq_slice)
        rotations = data_utils.matrix6D_to_9D_torch(rotations)
        # GEO:
        if geo is not None:
            positions=torch.cat([torch.zeros([positions.shape[0],SEQNUM_GEO,*positions.shape[2:]],dtype=dtype,device=device),positions],dim=1) # GEO
            rotations=torch.cat([torch.zeros([rotations.shape[0],SEQNUM_GEO,*rotations.shape[2:]],dtype=dtype,device=device),rotations],dim=1) # GEO
        # new pos and rot
        pos_new = train_utils.get_new_positions(
            positions, state_out, indices, seq_slice)
        rot_new = train_utils.get_new_rotations(
            state_out, indices, rotations, seq_slice)

        
        return pos_new, rot_new,c_out
