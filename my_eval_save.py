import os
import sys
import argparse


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)
import torch

from motion_inbetween_space import benchmark
from motion_inbetween_space.model import STTransformer
from motion_inbetween_space.config import load_config_by_name
from motion_inbetween_space.train import rmi
from motion_inbetween_space.train import context_model
from motion_inbetween_space.train import utils as train_utils
from motion_inbetween_space.data import utils_torch as data_utils
# for short prediction + pos error + rot error
# python my_eval_save.py apred_context_modelgeo2dAlibiNoabsGeo0_sp10_56rw_addtgt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVAL context model.")
    parser.add_argument("config", help="config name")
    parser.add_argument("-s", "--dataset",
                        help="dataset name (default=benchmark)",
                        default="benchmark")
    parser.add_argument("-t", "--trans", type=int, default=23,
                        help="transition length (default=30)")
    args = parser.parse_args()


    config = load_config_by_name(args.config)
    eval_trans=[5,10,15,20,25,30,45]
    context_model.ATTENTION_MODE = config["train"]["attention_mode"]
    context_model.INIT_INTERP = config["train"]["init_interp"]
    context_model.zscore_MODE = config["train"]["zscore_MODE"]    # normal/none/seq
    context_model.Data_Mask_MODE = config["train"]["data_mask_set"] 
    context_model.framework_MODE = config["train"]["framework_MODE"] # in/pred
    context_model.TGT_condition = config["train"]["add_tgt"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, args.dataset, device=device, shuffle=False,add_geo=config["train"]["add_geo"],add_tgt=config["train"]["add_tgt"])
    print("config:{}, geo:{}".format(args.config,config["train"]["add_geo"]))
    SEQNUM_GEO=12 if config["train"]["add_geo"] else 0
    if context_model.TGT_condition:
        SEQNUM_GEO+=1
    context_model.SEQNUM_GEO=SEQNUM_GEO
    add_geo_FLAG = config["train"]["add_geo"]
    # initialize model
    model = STTransformer(config["model"]).to(device)
    # load checkpoint
    epoch, iteration = train_utils.load_checkpoint(config, model,suffix=".80000val_min")
    gpos_list=[]
    gquat_list=[]
    npss_list=[]
    ploss_list=[]
    smoothloss_list=[]
    pos_error_list=[]
    rot_error_list=[]
    for trans in eval_trans:
        gpos_loss, gquat_loss, npss_loss, val_ploss,val_smoothloss,pos_error,rot_error = context_model.eval_on_dataset(
                                config, data_loader, model, trans,angle_save=True)
        gpos_list.append(gpos_loss)
        gquat_list.append(gquat_loss)
        npss_list.append(npss_loss)
        ploss_list.append(val_ploss)
        smoothloss_list.append(val_smoothloss)
        pos_error_list.append(pos_error)
        rot_error_list.append(rot_error)
        print("trans:[{}]:\ngpos: {:6f}, gquat: {:6f}, npss: {:.6f} l2p:{:.6f} smooth:{:.6f} pe:{:.6f} re:{:.6f}".format(trans, gpos_loss,
                                                            gquat_loss, npss_loss,val_ploss,val_smoothloss,pos_error,rot_error))
    for i in range(len(eval_trans)):
        print("trans:",eval_trans[i])
        print("gpos\tgquat\tnpss\tl2p\tsmooth\tpe\tre")
        print("{:6f}, {:6f}, {:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(gpos_list[i],gquat_list[i], npss_list[i],
                                                                        ploss_list[i],smoothloss_list[i],pos_error_list[i],rot_error_list[i]))
