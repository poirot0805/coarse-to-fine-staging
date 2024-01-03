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
    eval_trans=[5,10,15,20,25,30] if args.trans==23 else [1,2,3,4,5]
    context_model.ATTENTION_MODE = config["train"]["attention_mode"]
    context_model.zscore_MODE = config["train"]["zscore_MODE"]
    context_model.INIT_INTERP = config["train"]["init_interp"]
    context_model.Data_Mask_MODE=config["train"]["data_mask_set"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, args.dataset, device=device, shuffle=False,add_geo=config["train"]["add_geo"])
    print("config:{}, geo:{}".format(args.config,config["train"]["add_geo"]))
    SEQNUM_GEO=12 if config["train"]["add_geo"] else 0
    add_geo_FLAG = config["train"]["add_geo"]
    # initialize model
    model = STTransformer(config["model"]).to(device)
    # load checkpoint
    epoch, iteration = train_utils.load_checkpoint(config, model,suffix=".440000val_min")
    gpos_list=[]
    gquat_list=[]
    npss_list=[]
    ploss_list=[]
    smoothloss_list=[]
    for trans in eval_trans:
        gpos_loss, gquat_loss, npss_loss, val_ploss,val_smoothloss = context_model.eval_on_dataset(
                                config, data_loader, model, trans)
        gpos_list.append(gpos_loss)
        gquat_list.append(gquat_loss)
        npss_list.append(npss_loss)
        ploss_list.append(val_ploss)
        smoothloss_list.append(val_smoothloss)
        print("trans:[{}]:\ngpos: {:6f}, gquat: {:6f}, npss: {:.6f} l2p:{:.6f} smooth:{:.6f}".format(trans, gpos_loss,
                                                            gquat_loss, npss_loss,val_ploss,val_smoothloss))
    print(gpos_list)
    print(gquat_list)
    print(npss_list)
    print(ploss_list)
    print(smoothloss_list)