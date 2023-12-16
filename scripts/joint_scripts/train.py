# HACK ignore warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig, SunToScannetDatasetConfig
from data.sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
from lib.joint.dataset import ScannetReferenceDataset
from lib.joint.solver import Solver
from config.config import CONF
from models.jointnet.jointnet import JointNet
from scripts.utils.AdamW import AdamW
from scripts.utils.script_utils import set_params_lr_dict
from torch.nn.parallel import DistributedDataParallel

# SCANREFER_DUMMY = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_dummy.json")))

# extracted ScanNet object rotations from Scan2CAD 
# NOTE some scenes are missing in this annotation!!!
# SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "Masked_ScanRefer_filtered_train.json")))
# SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
SCAN2CAD_ROTATION = None # json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

# constants
# DC = SunToScannetDatasetConfig()
DC = ScannetDatasetConfig() if CONF.pretrain_data == "scannet" else SunToScannetDatasetConfig()


def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config, augment, shuffle=True,
                   scan2cad_rotation=None):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_new=scanrefer_new,
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        name=args.dataset,
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max,
        augment=augment,
        shuffle=shuffle,
        scan2cad_rotation=scan2cad_rotation
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    if split == "train":
        bs = args.batch_size
    else:
        bs = args.val_batch_size

    if args.distribute and split == "train":
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=bs,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 worker_init_fn=my_worker_init_fn,
                                                 sampler=sampler,
                                                 drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True if split == "train" else False, num_workers=0)
    # dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=1)

    return dataset, dataloader


def get_model(args, dataset):
    # initiate model
    # input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    input_channels = int(not args.no_height) + int(args.use_color) * 3
    model = JointNet(
        # num_class=DC.num_class,
        vocabulary=dataset.vocabulary,
        # embeddings=dataset.glove,
        # num_heading_bin=DC.num_heading_bin,
        # num_size_cluster=DC.num_size_cluster,
        # mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        width=args.width,
        # sampling=args.sampling,
        hidden_size=args.hidden_size,
        num_proposal=args.num_proposals,
        num_target=args.num_target,
        no_caption=args.no_caption,
        use_topdown=args.use_topdown,
        num_locals=args.num_locals,
        query_mode=args.query_mode,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        dataset_config=DC,
        args=args
    )

    if args.pretrain_model_on:
        if args.pretrain_model == "groupfree":
            print("loading predtrained GroupFree...")
            pretrained_groupfree_weights = torch.load(CONF.PATH.GROUPFREE_PRETRAIN)
            # print(pretrained_groupfree_weights.keys())
            model.group_free.load_state_dict(pretrained_groupfree_weights, strict=False)
        if args.pretrain_model == "votenet":
            print("loading pretrained VoteNet...")
            pretrained_votenet_weights = torch.load(CONF.PATH.VOTENET_PRETRAIN)
            # print(pretrained_votenet_weights["model_state_dict"].keys())
            model.load_state_dict(pretrained_votenet_weights["model_state_dict"], strict=False)

    # print("loading pretrained LangModule weights...")
    # pretrained_lang_weights = torch.load(CONF.PATH.LANGMODULE_PRETRAIN)
    # # print(pretrained_lang_weights.keys())
    # model.lang.load_state_dict(pretrained_lang_weights)

    # multi-GPU
    # if torch.cuda.device_count() > 1:
    #     print("using {} GPUs...".format(torch.cuda.device_count()))
    #     model = torch.nn.DataParallel(model)

    # to device
    model.cuda()

    if args.distribute:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataset, dataloader):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args, dataset["train"])
    weight_dict = {
        # 'lang': {'lr': 0.003},
        # 'relation': {'lr': 0.003},
        # 'match': {'lr': 0.003},
        # 'contranet': {'lr': 0.003},
        # 'recnet': {'lr': 0.003}
    }
    params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

    checkpoint_best = None

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint_best = checkpoint["best"]
    else:
        if args.tag:
            stamp = args.tag.upper()
        else:
            stamp = ""
        stamp += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    # LR_DECAY_STEP = [80, 120, 160] if args.no_caption else None
    # if args.coslr:
    LR_DECAY_STEP = {
        'type': 'cosine',
        'T_max': args.epoch,
        'eta_min': 1e-5,
    }
    LR_DECAY_RATE = 0.1 if args.no_caption else None
    BN_DECAY_STEP = 20 if args.no_caption else None
    BN_DECAY_RATE = 0.5 if args.no_caption else None

    print('LR&BN_DECAY', LR_DECAY_STEP, LR_DECAY_RATE, BN_DECAY_STEP, BN_DECAY_RATE, flush=True)
    print("criterion", args.criterion, flush=True)
    solver = Solver(
        model=model,
        config=DC, 
        dataset=dataset,
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_step=args.val_step,
        num_ground_epoch=args.num_ground_epoch,
        detection=not args.no_detection,
        caption=not args.no_caption, 
        orientation=args.use_orientation,
        distance=args.use_distance,
        use_tf=args.use_tf,
        reference=not args.no_reference,
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        criterion=args.criterion,
        checkpoint_best=checkpoint_best,
        distributed_rank=args.local_rank if args.distribute else None,
        opt_steps=args.opt_steps
    )
    num_params = get_num_params(model)

    return solver, num_params, root

def save_info(args, root, num_params, dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(dataset["train"])
    # info["num_eval_train"] = len(dataset["eval"]["train"])
    info["num_eval_val"] = len(dataset["eval"]["val"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    # info["num_eval_train_scenes"] = len(dataset["eval"]["train"].scene_list)
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):
    if args.dataset == "ScanRefer":
        annos_train = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer", "Masked_ScanRefer_filtered_train.json")))
        annos_val = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer", "ScanRefer_filtered_val.json")))
    elif args.dataset == "nr3d":
        annos_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d", "masked_nr3d_train.json")))
        annos_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d", "masked_nr3d_test.json")))
    elif args.dataset == "sr3d":
        annos_train = json.load(open(os.path.join(CONF.PATH.DATA, "sr3d", "masked_sr3d_train.json")))
        annos_val = json.load(open(os.path.join(CONF.PATH.DATA, "sr3d", "masked_sr3d_test.json")))
    else:
        raise ValueError("Invalid dataset.")

    if args.debug:
        annos_train = [annos_train[0]]
        annos_val = [annos_train[0]]

    if args.no_caption and args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        val_scene_list = get_scannet_scene_list("val")

        new_annos_train = []
        for scene_id in train_scene_list:
            data = deepcopy(annos_train[0])
            data["scene_id"] = scene_id
            new_annos_train.append(data)

        new_annos_val = []
        for scene_id in val_scene_list:
            data = deepcopy(annos_train[0])
            data["scene_id"] = scene_id
            new_annos_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in annos_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in annos_val])))

        # filter data in chosen scenes
        new_annos_train = []
        annos_train_new = []
        annos_train_new_scene = []
        scene_id = ""
        for data in annos_train:
            if data["scene_id"] in train_scene_list:
                new_annos_train.append(data)
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(annos_train_new_scene) > 0:
                        annos_train_new.append(annos_train_new_scene)
                    annos_train_new_scene = []
                if len(annos_train_new_scene) >= args.lang_num_max:
                    annos_train_new.append(annos_train_new_scene)
                    annos_train_new_scene = []
                annos_train_new_scene.append(data)
        annos_train_new.append(annos_train_new_scene)

        new_annos_val = annos_val
        annos_val_new = []
        annos_val_new_scene = []
        scene_id = ""
        for data in annos_val:
            # if data["scene_id"] not in scanrefer_val_new:
            # scanrefer_val_new[data["scene_id"]] = []
            # scanrefer_val_new[data["scene_id"]].append(data)
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(annos_val_new_scene) > 0:
                    annos_val_new_scene.append(annos_val_new_scene)
                annos_val_new_scene = []
            if len(annos_val_new_scene) >= args.lang_num_max:
                annos_val_new.append(annos_val_new_scene)
                annos_val_new_scene = []
            annos_val_new_scene.append(data)
        annos_val_new.append(annos_val_new_scene)

    print("annos_train_new", len(annos_train_new), len(annos_train_new[0]))
    print("annos_val_new", len(annos_val_new))
    sum = 0
    for i in range(len(annos_train_new)):
        sum += len(annos_train_new[i])
    print("sum", sum)  # 1418 363

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("using {} dataset".format(args.dataset))
    print("train on {} samples from {} scenes".format(len(new_annos_train), len(train_scene_list)))
    print("eval on {} scenes from val".format(len(new_annos_val)))

    return new_annos_train, new_annos_val, all_scene_list, annos_train_new, annos_val_new

def train(args):
    # init training dataset
    print("preparing data...")
    annos_train, annos_val, all_scene_list, annos_train_new, annos_val_new = get_scanrefer(args)

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, annos_train, annos_train_new, all_scene_list, "train", DC, True, SCAN2CAD_ROTATION)
    eval_val_dataset, eval_val_dataloader = get_dataloader(args, annos_val, annos_val_new, all_scene_list, "val", DC, False, shuffle=False)

    dataset = {
        "train": train_dataset,
        "eval": {
            # "train": eval_train_dataset,
            "val": eval_val_dataset,
        }
    }
    dataloader = {
        "train": train_dataloader,
        "eval": {
            # "train": eval_train_dataloader,
            "val": eval_val_dataloader,
        }
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataset, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, dataset)
    solver(args.epoch, args.verbose)

if __name__ == "__main__":

    # # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # reproducibility
    os.environ['PYTHONHASHSEED'] = str(CONF.seed)
    np.random.seed(CONF.seed)
    torch.manual_seed(CONF.seed)
    torch.cuda.manual_seed(CONF.seed)
    torch.cuda.manual_seed_all(CONF.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    if CONF.distribute:
        torch.cuda.set_device(CONF.local_rank)
        torch.backends.cudnn.benchmark = False
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    train(CONF)
