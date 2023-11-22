import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.configs.config import CONF
from lib.joint.dataset import ScannetReferenceDataset
from lib.loss_helper.loss_joint import get_joint_loss
from lib.joint.eval_ground import get_eval
from models.jointnet.jointnet import JointNet
from data.scannet.model_util_scannet import ScannetDatasetConfig, SunToScannetDatasetConfig
from torch.profiler import profile, record_function, ProfilerActivity

print('Import Done', flush=True)
if CONF.dataset == "ScanRefer":
    SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "Masked_ScanRefer_filtered_train.json")))
    SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
elif CONF.dataset == "nr3d":
    SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d", "masked_nr3d_train.json")))
    SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d", "masked_nr3d_test.json")))
elif CONF.dataset == "sr3d":
    SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "sr3d", "masked_sr3d_train.json")))
    SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "sr3d", "masked_sr3d_test.json")))
# SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_test.json")))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_new=scanrefer_new,
        scanrefer_all_scene=all_scene_list, 
        split=split,
        name=args.dataset,
        num_points=args.num_points, 
        use_color=args.use_color, 
        use_height=(not args.no_height),
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max,
    )
    print("evaluate on {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataset, dataloader

def get_model(args, DC, dataset):
    # load model
    # input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    input_channels = int(not args.no_height) + int(args.use_color) * 3
    model = JointNet(
        vocabulary=dataset.vocabulary,
        input_feature_dim=input_channels,
        width=args.width,
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
    ).cuda()

    model_name = "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):
    if not args.no_detection:
        scene_list = get_scannet_scene_list("val")
        scanrefer = []
        for scene_id in scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            scanrefer.append(data)
    else:
        scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
        scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
        if args.num_scenes != -1:
            scene_list = scene_list[:args.num_scenes]

        scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

        new_scanrefer_val = scanrefer
        scanrefer_val_new = []
        scanrefer_val_new_scene = []
        scene_id = ""
        for data in scanrefer:
            # if data["scene_id"] not in scanrefer_val_new:
            # scanrefer_val_new[data["scene_id"]] = []
            # scanrefer_val_new[data["scene_id"]].append(data)
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_val_new_scene) > 0:
                    scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            if len(scanrefer_val_new_scene) >= args.lang_num_max:
                scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            scanrefer_val_new_scene.append(data)
        if len(scanrefer_val_new_scene) > 0:
            scanrefer_val_new.append(scanrefer_val_new_scene)

        new_scanrefer_eval_val2 = []
        scanrefer_eval_val_new2 = []
        for scene_id in scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val2.append(data)
            scanrefer_eval_val_new_scene2 = []
            for i in range(args.lang_num_max):
                scanrefer_eval_val_new_scene2.append(data)
            scanrefer_eval_val_new2.append(scanrefer_eval_val_new_scene2)

    return scanrefer, scene_list, scanrefer_val_new

def eval_ref(args):
    print("evaluate localization...")
    # constant
    DC = ScannetDatasetConfig() if args.pretrain_data == "scannet" else SunToScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list, scanrefer_val_new = get_scanrefer(args)

    # dataloader
    #_, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)
    dataset, dataloader = get_dataloader(args, scanrefer, scanrefer_val_new, scene_list, "val", DC)

    # model
    model = get_model(args, DC, dataset)
    print("\nparam stats:")
    print("model:", count_parameters(model))
    print("backbone:", count_parameters(model.group_free))
    print("recon:", count_parameters(model.recnet))
    print("match:", count_parameters(model.match))

    # config
    POST_DICT = None

    # random seeds
    seeds = [args.seed] + [2 * i for i in range(args.repeat - 1)]
    mean_forward_time = 0

    # evaluate
    print("evaluating...")
    score_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "scores.p")
    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "predictions.p")
    gen_flag = (not os.path.exists(score_path)) or args.force or args.repeat > 1
    if gen_flag:
        # ref_acc_all = []
        ious_all = []
        topk_ious_all = []
        rec_ious_all = []
        topk_rec_ious_all = []
        rand_ious_all = []
        topk_rand_ious_all = []
        best_ious_all = []
        masks_all = []
        others_all = []
        lang_acc_all = []
        for seed in seeds:
            # reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            print("generating the scores for seed {}...".format(seed))
            # ref_acc = []
            ious = []
            topk_ious = []
            rec_ious = []
            topk_rec_ious = []
            rand_ious = []
            topk_rand_ious = []
            best_ious = []

            masks = []
            others = []
            lang_acc = []
            predictions = {}
            forward_times = []
            for data in tqdm(dataloader):
                for key in data:
                    data[key] = data[key].cuda()

                # feed
                with torch.no_grad():
                    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    data["epoch"] = 0
                    start = time.time()
                    with record_function("model_inference"):
                        data = model(data, is_eval=args.is_eval)
                    # with record_function("loss_calc"):
                        data = get_joint_loss(
                            data_dict=data,
                            config=DC,
                            is_eval=args.is_eval
                        )
                    with record_function("eval"):
                        data = get_eval(
                            data_dict=data,
                            config=DC,
                            reference=True,
                            use_lang_classifier=not args.no_lang_cls,
                            use_cat_rand=False,
                            use_best=args.use_best,
                            use_random=args.eval_rand,
                            is_eval=args.is_eval,
                            k=args.topk
                        )
                    forward_times.append(time.time() - start)
                    # ref_acc += data["ref_acc"]
                    ious += data["ref_iou"]
                    topk_ious += data["topk_iou"]
                    rec_ious += data["rec_iou"]
                    topk_rec_ious += data["topk_rec_iou"]
                    rand_ious += data["rand_iou"]
                    topk_rand_ious += data["topk_rand_iou"]
                    best_ious += data["best_iou"]
                    masks += data["ref_multiple_mask"]
                    others += data["ref_others_mask"]
                    lang_acc.append(data["lang_acc"].item())
                    # store predictions
                    ids = data["scan_idx"].detach().cpu().numpy()
                    for i in range(ids.shape[0]):
                        idx = ids[i]
                        scene_id = scanrefer[idx]["scene_id"]
                        object_id = scanrefer[idx]["object_id"]
                        ann_id = scanrefer[idx]["ann_id"]

                        if scene_id not in predictions:
                            predictions[scene_id] = {}

                        if object_id not in predictions[scene_id]:
                            predictions[scene_id][object_id] = {}

                        if ann_id not in predictions[scene_id][object_id]:
                            predictions[scene_id][object_id][ann_id] = {}

                        predictions[scene_id][object_id][ann_id]["pred_bbox"] = data["pred_bboxes"][i]
                        predictions[scene_id][object_id][ann_id]["gt_bbox"] = data["gt_bboxes"][i]
                        predictions[scene_id][object_id][ann_id]["iou"] = data["ref_iou"][i]
                # print(prof.key_averages().table(sort_by="cuda_time_total"))
                # print(prof.key_averages().table(sort_by="cpu_time_total"))
            # print(prof.key_averages().table(sort_by="cuda_time_total"))
            # print(prof.key_averages().table(sort_by="cpu_time_total"))
            mean_forward_time = np.mean(forward_times)
            print("mean forward time:", mean_forward_time)
            # save the last predictions
            with open(pred_path, "wb") as f:
                pickle.dump(predictions, f)

            # save to global
            ious_all.append(ious)
            topk_ious_all.append(topk_ious)
            rec_ious_all.append(rec_ious)
            topk_rec_ious_all.append(topk_rec_ious)
            rand_ious_all.append(rand_ious)
            topk_rand_ious_all.append(topk_rand_ious)
            best_ious_all.append(best_ious)
            masks_all.append(masks)
            others_all.append(others)
            lang_acc_all.append(lang_acc)

        # convert to numpy array
        ious = np.array(ious_all)
        topk_ious = np.array(topk_ious_all)
        rec_ious = np.array(rec_ious_all)
        topk_rec_ious = np.array(topk_rec_ious_all)
        rand_ious = np.array(rand_ious_all)
        topk_rand_ious = np.array(topk_rand_ious_all)
        best_ious = np.array(best_ious_all)
        masks = np.array(masks_all)
        others = np.array(others_all)
        lang_acc = np.array(lang_acc_all)

        # save the global scores
        with open(score_path, "wb") as f:
            scores = {
                "ious": ious_all,
                "topk_ious": topk_ious_all,
                "rec_ious": rec_ious_all,
                "topk_rec_ious": topk_rec_ious_all,
                "rand_ious": rand_ious_all,
                "topk_rand_ious": topk_rand_ious_all,
                "masks": masks_all,
                "others": others_all,
                "lang_acc": lang_acc_all
            }
            pickle.dump(scores, f)

    else:
        print("loading the scores...")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)

            # unpack
            ious = np.array(scores["ious"])
            topk_ious = np.array(scores["topk_ious"])
            if not args.is_eval:
                rec_ious = np.array(scores["rec_ious"])
                topk_rec_ious = np.array(scores["topk_rec_ious"])
            if args.eval_rand:
                rand_ious = np.array(scores["rand_ious"])
                topk_rand_ious = np.array(scores["topk_rand_ious"])
            masks = np.array(scores["masks"])
            others = np.array(scores["others"])
            lang_acc = np.array(scores["lang_acc"])

    if args.eval_rand:
        ious = rand_ious
        topk_ious = topk_rand_ious

    if args.use_best:
        ious = best_ious

    multiple_dict = {
        "unique": 0,
        "multiple": 1
    }
    others_dict = {
        "not_in_others": 0,
        "in_others": 1
    }

    # evaluation stats
    stats = {k: np.sum(masks[0] == v) for k, v in multiple_dict.items()}
    stats["overall"] = masks[0].shape[0]
    stats = {}
    for k, v in multiple_dict.items():
        stats[k] = {}
        for k_o, v_o in others_dict.items():
            stats[k][k_o] = np.sum(np.logical_and(masks[0] == v, others[0] == v_o))

        stats[k]["overall"] = np.sum(masks[0] == v)

    stats["overall"] = {}
    for k_o, v_o in others_dict.items():
        stats["overall"][k_o] = np.sum(others[0] == v_o)
    
    stats["overall"]["overall"] = masks[0].shape[0]

    # aggregate scores
    scores = {}
    for k, v in multiple_dict.items():
        for k_o in others_dict.keys():
            topk_acc_025ious, topk_acc_05ious, acc_025ious, acc_05ious = [], [], [], []
            for i in range(masks.shape[0]):
                running_acc_025iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.25)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.5)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_topk_acc_025iou = topk_ious[i][np.logical_and(
                    np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]),
                    topk_ious[i] >= 0.25)].shape[0] / topk_ious[i][np.logical_and(masks[i] == multiple_dict[k],
                                                                                  others[i] == others_dict[k_o])].shape[
                                              0] if np.sum(
                    np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_topk_acc_05iou = topk_ious[i][np.logical_and(
                    np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]),
                    topk_ious[i] >= 0.5)].shape[
                                             0] / topk_ious[i][np.logical_and(masks[i] == multiple_dict[k],
                                                                              others[i] == others_dict[k_o])].shape[
                                             0] if np.sum(
                    np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                # store
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)
                topk_acc_025ious.append(running_topk_acc_025iou)
                topk_acc_05ious.append(running_topk_acc_05iou)

            if k not in scores:
                scores[k] = {k_o: {} for k_o in others_dict.keys()}

            scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)
            scores[k][k_o]["topk_acc@0.25iou"] = np.mean(topk_acc_025ious)
            scores[k][k_o]["topk_acc@0.5iou"] = np.mean(topk_acc_05ious)

        topk_acc_025ious, topk_acc_05ious, acc_025ious, acc_05ious = [], [], [], []
        for i in range(masks.shape[0]):
            running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_topk_acc_025iou = \
            topk_ious[i][np.logical_and(masks[i] == multiple_dict[k], topk_ious[i] >= 0.25)].shape[0] \
            / topk_ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(
                masks[i] == multiple_dict[k]) > 0 else 0
            running_topk_acc_05iou = \
            topk_ious[i][np.logical_and(masks[i] == multiple_dict[k], topk_ious[i] >= 0.5)].shape[0] \
            / topk_ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(
                masks[i] == multiple_dict[k]) > 0 else 0

            # store
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)
            topk_acc_025ious.append(running_topk_acc_025iou)
            topk_acc_05ious.append(running_topk_acc_05iou)

        scores[k]["overall"] = {}
        scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)
        scores[k]["overall"]["topk_acc@0.25iou"] = np.mean(topk_acc_025ious)
        scores[k]["overall"]["topk_acc@0.5iou"] = np.mean(topk_acc_05ious)

    scores["overall"] = {}
    for k_o in others_dict.keys():
        topk_acc_025ious, topk_acc_05ious, acc_025ious, acc_05ious = [], [], [], []
        for i in range(masks.shape[0]):
            running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_topk_acc_025iou = \
            topk_ious[i][np.logical_and(others[i] == others_dict[k_o], topk_ious[i] >= 0.25)].shape[0] \
            / topk_ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(
                others[i] == others_dict[k_o]) > 0 else 0
            running_topk_acc_05iou = \
            topk_ious[i][np.logical_and(others[i] == others_dict[k_o], topk_ious[i] >= 0.5)].shape[0] \
            / topk_ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(
                others[i] == others_dict[k_o]) > 0 else 0

            # store
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)
            topk_acc_025ious.append(running_topk_acc_025iou)
            topk_acc_05ious.append(running_topk_acc_05iou)

        # aggregate
        scores["overall"][k_o] = {}
        scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)
        scores["overall"][k_o]["topk_acc@0.25iou"] = np.mean(topk_acc_025ious)
        scores["overall"][k_o]["topk_acc@0.5iou"] = np.mean(topk_acc_05ious)
   
    topk_acc_025ious, topk_acc_05ious, acc_025ious, acc_05ious = [], [], [], []
    topk_rec_acc_025ious, topk_rec_acc_05ious, rec_acc_025ious, rec_acc_05ious = [], [], [], []
    # topk_rand_acc_025ious, topk_rand_acc_05ious, rand_acc_025ious, rand_acc_05ious = [], [], [], []
    for i in range(masks.shape[0]):
        running_acc_025iou = ious[i][ious[i] >= 0.25].shape[0] / ious[i].shape[0]
        running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]
        running_topk_acc_025iou = topk_ious[i][topk_ious[i] >= 0.25].shape[0] / topk_ious[i].shape[0]
        running_topk_acc_05iou = topk_ious[i][topk_ious[i] >= 0.5].shape[0] / topk_ious[i].shape[0]
        if not args.is_eval:
            running_rec_acc_025iou = rec_ious[i][rec_ious[i] >= 0.25].shape[0] / rec_ious[i].shape[0]
            running_rec_acc_05iou = rec_ious[i][rec_ious[i] >= 0.5].shape[0] / rec_ious[i].shape[0]
            running_topk_rec_acc_025iou = topk_rec_ious[i][topk_rec_ious[i] >= 0.25].shape[0] / topk_rec_ious[i].shape[0]
            running_topk_rec_acc_05iou = topk_rec_ious[i][topk_rec_ious[i] >= 0.5].shape[0] / topk_rec_ious[i].shape[0]
        # if args.eval_rand:
        #     running_rand_acc_025iou = rand_ious[i][rand_ious[i] >= 0.25].shape[0] / rand_ious[i].shape[0]
        #     running_rand_acc_05iou = rand_ious[i][rand_ious[i] >= 0.5].shape[0] / rand_ious[i].shape[0]
        #     running_topk_rand_acc_025iou = topk_rand_ious[i][topk_rand_ious[i] >= 0.25].shape[0] / topk_rand_ious[i].shape[0]
        #     running_topk_rand_acc_05iou = topk_rand_ious[i][topk_rand_ious[i] >= 0.5].shape[0] / topk_rand_ious[i].shape[0]

        # store
        acc_025ious.append(running_acc_025iou)
        acc_05ious.append(running_acc_05iou)
        topk_acc_025ious.append(running_topk_acc_025iou)
        topk_acc_05ious.append(running_topk_acc_05iou)
        if not args.is_eval:
            rec_acc_025ious.append(running_rec_acc_025iou)
            rec_acc_05ious.append(running_rec_acc_05iou)
            topk_rec_acc_025ious.append(running_topk_rec_acc_025iou)
            topk_rec_acc_05ious.append(running_topk_rec_acc_05iou)
        # if args.eval_rand:
        #     rand_acc_025ious.append(running_rand_acc_025iou)
        #     rand_acc_05ious.append(running_rand_acc_05iou)
        #     topk_rand_acc_025ious.append(running_topk_rand_acc_025iou)
        #     topk_rand_acc_05ious.append(running_topk_rand_acc_05iou)

    # aggregate
    scores["overall"]["overall"] = {}
    scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
    scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)
    scores["overall"]["overall"]["topk_acc@0.25iou"] = np.mean(topk_acc_025ious)
    scores["overall"]["overall"]["topk_acc@0.5iou"] = np.mean(topk_acc_05ious)
    if not args.is_eval:
        scores["overall"]["overall"]["rec_acc@0.25iou"] = np.mean(rec_acc_025ious)
        scores["overall"]["overall"]["rec_acc@0.5iou"] = np.mean(rec_acc_05ious)
        scores["overall"]["overall"]["topk_rec_acc@0.25iou"] = np.mean(topk_rec_acc_025ious)
        scores["overall"]["overall"]["topk_rec_acc@0.5iou"] = np.mean(topk_rec_acc_05ious)
    # if args.eval_rand:
    #     scores["overall"]["overall"]["rand_acc@0.25iou"] = np.mean(rand_acc_025ious)
    #     scores["overall"]["overall"]["rand_acc@0.5iou"] = np.mean(rand_acc_05ious)
    #     scores["overall"]["overall"]["topk_rand_acc@025iou"] = np.mean(topk_rand_acc_025ious)
    #     scores["overall"]["overall"]["topk_rand_acc@05iou"] = np.mean(topk_rand_acc_05ious)

    # report
    print("\nstats:")
    for k_s in stats.keys():
        for k_o in stats[k_s].keys():
            print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

    for k_s in scores.keys():
        print("\n{}:".format(k_s))
        for k_m in scores[k_s].keys():
            for metric in scores[k_s][k_m].keys():
                print("{} | {} | {}: {}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

    print("\nlanguage classification accuracy: {}".format(np.mean(lang_acc)))

    print("\n mean forward time: {}".format(mean_forward_time))


if __name__ == "__main__":
    assert CONF.lang_num_max == 1, 'lang max num == 1; avoid bugs'
    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # evaluate
    # if args.reference: eval_ref(args)
    # if args.detection: eval_det(args)
    eval_ref(CONF)

