# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.ap_helper.ap_helper_fcos import parse_predictions
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou
import torch.nn.functional as F

from lib.configs.config import CONF


def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one reference prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = box3d_iou(pred_bbox, gt_bbox)

    return iou


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d


@torch.no_grad()
def get_eval(data_dict, config, reference, is_eval=False, use_lang_classifier=False, use_cat_rand=False,
             use_best=False, post_processing=None, use_random=False, use_best_in_cat=False, k=3):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    objectness_preds_batch = data_dict["objectness_pred"]
    # objectness_labels_batch = data_dict['objectness_label'].long()
    sem_cls_pred = data_dict['sem_cls_scores'].argmax(-1)  # (B,K)
    num_proposal = sem_cls_pred.shape[1]

    if post_processing:
        _ = parse_predictions(data_dict, post_processing)
        nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()

        # construct valid mask
        pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        # label_masks = (objectness_labels_batch == 1).float()
    else:
        # construct valid mask
        pred_masks = (objectness_preds_batch == 1).float()
        # label_masks = (objectness_labels_batch == 1).float()

    batch_size, len_num_max = data_dict['ref_center_label_list'].shape[:2]

    pred_mask1 = pred_masks.reshape(batch_size, 1, -1).repeat(1, len_num_max, 1).reshape(batch_size * len_num_max, -1)

    if not is_eval:
        target_ids = data_dict["target_ids"]
        bs, len_num_max, num_target = target_ids.shape
        target_ids = target_ids.resize(bs * len_num_max, num_target)
        target_object_mask = pred_mask1.new_zeros(*pred_mask1.size())
        target_object_mask.scatter_(1, target_ids, 1.)

        rec_score = pred_mask1.new_zeros(*pred_mask1.size())
        rec_score.scatter_(1, target_ids, data_dict["rec_score"])
        max_score = torch.max(data_dict["rec_score"], dim=-1, keepdim=True)[0] + 1.
        masked_pred_rec = rec_score * target_object_mask + (1-target_object_mask) * max_score
        pred_ref_rec = torch.argmin(masked_pred_rec, 1)
        pred_ref_rec_topk = torch.topk(masked_pred_rec, k=k, dim=1, largest=False)[1]

    if use_cat_rand:
        target_ids = data_dict["target_ids"]
        bs, len_num_max, num_target = target_ids.shape
        target_ids = target_ids.resize(bs * len_num_max, num_target)
        target_object_mask = pred_mask1.new_zeros(*pred_mask1.size())
        target_object_mask.scatter_(1, target_ids, 1.)
        candidate_mask = pred_mask1 * target_object_mask  # bs * len_num_max, num_proposal

        cluster_preds = torch.zeros(bs*len_num_max, num_proposal).cuda()
        for i in range(cluster_preds.shape[0]):
            candidates = torch.arange(num_proposal).cuda()[candidate_mask[i].bool()]
            if candidates.shape[0] == 0:
                candidates = torch.arange(num_proposal).cuda()[target_object_mask[i].bool()]
            try:
                chosen_idx = torch.randperm(candidates.shape[0]).cuda()[0]
                chosen_candidate = candidates[chosen_idx]
                cluster_preds[i, chosen_candidate] = 1
            except IndexError:
                cluster_preds[i, candidates] = 1
        pred_ref_rand = torch.argmax(cluster_preds, -1)  # (B,)

    # masked_pred = data_dict['cluster_ref'] * candidate_mask + data_dict['cluster_ref'] * target_object_mask * 1e-7
    masked_pred = data_dict["cluster_ref"] * pred_mask1
    pred_ref = torch.argmax(masked_pred, 1)  # (B,)
    pred_ref_topk = torch.topk(masked_pred, k=k, dim=1)[1]

    if CONF.no_distill:
        # pred_ref = torch.empty(batch_size*len_num_max).long()
        # pred_ref_topk = torch.empty(batch_size*len_num_max, k).long()
        # object_feat = data_dict["bbox_feature"]  # bs, N, dim
        # lang_feat = data_dict["lang_emb"]  # bs*num, dim
        # if CONF.mil_type == "nce":
        #     for i in range(batch_size):
        #         for j in range(len_num_max):
        #             idx = i * len_num_max + j
        #             x = torch.matmul(lang_feat[idx], object_feat[i].t())
        #             pred_ref[idx] = torch.argmax(x, 0)
        #             pred_ref_topk[idx, :] = torch.topk(x, k=k, dim=0)[1]
        #             # print(pred_ref[i])
        # else:
        #     lang_feat = lang_feat.unsqueeze(1).expand(-1, num_proposal, -1)
        #     for i in range(batch_size):
        #         for j in range(len_num_max):
        #             idx = i * len_num_max + j
        #             x = F.cosine_similarity(lang_feat[idx], object_feat[i])
        #             pred_ref[idx] = torch.argmax(x, 0)
        #             pred_ref_topk[idx, :] = torch.topk(x, k=k, dim=0)[1]
        weights = data_dict["coarse_weights"]
        pred_ref = torch.argmax(weights, 1)
        pred_ref_topk = torch.topk(weights, k=k, dim=1)[1]


    pred_heading = data_dict['pred_heading'].detach().cpu().numpy() # B,num_proposal
    pred_center = data_dict['pred_center'].detach().cpu().numpy() # (B, num_proposal)
    pred_box_size = data_dict['pred_size'].detach().cpu().numpy() # (B, num_proposal, 3)


    # store
    # data_dict["pred_mask"] = pred_masks
    # data_dict["label_mask"] = label_masks

    #print("ref_box_label", data_dict["ref_box_label"].shape, data_dict["ref_box_label_list"].shape)
    #gt_ref = torch.argmax(data_dict["ref_box_label"], 1)
    gt_ref = torch.argmax(data_dict["ref_box_label_list"], -1)
    gt_center = data_dict['center_label']  # (B,MAX_NUM_OBJ,3)
    gt_heading_class = data_dict['heading_class_label']  # B,K2
    gt_heading_residual = data_dict['heading_residual_label']  # B,K2
    gt_size_class = data_dict['size_class_label']  # B,K2
    gt_size_residual = data_dict['size_residual_label']  # B,K2,3
    lang_num = data_dict["lang_num"]

    ious = []
    multiple = []
    others = []
    pred_bboxes = []
    gt_bboxes = []
    #print("pred_ref", pred_ref.shape, gt_ref.shape)
    pred_ref = pred_ref.reshape(batch_size, len_num_max)

    topk_ious = []
    pred_ref_topk = pred_ref_topk.reshape(batch_size, len_num_max, -1)

    if not is_eval:
        rec_ious = []
        topk_rec_ious = []
        pred_ref_rec = pred_ref_rec.reshape(batch_size, len_num_max)
        pred_ref_rec_topk = pred_ref_rec_topk.reshape(batch_size, len_num_max, -1)
    if use_random:
        topk_rand_ious = []
        rand_ious = []
    if use_cat_rand:
        cat_rand_ious = []
        pred_ref_rand = pred_ref_rand.reshape(batch_size, len_num_max)
    if use_best:
        best_ious = []
    if use_best_in_cat:
        best_cat_ious = []

    # pred
    for i in range(batch_size):
        # candidates = torch.arange(num_proposal)[pred_masks[i].bool()]
        for j in range(len_num_max):
            if j < lang_num[i]:
                gt_ref_idx = gt_ref[i][j]
                gt_obb = config.param2obb(
                    gt_center[i, gt_ref_idx, 0:3].detach().cpu().numpy(),
                    gt_heading_class[i, gt_ref_idx].detach().cpu().numpy(),
                    gt_heading_residual[i, gt_ref_idx].detach().cpu().numpy(),
                    gt_size_class[i, gt_ref_idx].detach().cpu().numpy(),
                    gt_size_residual[i, gt_ref_idx].detach().cpu().numpy()
                )
                gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])
                pred_ref_idx = pred_ref[i][j]
                pred_center_ids = pred_center[i][pred_ref_idx]
                pred_heading_ids = pred_heading[i][pred_ref_idx]
                pred_box_size_ids = pred_box_size[i][pred_ref_idx]
                pred_bbox = get_3d_box(pred_box_size_ids, pred_heading_ids, pred_center_ids)
                iou = eval_ref_one_sample(pred_bbox, gt_bbox)
                pred_bbox = construct_bbox_corners(pred_center_ids, pred_box_size_ids)
                ious.append(iou)

                # NOTE: get_3d_box() will return problematic bboxes
                gt_bbox = construct_bbox_corners(gt_obb[0:3], gt_obb[3:6])
                pred_bboxes.append(pred_bbox)
                gt_bboxes.append(gt_bbox)

                # construct the multiple mask
                if CONF.dataset == "ScanRefer":
                    multiple.append(data_dict["unique_multiple_list"][i][j].item())
                else:
                    multiple.append(data_dict["easy_hard_list"][i][j].item())

                # construct the others mask
                if CONF.dataset == "ScanRefer":
                    flag = 1 if data_dict["object_cat_list"][i][j] == 17 else 0
                    others.append(flag)
                else:
                    others.append(data_dict["dep_indep_list"][i][j].item())

                # topk
                max_iou = 0
                for pred_ref_idx in pred_ref_topk[i][j]:
                    pred_center_ids = pred_center[i][pred_ref_idx]
                    pred_heading_ids = pred_heading[i][pred_ref_idx]
                    pred_box_size_ids = pred_box_size[i][pred_ref_idx]
                    pred_bbox = get_3d_box(pred_box_size_ids, pred_heading_ids, pred_center_ids)
                    iou = eval_ref_one_sample(pred_bbox, gt_bbox)
                    if iou > max_iou:
                        max_iou = iou
                topk_ious.append(max_iou)

                # rand in cat
                if use_cat_rand:
                    pred_ref_idx = pred_ref_rand[i][j]
                    pred_center_ids = pred_center[i][pred_ref_idx]
                    pred_heading_ids = pred_heading[i][pred_ref_idx]
                    pred_box_size_ids = pred_box_size[i][pred_ref_idx]
                    pred_bbox = get_3d_box(pred_box_size_ids, pred_heading_ids, pred_center_ids)
                    iou = eval_ref_one_sample(pred_bbox, gt_bbox)
                    cat_rand_ious.append(iou)

                # best in cat
                if use_best_in_cat:
                    max_iou = 0
                    for k in range(target_ids.shape[1]):
                        pred_ref_idx = target_ids[i * len_num_max + j][k]
                        pred_center_ids = pred_center[i][pred_ref_idx]
                        pred_heading_ids = pred_heading[i][pred_ref_idx]
                        pred_box_size_ids = pred_box_size[i][pred_ref_idx]
                        pred_bbox = get_3d_box(pred_box_size_ids, pred_heading_ids, pred_center_ids)
                        iou = eval_ref_one_sample(pred_bbox, gt_bbox)
                        if iou > max_iou:
                            max_iou = iou
                    best_cat_ious.append(max_iou)

                # use rec
                if not is_eval:
                    pred_ref_rec_idx = pred_ref_rec[i][j]
                    pred_center_rec_ids = pred_center[i][pred_ref_rec_idx]
                    pred_heading_rec_ids = pred_heading[i][pred_ref_rec_idx]
                    pred_box_size_rec_ids = pred_box_size[i][pred_ref_rec_idx]
                    pred_bbox_rec = get_3d_box(pred_box_size_rec_ids, pred_heading_rec_ids, pred_center_rec_ids)
                    rec_iou = eval_ref_one_sample(pred_bbox_rec, gt_bbox)
                    rec_ious.append(rec_iou)

                    max_iou = 0.
                    for pred_ref_rec_idx in pred_ref_rec_topk[i][j]:
                        pred_center_rec_ids = pred_center[i][pred_ref_rec_idx]
                        pred_heading_rec_ids = pred_heading[i][pred_ref_rec_idx]
                        pred_box_size_rec_ids = pred_box_size[i][pred_ref_rec_idx]
                        pred_bbox_rec = get_3d_box(pred_box_size_rec_ids, pred_heading_rec_ids, pred_center_rec_ids)
                        iou = eval_ref_one_sample(pred_bbox_rec, gt_bbox)
                        if iou > max_iou:
                            max_iou = iou
                    topk_rec_ious.append(max_iou)

                # use best
                if use_best:
                    max_iou = 0
                    for pred_ref_idx in range(256):
                        pred_center_ids = pred_center[i][pred_ref_idx]
                        pred_heading_ids = pred_heading[i][pred_ref_idx]
                        pred_box_size_ids = pred_box_size[i][pred_ref_idx]
                        pred_bbox = get_3d_box(pred_box_size_ids, pred_heading_ids, pred_center_ids)
                        iou = eval_ref_one_sample(pred_bbox, gt_bbox)
                        if iou > max_iou:
                            max_iou = iou
                    best_ious.append(max_iou)

                # use random
                if use_random:
                    pred_ref_idx = random.randint(0, 255)
                    pred_center_ids = pred_center[i][pred_ref_idx]
                    pred_heading_ids = pred_heading[i][pred_ref_idx]
                    pred_box_size_ids = pred_box_size[i][pred_ref_idx]
                    pred_bbox = get_3d_box(pred_box_size_ids, pred_heading_ids, pred_center_ids)
                    iou = eval_ref_one_sample(pred_bbox, gt_bbox)
                    rand_ious.append(iou)

                    max_iou = 0.
                    for _ in range(k):
                        pred_ref_idx = random.randint(0, 255)
                        pred_center_ids = pred_center[i][pred_ref_idx]
                        pred_heading_ids = pred_heading[i][pred_ref_idx]
                        pred_box_size_ids = pred_box_size[i][pred_ref_idx]
                        pred_bbox = get_3d_box(pred_box_size_ids, pred_heading_ids, pred_center_ids)
                        iou = eval_ref_one_sample(pred_bbox, gt_bbox)
                        if iou > max_iou:
                            max_iou = iou
                    topk_rand_ious.append(max_iou)


    # lang
    # if use_lang_classifier:
    #     object_cat = data_dict["object_cat_list"].reshape(batch_size*len_num_max)
    #     data_dict["lang_acc"] = (torch.argmax(data_dict['lang_scores'], 1) == object_cat).float().mean()
    # else:
    #     data_dict["lang_acc"] = torch.zeros(1)[0].cuda()

    # store
    data_dict["ref_iou"] = ious
    data_dict["ref_iou_0.1"] = np.array(ious)[np.array(ious) >= 0.1].shape[0] / np.array(ious).shape[0]
    data_dict["ref_iou_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
    data_dict["ref_iou_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]

    data_dict["topk_iou"] = topk_ious
    data_dict["topk_iou_0.1"] = np.array(topk_ious)[np.array(topk_ious) >= 0.1].shape[0] / np.array(topk_ious).shape[0]
    data_dict["topk_iou_0.25"] = np.array(topk_ious)[np.array(topk_ious) >= 0.25].shape[0] / np.array(topk_ious).shape[0]
    data_dict["topk_iou_0.5"] = np.array(topk_ious)[np.array(topk_ious) >= 0.5].shape[0] / np.array(topk_ious).shape[0]
    # data_dict["topk_iou_0.1"] = 0.
    # data_dict["topk_iou_0.25"] = 0.
    # data_dict["topk_iou_0.5"] = 0.

    if not is_eval:
        data_dict["rec_iou"] = rec_ious
        data_dict["rec_iou_0.1"] = np.array(rec_ious)[np.array(rec_ious) >= 0.1].shape[0] / np.array(rec_ious).shape[0]
        data_dict["rec_iou_0.25"] = np.array(rec_ious)[np.array(rec_ious) >= 0.25].shape[0] / np.array(rec_ious).shape[0]
        data_dict["rec_iou_0.5"] = np.array(rec_ious)[np.array(rec_ious) >= 0.5].shape[0] / np.array(rec_ious).shape[0]
        data_dict["topk_rec_iou"] = topk_rec_ious
    else:
        data_dict["rec_iou"] = []
        data_dict["rec_iou_0.1"] = 0.
        data_dict["rec_iou_0.25"] = 0.
        data_dict["rec_iou_0.5"] = 0.
        data_dict["topk_rec_iou"] = []

    if use_cat_rand:
        data_dict["rand_iou_0.1"] = np.array(cat_rand_ious)[np.array(cat_rand_ious) >= 0.1].shape[0] / np.array(cat_rand_ious).shape[0]
        data_dict["rand_iou_0.25"] = np.array(cat_rand_ious)[np.array(cat_rand_ious) >= 0.25].shape[0] / np.array(cat_rand_ious).shape[0]
        data_dict["rand_iou_0.5"] = np.array(cat_rand_ious)[np.array(cat_rand_ious) >= 0.5].shape[0] / np.array(cat_rand_ious).shape[0]
    else:
        data_dict["rand_iou_0.1"] = 0.
        data_dict["rand_iou_0.25"] = 0.
        data_dict["rand_iou_0.5"] = 0.

    if use_best_in_cat:
        data_dict["upper_iou_0.1"] = np.array(best_cat_ious)[np.array(best_cat_ious) >= 0.1].shape[0] / np.array(best_cat_ious).shape[0]
        data_dict["upper_iou_0.25"] = np.array(best_cat_ious)[np.array(best_cat_ious) >= 0.25].shape[0] / np.array(best_cat_ious).shape[0]
        data_dict["upper_iou_0.5"] = np.array(best_cat_ious)[np.array(best_cat_ious) >= 0.5].shape[0] / np.array(best_cat_ious).shape[0]
    else:
        data_dict["upper_iou_0.1"] = 0.
        data_dict["upper_iou_0.25"] = 0.
        data_dict["upper_iou_0.5"] = 0.

    if use_random:
        data_dict["ref_iou_0.1"] = np.array(rand_ious)[np.array(rand_ious) >= 0.1].shape[0] / np.array(rand_ious).shape[0]
        data_dict["ref_iou_0.25"] = np.array(rand_ious)[np.array(rand_ious) >= 0.25].shape[0] / np.array(rand_ious).shape[0]
        data_dict["ref_iou_0.5"] = np.array(rand_ious)[np.array(rand_ious) >= 0.5].shape[0] / np.array(rand_ious).shape[0]
        data_dict["rand_iou"] = rand_ious
        data_dict["topk_rand_iou"] = topk_rand_ious
    else:
        data_dict["rand_iou"] = []
        data_dict["topk_rand_iou"] = []

    if use_best:
        data_dict["ref_iou_0.1"] = np.array(best_ious)[np.array(best_ious) >= 0.1].shape[0] / np.array(best_ious).shape[0]
        data_dict["ref_iou_0.25"] = np.array(best_ious)[np.array(best_ious) >= 0.25].shape[0] / np.array(best_ious).shape[0]
        data_dict["ref_iou_0.5"] = np.array(best_ious)[np.array(best_ious) >= 0.5].shape[0] / np.array(best_ious).shape[0]
        data_dict["best_iou"] = best_ious
    else:
        data_dict["best_iou"] = []

    data_dict["ref_multiple_mask"] = multiple
    data_dict["ref_others_mask"] = others
    data_dict["pred_bboxes"] = pred_bboxes
    data_dict["gt_bboxes"] = gt_bboxes

    # if use_cat_rand or use_random or use_best_in_cat:
    #     print(data_dict["ref_iou_0.1"], data_dict["ref_iou_0.25"], data_dict["ref_iou_0.5"])
    # --------------------------------------------
    # Some other statistics
    # obj_pred_val = torch.argmax(data_dict['objectness_scores'], 2)  # B,K
    # obj_acc = torch.sum(
    #     (obj_pred_val == data_dict['objectness_label'].long()).float() * data_dict['objectness_mask']) / (
    #                       torch.sum(data_dict['objectness_mask']) + 1e-6)
    data_dict['obj_acc'] = torch.zeros(1)[0].cuda()
    # detection semantic classification
    # sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1,
    #                              data_dict['object_assignment'])  # select (B,K) from (B,K2)
    # sem_match = (sem_cls_label == sem_cls_pred).float()
    # data_dict["sem_acc"] = (sem_match * data_dict["pred_mask"]).sum() / (data_dict["pred_mask"].sum() + 1e-7)
    data_dict["sem_acc"] = torch.zeros(1)[0].cuda()

    return data_dict
