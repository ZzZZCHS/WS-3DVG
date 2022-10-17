# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

from lib.configs.config_joint import CONF
from .loss_detection import compute_vote_loss, compute_objectness_loss, compute_box_loss, compute_box_and_sem_cls_loss
from .loss_captioning import compute_cap_loss
from .loss_grounding import compute_reference_loss, compute_lang_classification_loss
from .loss_reconstruct import reconstruct_loss, weakly_supervised_loss, reconstruct_score

FAR_THRESHOLD = 0.3
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


def get_joint_loss(data_dict, config, is_eval=False):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    data_dict["pos_ratio"] = torch.sum(objectness_label.float())/float(total_num_proposal)
    data_dict["neg_ratio"] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict["pos_ratio"]

    # Box loss and sem cls loss
    # heading_cls_loss, heading_reg_loss, size_distance_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    # box_loss = 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * sem_cls_loss
    # box_loss = box_loss + 20 * size_distance_loss
    # center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(
    #     data_dict, config)
    # # box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss
    # box_loss = center_loss + heading_reg_loss + size_reg_loss
    # # print(center_loss, heading_reg_loss, size_reg_loss)
    # # objectness; Nothing
    # obj_pred_val = torch.argmax(data_dict["objectness_scores"], 2) # B,K
    # obj_acc = torch.sum((obj_pred_val==data_dict["objectness_label"].long()).float()*data_dict["objectness_mask"])/(torch.sum(data_dict["objectness_mask"])+1e-6)
    # data_dict["obj_acc"] = obj_acc
    #
    # data_dict["vote_loss"] = vote_loss
    # data_dict["objectness_loss"] = objectness_loss
    # data_dict["center_loss"] = center_loss
    # data_dict["heading_cls_loss"] = heading_cls_loss
    # data_dict["heading_reg_loss"] = heading_reg_loss
    # data_dict["size_cls_loss"] = size_cls_loss
    # data_dict["size_reg_loss"] = size_reg_loss
    # # data_dict["size_distance_loss"] = size_distance_loss
    # data_dict["sem_cls_loss"] = sem_cls_loss
    # data_dict["box_loss"] = box_loss

    # data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
    # data_dict["cap_loss"] = torch.zeros(1)[0].to(device)
    # data_dict["cap_acc"] = torch.zeros(1)[0].to(device)
    # data_dict["pred_ious"] = torch.zeros(1)[0].to(device)
    # data_dict["ori_loss"] = torch.zeros(1)[0].to(device)
    # data_dict["ori_acc"] = torch.zeros(1)[0].to(device)
    # data_dict["dist_loss"] = torch.zeros(1)[0].to(device)
    # data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    if not is_eval:
        rec_word_logits = data_dict["rec_word_logits"]
        gt_idx = data_dict["ground_lang_ids_list"]
        # rec_word_feat = data_dict["rec_word_feat"]
        # ori_feat = data_dict["ground_lang_feat_list"]
        # ori_feat = data_dict["enc_lang_feat"]
        masks_list = data_dict["all_masks_list"]
        target_obj_scores = data_dict["target_scores"]
        all_scores = data_dict["cluster_ref"]
        # target_ids = data_dict["target_ids"].resize(*target_obj_scores.shape)
        # print(rec_word_logits.shape, gt_idx.shape, masks_list.shape, target_obj_scores.shape)
        num_target = target_obj_scores.shape[1]
        rec_score = torch.zeros_like(target_obj_scores).to(target_obj_scores.device)
        # zeros = torch.zeros_like(target_ids).to(target_ids.device).float()
        # all_scores = all_scores.softmax(dim=-1)
        # all_scores = all_scores.scatter(dim=1, src=zeros, index=target_ids)
        for i in range(num_target):
            rec_word_logits_i = rec_word_logits[:, :, i, :, :]
            # rec_word_feat_i = rec_word_feat[:, :, i, :, :]
            rec_score[:, i] = reconstruct_score(rec_word_logits_i, gt_idx, masks_list)
            # rec_score[:, i] = reconstruct_score(rec_word_feat_i, ori_feat, masks_list)
        data_dict["rec_score"] = rec_score
        weak_loss = weakly_supervised_loss(rec_score, all_scores, data_dict)
        rec_loss = reconstruct_loss(rec_score)
        data_dict["rec_loss"] = rec_loss
        data_dict["weak_loss"] = weak_loss

    data_dict, _, _, cluster_labels = compute_reference_loss(data_dict, config, no_reference=True)
    # lang_count = data_dict['ref_center_label_list'].shape[1]
    # data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda().repeat(lang_count, 1)
    data_dict["cluster_labels"] = cluster_labels
    # print(data_dict["cluster_ref"][0])
    # data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda().repeat(lang_count, 1)
    # print(data_dict["cluster_ref"][0])

    # if reference and use_lang_classifier:
    #     data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    # else:
    # data_dict["lang_loss"] = torch.zeros(1)[0].cuda()
    data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)

    # Final loss function
    loss = torch.zeros(1)[0].cuda()
    if not is_eval:
        loss += 3 * data_dict["lang_loss"]
        if data_dict["epoch"] >= 3:
            loss += data_dict["rec_loss"]
        if data_dict["epoch"] >= 4:
            loss += data_dict["weak_loss"]
    # if use_lang_classifier:
    #     loss += data_dict["lang_loss"]

    data_dict["loss"] = loss

    return data_dict

