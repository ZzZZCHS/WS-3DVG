# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lib.configs.config import CONF
from .loss_detection import compute_objectness_loss
from .loss_grounding import compute_reference_loss, compute_lang_classification_loss
from .loss_reconstruct import reconstruct_loss, weakly_supervised_loss, reconstruct_score
from .mil import MILNCELoss, MILMARGINLoss

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

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    data_dict["pos_ratio"] = torch.sum(objectness_label.float())/float(total_num_proposal)
    data_dict["neg_ratio"] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict["pos_ratio"]

    lang_num = data_dict["lang_num"]
    # print(lang_num.float().mean())
    bs, len_num_max = data_dict["ground_lang_feat_list"].shape[:2]
    len_num_mask = torch.ones(bs, len_num_max).to(data_dict["all_masks_list"].device)
    for i in range(bs):
        len_num_mask[i][lang_num[i]:].fill_(0.)

    if not is_eval:
        rec_word_logits = data_dict["rec_word_logits"]
        gt_idx = data_dict["ground_lang_ids_list"]
        masks_list = data_dict["all_masks_list"]
        target_ids = data_dict["target_ids"].flatten(0, 1)
        all_scores = data_dict["cluster_ref"]
        num_target = target_ids.shape[1]
        rec_score = torch.zeros(target_ids.shape).to(target_ids.device)
        for i in range(num_target):
            rec_word_logits_i = rec_word_logits[:, :, i, :, :]
            rec_score[:, i] = reconstruct_score(rec_word_logits_i, gt_idx, masks_list, len_num_mask)
        data_dict["rec_score"] = rec_score
        weak_loss = weakly_supervised_loss(rec_score, all_scores, data_dict, len_num_mask, CONF)
        rec_loss = reconstruct_loss(rec_score, data_dict["epoch"], len_num_mask)
        data_dict["rec_loss"] = rec_loss
        data_dict["weak_loss"] = weak_loss

    data_dict, _, _, cluster_labels = compute_reference_loss(data_dict, config, no_reference=True)
    data_dict["cluster_labels"] = cluster_labels
    data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)

    # data_dict["contra_loss"] = contra_loss(data_dict)
    # if torch.isnan(data_dict["contra_loss"]):
    #     print(data_dict["contra_loss"])
    if CONF.mil_type == "nce":
        data_dict["nce_loss"] = MILNCELoss(data_dict, len_num_mask)
    else:
        data_dict["nce_loss"] = MILMARGINLoss(data_dict, len_num_mask)

    # Final loss function
    loss = torch.zeros(1)[0].cuda()
    if not is_eval:
        if not CONF.no_mil:
            loss += 2 * data_dict["nce_loss"]
        if not CONF.no_text:
            loss += 2 * data_dict["lang_loss"]
        if not CONF.no_recon and data_dict["epoch"] >= 0:
            loss += data_dict["rec_loss"]
        if not CONF.no_distill and data_dict["epoch"] >= 0:
            loss += data_dict["weak_loss"]

    data_dict["loss"] = loss

    return data_dict

