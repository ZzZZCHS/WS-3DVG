import torch
import torch.nn as nn
import numpy as np
import sys
import os

from models.base_module.backbone_module import Pointnet2Backbone
from models.base_module.voting_module import VotingModule
from models.base_module.lang_module import LangModule

from models.proposal_module.proposal_module_fcos import ProposalModule
from models.proposal_module.relation_module import RelationModule
from models.refnet.match_module import MatchModule
from models.capnet.caption_module import SceneCaptionModule, TopDownSceneCaptionModule
from models.recnet.reconstruct_module import ReconstructModule
from models.recnet.contra_module import ContraModule
from models.groupfree import GroupFreeDetector

from lib.ap_helper.ap_helper_fcos import parse_predictions


class JointNet(nn.Module):
    def __init__(self, vocabulary,
                 input_feature_dim=0, width=1,
                 num_proposal=128, num_target=32, num_rec_other=16, num_locals=-1, vote_factor=1, sampling="vote_fps",
                 no_caption=False, use_topdown=False, query_mode="corner",
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, hidden_size=256, dataset_config=None, args=None):
        super().__init__()

        # self.num_class = num_class
        # self.num_heading_bin = num_heading_bin
        # self.num_size_cluster = num_size_cluster
        # self.mean_size_arr = mean_size_arr
        # assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.no_caption = no_caption
        self.dataset_config = dataset_config
        self.num_target = num_target
        self.num_other = num_proposal - num_target
        self.num_rec_other = min(num_rec_other, self.num_other)
        self.vocab_size = len(vocabulary["idx2word"])
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        # --------- VOTENET PROPOSAL GENERATION ---------
        # Backbone point feature learning
        # self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        # self.vgen = VotingModule(self.vote_factor, 256)

        # Fix VoteNet
        # for _p in self.parameters():
        #     _p.requires_grad = False

        # Vote aggregation and object proposal
        # self.pnet = ProposalModule(num_proposal=num_proposal, sampling=sampling, dataset_config=dataset_config, num_target=num_target)

        # --------- GroupFree PROPOSAL GENERATION ---------
        self.group_free = GroupFreeDetector(
            input_feature_dim=input_feature_dim,
            width=width,
            num_proposal=num_proposal,
            dataset_config=dataset_config
        )
        for _p in self.group_free.parameters():
            _p.requires_grad = False

        self.relation = RelationModule(hidden_size=hidden_size, num_proposals=num_proposal, det_channel=hidden_size)  # bef 256

        self.lang = LangModule(dataset_config.num_class, use_lang_classifier, use_bidir, emb_size, hidden_size)
        # for _p in self.lang.parameters():
        #     _p.requires_grad = False
        self.contranet = ContraModule(hidden_size=hidden_size)

        self.recnet = ReconstructModule(vocab_size=self.vocab_size, hidden_size=hidden_size)

        self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size, num_target=num_target, hidden_size=hidden_size)  # bef 256

        # self.caption = TopDownSceneCaptionModule(vocabulary, embeddings, emb_size, 128, caption_hidden_size,
        #                                          num_proposal, num_locals, query_mode, use_relation)

        if args.distribute:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward(self, data_dict, use_tf=True, is_eval=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds,
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        # # hough voting
        # data_dict = self.backbone_net(data_dict)
        # xyz = data_dict["fp2_xyz"]
        # features = data_dict["fp2_features"]
        # data_dict["seed_inds"] = data_dict["fp2_inds"]
        # data_dict["seed_xyz"] = xyz
        # data_dict["seed_features"] = features
        # xyz, features = self.vgen(xyz, features)
        # features_norm = torch.norm(features, p=2, dim=1)
        # features = features.div(features_norm.unsqueeze(1))
        # data_dict["vote_xyz"] = xyz
        # data_dict["vote_features"] = features
        # # proposal generation
        # data_dict = self.pnet(xyz, features, data_dict)
        data_dict = self.group_free(data_dict)
        # text encode
        data_dict = self.lang(data_dict)

        data_dict = self.relation(data_dict)

        data_dict = self.contranet(data_dict)

        self.divide_proposals(data_dict)

        data_dict = self.match(data_dict)

        # reconstruction
        if not is_eval:
            self.reconstruct(data_dict)

        return data_dict

    def divide_proposals(self, data_dict):
        att_weight = data_dict["contra_att_weight"]
        len_num_max = att_weight.shape[1]
        att_weight = att_weight.flatten(0, 1)

        bbox_feature = data_dict["bbox_feature"]  # bs, num_proposal, hidden_dim
        sem_cls_scores = data_dict["sem_cls_scores"]  # bs, num_proposal, 18
        # pred_lang_cat = torch.argmax(data_dict["lang_scores"], 1)  # bs*len_num_max
        pred_lang_cat = self.lang.eval_lang_cls(data_dict)
        object_cat = data_dict["object_cat_list"].flatten(0, 1)
        data_dict["lang_acc"] = (pred_lang_cat == object_cat).float().mean()
        bs, num_proposal, hidden_dim = bbox_feature.shape
        # num_class = sem_cls_scores.shape[2]
        # len_num_max = pred_lang_cat.shape[0] // bs
        bbox_feature = bbox_feature.unsqueeze(1).expand(-1, len_num_max, -1, -1).flatten(0, 1)
        sem_cls_scores = sem_cls_scores.unsqueeze(1).expand(-1, len_num_max, -1, -1).flatten(0, 1)
        pred_by_target_cls = torch.gather(sem_cls_scores, 2, pred_lang_cat.unsqueeze(-1).unsqueeze(-1).expand(-1, num_proposal, -1)).squeeze(-1)  # bs*len_num_max, num_proposal
        pred_by_target_cls = torch.softmax(pred_by_target_cls, -1)

        # objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()
        objectness_preds_batch = torch.round(data_dict["objectness_scores"].sigmoid()).squeeze(-1).long()
        non_objectness_masks = (objectness_preds_batch == 0).bool()  # bs, num_proposal
        non_objectness_masks = non_objectness_masks.unsqueeze(1).expand(-1, len_num_max, -1).flatten(0, 1)
        # weights = pred_by_target_cls + att_weight
        weights = pred_by_target_cls
        weights.masked_fill_(non_objectness_masks, -float('inf'))
        # pred_by_target_cls.masked_fill_(non_objectness_masks, -float('inf'))
        # att_weight.masked_fill_(non_objectness_masks, 0.)

        # sorted_ids = torch.sort(pred_by_target_cls, descending=True, dim=1)[1]
        # target_ids = sorted_ids[:, :self.num_target]
        # other_ids = sorted_ids[:, self.num_target:]
        # target_feat = torch.gather(bbox_feature, 1, target_ids.unsqueeze(-1).expand(-1, -1, hidden_dim))  # bs*len_num_max, num_target, hiddem_dim
        # other_feat = torch.gather(bbox_feature, 1, other_ids.unsqueeze(-1).expand(-1, -1, hidden_dim))

        sorted_ids = torch.sort(weights, descending=True, dim=1)[1]
        target_ids = sorted_ids[:, :self.num_target]
        other_ids = sorted_ids[:, self.num_target:]
        target_feat = torch.gather(bbox_feature, 1, target_ids.unsqueeze(-1).expand(-1, -1, hidden_dim))  # bs*len_num_max, num_target, hiddem_dim
        other_feat = torch.gather(bbox_feature, 1, other_ids.unsqueeze(-1).expand(-1, -1, hidden_dim))

        data_dict["target_ids"] = target_ids.reshape(bs, len_num_max, self.num_target)
        data_dict["other_ids"] = other_ids.reshape(bs, len_num_max, self.num_other)
        data_dict["target_feat"] = target_feat.reshape(bs, len_num_max, self.num_target, hidden_dim)
        data_dict["other_feat"] = other_feat.reshape(bs, len_num_max, self.num_other, hidden_dim)

    def reconstruct(self, data_dict):
        target_feat = data_dict["target_feat"]  # bs, len_num_max, num_target, dim
        other_feat = data_dict["other_feat"]
        target_ids = data_dict["target_ids"]  # bs, len_num_max, num_target
        other_ids = data_dict["other_ids"]
        # words_feat = data_dict["ground_lang_feat_list"]
        words_feat = self.lang.masked_lang_feat(data_dict)
        # print("words_feat", words_feat.shape)
        # masks_list = data_dict["all_masks_list"]
        bs, len_num_max, _, hidden_dim = target_feat.shape
        max_des_len = words_feat.shape[2]
        masks_list = torch.empty(bs, len_num_max, max_des_len).to(target_feat.device)
        masks_list = masks_list.bernoulli_(p=0.3)
        masks_list = masks_list.masked_fill(data_dict["all_masks_list"][:, :, :max_des_len] == 2, 2)
        data_dict["rand_masks_list"] = masks_list
        # masks_list = masks_list[:, :, :max_des_len]
        xyz = data_dict['center'].unsqueeze(1).expand(-1, len_num_max, -1, -1).flatten(0, 1)  # bs * len_num_max, num_proposal, 3
        target_ids = target_ids.flatten(0, 1)  # bs*len_num_max, num_target
        other_ids = other_ids.flatten(0, 1)
        target_xyzs = torch.gather(xyz, dim=1, index=target_ids.unsqueeze(-1).expand(-1, -1, 3))  # bs*len_num_max, num_target, 3
        other_xyzs = torch.gather(xyz, dim=1, index=other_ids.unsqueeze(-1).expand(-1, -1, 3))
        other_feat = other_feat.reshape(bs*len_num_max, self.num_other, hidden_dim)
        # nms_masks = self.get_nms_masks(data_dict, len_num_max)
        device = target_feat.device
        word_logits = torch.zeros(bs, len_num_max, self.num_target, max_des_len, self.vocab_size).to(device)
        # rec_word_feat = torch.zeros(bs, len_num_max, self.num_target, max_des_len, 128).to(device)
        for i in range(self.num_target):
            center_xyz = target_xyzs[:, i:i+1, :]  # bs*len_num_max, 1, 3
            dist = n_distance(center_xyz, other_xyzs)  # bs*len_num_max, num_other
            # dist = n_distance(center_xyz, xyz)  # bs*len_num_max, num_proposal
            # dist = dist.masked_fill(nms_masks == 0, 0.) + 10. * dist.masked_fill(nms_masks == 1, 0)
            # dist = torch.gather(dist, dim=1, index=other_ids)
            min_dist_ids = torch.topk(dist, self.num_rec_other, largest=False)[1]  # bs*len_num_max, num_rec_other
            min_other_feat = torch.gather(other_feat, 1, min_dist_ids.unsqueeze(-1).expand(-1, -1, hidden_dim)).reshape(bs, len_num_max, self.num_rec_other, hidden_dim)
            object_feat = torch.cat((target_feat[:, :, i:i + 1, :], min_other_feat), dim=2)
            word_logit = self.recnet(words_feat, object_feat, masks_list)  # bs, len_num_max, max_des_len, vocab_size
            word_logits[:, :, i, :, :] = word_logit
            # rec_word_feat[:, :, i, :, :] = self.recnet(words_feat, object_feat, masks_list)

        data_dict["rec_word_logits"] = word_logits
        # data_dict["rec_word_feat"] = rec_word_feat

    def get_nms_masks(self, data_dict, len_num_max):
        POST_DICT = {
            "remove_empty_box": True,
            "use_3d_nms": True,
            "nms_iou": 0.25,
            "use_old_type_nms": False,
            "cls_nms": True,
            "per_class_proposal": True,
            "conf_thresh": 0.05,
            "dataset_config": self.dataset_config
        }

        parse_predictions(data_dict, POST_DICT)
        nms_masks = torch.LongTensor(data_dict["pred_mask"]).cuda()
        # print("nms_mask", nms_masks.float().sum(-1).mean())
        # obj_masks = torch.argmax(data_dict["objectness_scores"], 2).long()
        obj_masks = torch.round(data_dict["objectness_scores"].sigmoid()).squeeze(-1).long()
        # print("obj_mask", obj_masks.float().sum(-1).mean())
        nms_masks = nms_masks * obj_masks
        # print("all_mask", nms_masks.float().sum(-1).mean())
        nms_masks = nms_masks.unsqueeze(1).expand(-1, len_num_max, -1).flatten(0, 1)
        return nms_masks


def n_distance(center, points):
    diff = points - center
    dist = torch.sum(diff**2, dim=-1)
    return dist
