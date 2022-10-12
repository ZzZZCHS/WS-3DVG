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


class JointNet(nn.Module):
    def __init__(self, vocabulary, embeddings,
                 input_feature_dim=0, num_proposal=128, num_target=32, num_locals=-1, vote_factor=1, sampling="vote_fps",
                 no_caption=False, use_topdown=False, query_mode="corner", num_graph_steps=0, use_relation=False,
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, ground_hidden_size=256, caption_hidden_size=512, dataset_config=None, args=None):
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
        self.vocab_size = len(vocabulary["idx2word"])

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Fix VoteNet
        for _p in self.parameters():
            _p.requires_grad = False

        # Vote aggregation and object proposal
        self.pnet = ProposalModule(num_proposal=num_proposal, sampling=sampling, dataset_config=dataset_config, num_target=num_target)

        # self.relation = RelationModule(num_proposals=num_proposal, det_channel=128)  # bef 256

        self.lang = LangModule(dataset_config.num_class, use_lang_classifier, use_bidir, emb_size, ground_hidden_size)
        for _p in self.lang.parameters():
            _p.requires_grad = False

        self.recnet = ReconstructModule(vocab_size=self.vocab_size)

        self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * ground_hidden_size, det_channel=128, num_target=num_target)  # bef 256

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

        # hough voting
        data_dict = self.backbone_net(data_dict)
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # text encode
        data_dict = self.lang(data_dict)

        # proposal generation
        data_dict = self.pnet(xyz, features, data_dict)

        data_dict = self.match(data_dict)

        if not is_eval:
            # reconstruction
            target_feat = data_dict["target_feat"]
            other_feat = data_dict["other_feat"]
            target_ids = data_dict["target_ids"]
            words_feat = data_dict["ground_lang_feat_list"]
            masks_list = data_dict["all_masks_list"]
            bs, len_num_max, _, hidden_dim = target_feat.shape
            max_des_len = words_feat.shape[2]
            device = target_feat.device
            word_logits = torch.zeros(bs, len_num_max, self.num_target, max_des_len, self.vocab_size).to(device)
            for i in range(self.num_target):
                object_feat = torch.cat((target_feat[:, :, i:i+1, :], other_feat), dim=2)
                word_logit = self.recnet(words_feat, object_feat, masks_list)  # bs, len_num_max, max_des_len, vocab_size
                word_logits[:, :, i, :, :] = word_logit

            data_dict["rec_word_logits"] = word_logits

            # all_score = data_dict["cluster_ref"]
            # data_dict["target_scores"] = torch.gather(all_score, 1, target_ids.resize(bs * len_num_max, self.num_target))

        return data_dict
