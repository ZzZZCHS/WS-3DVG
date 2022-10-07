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
                 emb_size=300, ground_hidden_size=256, caption_hidden_size=512, dataset_config=None):
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

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.pnet = ProposalModule(num_proposal=num_proposal, sampling=sampling, dataset_config=dataset_config, num_target=num_target)

        # Fix VoteNet
        for _p in self.parameters():
            _p.requires_grad = False

        self.relation = RelationModule(num_proposals=num_proposal, det_channel=128)  # bef 256

        self.lang = LangModule(dataset_config.num_class, use_lang_classifier, use_bidir, emb_size, ground_hidden_size)

        self.recnet = ReconstructModule(vocab_size=len(vocabulary["idx2word"]))

        self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * ground_hidden_size, det_channel=128)  # bef 256

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

        # data_dict = self.relation(data_dict)

        data_dict = self.match(data_dict)


        return data_dict
