import os
import sys
from easydict import EasyDict
import argparse
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--config", type=str, default="config/groupfree.yaml", help="path to config file")
        self.parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
        self.parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer, nr3d or sr3d", default="nr3d")
        self.parser.add_argument("--gpu", type=str, help="gpu", default="0")
        self.parser.add_argument("--seed", type=int, default=3407, help="random seed")
        self.parser.add_argument("--force", action="store_true", help="enforce the generation of results")
        self.parser.add_argument("--repeat", type=int, default=1, help="number of time for evaluation")
        self.parser.add_argument("--distribute", action="store_true", help="distributed training")
        self.parser.add_argument("--local_rank", type=int, help="local ran for DistributedDataParallel")
        self.parser.add_argument("--opt_steps", type=int, default=1, help="optimizer steps")
        self.parser.add_argument("--folder", type=str, help="folder containing the model")
        self.parser.add_argument("--use_train", action="store_true", help="use train data when eval")
        self.parser.add_argument("--is_eval", action="store_true", help="is eval")
        self.parser.add_argument("--eval_rand", action="store_true", help="eval rand")
        self.parser.add_argument("--use_best", action="store_true", help="use best")
        self.parser.add_argument("--mil_type", type=str, default="nce", help="mil type (nce or margin)")
        self.parser.add_argument("--topk", type=int, default=3, help="k")

        self.parser.add_argument("--batch_size", type=int, help="batch size", default=12)
        self.parser.add_argument("--val_batch_size", type=int, help="val batch size", default=1)
        self.parser.add_argument("--epoch", type=int, help="number of epochs", default=20)
        self.parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=50)
        self.parser.add_argument("--val_step", type=int, help="iterations of validating", default=500)
        self.parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
        self.parser.add_argument("--wd", type=float, help="weight decay", default=5e-4)
        self.parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")

        # self.parser.add_argument("--hidden_size", type=int, help="hidden size", default=288)
        self.parser.add_argument("--lang_num_max", type=int, help="lang num max", default=8)
        # self.parser.add_argument("--num_points", type=int, default=50000, help="Point Number [default: 40000]")
        self.parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
        self.parser.add_argument("--num_target", type=int, default=8, help="Target proposal number [default: 8]")
        self.parser.add_argument("--num_locals", type=int, default=20, help="Number of local objects [default: -1]")
        self.parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
        self.parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")
        self.parser.add_argument("--num_ground_epoch", type=int, default=100, help="Number of ground epoch [default: 50]")
        self.parser.add_argument("--width", type=int, default=1, help="backbone width")

        self.parser.add_argument("--no_mil", action="store_true", help="no multi-instance learning")
        self.parser.add_argument("--no_recon", action="store_true", help="no reconstruct module")
        self.parser.add_argument("--no_text", action="store_true", help="no object-query class similarity")
        self.parser.add_argument("--no_distill", action="store_true", help="no distill")

        self.parser.add_argument("--criterion", type=str, default="sum", \
                            help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor, sum]")

        self.parser.add_argument("--query_mode", type=str, default="center",
                            help="Mode for querying the local context, [choices: center, corner]")
        self.parser.add_argument("--graph_mode", type=str, default="edge_conv",
                            help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
        self.parser.add_argument("--graph_aggr", type=str, default="add",
                            help="Mode for aggregating features, [choices: add, mean, max]")

        self.parser.add_argument("--coslr", action='store_true', help="cosine learning rate")
        # self.parser.add_argument("--no_height", action="store_true", default=True, help="Do NOT use height signal in input.")
        self.parser.add_argument("--no_augment", action="store_true", default=True,
                            help="Do NOT use height signal in input.")
        self.parser.add_argument("--no_detection", action="store_true", default=True,
                            help="Do NOT train the detection module.")
        self.parser.add_argument("--no_caption", action="store_true", default=True, help="Do NOT train the caption module.")
        self.parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
        self.parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")

        self.parser.add_argument("--use_tf", action="store_true", help="enable teacher forcing in inference.")
        self.parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
        self.parser.add_argument("--use_normal", action="store_true", default=True, help="Use RGB color in input.")
        self.parser.add_argument("--use_multiview", action="store_true", default=True, help="Use multiview images.")
        self.parser.add_argument("--use_topdown", action="store_true", default=True,
                            help="Use top-down attention for captioning.")
        self.parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
        self.parser.add_argument("--use_new", action="store_true", help="Use new Top-down module.")
        self.parser.add_argument("--use_orientation", action="store_true",
                            help="Use object-to-object orientation loss in graph.")
        self.parser.add_argument("--use_distance", action="store_true", help="Use object-to-object distance loss in graph.")
        self.parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
        self.parser.add_argument("--use_pretrained", type=str,
                            help="Specify the folder name containing the pretrained detection module.")
        self.parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")

        self.parser.add_argument("--pretrain_data", type=str, default="scannet", help="pretrained dataset")
        self.parser.add_argument("--pretrain_model", type=str, default="groupfree", help="pretrained model")
        self.parser.add_argument("--pretrain_model_on", action="store_true", default=True, help="pretrained model on")

        self.parser.add_argument("--debug", action="store_true", help="Debug mode.")

    def get_config(self):
        cfgs = self.parser.parse_args()
        if cfgs.config is not None:
            with open(cfgs.config, 'r') as f:
                config = yaml.safe_load(f)
            for key in config:
                for k, v in config[key].items():
                    setattr(cfgs, k, v)
        self.set_paths_cfg(cfgs)
        return cfgs

    def set_paths_cfg(self, CONF):
        # path
        CONF.PATH = EasyDict()
        CONF.PATH.BASE = ROOT_DIR
        CONF.PATH.CLUSTER = "" # TODO: change this
        CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
        CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
        CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
        CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
        CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

        # append to syspath
        for _, path in CONF.PATH.items():
            sys.path.append(path)
        # print(sys.path, 'sys path', flush=True)

        # scannet data
        CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
        CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
        CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

        # Scan2CAD
        # CONF.PATH.SCAN2CAD = os.path.join(CONF.PATH.DATA, "Scan2CAD_dataset") # TODO change this

        # data
        CONF.SCANNET_DIR =  CONF.PATH.DATA + "/scannet/scans" # TODO change this
        CONF.SCANNET_FRAMES_ROOT = CONF.PATH.DATA + "/scanrefer/frames_square/" # TODO change this
        CONF.PROJECTION = CONF.PATH.DATA + "/multiview_projection_scanrefer" # TODO change this
        CONF.ENET_FEATURES_ROOT = CONF.PATH.DATA + "/scanrefer/enet_features" # TODO change this

        CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}") # scene_id
        CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy") # frame_id
        CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode
        # CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))
        CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.BASE, "data/scannetv2_enet.pth")
        # CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
        CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")
        CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

        # scannet
        CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
        CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
        CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
        CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

        # output
        CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs/test")
        CONF.PATH.AXIS_ALIGNED_MESH = os.path.join(CONF.PATH.OUTPUT, "ScanNet_axis_aligned_mesh")

        CONF.TRAIN = EasyDict()
        CONF.TRAIN.MAX_DES_LEN = 30
        CONF.TRAIN.MAX_GROUND_DES_LEN = 126
        CONF.TRAIN.SEED = 42
        CONF.TRAIN.OVERLAID_THRESHOLD = 0.5
        CONF.TRAIN.MIN_IOU_THRESHOLD = 0.25
        CONF.TRAIN.NUM_BINS = 6

        # eval
        CONF.EVAL = EasyDict()
        CONF.EVAL.MIN_IOU_THRESHOLD = 0.5

        # pretrained
        CONF.PATH.PRETRAINED = os.path.join(CONF.PATH.BASE, "pretrained")

        # Pretrained features
        CONF.PATH.GT_FEATURES = os.path.join(CONF.PATH.CLUSTER, "gt_{}_features") # dataset
        # CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_features")
        CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_{}_predictions") # dataset

        if CONF.pretrain_data == "scannet" and CONF.pretrain_model == "votenet":
            CONF.PATH.VOTENET_PRETRAIN = os.path.join(CONF.PATH.PRETRAINED, "votenet", "pretrained_votenet_on_scannet.tar")
        if CONF.pretrain_data == "sunrgbd" and CONF.pretrain_model == "votenet":
            CONF.PATH.VOTENET_PRETRAIN = os.path.join(CONF.PATH.PRETRAINED, "votenet", "pretrained_votenet_on_sunrgbd.tar")
        if CONF.pretrain_data == "scannet" and CONF.pretrain_model == "groupfree":
            CONF.PATH.GROUPFREE_PRETRAIN = os.path.join(CONF.PATH.PRETRAINED, "groupfree", "scannet_l6o256.pth")
            CONF.PATH.PRETRAINED_TRAIN_DATA = os.path.join(CONF.PATH.PRETRAINED, "groupfree", "scannet_train_l6o256.pth")
            CONF.PATH.PRETRAINED_VAL_DATA = os.path.join(CONF.PATH.PRETRAINED, "groupfree", "scannet_val_l6o256.pth")
        if CONF.pretrain_data == "sunrgbd" and CONF.pretrain_model == "groupfree":
            CONF.PATH.GROUPFREE_PRETRAIN = os.path.join(CONF.PATH.PRETRAINED, "groupfree", "sunrgbd_l6o256_cls_agnostic.pth")
            CONF.PATH.PRETRAINED_TRAIN_DATA = os.path.join(CONF.PATH.PRETRAINED, "groupfree", "sunrgbd_train_l6o256.pth")
            CONF.PATH.PRETRAINED_VAL_DATA = os.path.join(CONF.PATH.PRETRAINED, "groupfree", "sunrgbd_val_l6o256.pth")
        CONF.PATH.LANGMODULE_PRETRAIN = os.path.join(CONF.PATH.PRETRAINED, "lang", "model.pth")

        CONF.PRETRAINED_LIST = [
            "pred_bbox_feature", "objectness_scores", "sem_cls_scores",
            "pred_heading", "pred_center", "pred_size", "pred_bbox_corner",
            "query_points_xyz", "query_points_feature", "query_points_sample_inds",
            "seed_inds"
        ]

CONF = Config().get_config()
