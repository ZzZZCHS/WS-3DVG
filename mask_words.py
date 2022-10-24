import nltk
import torch
import json
import os
import csv
import sys
from nltk.parse.stanford import StanfordParser
from nltk.tree import *

stanford_parser_dir = r'/home/huanghaifeng/stanford-parser-full-2015-12-09'
eng_model_path = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
path_to_models_jar = stanford_parser_dir + "/stanford-parser-3.6.0-models.jar"
path_to_jar = stanford_parser_dir + "/stanford-parser.jar"
PATH_DATA = '/home/huanghaifeng/3DWSVG/data'
# SCANREFER_TRAIN = json.load(open(os.path.join(PATH_DATA, "ScanRefer_filtered_train.json")))
# SCANREFER_VAL = json.load(open(os.path.join(PATH_DATA, "ScanRefer_filtered_val.json")))[0:10]

NR3D_DATA = []
NR3D_IDS = None

with open(os.path.join(PATH_DATA, "nr3d/nr3d.csv")) as f:
    csvreader = csv.reader(f)
    NR3D_IDS = csvreader.__next__()
    for row in csvreader:
        NR3D_DATA.append(row)

# print(NR3D_IDS)
print(len(NR3D_DATA))


def MultiTreePaths(root):
    def helper(root, path, res):
        if type(root)==str:
            res.append(path +str(root))
            return
        l = len(root)
        for i in range(l):
                if len(root)>=i:
                    if root[i]:
                        helper(root[i], path +str(root.label()) +'->', res)
    if root is None:
        return []
    l = []
    helper(root, '', l)
    return l


def find_JJ(tokens, leaf_nodes):
    masked_tokens = tokens
    mask_list = []
    for i, word in enumerate(leaf_nodes):
        if 'JJ' in word:
            mask_list.append(1)
            masked_tokens[i] = '[MASK]'
        else:
            mask_list.append(0)
    return masked_tokens, mask_list


def find_PPNN(tokens, leaf_nodes):
    masked_tokens = tokens
    mask_list = []
    for i, word in enumerate(leaf_nodes):
        if 'PP' in word:
            if 'NN' in word:
                mask_list.append(1)
                masked_tokens[i] = '[MASK]'
            else:
                mask_list.append(0)
        else:
            mask_list.append(0)
    return masked_tokens, mask_list


parser = StanfordParser(model_path=eng_model_path, path_to_models_jar=path_to_models_jar, path_to_jar=path_to_jar)

NR3D_IDS.extend(["masked_sentence", "mask_list"])

for i, data in enumerate(NR3D_DATA):
    sentence = data[2]
    tokens = nltk.word_tokenize(sentence)

    tree = list(parser.parse(tokens))[0]
    leaf_nodes = MultiTreePaths(tree[0])
    PPNN_masked_tokens, PPNN_mask_list = find_PPNN(tokens, leaf_nodes)
    data.append(' '.join(PPNN_masked_tokens))  # masked_sentence
    data.append(PPNN_mask_list)

    print(i)
    print(data[-2:])

with open(os.path.join(PATH_DATA, "nr3d/masked_nr3d.csv")) as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(NR3D_IDS)
    csvwriter.writerows(NR3D_DATA)


# NEW_SCANREFER_TRAIN = []
# for data in SCANREFER_TRAIN:
#     sentence = data['description']
#     first_sentence = nltk.sent_tokenize(sentence)[0]
#     tokens = nltk.word_tokenize(sentence)
#     first_tokens = nltk.word_tokenize(first_sentence)
#
#     first_tree = list(parser.parse(first_tokens))[0]
#     first_leaf_nodes = MultiTreePaths(first_tree[0])
#     JJ_masked_tokens, JJ_mask_list = find_JJ(first_tokens, first_leaf_nodes)
#     data['masked_first_sentence'] = ' '.join(JJ_masked_tokens)
#     data['mask_first_list'] = JJ_mask_list
#
#     tree = list(parser.parse(tokens))[0]
#     leaf_nodes = MultiTreePaths(tree[0])
#     PPNN_masked_tokens, PPNN_mask_list = find_PPNN(tokens, leaf_nodes)
#     data['masked_sentence'] = ' '.join(PPNN_masked_tokens)
#     data['mask_list'] = PPNN_mask_list
#
#     NEW_SCANREFER_TRAIN.append(data)
#
#
# with open(os.path.join(PATH_DATA,'scanrefer/Masked_ScanRefer_filtered_train.json'), 'w') as f:
#     f.write(json.dumps(NEW_SCANREFER_TRAIN, indent=4, ensure_ascii=False))

