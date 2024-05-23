# WS-3DVG: Distilling Coarse-to-Fine Semantic Matching Knowledge for Weakly Supervised 3D Visual Grounding

This is an official repository for the **ICCV 2023** paper "[Distilling Coarse-to-Fine Semantic Matching Knowledge for Weakly Supervised 3D Visual Grounding](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Distilling_Coarse-to-Fine_Semantic_Matching_Knowledge_for_Weakly_Supervised_3D_Visual_ICCV_2023_paper.pdf)"


## Introduction

3D visual grounding involves finding a target object in a 3D scene that corresponds to a given sentence query. Although many approaches have been proposed and achieved impressive performance, they all require dense object-sentence pair annotations in 3D point clouds, which are both time-consuming and expensive. To address the problem that fine-grained annotated data is difficult to obtain, we propose to leverage weakly supervised annotations to learn the 3D visual grounding model, i.e., only coarse scene-sentence correspondences are used to learn object-sentence links. To accomplish this, we design a novel semantic matching model that analyzes the semantic similarity between object proposals and sentences in a coarse-to-fine manner. Specifically, we first extract object proposals and coarsely select the top-K candidates based on feature and class similarity matrices. Next, we reconstruct the masked keywords of the sentence using each candidate one by one, and the reconstructed accuracy finely reflects the semantic similarity of each candidate to the query. Additionally, we distill the coarse-to-fine semantic matching knowledge into a typical two-stage 3D visual grounding model, which reduces inference costs and improves performance by taking full advantage of the well-studied structure of the existing architectures. We conduct extensive experiments on ScanRefer, Nr3D, and Sr3D, which demonstrate the effectiveness of our proposed method.


## News

[2023.12.16] 🔥 Code release.

## Dataset & Setup

### Data preparation

*This codebase is built based on the initial [ScanRefer](https://github.com/daveredrum/ScanRefer) and [3DJCG](https://github.com/zlccccc/3DVL_Codebase/blob/main/README_3DJCG.md) codebase. Please refer to them for more data preprocessing details.*

1. Download the ScanRefer dataset and unzip it under `data/`. 
2. Downloadand the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](https://github.com/ScanNet/ScanNet) for downloading the ScanNet dataset).

> After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`

4. Pre-process ScanNet data. A folder named `scannet_data/` will be generated under `data/scannet/` after running the following command. Roughly 3.8GB free space is needed for this step:

```shell
cd data/scannet/
python batch_load_scannet_data.py
```

> After this step, you can check if the processed scene data is valid by running:
>
> ```shell
> python visualize.py --scene_id scene0000_00
> ```


5. (Optional) Pre-process the **multiview features** from ENet.

- Download:
    Download the ENet [multiview features (~36GB, hdf5 database)](http://kaldir.vc.in.tum.de/enet_feats.hdf5) and put it under `data/scannet/scannet_data/`

- Projection:

   a. Download [the ENet pretrained weights (1.4MB)](http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth) and put it under `data/`
   b. Download and decompress [the extracted ScanNet frames (~13GB)](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip).
   c. Change the data paths in `config/config.py` marked with __TODO__ accordingly.
   d. Project ENet features from ScanNet frames to point clouds (~36GB, hdf5 database).

> ```shell
> python script/multiview_compute/compute_multiview_features.py
> python script/multiview_compute/project_multiview_features.py --maxpool --gpu 1
> ```

6. Download pretrained GroupFree checkpoints from their [repo](https://github.com/zeliu98/Group-Free-3D). Put them under `pretrained/groupfree/`.
- We used the (L6, O256) version pretrained on ScanNet v2.
- Note that loading the downloaded checkpoints directly may result in mismatched key names. You can download our processed version from [Google Drive](https://drive.google.com/file/d/1m3nFoqreE_44geoDjFQxzo7vcaeWh4j_/view?usp=sharing).
- We use GroupFree as the 3D encoder in our default code. We also provide code for loading VoteNet. If you want to use VoteNet, download the pretrained checkpoints from their [repo](https://github.com/facebookresearch/votenet). 

7. Masked annotations: You can download the masked scanrefer annotation [here](https://drive.google.com/drive/folders/1Erz6fMwwwWd6Dj_jjXPnU4_2clSlm3O7?usp=drive_link) and put them under `data/scanrefer/`.


### Setup

```shell
git clone --depth 1 https://github.com/ZzZZCHS/WS-3DVG.git

# Create conda environment with PyTorch 1.9.0 & CUDA 10.2
conda create -n WS-3DVG python=3.8.13
conda activate WS-3DVG
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

# Install necessary packages
pip install -r requirements.txt

# Install pointnet2
cd lib/pointnet2
python setup.py install
```

## Usage

### Training
To train the WS-3DVG model with default setting:
```shell
python scripts/joint_scripts/train.py
```

### Evaluation
To evaluate the trained models, please find the folder under `outputs/` and run:
```shell
python scripts/joint_scripts/ground_eval.py --folder <folder_name> --lang_num_max 1 --is_eval --force
```
Note that the flags must match the ones set before training. The training information is stored in `outputs/<folder_name>/info.json`


## Citation

If you find this project useful in your research, please consider cite:

```
@inproceedings{wang2023distilling,
  title={Distilling coarse-to-fine semantic matching knowledge for weakly supervised 3d visual grounding},
  author={Wang, Zehan and Huang, Haifeng and Zhao, Yang and Li, Linjun and Cheng, Xize and Zhu, Yichen and Yin, Aoxiong and Zhao, Zhou},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2662--2671},
  year={2023}
}
```

## Acknowledgement

Thanks to the open source of the following projects:

[3DJCG](https://github.com/zlccccc/3DVL_Codebase/blob/main/README_3DJCG.md), [ScanRefer](https://github.com/daveredrum/ScanRefer), [ReferIt3D](https://github.com/referit3d/referit3d), [ScanNet](https://github.com/ScanNet/ScanNet), [VoteNet](https://github.com/facebookresearch/votenet), [GroupFree](https://github.com/zeliu98/Group-Free-3D)

