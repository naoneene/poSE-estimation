# 3D Human Pose Estimation with simple Self-Supervised Learning

This repository holds the Pytorch implementation of [3D Human Pose Estimation with simple Self-Supervised Learning](https://) by Pham Le Minh Hoang.
![enter image description here](https://imgur.com/NM3LFaU.gif)

## Introduction

In this repository, we present a solution for single-view 3D human skeleton estimation based on deep learning method. Our network contains two separate model to fully regress and enhance the resulting poses. We utilize a newly proposed model whose name is Squeeze and Excitation Network (SE-net) as to construct our pose estimation network in order to estimate the corresponding pose from a colour image; then a model consisting of several blocks of fully-connected networks and a novel semantic graph convolutional networks featuring self-supervision to reconstruct 3D human pose. We demonstrate the effectiveness of our approach on standard datasets for benchmark where we achieved comparable results to some recent state-of-the-art methods.

### Results on Human3.6M

Under Protocol #1 (mean per-joint position error) and Protocol #2 (mean-per-joint position error after rigid alignment).

### Requirements

This repository is build upon Python 3.9.12 and Pytorch 1.4.0 on Ubuntu 20.04 LTS. NVIDIA GPUs are needed to train and test. We recommend installing Python 3.9.12 from [Anaconda](https://www.anaconda.com/), and installing PyTorch (>= 1.1.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. See [`requirements.txt`]() for other dependencies.

## Quick start

### Installation

1. You can install dependencies with the following commands using  `pip`.
   
   ```
   pip install -r requirements.txt
   ```

2. Download annotation files from [GoogleDrive]() (1024 MB) as a zip file under `${ROOT}` folder and `${ROOT}/refiner` for each annotation. The annotation for estimation module is marked as `_5hz`.

3. Finally prepare your workspace by running:
   
   ```
   mkdir output
   mkdir models
   ```
   
   At the end, your directory tree should like this.
   
   ```
   ${ROOT}
   ├── data/
   ├── experiments/
   ├── lib/
   ├── models/
   ├── output/
   ├── refiner/
   ├── src/
   ├── README.md
   └── requirements.txt
   ```
   
   ### Dataset preparation
   
   You would need Human3.6M data to train or test our model. **For Human3.6M data**, please download from [Human 3.6 M dataset](http://vision.imar.ro/human3.6m/description.php). You have to create an account to get download permission. For pre-processing, run `preprocess_h36m.m` to preprocess Human3.6M dataset. It converts videos to images and save meta data for each frame. `data` in `Human36M` contains the preprocessed data. The code for data preparation is borrowed from [Integral Human Pose Regression for 3D Human Pose Estimation](https://github.com/mks0601/Integral-Human-Pose-Regression-for-3D-Human-Pose-Estimation).

We provide annotation files for both of the model for training and testing in two stages or you can export manually the annotations by providing codes.

During training, we make use of [synthetic-occlusion](https://github.com/isarandi/synthetic-occlusion). If you want to use it please download the Pascal VOC dataset as instructed in their [repo](https://github.com/isarandi/synthetic-occlusion#getting-started) and update the `VOC` parameter in configuration files.

After downloading the datasets your  `data`  directory tree should look like this:

```
${ROOT}
|── data/
├───├── mpii/
|   └───├── annot/
|       └── images/
|       
└───├── h36m/
|   └───├── annot/
|           ├── train_5hz.pkl
|           ├── valid_5hz.pkl
|       └── images/
|           ├── s_01_act_02_subact_01_ca_01/
|           └── s_01_act_02_subact_01_ca_01/
|           ...
└───├── refiner/
    └───├── data/
            ├── train.pkl
            ├── valid.pkl
```

### Evaluating our pretrained models

The pretrained models can be downloaded from [Google Drive](). Put `model` in the project root directory. These models allow you to reproduce our top-performing baselines, which results in 47.3 mm for Human3.6M.

In order to run evaluation script with a self-supervised model, update the `MODEL.RESUME` field of [`experiments/h36m/eval.yaml`] with the path to the pretrained weight and run:

```
python src/eval.py --cfg experiments/h36m/eval.yaml
```

### Training from scratch

If you want to reproduce the results of our pretrained models, run the following commands.
For estimation module:

```
python src/train.py --cfg experiments/h36m/train.yaml
```

For regression module (upper and lower branch respectively):

```
python refiner/main_gcn.py --mode train
python refiner/main_linear.py --mode train
```

### Visualization

For application, we prepare a little bit different setup which mainly focus on visualizing the human pose coupling with a human/objection detection and tracker for real-field experiment. This set of preparation for application can run independently.

```
${ROOT}
├── CenterNet/
├── deep/
├── experiments/
├── lib/
├── models/
├── output/
├── refiner/
├── sort/
├── deep_sort.py
├── README.md
└── requirements.txt
└── viz.py
```

You can generate visualizations of the model predictions from original Human3.6M videos or any other video with full-body human by running:

```
python viz.py --viz_video CenterNet/images/h36m_3.mp4 --viz_output output/demo_h36m_3.gif
```

The script can also export MP4 videos, and supports a variety of parameters (e.g. downsampling/FPS, size, bitrate). 

## Acknowledgement

Part of our code is borrowed from the following repositories:

- [EpipolarPose](https://github.com/mkocabas/EpipolarPose)
- [Integral Human Pose](https://github.com/JimmySuen/integral-human-pose)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [centerNet + deep sort with pytorch](https://github.com/kimyoon-young/centerNet-deep-sort)

We appreciate the authors for releasing their codes. Please also consider citing their works.
