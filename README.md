# Mocap-to-SMPLX

This repository contains code for converting mocap data(3D joint positions) to SMPL-X parameters used in the paper [[CVPR 2024]Inter-X: Towards Versatile Human-human Interaction Analysis](https://github.com/liangxuy/Inter-X). The code is mainly based on the [smplify-x](https://github.com/vchoutas/smplify-x) repository. Consider citing Inter-X and smplify-x if you find this code useful.

## Environment setup
The code is tested on Ubuntu 20.04 with Python 3.8, Pytorch 1.11.0. You can create a new conda environment and install the dependencies using the following commands:
```bash
conda create -n mocap2smplx python=3.8
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Preparation
Download the SMPL-X model from the [official website](https://smpl-x.is.tue.mpg.de/) and put it in the `./body_models` folder. It should look like this:
```
./body_models
    ├── smplx
    │   ├── SMPLX_NEUTRAL.npz
    │   ├── SMPLX_FEMALE.npz
    │   └──...
```

## How to use
Currently, we assume the shape parameters(shape.pkl) is ready. Below we will test on the sample data provided in the `./test_data` folder. The `./test_data` folder contains a `.npy` file with 3D joint positions from the mocap data and a `.pkl` file of shape parameters and gender information.

The fitting process can be run using the following command:
```bash
python main.py --npy ./test_data/P2.npy # .npy file that contains 3D joints
               --shape_pkl ./test_data/P2.pkl # .pkl file that contains shape parameters
               --save_path ./test_data/P2_smplx.npz # path to save the SMPL-X parameters
               --vis_smplx # visualize the SMPL-X result
               --vis_kp3d # visualize the 3D keypoints
```
After that, you will find `P2_smplx.npz` in the `./test_data` folder. The file contains the desired SMPL-X parameters.

## Customization
If you'd like to modify the code to meet your needs, here's several **KEY_CODE** you may refer to.

1. `./smplifyx/config.py`: Here we define the weights for the optimization process in each stage. You can modify the weights to get better results.
2. `./smplifyx/loss.py`: Here we define the loss functions used in the optimization process. More types of loss functions can be added here.
3. `./utils/mapping.py`: Here we map the 3D joints of Optitrack Skeleton to SMPL-X joints. You can modify the mapping to fit your own data.
4. `./utils/limbs.py`: Here we define the limbs of the human body for visualization. You can modify the limbs to fit your own data.

## Citation
If you find the Inter-X dataset is useful for your research, please cite us:

```
@inproceedings{xu2023inter,
  title={Inter-X: Towards Versatile Human-Human Interaction Analysis},
  author={Xu, Liang and Lv, Xintao and Yan, Yichao and Jin, Xin and Wu, Shuwen and Xu, Congsheng and Liu, Yifan and Zhou, Yizhou and Rao, Fengyun and Sheng, Xingdong and Liu, Yunhui and Zeng, Wenjun and Yang, Xiaokang},
  booktitle={CVPR},
  year={2024}
}
```