<!-- Template from https://github.com/YuliangXiu/ECON -->

<p align="center">

  <h1 align="center">Semantic Human Mesh Reconstruction with Textures</h1>
  <p align="center">
    <a href="https://zhanxy.xyz/" rel="external nofollow noopener" target="_blank"><strong>Xiaoyu Zhan</strong></a>
    ·
    <a href="https://jason-yangjx.github.io/" rel="external nofollow noopener" target="_blank"><strong>Jianxin Yang</strong></a>
    ·
    <a href="http://www.njumeta.com/liyq/" rel="external nofollow noopener" target="_blank"><strong>Yuanqi Li</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=Sx4PQpQAAAAJ&hl=en" rel="external nofollow noopener" target="_blank"><strong>Jie Guo</strong></a>
    ·
    <a href="https://cs.nju.edu.cn/ywguo/index.htm" rel="external nofollow noopener" target="_blank"><strong>Yanwen Guo</strong></a>
    ·
    <a href="https://www.cs.hku.hk/people/academic-staff/wenping" rel="external nofollow noopener" target="_blank"><strong>Wenping Wang</strong></a>
  </p>
  <p align="center">
    <a href='https://zhanxy.xyz/projects/shert' rel="external nofollow noopener" target="_blank">
        <img src='https://img.shields.io/badge/Project-Page-green' alt='Project Page'></a>
    <a href="https://arxiv.org/abs/2403.02561v2" rel="external nofollow noopener" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2403.02561-B31B1B" alt='Arxiv Link'></a>
  </p>
  <br>
  <p>This repository contains the official <b>PyTorch</b> implementation for <b>Semantic Human Mesh Reconstruction with Textures</b>.</p>
  <div align="center">
    <img src="./assets/teaser_v11.jpg" alt="teaser" width="100%">
  </div>
  
</p>

  ## (Preparing, coming soon)
  ### Thank you for your interest in this repository. We are currently in the process of converting model formats and finalizing scripts. The code will be released at a later time.

## Overview
- [1 - Installation](#installation)
- [2 - Demo](#demo)
- [3 - Acknowledgments](#acknowledgments)
- [4 - Citation](#citation)
- [5 - Contact](#contact)

## Installation

### 1. Get Started
Start from creating a conda environment.
```bash
git clone https://github.com/ZhanxyR/SHERT.git
cd SHERT
conda create -n shert python=3.8
conda activate shert
```
### 2. Install Pytorch
Follow [Pytorch](https://pytorch.org/get-started/previous-versions/).

We recommend to use `Pytorch >= 2.0`. While lower versions (lowest tested on `1.13`) may require more GPU memory in texture inpainting.

### 3. Install Open3d

> [!IMPORTANT]
> We recommend you to install a specific version of open3d manually to avoid any problem.

(We will fix the bugs to adapt to higher versions later.)
```bash
pip install open3d==0.10.0
```

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 5. Build Pytorch3D
Follow [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#building--installing-from-source). We recommend to build from source code. 

The version we used is `v0.7.6`, while the lower versions should also be applicable.

> [!NOTE]
> If you have any troubles in building the package, you could just set the `refine_iter` to `1` in corresponding `config.yaml` to avoid using Pytorch3D.
<br>(e.g. `./examples/scan/config.yaml`)

### 6. Download Required Data and Checkpoints

Selective to dowload `data/smplx`, `data/face`, and `save/ckpt` from [NJU-Box](https://box.nju.edu.cn/d/a7feb0bf42014f97ae50/) or [Google-Drive](https://drive.google.com/drive/folders/1UokzpgQNGe3q-vdvrWQpD3FreFNy7MmS?usp=sharing).

Dowload `SMPL-X v1.1 Models` (Male, Female, Neutral) from [SMPL-X](https://smpl-x.is.tue.mpg.de) and put them to `data/models`.

The completed structure should be like:

```
|-- SHERT
    |-- data
        |-- cameras
        |-- masks
        |-- smplx
        |-- face
        |-- models
            |-- smplx
                |-- SMPLX_*.npz
    |-- save
        |-- ckpt
            |-- inpaint.pth     # For mesh completion
            |-- refine.pth      # For mesh refinement
            |-- texture_local   # For texture inpainting
            |-- texture_global  # For texture repainting
```


## Demo

The whole processes include two steps: `reconstruction` and `texture inpainting`.

⚡ Run `quick_demo` to test `reconstruction` in given resources. The results will be saved to `./examples/$subject$/results`.

```bash
# Use ECON-pred mesh and fitted smplx.
python -m apps.quick_demo

# Use THuman scan and fitted smplx.
python -m apps.quick_demo -e scan

# Given only the image and predict all inputs with ECON.
python -m apps.quick_demo -e image
```


## Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 62032011) and the Natural Science Foundation of Jiangsu Province (No. BK20211147).

There are also many powerful resources that greatly benefit our work:

- [ICON](https://github.com/YuliangXiu/ICON)
- [ECON](https://github.com/YuliangXiu/ECON)
- [SMPL-X](https://github.com/vchoutas/smplx)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion)
- [EMOCA](https://github.com/radekd91/emoca)
- [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset)
- [PIFu](https://github.com/shunsukesaito/PIFu)
- [PIFuHD](https://github.com/facebookresearch/pifuhd)
- [Open-PIFuhd](https://github.com/lingtengqiu/Open-PIFuhd)
- [DecoMR](https://github.com/zengwang430521/DecoMR)
- [Densebody](https://github.com/Lotayou/densebody_pytorch)


## Citation

```bibtex
@inproceedings{zhan2024shert,
    title     = {Semantic Human Mesh Reconsturction with Textures},
    author    = {Zhan, Xiaoyu and Yang, Jianxin and Li, Yuanqi and Guo, Jie and Guo, Yanwen and Wang, Wenping},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024},
}
```


## Contact
Zhan, Xiaoyu (zhanxy@smail.nju.edu.cn) and Yang, Jianxin (jianxin-yang@smail.nju.edu.cn)