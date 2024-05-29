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
    <a href="https://cs.nju.edu.cn/ywguo/index.htm" rel="external nofollow noopener" target="_blank"><strong>Yanwen Guo*</strong></a>
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

## Installation

- Build environment. See [install](docs/install.md).
- Download required data. See [resources](docs/resources.md).

## Demo

The whole processes include two steps: `reconstruction` and `texture inpainting`.

Run `quick_demo` to test `reconstruction` in given resources. The results will be saved to `./examples/$subject$/results`.

```bash
# Use ECON-pred mesh and fitted smplx.
python -m apps.quick_demo

# Use THuman scan and fitted smplx.
python -m apps.quick_demo -e scan

# Given only image and predict all inputs with ECON.
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