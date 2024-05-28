
### 1. Start
Start from creating a conda environment.
```bash
git clone https://github.com/ZhanxyR/SHERT.git
cd SHERT
conda create -n shert python=3.8
conda activate shert
```
### 2. Install Pytorch
Follow [Pytorch](https://pytorch.org/get-started/previous-versions/).

We recommend to use Pytorch >= 2.0. While lower version may require more GPU memory in texture inpainting.

### 3. Install Open3d
We recommend you to install a specific version of opend3d manually to avoid any problem.

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

The version we used is v0.7.6, while the lower versions should also be applicable.

(If you have any troubles in building the package, you could just set the `refine_iter` to `1` in corresponding `config.yaml` to avoid using Pytorch3D.)