import logging
import os

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import open3d as o3d
from ..utils.smplx_util import back_to_econ_axis
from ..utils.mesh_util import load_obj


class RefineEvalDataset(Dataset):
    def __init__(self, inputs, cfg, cfg_resources):
        self.cfg = cfg
        self.resources = cfg_resources
        self.inputs = inputs
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.uv_mask = cv2.imread(self.resources.masks.mask_wo_head_eyes_hands_feet, cv2.IMREAD_GRAYSCALE)/255
        self.uv_mask = self.uv_mask.reshape(self.cfg.load_size, self.cfg.load_size, 1)

        _, self.subsmplx_vts, self.subsmplx_faces = load_obj(cfg_resources.models.smplx_div2_template)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        data = self.inputs[idx]

        if 'camera' in data.keys():
            render_param = np.load(data['camera'], allow_pickle=True).item()
        else:
            render_param = {'center': np.asarray([0, 0, 0]), 'scale': 100}

        smplx_mesh = o3d.io.read_triangle_mesh(data['subsmplx'])
        smplx_mesh.vertices = o3d.utility.Vector3dVector(back_to_econ_axis(smplx_mesh.vertices, render_param, 0))
        smplx_mesh.compute_vertex_normals()
        smplx_normals = np.asarray(smplx_mesh.vertex_normals)
        smplx_vertices = np.asarray(smplx_mesh.vertices)

        inpaint_mesh = o3d.io.read_triangle_mesh(data['completed_mesh'])
        inpaint_mesh.vertices = o3d.utility.Vector3dVector(back_to_econ_axis(inpaint_mesh.vertices, render_param, 0))
        input_vertices = np.asarray(inpaint_mesh.vertices)
        if self.cfg.smooth_iter > 0:
            inpaint_mesh = inpaint_mesh.filter_smooth_laplacian(self.cfg.smooth_iter, self.cfg.smooth_lambda)
        inpaint_mesh.compute_vertex_normals()
        inpaint_normals = np.asarray(inpaint_mesh.vertex_normals)
        inpaint_vertices = np.asarray(inpaint_mesh.vertices)

        normal_mask = cv2.imread(data['mask']).astype(float)
        if normal_mask.shape[0] != self.cfg.load_size:
            normal_mask = cv2.resize(normal_mask, (self.cfg.load_size, self.cfg.load_size), interpolation=cv2.INTER_LINEAR)

        render_img = cv2.imread(data['image']).astype(float)
        if render_img.shape[0] != self.cfg.load_size:
            render_img = cv2.resize(render_img, (self.cfg.load_size, self.cfg.load_size), interpolation=cv2.INTER_LINEAR)
        render_img = render_img * (normal_mask / 255.)

        front_normal = cv2.imread(data['front_normal']).astype(float)
        back_normal = cv2.imread(data['back_normal']).astype(float)
        
        if front_normal.shape[0] != self.cfg.load_size:
            front_normal = cv2.resize(front_normal, (self.cfg.load_size, self.cfg.load_size), interpolation=cv2.INTER_LINEAR)
            back_normal = cv2.resize(back_normal, (self.cfg.load_size, self.cfg.load_size), interpolation=cv2.INTER_LINEAR)

        if self.cfg.normal_flip:
            back_normal = np.ascontiguousarray(np.flip(back_normal, axis=1))

        # transform
        # if self.transform:
        render_img = self.transform(render_img) / 255.
        front_normal = self.transform(front_normal) / 255.
        back_normal = self.transform(back_normal) / 255.
        normal_mask = torch.from_numpy(normal_mask) / 255.

        if self.cfg.normal_flip:
            front_normal = ((front_normal.permute(1, 2, 0) * 2 - 1) * normal_mask).permute(2, 0, 1)
            back_normal = ((back_normal.permute(1, 2, 0) * 2 - 1) * torch.tensor([-1, 1, -1]) * normal_mask).permute(2, 0, 1) 
        else:
            front_normal = ((front_normal.permute(1, 2, 0) * 2 - 1) * torch.tensor([1, -1, 1]) *normal_mask).permute(2, 0, 1)
            back_normal = ((back_normal.permute(1, 2, 0) * 2 - 1) * torch.tensor([1, -1, 1]) * normal_mask).permute(2, 0, 1)

        return {"input_vertices": torch.from_numpy(input_vertices).float(),
                "inpaint_vertices": torch.from_numpy(inpaint_vertices).float(),
                "inpaint_normals": torch.from_numpy(inpaint_normals).float(),
                "smplx_normals": torch.from_numpy(smplx_normals).float(),
                "smplx_vertices": torch.from_numpy(smplx_vertices).float(),
                "render_img": render_img.float(),
                "front_normal": front_normal.float(),
                "back_normal": back_normal.float(),
                "center": render_param['center'],
                "scale": render_param['scale']}
