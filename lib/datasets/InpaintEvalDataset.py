
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
import open3d as o3d
import pickle
import  logging
from ..utils.mesh_util import load_obj, projection_length
from ..utils.smplx_util import transfer_to_std_smplx, load_smplx_param

np.seterr(divide='ignore', invalid='ignore')

class InapintEvalDataset(Dataset):
    def __init__(self, inputs, cfg, cfg_resources):

        self.cfg = cfg
        self.resources = cfg_resources
        self.inputs = inputs
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.cfg.load_size)])

        if cfg.use_smplx_face:
            self.uv_mask = cv2.imread(self.resources.masks.mask_wo_head_eyes_hands_feet, cv2.IMREAD_GRAYSCALE)/255
        else:
            self.uv_mask = cv2.imread(self.resources.masks.mask_wo_eyes_hands_feet, cv2.IMREAD_GRAYSCALE)/255
        self.uv_mask = self.uv_mask.reshape(self.cfg.load_size, self.cfg.load_size, 1)

        _, self.subsmplx_vts, self.subsmplx_faces = load_obj(cfg_resources.models.smplx_div2_template)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        data = self.inputs[idx]

        smplx_param = load_smplx_param(data['smplx_param'])

        smplx_star_mesh = o3d.io.read_triangle_mesh(data['smplx_star'])
        smplx_star_mesh.vertices = o3d.utility.Vector3dVector(transfer_to_std_smplx(smplx_star_mesh.vertices,  smplx_param["scale"], smplx_param["translation"]))
        smplx_star_vertices = np.asarray(smplx_star_mesh.vertices)
        smplx_star_mesh.compute_vertex_normals()
        smplx_star_normals = np.asarray(smplx_star_mesh.vertex_normals)

        smplx_mesh = o3d.io.read_triangle_mesh(data['subsmplx'])
        smplx_mesh.vertices = o3d.utility.Vector3dVector(transfer_to_std_smplx(smplx_mesh.vertices,  smplx_param["scale"], smplx_param["translation"]))
        smplx_vertices = np.asarray(smplx_mesh.vertices)
        smplx_mesh.compute_vertex_normals()
        smplx_normals = np.asarray(smplx_mesh.vertex_normals)

        sns_mesh = o3d.io.read_triangle_mesh(data['sampled_mesh'])
        sns_mesh.vertices = o3d.utility.Vector3dVector(transfer_to_std_smplx(sns_mesh.vertices,  smplx_param["scale"], smplx_param["translation"]))
        sns_vertices = np.asarray(sns_mesh.vertices)

        # TODO: add dilation
        error_uv_mask = cv2.imread(data['error_uv']).astype(float) * self.uv_mask

        displacement = projection_length(sns_vertices - smplx_vertices, smplx_normals)
        displacement = np.repeat(displacement, 3, axis=-1)
        star_vertices = smplx_star_vertices + displacement * smplx_star_normals

        # transform
        if self.transform:
            error_uv_mask = self.transform(error_uv_mask) / 255.

        return {
                "smplx_star_normals": torch.from_numpy(smplx_star_normals).float(),
                "smplx_star_vertices": torch.from_numpy(smplx_star_vertices).float(),
                "smplx_vertices": torch.from_numpy(smplx_vertices).float(),
                "smplx_normals": torch.from_numpy(smplx_normals).float(),
                "sns_vertices": torch.from_numpy(sns_vertices).float(),
                "displacement": torch.from_numpy(displacement).float(),
                "star_vertices": torch.from_numpy(star_vertices).float(),
                "error_uv_mask": error_uv_mask.float(),
                "param_scale": smplx_param["scale"],
                "param_translation": smplx_param["translation"],
                }
