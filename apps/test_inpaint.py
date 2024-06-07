import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from lib.models.InpaintUNet import InpaintUNet
from lib.datasets.InpaintEvalDataset import InapintEvalDataset
from lib.utils.uv_sample.divided_uv_generator import Index_UV_Generator
from lib.utils.mesh_util import load_obj, save_obj
from lib.utils.smplx_util import back_to_org_smplx, load_smplx_param
from lib.utils.config import  get_cfg_defaults


def evaluate_inpaint(sampler, model, dataset, cfg, cfg_resources, device, save_root=None):

    cfg_test = cfg.test

    eval_loader_args = dict(batch_size=cfg_test.batch_size, num_workers=8, pin_memory=True)
    eval_loader = DataLoader(dataset, **eval_loader_args) 

    smplx_faces = dataset.subsmplx_faces
    uv_mask = dataset.uv_mask
    uv_mask = torch.from_numpy(uv_mask).unsqueeze(0).to(device)

    for batch in tqdm(eval_loader):
        smplx_normals = batch['smplx_normals'].to(device)
        smplx_vertices = batch['smplx_vertices'].to(device)
        smplx_star_vertices = batch["smplx_star_vertices"].to(device)
        smplx_star_normals = batch["smplx_star_normals"].to(device)
        displacement = batch['displacement'].to(device)
        star_vertices = batch['star_vertices'].to(device)
        error_uv_mask = batch['error_uv_mask'].to(device)
    
        smplx_star_uv = sampler.get_UV_map(smplx_star_vertices)
        smplx_star_normal_uv = sampler.get_UV_map(smplx_star_normals) * uv_mask
        displacement_uv = sampler.get_UV_map(displacement) * uv_mask
        star_uv = sampler.get_UV_map(star_vertices) * uv_mask

        error_uv_mask = error_uv_mask.permute(0, 2, 3, 1) * uv_mask
        correct_uv_mask = error_uv_mask.clone() + 1
        correct_uv_mask[correct_uv_mask == 2] = 0
        correct_uv_mask = correct_uv_mask * uv_mask

        correct_displacement_uv = displacement_uv * correct_uv_mask
        correct_star_uv = star_uv * correct_uv_mask

        # inpaint error area based on surrounding good area on template pose
        input_features = torch.cat(
            [correct_uv_mask, correct_star_uv, smplx_star_normal_uv, smplx_star_uv, correct_displacement_uv],
            dim=3).permute(0, 3, 1, 2).float()

        # model
        pred_full_disp_uv = model(input_features)

        # resample
        pred_full_disp_uv = pred_full_disp_uv.permute(0, 2, 3, 1)
        pred_disp_uv = pred_full_disp_uv * error_uv_mask + correct_displacement_uv
        pred_star_uv = smplx_star_uv + pred_disp_uv * smplx_star_normal_uv
        pred_star_vertices = sampler.resample(pred_star_uv)

        smplx_uv = sampler.get_UV_map(smplx_vertices)
        smplx_normal_uv = sampler.get_UV_map(smplx_normals)
        pred_org_uv = smplx_uv + pred_disp_uv * smplx_normal_uv
        pred_org_verts = sampler.resample(pred_org_uv)

        # back to org (test batch = 1)
        # TODO: Batch
        param_scale_np = batch["param_scale"][0].detach().cpu().numpy()
        param_translation_np = batch["param_translation"][0].detach().cpu().numpy()
        
        pred_star_vertices_np = pred_star_vertices[0].detach().cpu().numpy()
        pred_star_vertices_np = back_to_org_smplx(pred_star_vertices_np, param_scale_np, param_translation_np)

        pred_org_verts_np = pred_org_verts[0].detach().cpu().numpy()
        pred_org_verts_np = back_to_org_smplx(pred_org_verts_np, param_scale_np, param_translation_np)
        
        if save_root is not None:
            os.makedirs(save_root, exist_ok=True)

            # TODO: (optional) vt
            star_path = os.path.join(save_root, "pred_star_inpaint.obj")
            save_obj(pred_star_vertices_np, smplx_faces, star_path, single=True)
            org_path = os.path.join(save_root, "pred_org_inpaint.obj")
            save_obj(pred_org_verts_np, smplx_faces, org_path, single=True)

            return pred_star_vertices_np, pred_org_verts_np, star_path, org_path
        
        return pred_star_vertices_np, pred_org_verts_np


