import os
import sys
import logging
import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import open3d as o3d

from lib.utils.config import get_cfg_defaults
from lib.models.RefineUNet import RefineUNet
from lib.datasets.RefineEvalDataset import RefineEvalDataset
from lib.utils.uv_sample.divided_uv_generator import Index_UV_Generator
from lib.utils.mesh_util import load_obj, save_obj
from lib.tools.feature_projection import orthogonal, index


def evaluate_refine(sampler, model, dataset, cfg, cfg_resources, device, save_root=None):
    # 1. Create dataset
    cfg_test = cfg.test
    batch_size = cfg_test.batch_size

    eval_loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    eval_loader = DataLoader(dataset, **eval_loader_args)

    # (Initialize logging)
    logging.info(f'Data: {dataset.__len__()}| Iter: {cfg_test.refine_iter} | Smooth: {cfg_test.smooth_iter}')

    uv_mask = dataset.uv_mask
    uv_mask = uv_mask[..., 0]
    uv_mask = torch.from_numpy(uv_mask.astype(float)).unsqueeze(dim=-1).to(device=device).expand(batch_size, *uv_mask.shape, 3).float()

    smplx_faces = dataset.subsmplx_faces
    smplx_faces = smplx_faces
    smplx_faces_tensor = torch.from_numpy(smplx_faces[..., 0]).expand(batch_size, *(smplx_faces.shape[:-1])).to(device)
    econ_clib = np.load(cfg_resources.econ_calib)
    econ_clib = torch.from_numpy(econ_clib).expand(batch_size, *econ_clib.shape).float().to(device=device)

    for batch in tqdm(eval_loader):
        inpaint_uv = sampler.get_UV_map(batch["inpaint_vertices"].to(device=device))
        inpaint_normal_uv = sampler.get_UV_map(batch["inpaint_normals"].to(device=device))
        smplx_uv = sampler.get_UV_map(batch["smplx_vertices"].to(device=device))
        smplx_normal_uv = sampler.get_UV_map(batch["smplx_normals"].to(device=device))

        render_img = batch["render_img"].to(device=device)
        front_normal = batch["front_normal"].to(device=device)
        back_normal = batch["back_normal"].to(device=device)

        inpaint_uv_vertices = (inpaint_uv * uv_mask).permute(0, 3, 1, 2).reshape(batch_size, 3, -1)
        query_points = orthogonal(inpaint_uv_vertices, econ_clib, None)
        xy = query_points[:, :2, :]
        z = query_points[:, 2:3, :]


        save_path = save_root
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if save_root is not None:
            save_obj(batch["inpaint_vertices"][0].clone().detach().cpu().numpy(), smplx_faces,  os.path.join(save_root, 'refine_smooth.obj'))

        for i in range(cfg_test.refine_iter):
            pred_disp = model(
                torch.cat([(inpaint_uv * uv_mask).permute(0, 3, 1, 2),
                           (inpaint_normal_uv * uv_mask).permute(0, 3, 1, 2),
                           (smplx_uv * uv_mask).permute(0, 3, 1, 2), (smplx_normal_uv * uv_mask).permute(0, 3, 1, 2)], dim=1),
                render_img,
                torch.cat([front_normal, back_normal], dim=1),
                xy, z).permute(0, 2, 3, 1)

            pred_full_uv = pred_disp * uv_mask * inpaint_normal_uv + inpaint_uv
            pred_full_verts = sampler.resample(pred_full_uv)

            if cfg_test_refine_iter > 1:
                import pytorch3d.structures
                pred_mesh = pytorch3d.structures.Meshes(verts=pred_full_verts, faces=smplx_faces_tensor - 1)
                pred_vertex_normals = pred_mesh.verts_normals_padded()
                pred_normal_uv = sampler.get_UV_map(pred_vertex_normals)

                # example iter
                inpaint_uv = pred_full_uv
                inpaint_normal_uv = pred_normal_uv

            if save_root is not None:
                # econ_calib
                trans_pred = pred_full_verts[0].clone().detach().cpu().numpy() * 100 / batch['scale'][0] + batch['center'][0]
                if i > 0:
                    result = o3d.geometry.TriangleMesh()
                    result.vertices = o3d.utility.Vector3dVector(trans_pred)
                    result.triangles = o3d.utility.Vector3iVector(smplx_faces[..., 0] - 1)
                    # smooth for better results
                    result = result.filter_smooth_laplacian(1, 0.5)
                    save_obj(np.asarray(result.vertices), smplx_faces, os.path.join(save_root, 'refine_s%d_i%d.obj' % (cfg_test.smooth_iter, i)))
                else:
                    save_obj(trans_pred, smplx_faces, os.path.join(save_root, 'refine_s%d_i%d.obj' % (cfg_test.smooth_iter, i)))



# if __name__ == '__main__':

#     cfg_resources = get_cfg_defaults("lib/configs/resources.yaml")
#     cfg_refine = get_cfg_defaults("lib/configs/refine.yaml")

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     torch.cuda.empty_cache()
#     torch.cuda.set_device(device)

#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#     model = RefineUNet(cfg_refine.model.input_channels, 3, 6, 1, bilinear=cfg_refine.model.bilinear)

#     model.requires_grad_(False)
#     model.eval()
#     model.cuda()
#     state_dict = torch.load(cfg_refine.test.ckpt_path, map_location=device)
#     model.load_state_dict(state_dict['model'])
#     logging.info(f'load refine model from {cfg_refine.test.ckpt_path} on {device}')

#     evaluate_refine(cfg_refine, cfg_resources, model, device)





