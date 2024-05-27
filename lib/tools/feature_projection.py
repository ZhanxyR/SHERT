'''
This includes the module to project the image colors to the reconstructed mesh.
Part of the codes is adapted from https://github.com/shunsukesaito/PIFu
'''

import numpy as np
import torch
import cv2
import os
import open3d as o3d
def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts

def index(feat, uv):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

if __name__=="__main__":
    mesh = o3d.io.read_triangle_mesh("../60_0_00_full.obj")
    vertices = np.asarray(mesh.vertices)
    vertices = torch.from_numpy(vertices).float().unsqueeze(dim=0).permute(0,2,1)


    calib = np.load("./data/econ_calib.npy")
    calib = torch.from_numpy(calib).float().unsqueeze(dim=0)
    crop_query_points = orthogonal(vertices, calib, None)
    print("vertices shape {}".format(vertices.shape))
    print("crop_query_points shape {}".format(crop_query_points.shape))
    xy = crop_query_points[:, :2, :]
    z = crop_query_points[:, 2:3, :]

    print("xy shape {}".format(xy.shape))
    # in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
    # print("in_img shape {}".format(in_img.shape))
    img = cv2.imread("60_0_00.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) /255.
    img = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(dim=0)
    sample_fea = index(img, xy).permute(0,2,1)[0].numpy()
    print("sampler fea shape {}".format(sample_fea.shape))
    mesh.vertex_colors = o3d.utility.Vector3dVector(sample_fea)


    o3d.visualization.draw_geometries([mesh],window_name="mesh")



