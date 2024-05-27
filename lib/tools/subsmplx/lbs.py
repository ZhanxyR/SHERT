# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# import pickle
from typing import Tuple, List
import numpy as np

import torch
import torch.nn.functional as F

from smplx.utils import rot_mat_to_euler, Tensor
from smplx.lbs import batch_rigid_transform, batch_rodrigues, vertices2joints, blend_shapes


# standard template pose back to specific pose
def lbs_back(t_verts, T):
    batch_size = t_verts.shape[0]
    device, dtype = t_verts.device, t_verts.dtype
    T = T.to(device=device, dtype=dtype)
    homogen_coord = torch.ones([batch_size, t_verts.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([t_verts, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts

# REVISE: design for subdivided smplx
def lbs_t(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    src_verts: Tensor,
    toT : bool = True,
    pose2rot: bool = True,
    skinning_weight_path: str = "./data/smplx/lbs_weights_divide2.npy",
) -> Tuple[Tensor, Tensor, Tensor]:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype


    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    # print(v_shaped.shape)
    # v_shaped = src_verts.float().clone()

    # 2. Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        # pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        # pose_offsets = torch.matmul(
            # pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        # pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        # pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    # posedirs).view(batch_size, -1, 3)

    # REVISE: original v_posed represents shaped smplx vertices with pose_offsets, repalce it with input detailed vertices
    v_posed = src_verts.float()


    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)

    # REVISE: save W in original smplx lbs function as skinning weights
    # load skinning weights for subdivided model(subdivide skinning weights like the model)
    loaded_lbs_weights = np.load(skinning_weight_path)
    loaded_lbs_weights = torch.from_numpy(loaded_lbs_weights).to(device=device, dtype=dtype)
    W = loaded_lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])

    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)

    # REVISE
    if toT: # input pose -> standard template pose
        v_homo = torch.matmul(torch.linalg.inv(T), torch.unsqueeze(v_posed_homo, dim=-1))
    else: # standard template pose -> specific pose (same as original lbs)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, T

