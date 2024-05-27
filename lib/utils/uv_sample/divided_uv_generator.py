'''
This includes the module to realize the transformation between mesh and location map.
Part of the codes is adapted from https://github.com/Lotayou/densebody_pytorch
'''

from time import time
from tqdm import tqdm
from numpy.linalg import solve
from os.path import join
import torch
from torch import nn
import os
import torch.nn.functional as F

import pickle
import numpy as np

'''
Index_UV_Generator is used to transform mesh and location map
The verts is in shape (B * V *C)
The UV map is in shape (B * H * W * C)
B: batch size;     V: vertex number;   C: channel number
H: height of uv map;  W: width of uv map
'''
class Index_UV_Generator(nn.Module):
    def __init__(self, UV_height=1024, UV_width=-1, uv_type='SMPLX', data_dir=None):
        super(Index_UV_Generator, self).__init__()

        if uv_type == 'SMPLX':
            obj_file = 'smplx_div2_template.obj'
        # elif uv_type == 'SMPL':
        #     obj_file = 'smpl_fbx_template.obj'
        # elif uv_type == 'BF':
        #     obj_file = 'smpl_boundry_free_template.obj'

        self.uv_type = uv_type

        if data_dir is None:
            data_dir = 'data/smplx'

        self.data_dir = data_dir
        self.h = UV_height
        self.w = self.h if UV_width < 0 else UV_width
        self.obj_file = obj_file
        self.para_file = 'paras_h{:04d}_w{:04d}_{}.npz'.format(self.h, self.w, self.uv_type)

        if not os.path.isfile(join(data_dir, self.para_file)):
            self.process()

        para = np.load(join(data_dir, self.para_file))

        self.v_index = torch.LongTensor(para['v_index'])
        self.bary_weights = torch.FloatTensor(para['bary_weights'])
        self.vt2v = torch.LongTensor(para['vt2v'])
        self.vt_count = torch.FloatTensor(para['vt_count'])
        self.texcoords = torch.FloatTensor(para['texcoords'])
        self.texcoords = 2 * self.texcoords - 1
        self.mask = torch.ByteTensor(para['mask'].astype('uint8'))

    def process(self):
        ##############################################################################
        # Load template obj file
        print('Loading obj file')
        with open(join(self.data_dir, self.obj_file), 'r') as fin:
            lines = [l
                     for l in fin.readlines()
                     if len(l.split()) > 0
                     and not l.startswith('#')
                     ]

        # Load all vertices (v) and texcoords (vt)
        vertices = []
        texcoords = []

        for line in lines:
            lsp = line.split()
            if lsp[0] == 'v':
                x = float(lsp[1])
                y = float(lsp[2])
                z = float(lsp[3])
                vertices.append((x, y, z))
            elif lsp[0] == 'vt':
                u = float(lsp[1])
                v = float(lsp[2])
                # texcoords.append((1 - v, u))
                texcoords.append((u, v))


        # Stack these into an array
        vertices = np.vstack(vertices).astype(np.float32)
        texcoords = np.vstack(texcoords).astype(np.float32)

        # Load face data. All lines are of the form:
        # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
        # Store the texcoord faces and a mapping from texcoord faces to vertex faces
        vt_faces = []
        v_num = vertices.shape[0]
        vt_num = texcoords.shape[0]
        # vt2v = np.zeros(vt_num).astype('int64') - 1
        vt2v = np.zeros(vt_num).astype('int64') + 149921
        v2vt = [None] * v_num
        for i in range(v_num):
            v2vt[i] = set()

        for line in lines:
            vs = line.split()
            if vs[0] == 'f':
                v0 = int(vs[1].split('/')[0]) - 1
                v1 = int(vs[2].split('/')[0]) - 1
                v2 = int(vs[3].split('/')[0]) - 1
                vt0 = int(vs[1].split('/')[1]) - 1
                vt1 = int(vs[2].split('/')[1]) - 1
                vt2 = int(vs[3].split('/')[1]) - 1
                vt_faces.append((vt0, vt1, vt2))

                # print(v0, vt0, v1, vt1, v2, vt2) # 3 0 37497 40197 37498 40198
                # exit()

                vt2v[vt0] = v0
                vt2v[vt1] = v1
                vt2v[vt2] = v2

                v2vt[v0].add(vt0)
                v2vt[v1].add(vt1)
                v2vt[v2].add(vt2)

        # print(vt2v.shape) # (154161,)
        # print(len(v2vt)) # 149921
        # print(vt_num) # 154161


        vt_faces = np.vstack(vt_faces)
        vt_count = np.zeros(v_num)
        for v_id in range(v_num):
            vt_count[v_id] = len(v2vt[v_id])

        ############################################################################
        # Calculating the barycentric weights used for UV map generation
        print('Calculating barycentric weights')
        s = time()
        h = self.h
        w = self.w
        face_num = vt_faces.shape[0]

        face_id = np.zeros((h, w), dtype=int)
        bary_weights = np.zeros((h, w, 3), dtype=np.float32)
        # uvs = texcoords * np.array([[h - 1, w - 1]])
        uvs = texcoords * np.array([[w - 1, h - 1]])
        grids = np.ones((face_num, 3), dtype=np.float32)
        anchors = np.concatenate((
            uvs[vt_faces].transpose(0, 2, 1),
            np.ones((face_num, 1, 3), dtype=uvs.dtype)
        ), axis=1)  # [F * 3 * 3]

        ##############################
        # print(anchors.shape)
        # print(grids.shape)

        # print(anchors[:10])
        # np.linalg.pinv(anchors)
        det = np.linalg.det(anchors)
        # print(det.shape[0])
        for i in range(det.shape[0]):
            if det[i] == 0:
                # print(i)
                # print(anchors[i])
                anchors[i][0] += np.array([1e-7, 2e-7, 3e-7])
        # np.linalg.inv(anchors)
        ##############################

        # exit()

        _loop = tqdm(np.arange(h * w), ncols=80)
        for i in _loop:
            r = i // w
            c = i % w
            # grids[:, 0] = r
            # grids[:, 1] = c

            grids[:, 0] = c
            grids[:, 1] = r

            weights = solve(anchors, grids)  # not enough accuracy?
            inside = np.logical_and.reduce(weights.T > 1e-10)
            index = np.where(inside == True)[0]

            if 0 == index.size:
                face_id[r, c] = -1  # just assign random id with all zero weights.
            # elif index.size > 1:
            #    print('bad %d' %i)
            else:
                face_id[r, c] = index[0]
                bary_weights[r, c] = weights[index[0]]

        v_index = vt2v[vt_faces[face_id]]
        mask = face_id >= 0
        v_index[face_id < 0, :] = 0
        bary_weights[face_id < 0, :] = 0
        print('Calculating finished. Time elapsed: {}s'.format(time() - s))

        ############################################################################
        # Ensure the neighboring pixels of vt on the UV map are meaningful
        tex = torch.FloatTensor(texcoords) * torch.FloatTensor([[w-1, h-1]])
        u_grid = tex[:, 0]
        u_lo = u_grid.floor().long().clamp(min=0, max=w - 1)
        u_hi = (u_lo + 1).clamp(max=w - 1)
        u_grid = torch.min(u_hi.float(), u_grid)
        u_w = u_grid - u_lo.float()

        v_grid = tex[:, 1]
        v_lo = v_grid.floor().long().clamp(min=0, max=h - 1)
        v_hi = (v_lo + 1).clamp(max=h - 1)
        v_grid = torch.min(v_hi.float(), v_grid)
        v_w = v_grid - v_lo.float()

        w_vlo_ulo = (1.0 - u_w) * (1.0 - v_w)
        w_vlo_uhi = u_w * (1.0 - v_w)
        w_vhi_ulo = (1.0 - u_w) * v_w
        w_vhi_uhi = u_w * v_w

        w = torch.cat([w_vlo_ulo, w_vlo_uhi, w_vhi_ulo, w_vhi_uhi], dim=0)
        u = torch.cat([u_lo, u_hi, u_lo, u_hi], dim=0)
        v = torch.cat([v_lo, v_lo, v_hi, v_hi], dim=0)
        v_id = torch.LongTensor(vt2v).repeat(4)        # The function repeat for ndarray and tensor are different!

        w_sorted, sort_index = w.sort(dim=0, descending=True)
        u_sorted = u[sort_index]
        v_sorted = v[sort_index]
        v_id_sorted = v_id[sort_index]

        # asign the empty pixel with the value of the 1 vt with maximal weights
        n_expand = 0
        for i in range(len(w_sorted)):
            m = mask[v_sorted[i], u_sorted[i]]
            if not m:
                v_index[v_sorted[i], u_sorted[i], 0] = v_id_sorted[i]
                bary_weights[v_sorted[i], u_sorted[i], 0] = 1
                mask[v_sorted[i], u_sorted[i]] = 1
                n_expand += 1

        np.savez(join(self.data_dir, self.para_file),
                 v_index=v_index,
                 bary_weights=bary_weights,
                 texcoords=texcoords,
                 vt2v=vt2v,
                 vt_count=vt_count,
                 mask=mask,
                 )


    def get_UV_map(self, verts):
        self.bary_weights = self.bary_weights.type(verts.dtype).to(verts.device)

        change = np.where(self.v_index.cpu().numpy()==149921, 0, self.v_index.cpu().numpy())
        self.v_index = torch.from_numpy(change).to(verts.device)

        if verts.dim() == 2:
            verts = verts.unsqueeze(0)

        im = verts[:, self.v_index, :]
        bw = self.bary_weights[:, :, None, :]

        im = torch.matmul(bw, im).squeeze(dim=3)
        return im

    def resample(self, uv_map):
        batch_size, _, _, channel_num = uv_map.shape
        v_num = self.vt_count.shape[0]
        self.texcoords = self.texcoords.type(uv_map.dtype).to(uv_map.device)
        self.vt2v = self.vt2v.to(uv_map.device)
        self.vt_count = self.vt_count.type(uv_map.dtype).to(uv_map.device)

        uv_grid = self.texcoords[None, None, :, :].expand(batch_size, -1, -1, -1)

        vt = F.grid_sample(uv_map.permute(0, 3, 1, 2), uv_grid, mode='bilinear', align_corners=True)
        vt = vt.squeeze(2).permute(0, 2, 1)
        v = vt.new_zeros([batch_size, v_num+1, channel_num])

        index = np.argwhere(self.vt2v.cpu().numpy()==-1)

        v.index_add_(1, self.vt2v, vt)
        v = v[:,:149921,:]

        v = v / self.vt_count[None, :, None]
        return v

    # just used for the generation of GT UVmaps
    def forward(self, verts):
        return self.get_UV_map(verts)
