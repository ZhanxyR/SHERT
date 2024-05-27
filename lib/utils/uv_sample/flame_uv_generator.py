'''
This includes the module to realize the transformation between mesh and location map.
Part of the codes is adapted from https://github.com/Lotayou/densebody_pytorch and https://github.com/zengwang430521/DecoMR
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
class FLAME_UV_Generator(nn.Module):
    def __init__(self, UV_height, UV_width=-1, uv_type='MFLAME', data_dir=None):
        super(FLAME_UV_Generator, self).__init__()

        # obj_file = 'mflame_template.obj'

        self.uv_type = uv_type

        if data_dir is None:
            data_dir = 'data/face'

        self.data_dir = data_dir
        self.h = UV_height
        self.w = self.h if UV_width < 0 else UV_width
        # self.obj_file = obj_file
        self.para_file = 'paras_h{:04d}_w{:04d}_{}.npz'.format(self.h, self.w, self.uv_type)

        # if not os.path.isfile(join(data_dir, self.para_file)):
        #     self.process()

        para = np.load(join(data_dir, self.para_file))

        self.v_index = torch.LongTensor(para['v_index'])
        self.bary_weights = torch.FloatTensor(para['bary_weights'])
        self.vt2v = torch.LongTensor(para['vt2v'])
        self.vt_count = torch.FloatTensor(para['vt_count'])
        self.texcoords = torch.FloatTensor(para['texcoords'])
        self.texcoords = 2 * self.texcoords - 1
        self.mask = torch.ByteTensor(para['mask'].astype('uint8'))


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

        return vt

    # just used for the generation of GT UVmaps
    def forward(self, verts):
        return self.get_UV_map(verts)



