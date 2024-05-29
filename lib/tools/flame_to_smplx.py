import numpy as np
import torch
import cv2
from lib.utils.mesh_util import load_obj, save_obj
from lib.utils.uv_sample.FLAME import texture_flame2smplx
from lib.utils.config import  get_cfg_defaults
from lib.utils.uv_sample.flame_uv_generator import FLAME_UV_Generator
from lib.utils.uv_sample.divided_uv_generator import Index_UV_Generator

def flame_resample(verts, flame_sampler):

    flame_uv = flame_sampler.get_UV_map(verts)
    # visualization
    # write_pic($path$, np.flip(flame_uv[0].detach().cpu().numpy()))
    # vert_resample = flame_sampler.resample(flame_uv.unsqueeze(0))
    # _, vts, f = load_obj('data/face/mflame_template.obj')
    # save_mtl(vert_resample[0].detach().cpu().numpy(), f, vts, $path$)

    return flame_uv

def flame_to_smplx(verts, cfg_resources, smplx_sampler, device, flame_texture=None, mouse_correct=True):

    flame_sampler = FLAME_UV_Generator(UV_height=512, UV_width=-1, uv_type='MFLAME', data_dir='data/face').to(device)
    # smplx_sampler = Index_UV_Generator(UV_height=1024, UV_width=-1, uv_type='SMPLX', data_dir='data/smplx').to(device)
    cached_data = np.load(cfg_resources.index.flame_to_smplx_verts_index, allow_pickle=True, encoding = 'latin1').item()
    smplx_texture = np.zeros((1024, 1024, 3))

    flame_uv = flame_resample(torch.from_numpy(verts).unsqueeze(0), flame_sampler)
    flame_uv = flame_uv[0].detach().cpu().numpy()
    flame_uv = np.fliplr(flame_uv)
    flame_uv = np.flip(flame_uv)
    # write_pic($path$, flame_uv)

    smplx_uv = texture_flame2smplx(cached_data, flame_uv, smplx_texture)
    # write_pic($path$, smplx_uv)
    smplx_uv = np.flip(smplx_uv)
    smplx_uv = np.fliplr(smplx_uv)

    if mouse_correct:

        import cv2
        from skimage.restoration import inpaint

        # mouse_mask = cv2.imread("data/face/mouse_mask.png")
        mouse_mask = cv2.imread(cfg_resources.masks.mouse)
        kernel = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]]).astype(np.uint8)
        mouse_mask = cv2.dilate(mouse_mask.astype(np.uint8), kernel, iterations=2)
        mouse_mask = cv2.cvtColor(mouse_mask, cv2.COLOR_BGR2RGB) / 255.
        mouse_mask = mouse_mask[:, :, 0]
        smplx_uv = inpaint.inpaint_biharmonic(smplx_uv, mouse_mask, channel_axis=2)

    smplx_uv_t = torch.from_numpy(smplx_uv.copy()).float().to(device)
    smplx_vert_resample = smplx_sampler.resample(smplx_uv_t.unsqueeze(0))

    if flame_texture is not None:
        smplx_tex_uv = texture_flame2smplx(cached_data, flame_texture, smplx_texture)
    else:
        smplx_tex_uv = smplx_texture

    return smplx_vert_resample, smplx_tex_uv

# '''
# Input:  
#     |---FLAME-detailed:      
#         |---Vertices        [59315, 3]
#         |---Faces           [117380, 3]
#     |---Texture(FLAME)      [256, 256, 3] 

# Output:
#     |---SMPLX-d2:            
#         |---Vertices        [149921, 3]
#         |---Faces           [299712, 3]
#     |---Texture(SMPLX)      [1024, 1024, 3]
# '''
# if __name__=='__main__':

#     cfg_resources = get_cfg_defaults("lib/configs/resources.yaml")
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     v, _, _ = load_obj('zxy_debug/input/mesh_coarse_detail.obj')
#     flame_texture = cv2.imread('zxy_debug/input/deca.png')

#     # v, uv = flame_to_smplx(v, device)
#     v, uv = flame_to_smplx(v, cfg_resources, device, flame_texture=flame_texture)

#     v = v[0].detach().cpu().numpy()
#     _, vts, f = load_obj('data/smplx/smplx_div2_template.obj')
#     save_obj(v, f, "zxy_debug/output/smplx_resample.obj", vts=vts)

#     cv2.imwrite('zxy_debug/output/smplx_uv.png', uv)


