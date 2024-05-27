import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import torch
from .body_models import create
from lib.utils.smplx_util import transfer_to_std_smplx


'''
    important: some params to init smplx model need to adjusted according to the input smplx format 
    like `num_betas`, `num_pca_comps`(if `use_pca`=True), `num_expression_coeffs`, etc.
'''
# specific pose to standard template pose with std scale
def back_to_template_pose(verts, smplx_param, skinning_weight_path, gender="male", use_pca=True,num_pca_comps=12, num_betas=10):
    verts  = transfer_to_std_smplx(verts, smplx_param["scale"], smplx_param["translation"])

    # 1.  init smplx model
    model_init_params = dict(
        gender=gender,
        model_type='smplx',
        model_path='./data/models',
        create_global_orient=False,
        create_body_pose=False,
        create_betas=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_transl=False,
        use_pca=use_pca,
        num_pca_comps=num_pca_comps, # for Thuman smplx_param TODO: adjust according to your smplx param format
        num_betas=num_betas # for Thuman smplx_param TODO: adjust according to your smplx param format
    )
    smplx_model = create(**model_init_params)

    # 2. smplx forward (pose transformation)
    for key in smplx_param.keys():
        smplx_param[key] = torch.as_tensor(smplx_param[key]).to(torch.float32)
    model_forward_params = dict(betas=smplx_param['betas'],
                                global_orient=smplx_param['global_orient'],
                                body_pose=smplx_param['body_pose'],
                                left_hand_pose=smplx_param['left_hand_pose'],
                                right_hand_pose=smplx_param['right_hand_pose'],
                                jaw_pose=smplx_param['jaw_pose'],
                                leye_pose=smplx_param['leye_pose'],
                                reye_pose=smplx_param['reye_pose'],
                                expression=smplx_param['expression'],
                                return_verts=True,
                                )
    model_forward_params["src_verts"] = verts
    model_forward_params["skinning_weight_path"] = skinning_weight_path
    smplx_out, _, T_org_pose = smplx_model(**model_forward_params)
    t_verts = smplx_out.vertices[0].detach().cpu().numpy()
    return t_verts

# standard template pose to specific pose
def repose(t_verts, smplx_param, skinning_weight_path, gender="male",use_pca=True, num_pca_comps=12, num_betas=10, num_expression_coeffs=50):
    # 1. init smplx model
    model_init_params = dict(
        gender=gender,
        model_type='smplx',
        model_path='./data/models',
        create_global_orient=False,
        create_body_pose=False,
        create_betas=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        num_pca_comps=num_pca_comps, # TODO: adjust according to your smplx param format
        use_pca=use_pca, # TODO: adjust according to your smplx param format
        num_expression_coeffs=num_expression_coeffs,  # TODO: adjust according to your smplx param format
        num_betas=num_betas # TODO: adjust according to your smplx param format
    )
    smplx_model = create(**model_init_params)

    # smplx forward
    for key in smplx_param.keys():
        smplx_param[key] = torch.as_tensor(smplx_param[key]).to(torch.float32)
    model_forward_params = dict(betas=smplx_param['betas'],
                                global_orient=smplx_param['global_orient'],
                                body_pose=smplx_param['body_pose'],
                                left_hand_pose=smplx_param['left_hand_pose'],
                                right_hand_pose=smplx_param['right_hand_pose'],
                                jaw_pose=smplx_param['jaw_pose'],
                                expression=smplx_param['expression'],
                                return_verts=True,
                                return_shaped=False)
    model_forward_params["src_verts"] = t_verts
    model_forward_params["skinning_weight_path"] = skinning_weight_path
    model_forward_params["toT"] = False
    smpl_out, _, tranfered_pose = smplx_model(**model_forward_params)
    repose_verts = smpl_out.vertices[0].detach().cpu().numpy()
    return repose_verts

















