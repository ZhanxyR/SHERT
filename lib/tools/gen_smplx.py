from smplx import create
import torch
import trimesh
import os
from lib.utils.smplx_util import load_smplx_param
from lib.utils.mesh_util import save_obj
from lib.tools.subdivide_smplx import subdivide

def init_smplx_model(gender='neutral'):
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
        num_pca_comps=12,
        num_betas=10
    )
    return create(**model_init_params)


def gen_smplx(smplx_param_path, output_path):
    smplx_model = init_smplx_model()
    smplx_param = load_smplx_param(smplx_param_path)
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
    smplx_out = smplx_model(**model_forward_params)
    smplx_verts = (smplx_out.vertices[0] * smplx_param['scale'].detach() + smplx_param['translation']).detach().numpy()
    smplx_mesh = trimesh.Trimesh(smplx_verts, smplx_model.faces, process=False, maintain_order=True)
    smplx_mesh.export(output_path)


def gen_star_smplx(smplx_param_path, save_root, cfg_resources):
    smplx_model = init_smplx_model()
    smplx_param = load_smplx_param(smplx_param_path)
    for key in smplx_param.keys():
        smplx_param[key] = torch.as_tensor(smplx_param[key]).to(torch.float32)
    # TODO : only keep betas?
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
    model_forward_params['global_orient'] = torch.zeros_like(model_forward_params['global_orient']).reshape(1, -1)
    model_forward_params['body_pose'] = torch.zeros_like(smplx_param['body_pose']).reshape(1, -1)
    model_forward_params['body_pose'][0][2] = 0.5
    model_forward_params['body_pose'][0][5] = -0.5
    smplx_out = smplx_model(**model_forward_params)
    smplx_verts = (smplx_out.vertices[0] * smplx_param['scale'].detach() + smplx_param['translation']).detach().numpy()
    # smplx_mesh = trimesh.Trimesh(smplx_verts, smplx_model.faces, process=False, maintain_order=True)
    # smplx_mesh.export(output_path)
    os.makedirs(save_root, exist_ok=True)
    output_path = os.path.join(save_root, 'smplx_star.obj')
    save_obj(smplx_verts, smplx_model.faces + 1, output_path, single=True)
    verts, faces = subdivide(cfg_resources.models.smplx_eyes, output_path)
    save_obj(verts, faces, output_path, single=True)


    return output_path

if __name__=="__main__":
    gen_star_smplx("examples/demo_image_w_gt_smplx/smplx_param.pkl", "examples/demo_image_w_gt_smplx/smplx_star.obj")