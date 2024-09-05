import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
from lib.utils.mesh_util import load_obj, save_obj, save_mtl
from lib.utils.smplx_util import load_smplx_param
from lib.tools.subsmplx.lbs_transform import back_to_template_pose, repose

'''
    How to Animate Meshes Handled by SHERT
    1. org pose to standard template pose by `back_to_template_pose`
    2. standard template pose to specific pose by `repose`
'''
# org pose to specific pose
def parse_motion_param(motion_param_path):
    motion = np.load(motion_param_path,allow_pickle=True)
    # print(motion.shape)
    motion_parms = {
        'global_orient': motion[:, :3],  # controls the global root orientation
        'body_pose': motion[:, 3:3 + 63],  # controls the body
        'left_hand_pose': motion[:, 66:66 + 45],  # controls the finger articulation
        'right_hand_pose': motion[:, 66 + 45:66 + 90],  # controls the finger articulation
        'jaw_pose': motion[:, 66 + 90:66 + 93],  # controls the yaw pose
        'expression': motion[:, 159:159 + 50],  # controls the face expression
        'face_shape': motion[:, 209:209 + 100],  # controls the face shape
        'transl': motion[:, 309:309 + 3],  # controls the global body position
        'betas': motion[:, 312:],  # controls the body shape. Body shape is static
    }
    return motion_parms

if __name__ == '__main__':
    # 1. org pose to specific pose based on mesh from thuman
    src_mesh_path = "SHERT_RECON/examples/thuman/result_69/refine_s3_i1.obj"
    src_mesh_path = "SHERT_RECON/examples/thuman/result_280/refine_s3_i1.obj"
    src_mesh_path = "SHERT_RECON/examples/TEST/results/refine_s3_i1.obj"

    # src_smplx_param_path = "./examples/0280/smplx_param.pkl"
    # src_smplx_param_path = "/home/zxy/workspace/datasets/THuman2.0/THUman20_Release_Smpl-X_Paras/0069/smplx_param.pkl"
    src_smplx_param_path = "/home/zxy/workspace/datasets/THuman2.0/THUman20_Release_Smpl-X_Paras/0280/smplx_param.pkl"
    # target_smplx_param_path = "./examples/0513/smplx_param.pkl"
    skinning_weights_path= "./data/skinning_weights/lbs_weights_divide2.npy"

    src_verts, vts, faces = load_obj(src_mesh_path)
    src_smplx_param = load_smplx_param(src_smplx_param_path)
    # target_smplx_param = load_smplx_param(target_smplx_param_path)

    target_smplx_param = src_smplx_param.copy()


    # subject = 'D2_-_Wait_1_stageii'
    id = '0280'
    # subject = 'General_A5_-_Pick_Up_Box_stageii'
    subject = 'A3_-_Swing_stageii'
    # subject = 'walkdog_stageii'
    path = f'./examples/AMASS/{subject}.npz'
    # path = f'/home/zxy/workspace/datasets/AMASS/ACCADrenders/ACCAD/Male1General_c3d'

    bdata = np.load(path)
    # os.makedirs(f'./zxy_debug/animate/{subject}_0524', exist_ok=True)
    # save_path = f'./zxy_debug/animate/{subject}_0524'
    os.makedirs(f'./zxy_debug/animate/{subject}_{id}', exist_ok=True)
    save_path = f'./zxy_debug/animate/{subject}_{id}'

    # The subject of the mocap sequence is  neutral.
    # Body parameter vector shapes:
    # root_orient: torch.Size([1955, 3])
    # pose_body: torch.Size([1955, 63])
    # pose_hand: torch.Size([1955, 99])
    # trans: torch.Size([1955, 3])
    # betas: torch.Size([1955, 16])
    # time_length = 1955

    # 'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
    # 'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
    # 'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
    # 'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    # 'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    # # 'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics

    t_verts = back_to_template_pose(src_verts, src_smplx_param, skinning_weights_path, gender="male")

    save_obj(t_verts, faces, 'zxy_debug/animate/t_0280.obj', single=True)

    _, vts, vt_faces = load_obj('data/smplx/vt_example.obj')


    save_path_origin = save_path

    from tqdm import tqdm

    for i in tqdm(range(bdata['poses'].shape[0])):

        save_path = os.path.join(save_path_origin, f'{i}.obj')

        target_smplx_param['global_orient'] = bdata['poses'][i:i+1, :3]
        target_smplx_param['body_pose'] = bdata['poses'][i:i+1, 3:66]
        target_smplx_param['left_hand_pose'] = bdata['poses'][i:i+1, 66 + 9:66 + 9 + 45]
        target_smplx_param['right_hand_pose'] = bdata['poses'][i:i+1, 66 + 9 + 45:]
        target_smplx_param['tanslation'] = bdata['trans'][i]
        # target_smplx_param['tanslation'] = bdata['trans'][0]

        # betas (1, 10)
        # global_orient (1, 3)
        # body_pose (1, 63)
        # left_hand_pose (1, 12)
        # right_hand_pose (1, 12)
        # jaw_pose (1, 3)
        # leye_pose (1, 3)
        # reye_pose (1, 3)
        # expression (1, 10)
        # scale (1,)
        # translation (3,)

        target_smplx_param["betas"] = src_smplx_param["betas"]

        repose_verts = repose(t_verts.copy(), target_smplx_param.copy(), skinning_weights_path, gender="neutral", num_pca_comps=45, num_expression_coeffs=10)
        # repose_verts = repose(t_verts.copy(), target_smplx_param.copy(), skinning_weights_path, gender="male", num_pca_comps=45, num_expression_coeffs=10)

    #     def back_to_template_pose(verts, smplx_param, skinning_weight_path, gender="male", use_pca=True,num_pca_comps=12, num_betas=10):
    # verts  = transfer_to_std_smplx(verts, smplx_param["scale"], smplx_param["translation"])

        # save_obj(repose_verts, faces,save_path, single=True)
        save_mtl(repose_verts, vt_faces, vts, save_path, os.path.join(save_path_origin, 'texture.mtl'), 'texture.png')

        # 2. org pose to specific pose based on econ pred mesh
        # src_mesh_path = "./example"
        # src_smplx_param_path = ""
        # target_smplx_param_path = ""
        # skinning_weights_path = "./data/skinning_weights/lbs_weights_divide2.npy"
        # save_path = ""
        # src_verts, vts, faces = load_obj(src_mesh_path)
        # src_smplx_param = load_smplx_param(src_smplx_param_path)
        # target_smplx_param = load_smplx_param(target_smplx_param_path)
        # target_smplx_param["betas"] = src_smplx_param["betas"]
        #
        # t_verts = back_to_template_pose(src_verts, src_smplx_param, skinning_weights_path, gender="neutral", num_pca_comps=45, num_betas=200)
        # repose_verts = repose(t_verts, target_smplx_param, skinning_weights_path, num_expression_coeffs=50)
        #
        # save_obj(repose_verts, faces, save_path, single=True)

        # 3. animation sequences



