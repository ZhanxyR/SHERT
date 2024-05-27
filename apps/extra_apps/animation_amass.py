import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
from lib.utils.mesh_util import load_obj, save_obj
from lib.utils.smplx_util import load_smplx_param
from lib.tools.subsmplx.lbs_transform import back_to_template_pose, repose

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
    src_mesh_path = "./save/results/refine/0524/final_s3_l0.obj"
    src_mesh_path = "./examples/demo_image_w_gt_smplx/results/pred_org_inpaint.obj"

    src_smplx_param_path = "./examples/0524/smplx_param.pkl"
    target_smplx_param_path = "./examples/0513/smplx_param.pkl"
    skinning_weights_path= "./data/skinning_weights/lbs_weights_divide2.npy"
    save_path = "./zxy_debug/animate/0524_final_s3_l0_repose_to_0513_amass.obj"
    src_verts, vts, faces = load_obj(src_mesh_path)
    src_smplx_param = load_smplx_param(src_smplx_param_path)
    target_smplx_param = load_smplx_param(target_smplx_param_path)


    path = './examples/AMASS/Run1_stageii.npz'
    bdata = np.load(path)
    os.makedirs('./zxy_debug/animate/Run1_stageii_neutral', exist_ok=True)
    save_path = './zxy_debug/animate/Run1_stageii_neutral'

    t_verts = back_to_template_pose(src_verts, src_smplx_param, skinning_weights_path, gender="male")

    save_path_origin = save_path

    from tqdm import tqdm

    for i in tqdm(range(bdata['poses'].shape[0])):

        if i == 10:
            exit()

        save_path = os.path.join(save_path_origin, f'{i}.obj')

        target_smplx_param['global_orient'] = bdata['poses'][i:i+1, :3]
        target_smplx_param['body_pose'] = bdata['poses'][i:i+1, 3:66]
        target_smplx_param['left_hand_pose'] = bdata['poses'][i:i+1, 66 + 9:66 + 9 + 45]
        target_smplx_param['right_hand_pose'] = bdata['poses'][i:i+1, 66 + 9 + 45:]
        target_smplx_param['tanslation'] = bdata['trans'][i]
        target_smplx_param["betas"] = src_smplx_param["betas"]

        repose_verts = repose(t_verts.copy(), target_smplx_param.copy(), skinning_weights_path, gender="male", num_pca_comps=45, num_expression_coeffs=10)

        save_obj(repose_verts, faces,save_path, single=True)




