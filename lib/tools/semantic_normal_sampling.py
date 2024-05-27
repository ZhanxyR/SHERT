import os
import pickle
import sys
from tqdm import tqdm
import cv2
import numpy as np
import torch
import time
import open3d as o3d
import trimesh
import random
from lib.utils.mesh_util import save_obj_o3d, projection_length
from lib.utils.uv_sample.divided_uv_generator import Index_UV_Generator
from lib.tools.subsmplx.body_models import create
# from lib.tools.sns.smplx_t import create
from lib.utils.common_util import check_key
from lib.utils.image_util import write_pic
from lib.utils.config import  get_cfg_defaults


class Semantic_Normal_Sampling():
    
    def __init__(self, cfg_resources, uv_sampler, device, cfg_parameters=None):

        self.device = device
        self.sampler = uv_sampler

        parameters = self.update_parameters(cfg_resources.configs.default_sns, cfg_parameters)

        self.use_thuman_smplx_param = parameters.use_thuman_smplx_param 
        self.smplx_gender = parameters.smplx_gender
        self.t_max_sample_dis = parameters.t_max_sample_dis
        # thresholds for origin pose
        self.t_angle = parameters.t_angle
        self.t_area = parameters.t_area
        self.t_edge = parameters.t_edge
        # thresholds for star pose
        self.t_s_angle = parameters.t_s_angle
        self.t_s_area = parameters.t_s_area
        self.t_s_edge = parameters.t_s_edge
        self.t_s_connect = parameters.t_s_connect

        # resources
        self.skinning_weight_path = cfg_resources.others.smplx_div2_skinning_weights
        self.smplx_template_root = cfg_resources.models.smplx_template_root
        self.smplx_star_template_mesh_path = cfg_resources.models.smplx_div2_star_template
        self.full_mask = cv2.imread(cfg_resources.masks.mask_wo_eyes, cv2.IMREAD_GRAYSCALE) / 255

        # unnessesary
        self.hands_feet_verts_index = []
        self.face_verts_index = []

    def update_parameters(self, default_path, cfg):
        default_parameters = get_cfg_defaults(default_path)

        if check_key(cfg, ['parameters', 'sns']):
            for key in cfg.parameters.sns.keys():
                    default_parameters[key] = cfg.parameters.sns[key]
         
        return default_parameters

    def load_smplx_param(self, smplx_param_path):
        _, ext = os.path.splitext(smplx_param_path)
        ext = ext.lower()
        if ext == ".npy":
            smplx_param = np.load(smplx_param_path, allow_pickle=True).item()
        elif ext == ".pkl":
            smplx_param = pickle.load(open(smplx_param_path, "rb"))
        else:
            smplx_param = None
            print("smplx param ext {}".format(ext))
            raise Exception("smplx_param_path extension does not match!")
        return smplx_param

    def compute_triangle_angle(self, normal_a, normal_b):
        dot_product = np.einsum('ij,ij->i', normal_a, normal_b)
        length1 = np.linalg.norm(normal_a, axis=1)
        length2 = np.linalg.norm(normal_b, axis=1)

        cos = np.clip(dot_product / (length1 * length2), -1, 1)
        angle = np.arccos(cos)

        return angle

    def compute_triangle_area(self, vertices):
        a = vertices[:, 0, :]
        b = vertices[:, 1, :]
        c = vertices[:, 2, :]
        ab = b - a
        ac = c - a
        cross = np.cross(ab, ac)
        area = np.linalg.norm(cross, axis=1) / 2

        return area

    def compute_edge_ratio(self, vertices):
        edge_ratio = np.linalg.norm(vertices[:, [0, 0, 1]] - vertices[:, [1, 2, 2]], axis=2)
        edge_ratio = edge_ratio[:, [0, 0, 1, 1, 2, 2]] / edge_ratio[:, [1, 2, 0, 2, 0, 1]]

        return edge_ratio

    def culling(self, normals, smplx_normals, vertices, triangles, smplx_vertices, smplx_triangles, t_angle=2, t_area=3, t_edge=3):

        index = []
        
        # Triangle Angle
        if t_angle > 0:
            angles = self.compute_triangle_angle(smplx_normals, normals)

            for i in range(triangles.shape[0]):
                if angles[i] > t_angle:
                    index.append(i)

        # Area Ratio
        if t_area > 0:
            smplx_area = self.compute_triangle_area(smplx_vertices[smplx_triangles])
            mesh_area = self.compute_triangle_area(vertices[triangles])
            area_ratio = mesh_area / smplx_area

            for i in range(smplx_area.shape[0]):
                if area_ratio[i] > t_area:
                    index.append(i)

        # Edge Ratio
        if t_edge > 0:
            edge_ratio = self.compute_edge_ratio(vertices[triangles])

            for i in range(triangles.shape[0]):
                if edge_ratio[i].max() > t_edge:
                    index.append(i)

        deleted_triangles = np.delete(triangles.copy(), index, axis=0)

        return deleted_triangles

    def connectivity_detection(self, mesh, level=500):
        clusters, number, area = mesh.cluster_connected_triangles()
        number = np.asarray(number)

        index = []
        for i in range(np.asarray(clusters).shape[0]):
            if number[clusters[i]] < level:
                index.append(i)

        index = sorted(list(set(index)))
        triangles = np.asarray(mesh.triangles)
        deleted_triangles = np.delete(triangles, index, 0)

        return deleted_triangles

    def transfer_to_star_thuman(self, smplx_param, vertices):
        # init smplx model
        model_init_params = dict(
            gender=self.smplx_gender,
            model_type='smplx',
            model_path=self.smplx_template_root,
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
        smplx_model = create(**model_init_params)
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

        # affine to std smplx
        smplx_param["translation"] = smplx_param["translation"].numpy()
        smplx_param["scale"] = smplx_param["scale"].numpy()
        src_verts = vertices - smplx_param["translation"]
        src_verts = src_verts / smplx_param["scale"]
        model_forward_params["src_verts"] = src_verts
        model_forward_params["skinning_weight_path"] = self.skinning_weight_path


        # transfer to T pose
        smpl_out, _, T_org_pose = smplx_model(**model_forward_params)

        # T pose to star pose
        model_forward_params['global_orient'] = torch.zeros_like(model_forward_params['global_orient']).reshape(1, -1)
        model_forward_params['body_pose'] = torch.zeros_like(smplx_param['body_pose']).reshape(1, -1)
        model_forward_params['body_pose'][0][2] = 0.5
        model_forward_params['body_pose'][0][5] = -0.5
        t_vertices = smpl_out.vertices[0].detach().cpu().numpy()
        model_forward_params["toT"] = False
        model_forward_params["src_verts"] = t_vertices
        model_forward_params["skinning_weight_path"] = self.skinning_weight_path

        smpl_out, _, T_star = smplx_model(**model_forward_params)
        star_vertices = smpl_out.vertices[0].detach().cpu().numpy()
        return star_vertices, T_star, T_org_pose, smplx_model

    def transfer_to_star_econ(self, smplx_param, vertices):
        # init smplx model
        model_init_params = dict(
            gender=self.smplx_gender,
            model_type='smplx',
            model_path=self.smplx_template_root,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_expression=True,
            create_jaw_pose=False,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=False,
            num_pca_comps=45,
            num_betas=200
        )
        smplx_model = create(**model_init_params)

        for key in smplx_param.keys():
            smplx_param[key] = torch.as_tensor(smplx_param[key].reshape(1, -1)).to(torch.float32)

        model_forward_params = dict(betas=smplx_param['betas'],
                                    global_orient=smplx_param['global_orient'],
                                    body_pose=smplx_param['body_pose'],
                                    left_hand_pose=smplx_param['left_hand_pose'],
                                    right_hand_pose=smplx_param['right_hand_pose'],
                                    jaw_pose=smplx_param['jaw_pose'],
                                    return_verts=True)

        # Affine to std smplx
        smplx_param["transl"] = smplx_param["transl"].numpy()[0]
        smplx_param["scale"] = smplx_param["scale"].numpy()[0]
        src_verts = vertices * np.asarray([1.0, -1.0, -1.0], dtype=vertices.dtype)
        src_verts = src_verts / smplx_param["scale"] - smplx_param["transl"]
        model_forward_params["src_verts"] = src_verts
        model_forward_params["skinning_weight_path"] = self.skinning_weight_path


        # Transfer to T pose
        smplx_out, _, T_org_pose = smplx_model(**model_forward_params)

        # T pose to star pose
        model_forward_params['global_orient'] = torch.zeros_like(model_forward_params['global_orient']).reshape(1, -1)
        model_forward_params['body_pose'] = torch.zeros_like(smplx_param['body_pose']).reshape(1,-1)
        model_forward_params['body_pose'][0][2] = 0.5
        model_forward_params['body_pose'][0][5] = -0.5
        t_vertices = smplx_out.vertices[0].detach().cpu().numpy()
        model_forward_params["toT"] = False
        model_forward_params["src_verts"] = t_vertices
        model_forward_params["skinning_weight_path"] = self.skinning_weight_path

        smplx_out, _, T_star = smplx_model(**model_forward_params)
        star_vertices = smplx_out.vertices[0].detach().cpu().numpy()

        return star_vertices, T_star, T_org_pose, smplx_model

    def transfer_back_econ(self, mesh_star, T_star, T_org_pose, smplx_model, smplx_param):
        src_star_verts = np.asarray(mesh_star.vertices)
        src_star_verts = torch.from_numpy(src_star_verts).to(device=self.device, dtype=T_star.dtype).unsqueeze(dim=0).expand(
            [T_star.shape[0], -1, -1])

        # star pose  to T pose
        posed_verts = smplx_model.back_from_t(src_star_verts, torch.linalg.inv(T_star))

        # T pose to org pose
        posed_verts = smplx_model.back_from_t(posed_verts, T_org_pose)[0].detach().cpu().numpy()

        # affine to org scale
        posed_verts = ((posed_verts + smplx_param["transl"]) * smplx_param["scale"]) * np.asarray([1, -1, -1], dtype=np.float64)
        
        return posed_verts


    def transfer_back_thuman(self, mesh_star, T_star, T_org_pose, smplx_model, smplx_param):
        src_star_verts = np.asarray(mesh_star.vertices)
        src_star_verts = torch.from_numpy(src_star_verts).to(device=self.device, dtype=T_star.dtype).unsqueeze(
            dim=0).expand([T_star.shape[0], -1, -1])

        # star pose  to T pose
        posed_verts = smplx_model.back_from_t(src_star_verts, torch.linalg.inv(T_star))

        # T pose to org pose
        posed_verts = smplx_model.back_from_t(posed_verts, T_org_pose)[0].detach().cpu().numpy()

        # affine to org scale
        posed_verts = posed_verts * smplx_param["scale"] + smplx_param["translation"]

        return posed_verts

    def get_error_vertices(self, optimize_faces, smplx_faces):
        candidate_verts_index = np.unique(np.asarray(smplx_faces).flatten())
        remain_verts_index = np.unique(np.asarray(optimize_faces).flatten())
        error_verts_index = np.asarray(list(set(candidate_verts_index) - set(remain_verts_index) - set(self.face_verts_index) - set(self.hands_feet_verts_index)), dtype=int)
        return error_verts_index

    def generate_error_mask(self, sampler, optimize_vertices, smplx_verts, error_vertices_index):
        smplx_verts_t = torch.from_numpy(smplx_verts).unsqueeze(dim=0).float().to(device=self.device)
        smplx_uv = sampler.get_UV_map(smplx_verts_t)[0].detach().cpu().numpy()

        optimize_vertices_t = torch.from_numpy(optimize_vertices).unsqueeze(dim=0).float().to(device=self.device)
        uv_map_org = sampler.get_UV_map(optimize_vertices_t)[0].detach().cpu().numpy()

        # get error uv mask
        optimize_vertices[error_vertices_index] = -1.
        optimize_vertices_t = torch.from_numpy(optimize_vertices).unsqueeze(0).float().to(device=self.device)
        uv_map_min = sampler.get_UV_map(optimize_vertices_t)[0].detach().cpu().numpy()

        optimize_vertices[error_vertices_index] = 1.
        optimize_vertices_t = torch.from_numpy(optimize_vertices).unsqueeze(0).float().to(device=self.device)
        uv_map_max = sampler.get_UV_map(optimize_vertices_t)[0].detach().cpu().numpy()

        error_uv_mask = uv_map_max - uv_map_min
        error_uv_mask[error_uv_mask != 0] = 1
        error_uv_mask[self.full_mask == 0] = 0.

        uv_map_org[self.full_mask == 0] = 0.
        uv_map_org[error_uv_mask == 1] = 0.
  
        return uv_map_org, error_uv_mask

    def normal_based_sampling(self, target_mesh, source_mesh, threshold=0.1, cal_contain=True):

        source_mesh.compute_vertex_normals()
        src_verts = np.asarray(source_mesh.vertices).copy()
        src_normals = np.asarray(source_mesh.vertex_normals).copy()

        normals = np.asarray(target_mesh.vertex_normals)

        # adjust normal orientation
        if cal_contain:
            inside = target_mesh.contains(src_verts).astype(int) + 1
            inside = (-1) ** inside
            oriented_normals = inside[:, None] * src_normals
        else:
            oriented_normals = src_normals

        pbar = tqdm(range(src_verts.shape[0]))
        pbar.set_description('Sampling')
        for i in pbar:
            intersection = target_mesh.ray.intersects_id([src_verts[i]], [oriented_normals[i]], multiple_hits=False,
                                                    return_locations=True)
            # if len(intersection[2]) <= 0  or (np.linalg.norm(intersection[2][0] - vertices[i]) >= threshold):
            if len(intersection[2]) <= 0:
                intersection = target_mesh.ray.intersects_id([src_verts[i]], [oriented_normals[i] * np.array([-1, -1, -1])],
                                                        multiple_hits=False, return_locations=True)
            if len(intersection[2] > 0):
                if np.linalg.norm(intersection[2][0] - src_verts[i]) < threshold:
                    src_verts[i] = intersection[2][0]
                else:
                    src_verts[i] = src_verts[i] + 0.5 + random.random()
            else:
                src_verts[i] = src_verts[i] + 0.5 + random.random()

        result = o3d.geometry.TriangleMesh()
        result.vertices = o3d.utility.Vector3dVector(src_verts)
        result.triangles = source_mesh.triangles

        source_mesh.compute_vertex_normals()

        return result

    def mesh_culling_with_star_pose(self, mesh, smplx_mesh, smplx_param, smplx_star_template_mesh):
        # Mesh culling based on org pose
        mesh.compute_triangle_normals()
        smplx_mesh.compute_triangle_normals()
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)
        smplx_vertices = np.asarray(smplx_mesh.vertices)
        smplx_triangles = np.asarray(smplx_mesh.triangles)
        smplx_normals = np.asarray(smplx_mesh.triangle_normals)

        deleted_triangles = self.culling(normals, smplx_normals, vertices, triangles, smplx_vertices, smplx_triangles, self.t_angle, self.t_area, self.t_edge)

        # Transfer to star pose
        if self.use_thuman_smplx_param:
            star_vertices, T_star, T_org_pose, smplx_model = self.transfer_to_star_thuman(smplx_param, vertices)
        else:
            star_vertices, T_star, T_org_pose, smplx_model = self.transfer_to_star_econ(smplx_param, vertices)

        # Mesh culling based on star pose
        mesh_star = o3d.geometry.TriangleMesh()
        mesh_star.vertices = o3d.utility.Vector3dVector(star_vertices)
        mesh_star.triangles = o3d.utility.Vector3iVector(deleted_triangles)
        mesh_star.compute_triangle_normals()
        # o3d.io.write_triangle_mesh('zxy_debug/sns/mesh_star.obj', mesh_star)

        smplx_star_template_mesh.triangles = mesh_star.triangles
        smplx_star_template_mesh.compute_triangle_normals()
        # o3d.io.write_triangle_mesh('zxy_debug/sns/smplx_star.obj', smplx_star_template_mesh)

        vertices_star = np.asarray(mesh_star.vertices)
        triangles_star = np.asarray(mesh_star.triangles)
        normals_star = np.asarray(mesh_star.triangle_normals)
        smplx_vertices_star = np.asarray(smplx_star_template_mesh.vertices)
        smplx_triangles_star = np.asarray(smplx_star_template_mesh.triangles)
        smplx_normals_star = np.asarray(smplx_star_template_mesh.triangle_normals)

        deleted_triangles = self.culling(normals_star, smplx_normals_star, vertices_star, triangles_star, smplx_vertices_star, smplx_triangles_star, self.t_s_angle, self.t_s_area, self.t_s_edge)

        # Connectivity detection
        mesh_star.triangles = o3d.utility.Vector3iVector(deleted_triangles)
        deleted_triangles = self.connectivity_detection(mesh_star, self.t_s_connect)

        # Transfer back to org pose
        mesh_processed = mesh_star
        mesh_processed.triangles = o3d.utility.Vector3iVector(deleted_triangles)

        if self.use_thuman_smplx_param:
            posed_verts = self.transfer_back_thuman(mesh_processed, T_star, T_org_pose, smplx_model, smplx_param)
        else:
            posed_verts = self.transfer_back_econ(mesh_processed, T_star, T_org_pose, smplx_model, smplx_param)

        mesh_processed.vertices = o3d.utility.Vector3dVector(posed_verts)

        # Get error vertex index
        error_verts_index = self.get_error_vertices(mesh_processed.triangles, smplx_triangles)
        # print("error verts num {}".format(error_verts_index.shape[0]))

        # Generate error mask
        # error_uv_mask = self.generate_error_mask(posed_verts, smplx_vertices, error_vertex_index, device)
        uv, error_uv_mask = self.generate_error_mask(self.sampler, posed_verts, smplx_vertices, error_verts_index)
        
        return mesh_processed, error_uv_mask, error_verts_index

    def mesh_culling_only_org_pose(self, mesh, smplx_mesh):
        mesh.compute_triangle_normals()
        smplx_mesh.compute_triangle_normals()

        # Mesh culling based on org pose
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)
        smplx_vertices = np.asarray(smplx_mesh.vertices)
        smplx_triangles = np.asarray(smplx_mesh.triangles)
        smplx_normals = np.asarray(smplx_mesh.triangle_normals)

        deleted_triangles = self.culling(normals, smplx_normals, vertices, triangles, smplx_vertices, smplx_triangles)
        mesh.triangles = o3d.utility.Vector3iVector(deleted_triangles)

        # Connectivity detection
        deleted_triangles = self.connectivity_detection(mesh, self.t_s_connect)
        mesh.triangles = o3d.utility.Vector3iVector(deleted_triangles)

        # Get error vertex index
        error_verts_index = self.get_error_vertices(mesh.triangles, smplx_triangles)

        # Generate error mask
        uv, error_uv_mask = self.generate_error_mask(self.sampler, posed_verts, smplx_vertices, error_verts_index)

        return mesh, error_uv_mask, error_verts_index

    def sample(self, target_path, smplx_path, smplx_param_path, save_root=None, save_name='sns', return_data=False):

        target_mesh = trimesh.load(target_path)
        smplx_mesh = o3d.io.read_triangle_mesh(smplx_path)
        smplx_star_template_mesh = o3d.io.read_triangle_mesh(self.smplx_star_template_mesh_path)

        smplx_param = self.load_smplx_param(smplx_param_path)
        sampled_mesh = self.normal_based_sampling(target_mesh, smplx_mesh, self.t_max_sample_dis)

        # smplx_mesh = smplx_mesh.filter_smooth_laplacian(1, 5)

        mesh_processed, error_uv_mask, error_verts_index = self.mesh_culling_with_star_pose(sampled_mesh, smplx_mesh, smplx_param, smplx_star_template_mesh)

        if save_root is not None:
            os.makedirs(save_root, exist_ok=True)

            mesh_path = os.path.join(save_root, f'{save_name}_sample.obj')
            error_uv_path = os.path.join(save_root, f'{save_name}_error_uv_mask.png')
            error_index_path = os.path.join(save_root, f"{save_name}_error_vers_index.npy")

            save_obj_o3d(mesh_path, mesh_processed)
            write_pic(error_uv_path, error_uv_mask, type=2)
            np.save(error_index_path, error_verts_index)
            # TODO: save parameters

            if return_data:
                return mesh_path, error_uv_path, error_index_path, mesh_processed, error_uv_mask, error_verts_index
            else:
                return mesh_path, error_uv_path, error_index_path

        return mesh_processed, error_uv_mask, error_verts_index



# if __name__ == "__main__":

#     cfg_resources = get_cfg_defaults("lib/configs/resources.yaml")
#     # for single example
#     input_dir = "./examples/demo_image_w_gt_smplx"
#     input_dir = "./examples/demo_image"

#     output_dir = os.path.join(input_dir, 'results')

#     # use_thuman_smplx_param = True
#     use_thuman_smplx_param = False

#     mesh_path = os.path.join(input_dir, 'target.obj') #example/demo_input/demo_input_0_full.obj
#     smplx_path = os.path.join(input_dir, 'smplx_d2.obj')
#     smplx_param_path = os.path.join(input_dir, 'smplx_param.npy')

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)

#     # mesh_path = os.path.join(input_dir, '0524.obj') #example/demo_input/demo_input_0_full.obj
#     # smplx_path = os.path.join(input_dir, 'smplx_div2_org.obj')
#     # smplx_param_path = os.path.join(input_dir, 'smplx_param.pkl')

#     smplx_star_path = cfg_resources.models.smplx_div2_star_template
#     # smplx_star_path = "input/star_template_divide2.obj" 
#     # output_dir = input_dir
#     gpu_id = 0
#     device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    

#     target_mesh = trimesh.load(target_path)
#     smplx_mesh = o3d.io.read_triangle_mesh(smplx_path)
#     # smplx_mesh = smplx_mesh.filter_smooth_laplacian(1, 5)

#     smplx_star_template_mesh = o3d.io.read_triangle_mesh(smplx_star_path)

#     smplx_param = load_smplx_param(smplx_param_path)

#     start = time.time()
#     # normal sample
#     sampled_mesh = normal_based_sampling(target_mesh, smplx_mesh)
#     # o3d.io.write_triangle_mesh('zxy_debug/sns/mesh_sample.obj', sampled_mesh)

#     print('Culling')
#     # mesh culling
#     mesh_processed, error_uv_mask, error_vertex_index = mesh_culling_with_star_pose(sampled_mesh, smplx_mesh, smplx_param, smplx_star_template_mesh, device, use_thuman_smplx_param)
#     # mesh_processed, error_uv_mask, error_vertex_index = mesh_culling_only_org_pose(sampled_mesh, smplx_mesh, gpu_id)

#     # o3d.io.write_triangle_mesh(os.path.join(output_dir, 'sns.obj'), mesh_processed)
#     save_obj_o3d(os.path.join(output_dir, 'sns.obj'), mesh_processed)

#     write_pic(os.path.join(output_dir, 'error_uv_mask.png'), error_uv_mask, type=2)
#     np.save(os.path.join(output_dir, "error_vertex_index.npy"), error_vertex_index)

#     print(f'time consuming {time.time() - start} s')
    
    # for batch
    # mesh_dir = "/home/zxy/workspace/datasets/THuman2.0/THuman_standard_render/"
    # smplx_dir = "/home/zxy/workspace/datasets/THuman2.0/THUman20_Release_Smpl-X_Paras"
    # output_dir = "zxy_debug/sns"
    # subjects = [subject for subject in os.listdir(smplx_dir) if os.path.isdir(os.path.join(smplx_dir, subject))]
    # print("total subjects num : {}".format(len(subjects)))

    # gpu_id = 0
    # device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    # use_thuman_smplx_param = True

    # start = time.time()
    # for subject in tqdm(sorted(subjects)[500:]):
    #     if not os.path.exists(os.path.join(output_dir, subject)):
    #         os.makedirs(os.path.join(output_dir, subject), exist_ok=True)

    #     smplx_path = os.path.join(smplx_dir, subject, "divided_2.obj")  # smplx_div2_org.obj
    #     smplx_mesh = o3d.io.read_triangle_mesh(smplx_path)
    #     smplx_mesh = smplx_mesh.filter_smooth_laplacian(1, 5)

    #     # smplx_star_path = os.path.join(input_dir, 'smplx_div2_star.obj')
    #     # smplx_star_path = "data/smplx/star_template_divide2.obj"
    #     smplx_star_path = cfg_resources.models.smplx_div2_star_template

    #     smplx_star_template_mesh = o3d.io.read_triangle_mesh(smplx_star_path)

    #     smplx_param_path = os.path.join(smplx_dir, subject, "smplx_param.pkl")
    #     smplx_param = load_smplx_param(smplx_param_path)

    #     mesh_path = os.path.join(mesh_dir, subject, "GEO/OBJ", subject, "{}.obj".format(subject))  # 0000/GEO/OBJ/0000
    #     # normal sample
    #     sampled_mesh = semantic_normal_sample(mesh_path, smplx_path)

    #     # mesh culling
    #     mesh_processed, error_uv_mask, error_vertex_index = mesh_culling_with_star_pose(sampled_mesh, smplx_mesh,
    #                                                                                     smplx_param,
    #                                                                                     smplx_star_template_mesh, device,
    #                                                                                     use_thuman_smplx_param)
    #     # mesh_processed, error_uv_mask, error_vertex_index = mesh_culling_only_org_pose(sampled_mesh, smplx_mesh, gpu_id)

    #     o3d.io.write_triangle_mesh(os.path.join(output_dir, subject, 'sns.obj'), mesh_processed)
    #     write_pic(os.path.join(output_dir, subject, 'error_uv_mask.png'), error_uv_mask, type=2)
    #     np.save(os.path.join(output_dir, subject, "error_vertex_index.npy"), error_vertex_index)

    #     exit()

    # print(f'time consuming {time.time() - start} s')

