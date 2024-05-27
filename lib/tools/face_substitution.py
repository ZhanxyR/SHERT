import numpy as np
import os
import cv2
import torch
import open3d as o3d
from lib.utils.config import  get_cfg_defaults
from lib.utils.mesh_util import load_obj, save_obj, get_rotate_matrix
from lib.utils.mesh_util import norm_to_0_1, backup, norm_a_b, norm_with_scale, norm_to_center, back_center, norm_with_center_param
from lib.utils.uv_sample.flame_uv_generator import FLAME_UV_Generator
from lib.utils.uv_sample.divided_uv_generator import Index_UV_Generator


def face_icp_refine(source, target, vertex_index):

    # vertex_index = np.load('data/face/face_vertex_index_refine.npy')
    # source, vts, faces = load_obj(src)
    # target,_,_ = load_obj(target)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source)[vertex_index])
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(np.asarray(target))

    reg_p2p = o3d.registration.registration_icp(source_pcd, target_pcd, 5.0)
    # print("reg_p2p.inlier_rmse {}".format(reg_p2p.inlier_rmse))

    source_pcd.transform(reg_p2p.transformation)
    # o3d.visualization.draw_geometries([source_pcd, target_pcd])
    source[vertex_index] = np.asarray(source_pcd.points)

    # icp_res_path = "output/face_icp.obj"
    # save_obj(source, faces, icp_res_path, single=True, vts=vts)
    # return icp_res_path

    return source
    # o3d.io.write_triangle_mesh("face_icp.obj", source)
    # print("shpe of src_verts {}".format(np.asarray(source_pcd.points).shape))

def face_clip_smooth(verts, faces, index, smooth=True, factor=10):

    # face_vertex_index = set(np.load('data/face/face_vertex_index_refine.npy') + 1)

    face_vertex_index = set(index + 1)

    faces_v = faces[:, :, 0].reshape(-1, 3)

    vertex_index = set(np.sort(np.unique(faces_v)))
    removed_face_vertex_index = face_vertex_index - vertex_index
    vertex_index = np.asarray(list(face_vertex_index - removed_face_vertex_index))
    vertex_index_list = list(vertex_index)
    vertex_index_map = {}
    for i in range(len(vertex_index_list)):
        vertex_index_map[vertex_index_list[i]] = i
    assert len(vertex_index_map.keys()) == len(vertex_index_list)
    v_sample = verts[vertex_index - 1]
    vertex_index_map = {} # key:old_index ,value:new_index
    faces_new = []
    for i in range(len(vertex_index)):
        vertex_index_map[vertex_index[i]] = i + 1
    for i in range(len(faces_v)):
        face = []
        for j in range(3):
            if vertex_index_map.get(faces_v[i][j]) is not None:
                face.append(vertex_index_map[faces_v[i][j]])
        if len(face) == 3:
            faces_new.append(face)
    # faces[:,:,0] = faces_v
    faces_new = np.asarray(faces_new)

    if smooth:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v_sample)
        mesh.triangles = o3d.utility.Vector3iVector(faces_new-1)
        mesh = mesh.filter_smooth_laplacian(factor)
        v_sample = np.asarray(mesh.vertices)

    return v_sample, faces_new

def face_rotate_align(verts, faces, smplx_verts, smplx_faces, face_vertex_index, interim_out=False):

    # _, _, rot_face = load_obj("data/face/norm_v_d2_clip_format.obj")
    rot_y, rot_x = get_rotate_matrix(verts[face_vertex_index], faces)
    rot_v_face_sample = np.matmul(rot_y, verts.transpose()).transpose()
    rot_v_face_sample = np.matmul(rot_x, rot_v_face_sample.transpose()).transpose()

    rot_y, rot_x = get_rotate_matrix(smplx_verts, smplx_faces)
    rot_smplx_verts = np.matmul(rot_y, smplx_verts.transpose()).transpose()
    rot_smplx_verts = np.matmul(rot_x, rot_smplx_verts.transpose()).transpose()
    inv_rot_y = np.linalg.inv(rot_y)
    inv_rot_x = np.linalg.inv(rot_x)

    _, norm_v_face_sample = norm_a_b(rot_v_face_sample[face_vertex_index], rot_v_face_sample)
    norm_v_face_sample_clip, center_1, x_1, y_1, z_1 = norm_to_center(rot_v_face_sample[face_vertex_index])
    norm_v_face_sample = norm_with_center_param(rot_v_face_sample, center_1 ,x_1, y_1, z_1)

    # save_obj(norm_v_face_sample, f_clip, 'zxy_debug/example_out/norm_v_face_sample_clip.obj', single=True)

    norm_smplx_verts, x_01, y_01, z_01, scale_01 = norm_to_0_1(rot_smplx_verts)
    norm_smplx_verts, center, x, y, z = norm_to_center(norm_smplx_verts)
    
    # save_obj(rot_v_d2, faces_d2_new, 'zxy_debug/example_out/v_d2.obj', single=True)
    # save_obj(norm_v_d2_clip, faces_d2_new, 'zxy_debug/example_out/norm_v_d2.obj', single=True)

    icp_verts = face_icp_refine(norm_v_face_sample, norm_smplx_verts, face_vertex_index)

    norm_v_face_sample[face_vertex_index] = icp_verts[face_vertex_index]
    
    back_v_face_sample = back_center(norm_v_face_sample, center, x, y, z)
    back_v_face_sample = backup(back_v_face_sample, x_01, y_01, z_01, scale_01)
    
    # save_obj(back_v_face_sample, f_clip, 'zxy_debug/example_out/backup_face_sample.obj', single=True)

    rot_back_v_face_sample = np.matmul(inv_rot_x, back_v_face_sample.transpose()).transpose()
    rot_back_v_face_sample = np.matmul(inv_rot_y, rot_back_v_face_sample.transpose()).transpose()

    return rot_back_v_face_sample


def face_bound_smooth(verts, faces, face_mask, device, inner_iter=15, outer_iter=5, smooth_iter=15, smooth_lamba=0.5):

    sampler = Index_UV_Generator(UV_height=1024, UV_width=-1, uv_type='SMPLX', data_dir='data/smplx').to(device)

    kernel = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]]).astype(np.uint8)
    face_mask_d5 = cv2.dilate(face_mask.astype(np.uint8), kernel, iterations=outer_iter)
    face_mask_d5 = face_mask_d5 * (1 - face_mask)
    face_mask_d5[700:800,250:350] = 0.
    # cv2.imwrite(os.path.join(outout_dir, "face_mask_d5.png"), face_mask_d5 * 255.)
    face_mask_d10 = cv2.dilate(face_mask_d5.astype(np.uint8), kernel, iterations=inner_iter)
    face_mask_d10 = face_mask_d10 * (face_mask_d5 + face_mask)
    # print(np.unique(face_mask_d10))
    # cv2.imwrite(os.path.join(outout_dir, "face_mask_d10.png"), face_mask_d10*255.)

    mask = np.repeat(face_mask_d10[:,:,None],3,axis=-1)
    mask_tensor = torch.from_numpy(mask).unsqueeze(dim=0).to(device=device).float()
    mask_tensor[mask_tensor != 0.] = 1.
    verts_max = sampler.resample(mask_tensor)[0].detach().cpu().numpy()
    mask_tensor[mask_tensor == 1.] = -1.
    verts_min = sampler.resample(mask_tensor)[0].detach().cpu().numpy()
    verts_diff = verts_max - verts_min
    verts_diff = verts_diff.sum(axis=-1)
    # print("where", len(np.where(verts_diff != 0)))
    face_bound_verts_index = sorted(list(np.where(verts_diff != 0)[0]))
    # print(face_bound_verts_index)
    face_bound_verts_map = {}
    for i in range(len(face_bound_verts_index)):
        face_bound_verts_map[face_bound_verts_index[i]] = i
    face_bound_faces = []
    for i in range(faces.shape[0]):
        if face_bound_verts_map.get(faces[i][0]) is not None and face_bound_verts_map.get(
                faces[i][1]) is not None and face_bound_verts_map.get(faces[i][2]) is not None:
            face_bound_faces.append([face_bound_verts_map.get(faces[i][0]),face_bound_verts_map.get(faces[i][1]),face_bound_verts_map.get(faces[i][2])])

    face_bound_verts_index = np.asarray(face_bound_verts_index)
    face_bound_verts = verts[face_bound_verts_index]
    face_bound_faces = np.asarray(face_bound_faces)
    bound_mesh = o3d.geometry.TriangleMesh()
    bound_mesh.vertices = o3d.utility.Vector3dVector(face_bound_verts)
    bound_mesh.triangles = o3d.utility.Vector3iVector(face_bound_faces)
    # o3d.io.write_triangle_mesh(os.path.join(outout_dir, "face_bound.obj"), bound_mesh)

    smoothed_mesh = bound_mesh.filter_smooth_laplacian(smooth_iter, smooth_lamba)
    smoothed_bound_verts = np.asarray(smoothed_mesh.vertices)

    import math
    for org_vertex, mapped_vertex in face_bound_verts_map.items():
        # if face_template_verts_map.get(org_vertex) is not None:
        if math.isnan(smoothed_bound_verts[mapped_vertex][0]) or math.isnan(smoothed_bound_verts[mapped_vertex][1]) or math.isnan(smoothed_bound_verts[mapped_vertex][2]):
            continue
        verts[org_vertex] = smoothed_bound_verts[mapped_vertex]
    
    return face_mask_d10, verts


# if __name__=='__main__':

#     cfg_resources = get_cfg_defaults("lib/configs/resources.yaml")


#     # face_vertex_index = np.load('data/face/face_vertex_index_refine.npy')
#     face_vertex_index = np.load(cfg_resources.index.face_verts_index)

#     # SMPLX-d2
#     # v_sample, vts, faces = load_obj("zxy_debug/example/0524/0524_econ_comp.obj")
#     v_sample, vts, faces = load_obj("zxy_debug/example/0524/divided_2.obj")
#     v_d2, faces_d2_new = face_clip_smooth(v_sample, faces, face_vertex_index, smooth=True)
#     # save_obj(v_d2.copy(), faces_d2_new.copy(), "zxy_debug/example_out/smplx_face_format_econ_smooth_my.obj", single=True)

#     # FLAME-detailed
#     pred_face_path = "zxy_debug/example/0524/smplx_resample.obj"
#     v_face_sample, _, _ = load_obj(pred_face_path)

#     # SMPLX-d2-detailed
#     optimize_obj_path = "zxy_debug/example/0524/0524_comp_scan.obj"
#     v_opt, _, f_opt = load_obj(optimize_obj_path)

#     # align
#     # _, _, rot_face = load_obj("data/face/norm_v_d2_clip_format.obj")
#     _, _, rot_face = load_obj(cfg_resources.models.face_clip_norm)
#     verts = face_rotate_align(v_face_sample, rot_face - 1, v_d2, faces_d2_new - 1, face_vertex_index)
#     v_opt[face_vertex_index] = verts[face_vertex_index]
#     save_obj(v_opt, faces, 'zxy_debug/example_out/cat_face.obj', vts=vts)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     example_path = "zxy_debug/example_out"

#     outout_dir = example_path

#     # face_mask = cv2.imread('data/face_uv_refine.png')
#     face_mask = cv2.imread(cfg_resources.masks.face_uv_refine, cv2.IMREAD_GRAYSCALE) / 255

#     face_mask_d10, verts = face_bound_smooth(v_opt, faces[:,:,0] - 1, face_mask, device)

#     cv2.imwrite(os.path.join(outout_dir, "face_mask_d10.png"), face_mask_d10*255.)

#     mesh = o3d.io.read_triangle_mesh(os.path.join(example_path, "cat_face.obj"))
#     res_mesh = o3d.geometry.TriangleMesh()
#     res_mesh.vertices = o3d.utility.Vector3dVector(verts)
#     res_mesh.triangles = mesh.triangles
#     o3d.io.write_triangle_mesh(os.path.join(outout_dir, "smooth.obj"), res_mesh)





    

    
        