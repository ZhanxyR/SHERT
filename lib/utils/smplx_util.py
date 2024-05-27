import os
import numpy as np
import pickle

def load_smplx_param(smplx_param_path):
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

def transfer_to_std_smplx(vertices, scale, trans):
    vertices = np.asarray(vertices)
    vertices = vertices - trans
    vertices = vertices / scale
    return vertices

def back_to_org_smplx(vertices, scale, trans):
    vertices = np.asarray(vertices)
    vertices = vertices * scale
    vertices = vertices + trans
    return vertices

def inverse_rot(vertex, theta):
    theta = np.array((theta) * np.pi / 180.)
    rot_mat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]).astype(
        np.float32)
    return np.matmul(rot_mat, vertex.transpose(1, 0)).transpose(1, 0)


def back_to_econ_axis(vertices, param, view_id):
    vertices = np.asarray(vertices)
    vertices = vertices - param['center']
    vertices = vertices * param['scale'] / 100
    return inverse_rot(vertices, view_id)

def load_smplx_param(smplx_param_path):
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