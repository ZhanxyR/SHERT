import numpy as np
import os
from glob import glob
from tqdm import tqdm
from lib.utils.mesh_util import load_obj, save_obj

def read_eyes_info(path):
    
    eyes_faces = []
    eyes_verts = []

    with open(path, "r") as f:
        for line in f.readlines():
            line_spilts = line.replace("\n", "").split(" ")
            if line_spilts[0] == "f":
                eyes_faces.append(sorted(line_spilts[1:4]))
                for v_index in line_spilts[1:4]:
                    eyes_verts.append(int(v_index))

    return eyes_faces, eyes_verts

def remove_eyes(template_path, mesh_path):

    eyes_faces, eyes_verts = read_eyes_info(template_path)
    
    verts = []
    faces = []

    with open(mesh_path, "r") as f:
        verts_index = 0
        for line in f.readlines():
            line_spilts = line.replace("\n", "").split(" ")
            if line_spilts[0] == "v":
                verts_index += 1
                if verts_index not in eyes_verts:
                    verts.append([float(v) for v in line_spilts[1:4]])
            elif line_spilts[0] == "f":
                v_index = sorted([line_spilts[i].split("/")[0] for i in range(1, 4)])
                # vt_index = [int(line_spilts[i].split("/")[1]) for i in range(1, 4)]
                if v_index not in eyes_faces:
                    # no need to change verts_index of faces, because last 1092 verts belongs to eyes
                    faces.append([int(v_i.split('/')[0]) for v_i in line_spilts[1:4]])  # not ordered v_index
                    # face_vts.append(vt_index)
                    
    return np.asarray(verts), np.asarray(faces)


def divide_midpoint(p, f, iteration=2):
    '''
    Arrange the new triangles in order after the old ones.
    '''

    for i in range(iteration):

        solved = []
        nps = []
        nfs = []
        nvts = []
        loc = {}
        loc_vt = {}

        newgrid = []

        for fi in f:

            np1 = (p[fi[0] - 1] + p[fi[1] - 1]) / 2
            np2 = (p[fi[0] - 1] + p[fi[2] - 1]) / 2
            np3 = (p[fi[1] - 1] + p[fi[2] - 1]) / 2

            lp1 = -1
            lp2 = -1
            lp3 = -1

            key1 = str(min(fi[0], fi[1])) + "," + str(max(fi[0], fi[1]))
            key2 = str(min(fi[0], fi[2])) + "," + str(max(fi[0], fi[2]))
            key3 = str(min(fi[1], fi[2])) + "," + str(max(fi[1], fi[2]))

            # 0-1
            if loc.get(key1) == None:
                solved.append([fi[0], fi[1]])
                nps.append(np1)
                lp1 = len(p) + len(nps)
                loc[key1] = lp1
            else:
                lp1 = loc[key1]

            # 0-2
            if loc.get(key2) == None:
                solved.append([fi[0], fi[2]])
                nps.append(np2)
                lp2 = len(p) + len(nps)
                loc[key2] = lp2
            else:
                lp2 = loc[key2]

            # 1-2
            if loc.get(key3) == None:
                solved.append([fi[1], fi[2]])
                nps.append(np3)
                lp3 = len(p) + len(nps)
                loc[key3] = lp3
            else:
                lp3 = loc[key3]

            nf1 = [fi[0], lp1, lp2]
            nf2 = [lp1, lp3, lp2]
            nf3 = [fi[1], lp3, lp1]
            nf4 = [fi[2], lp2, lp3]

            nfs.append(nf1)
            nfs.append(nf2)
            nfs.append(nf3)
            nfs.append(nf4)

        p = np.concatenate([p, nps], axis=0)
        f = nfs

    return p, f

def subdivide(eyes_template_path, mesh_path, save_root=None):

    verts, faces = remove_eyes(eyes_template_path, mesh_path)
    verts, faces = divide_midpoint(verts, faces)

    if save_root is not None:
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, 'smplx_d2.obj')
        save_obj(verts, faces, save_path, single=True)
        
        return verts, faces, save_path

    return verts, faces


