import  numpy as np
import open3d as o3d

def load_obj(file):
    verts = []
    vts = []
    faces = []
    with open(file) as f:
        while True:
            line = f.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                verts.append((float(strs[1]), float(strs[2]), float(strs[3])))
            elif strs[0] == "vt":
                vts.append((float(strs[1]), float(strs[2])))
            elif strs[0] == 'f':
                faces.append([[int(s) for s in strs[1].split("/")], [int(s) for s in strs[2].split("/")],
                              [int(s) for s in strs[3].split("/")]])
            else:
                continue
    return np.array(verts, dtype=float), np.array(vts, dtype=float), np.array(faces, dtype=int)


def save_obj(verts, faces, path_out, single=False, vts=None, colors=None):
    with open(path_out, 'w') as fp:
        if colors is not None:
            for i in range(len(verts)):
                vi_np = np.array(verts[i])
                color_np = np.array(colors[i])
                # fp.write('v %f %f %f\n' % (vi_np[0], vi_np[1], vi_np[2]))
                fp.write('v %f %f %f %f %f %f\n' % (vi_np[0], vi_np[1], vi_np[2], color_np[0], color_np[1], color_np[2]))
        else:
            for vi in verts:
                vi_np = np.array(vi)
                fp.write('v %f %f %f\n' % (vi_np[0], vi_np[1], vi_np[2]))

        if vts is not None:
            for vt in vts:
                vt_np = np.array(vt)
                fp.write("vt %f %f\n" % (vt_np[0], vt_np[1]))

        for fi in faces:
            ft = np.array(fi)
            if not single:
                fp.write('f %d/%d %d/%d %d/%d\n' % (ft[0][0], ft[0][1], ft[1][0], ft[1][1], ft[2][0], ft[2][1]))
            else:
                if len(ft.shape) == 2:
                    ft = ft[..., 0]
                fp.write('f %d %d %d\n' % (ft[0], ft[1], ft[2]))

def save_obj_o3d(path_out, mesh):
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles) + 1
    with open(path_out, 'w') as fp:
        for vi in verts:
            fp.write('v %f %f %f\n' % (vi[0], vi[1], vi[2]))
        for fi in faces:
            fp.write('f %d %d %d\n' % (fi[0], fi[1], fi[2]))


def projection_length(a, b):
    dot_product = (a * b).sum(axis=1).reshape(-1, 1)
    b_norm = np.sqrt(np.power(b, 2).sum(axis=1)).reshape(-1, 1)

    return dot_product / b_norm

def save_mtl(p, faces, vts, path_out):
    with open(path_out, 'w') as fp:
        fp.write('mtllib deca.mtl\n')
        fp.write('usemtl FaceTexture\n')
        for vi in p:
            vi_np = np.array(vi)
            fp.write('v %f %f %f\n' % (vi_np[0], vi_np[1], vi_np[2]))
        for vt in vts:
            vt_np = np.array(vt)
            fp.write("vt %f %f\n" % (vt_np[0], vt_np[1]))
        for fi in faces:
            ft = np.array(fi)
            fp.write('f %d/%d %d/%d %d/%d\n' % (ft[0][0], ft[0][1], ft[1][0], ft[1][1], ft[2][0], ft[2][1]))



# center norm
def norm_point(points, align=True):

    x = (points[np.argmin(points[:, 0])][0] + points[np.argmax(points[:, 0])][0])/2
    y = (points[np.argmin(points[:, 1])][1] + points[np.argmax(points[:, 1])][1])/2
    z = (points[np.argmin(points[:, 2])][2] + points[np.argmax(points[:, 2])][2])/2

    # print(points[np.argmin(points[:, 0])][0], points[np.argmax(points[:, 0])][0])

    if align:
        points[:, 0] = points[:, 0] - x
        points[:, 1] = points[:, 1] - y
        points[:, 2] = points[:, 2] - z

    # return points

    return points/abs(points.ravel()[np.argmax(abs(points.ravel()))]), x, y, z, abs(points.ravel()[np.argmax(abs(points.ravel()))])

def norm_with_scale(points, x, y, z, scale, align=True):
    # print(points[np.argmin(points[:, 0])][0], points[np.argmax(points[:, 0])][0])

    if align:
        points[:, 0] = points[:, 0] - x
        points[:, 1] = points[:, 1] - y
        points[:, 2] = points[:, 2] - z

    return points/scale

def backup(points, x, y, z, scale):
    points = points * scale
    
    points[:, 0] = points[:, 0] + x
    points[:, 1] = points[:, 1] + y
    points[:, 2] = points[:, 2] + z

    return points 

def norm_to_0_1(points, align=True):
    x = points[np.argmin(points[:, 0])][0]
    y = points[np.argmin(points[:, 1])][1]
    z = points[np.argmin(points[:, 2])][2]

    # print(points[np.argmin(points[:, 0])][0], points[np.argmax(points[:, 0])][0])

    if align:
        points[:, 0] = points[:, 0] - x
        points[:, 1] = points[:, 1] - y
        points[:, 2] = points[:, 2] - z

    # return points

    return points/abs(points.ravel()[np.argmax(abs(points.ravel()))]), x, y, z, abs(points.ravel()[np.argmax(abs(points.ravel()))])

# norm a && norm b with a
def norm_a_b(a, b):
    norm_a, x, y, z, scale = norm_to_0_1(a, True)
    norm_b = norm_with_scale(b, x, y, z, scale, True)

    return norm_a, norm_b


def norm_to_center(point):
    center = point.mean(axis=0)
    x_scale = point[:, 0].max() - point[:, 0].min()
    y_scale = point[:, 1].max() - point[:, 1].min()
    z_scale = point[:, 2].max() - point[:, 2].min()
    point_norm = point - center[None, :]
    point_norm[:, 0] = point_norm[:, 0] / x_scale
    point_norm[:, 1] = point_norm[:, 1] / y_scale
    point_norm[:, 2] = point_norm[:, 2] / z_scale
    return point_norm, center, x_scale, y_scale, z_scale

def norm_with_center_param(point, center, x_scale, y_scale, z_scale):
    point_norm = point - center[None, :]
    point_norm[:, 0] = point_norm[:, 0] / x_scale
    point_norm[:, 1] = point_norm[:, 1] / y_scale
    point_norm[:, 2] = point_norm[:, 2] / z_scale
    return point_norm

def back_center(point, center, x_scale, y_scale, z_scale):
    point[:, 0] = point[:, 0] * x_scale
    point[:, 1] = point[:, 1] * y_scale
    point[:, 2] = point[:, 2] * z_scale
    point = point + center[None, :]
    return point

def tensor_norm_with_scale(points, param, align=True):

    if align:
        points[:, 0] = points[:, 0] - param[0]
        points[:, 1] = points[:, 1] - param[1]
        points[:, 2] = points[:, 2] - param[2]

    return points/param[3]

def cal_average_face_normals(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_triangle_normals(normalized=False)

    nx = np.mean(np.asarray(mesh.triangle_normals)[:, 0])
    ny = np.mean(np.asarray(mesh.triangle_normals)[:, 1])
    nz = np.mean(np.asarray(mesh.triangle_normals)[:, 2])

    return nx, ny, nz

def get_clock_angle(v1, v2):

    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2)/TheNorm))
    theta = np.rad2deg(np.arccos(np.dot(v1, v2)/TheNorm))

    return rho, theta

def get_rotate_matrix(vertices, faces):

    nx, ny, nz = cal_average_face_normals(vertices, faces)

    assert(not np.isnan(nx).any())

    y_dir, y_angle = get_clock_angle([nx, 0, nz], [0, 0, 1])
    x_dir, x_angle = get_clock_angle([0, ny, np.sqrt(nx**2 + nz**2)], [0, 0, 1])

    y_angle = y_dir[1]/np.abs(y_dir[1]) * y_angle
    x_angle = x_dir[0]/np.abs(x_dir[0]) * x_angle

    y_angle = np.array((y_angle) * np.pi / 180.)
    x_angle = np.array((x_angle) * np.pi / 180.)

    '''
    # z
    rot = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]).astype(np.float32)
    # y
    rot = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]).astype(np.float32)
    # x
    rot = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]]).astype(np.float32)
    '''
    theta = y_angle
    rot_y = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]).astype(np.float32)
    theta = x_angle
    rot_x = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]]).astype(np.float32)

    return rot_y, rot_x
