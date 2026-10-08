import trimesh
from kiui.mesh_utils import clean_mesh, decimate_mesh

def postprocess_mesh(mesh: trimesh.Trimesh, decimate_target=100000):
    vertices = mesh.vertices
    triangles = mesh.faces

    if vertices.shape[0] > 0 and triangles.shape[0] > 0:
        vertices, triangles = clean_mesh(vertices, triangles, remesh=False, min_f=25, min_d=5)
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target, optimalplacement=False)
        if vertices.shape[0] > 0 and triangles.shape[0] > 0:
            vertices, triangles = clean_mesh(vertices, triangles, remesh=False, min_f=25, min_d=5)

    mesh.vertices = vertices
    mesh.faces = triangles

    return mesh


def filter_mesh(mesh):
    submeshes = mesh.split(only_watertight=False)
    if len(submeshes) == 1:
        return submeshes[0]

    face_counts = [len(m.faces) for m in submeshes]
    max_faces = max(face_counts)

    filtered = [m for m in submeshes if len(m.faces) > max_faces*0.1]

    if len(filtered) == 1:
        return filtered[0]

    merged = trimesh.util.concatenate(filtered)
    merged.merge_vertices()
    _ = merged.vertex_normals
    return merged
