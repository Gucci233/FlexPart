from src.utils.typing_utils import *

import os
import numpy as np
import trimesh
import torch
from PIL import Image, ImageDraw
import pyrender
import cv2
import torch.nn.functional as F
from src.utils.render_utils import create_cube_face_camera_poses
from scipy.ndimage import binary_erosion,binary_opening

def normalize_mesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    scale: float = 2.0,
):
    # if not isinstance(mesh, trimesh.Trimesh) and not isinstance(mesh, trimesh.Scene):
    #     raise ValueError("Input mesh is not a trimesh.Trimesh or trimesh.Scene object.")
    bbox = mesh.bounding_box
    translation = -bbox.centroid
    scale = scale / bbox.primitive.extents.max()
    mesh.apply_translation(translation)
    mesh.apply_scale(scale)
    return mesh

def remove_overlapping_vertices(mesh: trimesh.Trimesh, reserve_material: bool = False):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input mesh is not a trimesh.Trimesh object.")
    vertices = mesh.vertices
    faces = mesh.faces
    unique_vertices, index_map, inverse_map = np.unique(
        vertices, axis=0, return_index=True, return_inverse=True
    )
    clean_faces = inverse_map[faces]
    clean_mesh = trimesh.Trimesh(vertices=unique_vertices, faces=clean_faces, process=True)
    if reserve_material:
        uv = mesh.visual.uv
        material = mesh.visual.material
        clean_uv = uv[index_map]
        clean_visual = trimesh.visual.TextureVisuals(uv=clean_uv, material=material)
        clean_mesh.visual = clean_visual
    return clean_mesh

RGB = [
    (82, 170, 220),
    (215, 91, 78),
    (45, 136, 117),
    (247, 172, 83),
    (124, 121, 121),
    (127, 171, 209),
    (243, 152, 101),
    (145, 204, 192),
    (150, 59, 121),
    (181, 206, 78),
    (189, 119, 149),
    (199, 193, 222),
    (200, 151, 54),
    (236, 110, 102),
    (238, 182, 212),
]


def get_colored_mesh_composition(
    meshes: Union[List[trimesh.Trimesh], trimesh.Scene],
    is_random: bool = True,
    is_sorted: bool = False,
    RGB: List[Tuple] = RGB
):
    if isinstance(meshes, trimesh.Scene):
        meshes = meshes.dump()
    if is_sorted:
        volumes = []
        for mesh in meshes:
            try:
                volume = mesh.volume
            except:
                volume = 0.0
            volumes.append(volume)
        # sort by volume from large to small
        meshes = [x for _, x in sorted(zip(volumes, meshes), key=lambda pair: pair[0], reverse=True)]
    colored_scene = trimesh.Scene()
    for idx, mesh in enumerate(meshes):
        if is_random:
            color = (np.random.rand(3) * 256).astype(int)
        else:
            color = np.array(RGB[idx % len(RGB)])
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=color,
        )
        colored_scene.add_geometry(mesh)
    return colored_scene

def mesh_to_surface(
    mesh: trimesh.Trimesh,
    num_pc: int = 204800,
    clip_to_num_vertices: bool = False,
    return_dict: bool = False,
):
    # if not isinstance(mesh, trimesh.Trimesh):
    #     raise ValueError("mesh must be a trimesh.Trimesh object")
    if clip_to_num_vertices:
        num_pc = min(num_pc, mesh.vertices.shape[0])
    points, face_indices = mesh.sample(num_pc, return_index=True)
    normals = mesh.face_normals[face_indices]
    if return_dict:
        return {
            "surface_points": points,
            "surface_normals": normals,
        }
    return points, normals

def scene_to_parts(
    mesh: trimesh.Scene,
    normalize: bool = True,
    scale: float = 2.0,
    num_part_pc: int = 204800,
    clip_to_num_part_vertices: bool = False,
    return_type: Literal["mesh", "point"] = "mesh",
) -> Union[List[trimesh.Geometry], List[Dict[str, np.ndarray]]]:
    if not isinstance(mesh, trimesh.Scene):
        raise ValueError("mesh must be a trimesh.Scene object")
    if normalize:
        mesh = normalize_mesh(mesh, scale=scale)
    parts: List[trimesh.Geometry] = mesh.dump()
    if return_type == "point":
        datas: List[Dict[str, np.ndarray]] = []
        for geom in parts:
            data = mesh_to_surface(
                geom,
                num_pc=num_part_pc,
                clip_to_num_vertices=clip_to_num_part_vertices,
                return_dict=True,
            )
            datas.append(data)
        return datas
    elif return_type == "mesh":
        return parts
    else:
        raise ValueError("return_type must be 'mesh' or 'point'")

def get_center(mesh: trimesh.Trimesh, method: Literal['mass', 'bbox']):
    if method == 'mass':
        return mesh.center_mass
    elif method =='bbox':
        return mesh.bounding_box.centroid
    else:
        raise ValueError('type must be mass or bbox')

def get_direction(vector: np.ndarray):
    return vector / np.linalg.norm(vector)

def move_mesh_by_center(mesh: trimesh.Trimesh, scale: float, method: Literal['mass', 'bbox'] = 'mass'):
    offset = scale - 1
    center = get_center(mesh, method)
    direction = get_direction(center)
    translation = direction * offset
    mesh = mesh.copy()
    mesh.apply_translation(translation)
    return mesh

def move_meshes_by_center(meshes: Union[List[trimesh.Trimesh], trimesh.Scene], scale: float):
    if isinstance(meshes, trimesh.Scene):
        meshes = meshes.dump()
    moved_meshes = []
    for mesh in meshes:
        moved_mesh = move_mesh_by_center(mesh, scale)
        moved_meshes.append(moved_mesh)
    moved_meshes = trimesh.Scene(moved_meshes)
    return moved_meshes

def get_series_splited_meshes(meshes: List[trimesh.Trimesh], scale: float, num_steps: int) -> List[trimesh.Scene]:
    series_meshes = []
    for i in range(num_steps):
        temp_scale = 1 + (scale - 1) * i / (num_steps - 1)
        temp_meshes = move_meshes_by_center(meshes, temp_scale)
        series_meshes.append(temp_meshes)
    return series_meshes

def load_surface(data, num_pc=204800):

    surface = data["surface_points"]  # Nx3
    normal = data["surface_normals"]  # Nx3

    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_pc, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    surface = torch.cat([surface, normal], dim=-1)

    return surface

def load_sharp_surfaces(surfaces, num_pc=204800):
    surfaces = [load_sharp_surface(surface, num_pc) for surface in surfaces]
    surfaces = torch.stack(surfaces, dim=0)
    return surfaces

def load_sharp_surface(data, num_pc=204800):

    surface = data["sharp_points"]  # Nx3
    normal = data["sharp_normals"]  # Nx3

    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_pc, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    surface = torch.cat([surface, normal], dim=-1)

    return surface

def load_surfaces(surfaces, num_pc=204800):
    surfaces = [load_surface(surface, num_pc) for surface in surfaces]
    surfaces = torch.stack(surfaces, dim=0)
    return surfaces



def get_camera(
    radius: float = 4.0,
    image_size: tuple = (2048, 2048),
    fov: float = 40.0,
    znear: float = 0.1,
    zfar: float = 10.0,
):
    camera = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(fov),
        aspectRatio=image_size[0]/image_size[1],
        znear=znear,
        zfar=zfar
    )

    camera_poses = create_cube_face_camera_poses(
        radius=radius,
    )

    return camera,camera_poses


def get_visible_part_masks(
    mesh_scene: trimesh.Scene,
    camera_pose: np.ndarray,
    camera: pyrender.PerspectiveCamera,
    image_size: Tuple[int, int]
) -> List[np.ndarray]:
    """
    为 3D 场景中的每个部件生成一个 2D 可见性掩码 (mask)。

    此函数执行一次 "ID 渲染"，通过 Z-buffering (深度测试) 自动处理遮挡。

    Args:
        mesh_scene (trimesh.Scene):
            输入的 3D 场景。
        camera_pose (np.ndarray):
            (4, 4) 相机外参 (Camera-to-World)。
        camera (pyrender.PerspectiveCamera):
            Pyrender 相机内参对象。
        image_size (Tuple[int, int]):
            渲染图像的 (width, height)。

    Returns:
        List[np.ndarray]:
            N 个部件的列表，其中 N = len(mesh_scene.dump())。
            列表中的每个元素是一个 (height, width) 的 bool 类型的掩码，
            标记了该部件在 2D 图像上的 *可见* 像素。
            (如果某个部件完全被遮挡，其掩码将全部为 False)。
    """
    width, height = image_size
    scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[0.0, 0.0, 0.0])
    renderer = pyrender.OffscreenRenderer(width, height)

    if isinstance(mesh_scene, trimesh.Trimesh):
        mesh_scene = trimesh.Scene(mesh_scene)

    parts: List[trimesh.Geometry] = mesh_scene.dump()
    num_parts = len(parts)

    if num_parts == 0:
        renderer.delete()
        return []

    for i, geom in enumerate(parts):
        if not isinstance(geom, trimesh.Trimesh) or geom.vertices is None or len(geom.vertices) == 0:
            continue

        part_id = i + 1

        r = (part_id & 0xFF) / 255.0
        g = ((part_id >> 8) & 0xFF) / 255.0
        b = ((part_id >> 16) & 0xFF) / 255.0

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[float(r), float(g), float(b), 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )

        pr_mesh = pyrender.Mesh.from_trimesh(geom, material=material)
        scene.add(pr_mesh)

    scene.add(camera, pose=camera_pose)

    flags = pyrender.constants.RenderFlags.FLAT
    color_img_uint8, depth_img = renderer.render(scene, flags=flags)

    renderer.delete()

    r_ch = color_img_uint8[:, :, 0].astype(np.uint32)
    g_ch = color_img_uint8[:, :, 1].astype(np.uint32)
    b_ch = color_img_uint8[:, :, 2].astype(np.uint32)

    id_map = r_ch + (g_ch << 8) + (b_ch << 16)

    all_masks: List[np.ndarray] = []

    for i in range(num_parts):
        part_id = i + 1
        mask = (id_map == part_id)
        mask = binary_opening(mask, structure=np.ones((2,2)))
        all_masks.append(mask)

    return all_masks



def visualize_masks(
    all_masks: List[np.ndarray],
    base_image: Optional[Union[np.ndarray, Image.Image]] = None,
    alpha: float = 0.9
) -> Image.Image:
    """
    将一个布尔掩码的列表，用不同的半透明颜色绘制到一个基础图像上。

    Args:
        all_masks (List[np.ndarray]):
            一个 (H, W) 布尔掩码的列表 (来自 get_visible_part_masks)。
        base_image (Optional[Union[np.ndarray, Image.Image]]):
            (H, W, 3) 的 uint8 numpy 图像，或 PIL 图像。
            如果为 None，将使用白色背景。
        alpha (float):
            掩码的透明度 (0.0 到 1.0)。

    Returns:
        Image.Image:
            合成了掩码的 PIL 图像 (RGB 模式)。
    """

    if not all_masks:
        # 如果列表为空
        H, W = (2048, 2048) # 默认大小
        if base_image is None:
            return Image.new("RGB", (W, H), (255, 255, 255))
        elif isinstance(base_image, np.ndarray):
            return Image.fromarray(base_image).convert("RGB")
        else:
            return base_image.convert("RGB")

    # 从第一个掩码获取形状
    H, W = all_masks[0].shape
    num_colors = len(RGB)

    # 1. 准备基础图像 (必须是 RGBA 才能合成)
    if base_image is None:
        base_rgba_pil = Image.new("RGBA", (W, H), (255, 255, 255, 255)) # 白色
    elif isinstance(base_image, np.ndarray):
        base_rgba_pil = Image.fromarray(base_image).convert("RGBA")
    else:
        base_rgba_pil = base_image.convert("RGBA")

    # 2. 创建一个空的“图层”用于绘制所有掩码
    overlay_rgba_pil = Image.new("RGBA", (W, H), (0, 0, 0, 0)) # 完全透明

    # 3. 遍历每个掩码并将其“粘贴”到图层上
    for i, mask_np in enumerate(all_masks):

        if mask_np.shape != (H, W):
            continue

        # 如果掩码全为 False，跳过绘制
        if not mask_np.any():
            continue

        # a) 获取颜色 (带透明度)
        color_rgb = RGB[i % num_colors]
        color_rgba = (*color_rgb, int(alpha * 255)) # e.g., (255, 0, 0, 128)

        # b) 将布尔掩码 (False/True) 转换为 PIL 'L' 掩码 (0/255)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

        # c) 将此颜色“粘贴”到图层上，*只*在掩码区域
        #    (我们创建一个纯色图像，然后用 mask_pil 作为粘贴蒙版)
        color_layer = Image.new("RGBA", (W, H), color_rgba)
        overlay_rgba_pil.paste(color_layer, mask=mask_pil)

    # 4. 将掩码图层合成到基础图像上
    final_image_pil = Image.alpha_composite(base_rgba_pil, overlay_rgba_pil)

    return final_image_pil.convert("RGB")


def mask_to_box(mask):
    """
    输入 mask: H×W，值为0/1或0/255
    输出: (x1, y1, x2, y2)
    """
    # 找到所有前景点
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None  # 空mask

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()

    return x1, y1, x2, y2


def filter_scene_meshes(input_scene: trimesh.Scene) -> trimesh.Scene:
    meshes = input_scene.dump()

    fixed_meshes = []

    for idx, geom in enumerate(meshes):
        if isinstance(geom, trimesh.Trimesh):
            fixed_meshes.append(geom)

    new_scene = trimesh.Scene(fixed_meshes)

    return new_scene

def count_non_white_pixels(img):
    if isinstance(img, Image.Image):
        img = np.array(img)

    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]

    mask = np.any(img < 250, axis=-1)
    return np.count_nonzero(mask)


def vector_to_obb_mesh(vector_10d: np.ndarray) -> trimesh.Trimesh:
    """
    将一个 10 维 OBB 向量转换为 trimesh.Trimesh Box 对象。

    Args:
        vector_10d: 一个 NumPy 数组，形状为 (10,)，包含 OBB 参数：
                    [cx, cy, cz, sx, sy, sz, qw, qx, qy, qz]
                    - cx, cy, cz: 中心坐标 (3 维)
                    - sx, sy, sz: 尺寸/范围 (3 维)
                    - qw, qx, qy, qz: 四元数表示的旋转 (4 维)

    Returns:
        一个 trimesh.Trimesh 对象，代表了由 10D 向量定义的 OBB。
    """
    if not isinstance(vector_10d, np.ndarray) or vector_10d.shape != (10,):
        raise ValueError("输入必须是一个形状为 (10,) 的 NumPy 数组。")

    center = vector_10d[0:3]
    size = vector_10d[3:6]
    quaternion = vector_10d[6:10]


    rotation_matrix = trimesh.transformations.quaternion_matrix(quaternion)

    transform_matrix = rotation_matrix
    transform_matrix[:3, 3] = center

    obb_mesh = trimesh.creation.box(extents=size, transform=transform_matrix)

    return obb_mesh.bounds
