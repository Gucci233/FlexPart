import os
import json
import argparse
from tqdm import tqdm
import sys
import numpy as np
import torch
from pathlib import Path
import trimesh
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image,ImageDraw
from typing import *
# --- 添加项目根目录 (请确保路径正确) ---
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 假设 prepare_image 位于 image_utils
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG
from src.utils.data_utils import scene_to_parts, mesh_to_surface, normalize_mesh,get_camera,\
get_visible_part_masks,visualize_masks,filter_scene_meshes,mask_to_box,count_non_white_pixels
from src.utils.render_utils import render_single_view,render_cube_views_mesh
from src.utils.metric_utils import compute_IoU_for_scene
# --- 全局常量定义 ---
DEFAULT_RADIUS = 4.0
DEFAULT_IMAGE_SIZE = (2048, 2048)
DEFAULT_LIGHT_INTENSITY = 2.5
DEFAULT_NUM_ENV_LIGHTS = 36
RMBG_WEIGHTS_DIR = "./weight/RMBG-1.4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (确保 DEBUG_COLOR_PALETTE 已在上面定义)
# --- 1. 新增: 全局调试调色板 ---
# (R, G, B)
DEBUG_COLOR_PALETTE = [
    (255, 0, 0),   # 1. 亮红
    (0, 255, 0),   # 2. 亮绿
    (0, 0, 255),   # 3. 亮蓝
    (255, 255, 0), # 4. 亮黄
    (0, 255, 255), # 5. 亮青
    (255, 0, 255), # 6. 亮紫
    (255, 128, 0), # 7. 橙色
    (128, 0, 255), # 8. 紫色
    (0, 128, 0),   # 9. 深绿
    (255, 0, 128), # 10. 粉红
    (0, 128, 128), # 11. 蓝绿
    (128, 128, 0), # 12. 橄榄色
]

def get_mesh_rotation_matrix(camera_pose: np.ndarray) -> np.ndarray:
    """
    根据相机的 c2w pose 计算 Mesh 需要旋转的矩阵。
    逻辑：Mesh 的旋转 = 相机旋转的逆 (Inverse/Transpose)
    """
    # 1. 提取相机 pose 的旋转部分 (左上角 3x3)
    # camera_pose shape: (4, 4)
    R_camera = camera_pose[:3, :3]

    # 2. 计算逆旋转 (对于旋转矩阵，转置即为逆)
    R_mesh = R_camera.T

    # 3. 构建 4x4 的 Mesh 变换矩阵
    mesh_transform = np.eye(4)
    mesh_transform[:3, :3] = R_mesh

    # 注意：通常只旋转物体，不平移，所以保持平移部分为 0 (即最后一列为 0,0,0,1)
    return mesh_transform

def process_mesh_to_data(input_path: str, output_base_dir: str):
    """
    终极简化版：直接提取 OBB 数据，不渲染 OBB 图像，逻辑完全内聚。
    """
    mesh_name = Path(input_path).stem
    output_dir = Path(output_base_dir) / mesh_name

    try:
        # --- 1. 加载与初步过滤 ---
        raw_mesh = trimesh.load(input_path, process=False)
        norm_mesh = normalize_mesh(filter_scene_meshes(raw_mesh))

        # 检查部件数量
        parts_count = len(norm_mesh.geometry) if isinstance(norm_mesh, trimesh.Scene) else 1
        if not (0 < parts_count <= 20):
            # print(f"跳过 [{mesh_name}]: 部件数量 ({parts_count}) 不在范围内")
            return None

        # --- 2. 视角寻优 (寻找最佳视角) ---
        # 渲染 6 个视角用于选择
        render_kwargs = {"radius": DEFAULT_RADIUS, "image_size": DEFAULT_IMAGE_SIZE}
        flat_geometry = norm_mesh.to_geometry() if isinstance(norm_mesh, trimesh.Scene) else norm_mesh
        cube_images = render_cube_views_mesh(flat_geometry, return_type='ndarray', **render_kwargs)

        pixel_counts = [count_non_white_pixels(img) for img in cube_images]
        best_view_idx = np.argmax(pixel_counts)
        best_render_img = cube_images[best_view_idx]

        # --- 3. 几何变换 (旋转 Mesh) ---
        camera, poses = get_camera(image_size=DEFAULT_IMAGE_SIZE)
        rotation_matrix = get_mesh_rotation_matrix(poses[best_view_idx])
        norm_mesh.apply_transform(rotation_matrix)

        # 更新展平后的 Mesh 用于保存 'object'
        final_flat_mesh = norm_mesh.to_geometry() if isinstance(norm_mesh, trimesh.Scene) else norm_mesh

        # --- 4. 提取核心数据 (Points, Masks, OBB) ---

        # A. 提取部件点云 (此时坐标已是旋转后的)
        # 假设 scene_to_parts 返回的是 [{'points': array, ...}, ...]
        parts_data_list = scene_to_parts(norm_mesh, return_type="point", normalize=False)

        # B. 提取 Mask (使用默认相机 pose[0])
        masks = get_visible_part_masks(
            mesh_scene=norm_mesh,
            camera_pose=poses[0],
            camera=camera,
            image_size=DEFAULT_IMAGE_SIZE
        )

        # C. 准备容器
        valid_data_entries = [] # 存储 (mask_size, part_dict, obb_vector)
        min_mask_size = 50

        # --- 5. 循环处理每个部件：计算 OBB 并结合 Mask ---
        for i, part_data in enumerate(parts_data_list):
            current_mask = masks[i]
            mask_size = np.sum(current_mask)

            # 过滤过小的 Mask
            if mask_size < min_mask_size:
                continue

            # --- 计算 OBB (10D) ---
            # 10D = Center(3) + Size(3) + Quaternion(4)
            try:
                points = part_data.get('points')
                if points is None or len(points) < 4:
                    obb_vector = np.zeros(10, dtype=np.float32)
                else:
                    # 使用 trimesh 计算 OBB
                    pc = trimesh.PointCloud(points)
                    obb = pc.bounding_box_oriented

                    center = obb.centroid
                    extents = obb.extents
                    # 获取旋转矩阵并转为四元数
                    transform = obb.transform
                    quaternion = trimesh.transformations.quaternion_from_matrix(transform) # [w, x, y, z]

                    obb_vector = np.concatenate([center, extents, quaternion]).astype(np.float32)
            except Exception as e:
                print(f"OBB计算警告 [{mesh_name} - part {i}]: {e}")
                obb_vector = np.zeros(10, dtype=np.float32)

            # --- 注入 2D 信息 ---
            part_data['2d_mask'] = current_mask
            part_data['2d_box'] = mask_to_box(current_mask)

            # 暂存数据用于排序
            valid_data_entries.append({
                "size": mask_size,
                "part": part_data,
                "obb": obb_vector
            })

        # --- 6. 排序与重组 ---
        # 按 Mask 大小降序排列
        valid_data_entries.sort(key=lambda x: x["size"], reverse=True)

        final_parts = [entry["part"] for entry in valid_data_entries]
        final_obbs = np.array([entry["obb"] for entry in valid_data_entries])

        # 如果没有有效部件，直接返回
        if not final_parts:
            return None

        # --- 7. 保存结果 ---
        output_dir.mkdir(parents=True, exist_ok=True)
        norm_mesh.export(output_dir / 'normalized_rotated_scene.glb')

        # 保存图片
        Image.fromarray(best_render_img).save(output_dir / "rendering.png")
        visualize_masks([p['2d_mask'] for p in final_parts]).save(output_dir / "mask.png")

        # 保存数据
        datas = {
            "object": mesh_to_surface(final_flat_mesh, return_dict=True),
            "parts": final_parts,
            "part_obbs": final_obbs
        }

        np.save(output_dir / 'points.npy', datas, allow_pickle=True)

        config = {"num_parts": len(final_parts)}
        with open(output_dir / 'num_parts.json', 'w') as f:
            json.dump(config, f, indent=4)

        return config

    except Exception as e:
        print(f"处理失败 [{input_path}]: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_cpu_tasks_for_mesh(mesh_path: str, output_base_dir: str):
    # (此处省略了函数体，使用您代码中的原函数即可)
    try:
        render_kwargs = {"radius": DEFAULT_RADIUS, "image_size": DEFAULT_IMAGE_SIZE, "light_intensity": DEFAULT_LIGHT_INTENSITY, "num_env_lights": DEFAULT_NUM_ENV_LIGHTS}
        config = process_mesh_to_data(mesh_path, output_base_dir)
        return config
    except Exception as e:
        print(e)
        return None

def validate_obbs(part_obbs) :
    """
    检查 OBB NumPy 数组 (N, 10) 是否有效 (非 None, 正确形状, 且不包含全零行)。
    """
    if part_obbs is None:
        # print("调试: OBB 数据为 None。") # 用于调试
        return False # 加载失败或键不存在，视为无效
    if not isinstance(part_obbs, np.ndarray) or part_obbs.ndim != 2 or part_obbs.shape[1] != 10:
        print(f"警告: OBB 格式不正确 (shape: {part_obbs.shape if isinstance(part_obbs, np.ndarray) else 'N/A'}), 视为无效。")
        return False
    if part_obbs.shape[0] == 0:
         print(f"警告: OBB 数组为空 (shape: {part_obbs.shape})，视为无效。")
         return False

    # 检查是否存在全零行
    if np.any(np.all(part_obbs == 0, axis=1)):
        # print(f"信息: 检测到全零 OBB 行，该样本将被过滤。") # 可以取消注释
        return False
    else:
        # 没有全零行，数据有效
        return True

def process_single_marker(marker_path, input_dir):
    """
    处理单个标记文件，验证并生成对应的配置字典。
    假定所有原始文件都在唯一的 input_dir 中。
    """
    try:
        mesh_output_path = marker_path.parent
        mesh_name = mesh_output_path.name

        # 验证所有必需的输出文件是否存在
        image_path = mesh_output_path / 'rendering.png'
        surface_path = mesh_output_path / 'points.npy'
        obb_image_path = mesh_output_path / 'render_OBB.png'
        mask_path = mesh_output_path / 'mask.png'
        if not all(p.exists() for p in [image_path, surface_path, mask_path,
                                       mesh_output_path / 'normalized_rotated_scene.glb']):
            return None
        # 在指定的單個輸入目錄中查找原始網格文件
        base_path = Path(input_dir)
        original_mesh_path = None

        candidates = list(base_path.rglob(f"{mesh_name}.glb"))
        if len(candidates) != 1:
            return None
        path_glb = candidates[0]
        if path_glb.is_file():
            original_mesh_path = str(path_glb)
        else:
            path_obj = base_path / f"{mesh_name}.obj"
            if path_obj.is_file():
                original_mesh_path = str(path_obj)

        if not original_mesh_path:
            return None # 未找到原始文件，返回 None

        # 读取JSON数据并构建配置
        # with open(marker_path, 'r') as f:
        #     num_parts = json.load(f)['num_parts']
        surface_data = np.load(surface_path, allow_pickle=True).item()
        if 'part_obbs' not in surface_data.keys():
            return None
        elif not validate_obbs(surface_data["part_obbs"]):
            return None
        parts_list = surface_data.get('parts', [1])
        if len(parts_list) >= 1:
            num_parts = len(parts_list)
        else:
            return None

        config = {
            "file": os.path.relpath(original_mesh_path, start=input_dir), # 直接使用 input_dir 計算相對路徑
            "mesh_path": original_mesh_path,
            "surface_path": str(surface_path),
            "image_path": str(image_path),
            "obb_image_path": str(obb_image_path) if obb_image_path.is_file() else None,
            "mask_path":str(mask_path),
            "num_parts": num_parts,
            "valid": True,
        }
        return config
    except Exception:
        return None



def check_single_model(mesh_path_output):
    """检查单个 mesh 是否需要重新处理"""
    mesh_path, output_root = mesh_path_output
    mesh_name = Path(mesh_path).stem

    render_path = os.path.join(output_root, mesh_name, 'rendering.png')
    points_path = os.path.join(output_root, mesh_name, 'points.npy')

    # 缺任意关键文件
    if not all(os.path.exists(os.path.join(output_root, mesh_name, name)) for name in
               ('rendering.png', 'points.npy', 'mask.png', 'normalized_rotated_scene.glb', 'num_parts.json')):
        return mesh_path

    try:
        with open(points_path, "rb") as f:
            data = np.load(f, allow_pickle=True).item()
        if not isinstance(data, dict) or 'part_obbs' not in data:
            return mesh_path
    except Exception:
        return mesh_path

    return None


def collect_cpu_todo(all_model_paths, output_root, num_workers=None):
    num_workers = num_workers or max(1, cpu_count() - 1)
    cpu_todo_paths = []

    tasks = [(p, output_root) for p in all_model_paths]
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(check_single_model, tasks),
                total=len(tasks),
                desc="检查CPU任务进度",
                dynamic_ncols=True
            )
        )

    cpu_todo_paths = [r for r in results if r is not None]
    return cpu_todo_paths

# ===============================================================
#  主执行逻辑 (已更新)
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="对3D模型数据集进行可中断的、并行的预处理。")
    parser.add_argument('--input', type=str, default='./data/raw', help="包含 .glb 文件的根目录。")
    parser.add_argument('--output', type=str, default='./data/preprocessed', help="预处理后文件的输出目录。")
    parser.add_argument('--workers', type=int, default=1, help="要使用的工作进程数。默认为系统CPU核心数。")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error('--workers must be at least 1')
    if not Path(args.input).is_dir():
        parser.error('--input must be an existing directory')

    os.makedirs(args.output, exist_ok=True)

    # --- 改造点 1: 任务开始前，过滤掉已完成的CPU任务 ---
    print("正在查找所有 .glb 文件...")
    all_model_paths = sorted(path.as_posix() for path in Path(args.input).rglob("*.glb"))
    names = [Path(path).stem for path in all_model_paths]
    if len(names) != len(set(names)):
        parser.error('Input GLB files must have unique stems to avoid output collisions')
    print(f"共找到 {len(all_model_paths)} 个模型。")

    cpu_todo_paths = collect_cpu_todo(all_model_paths, args.output)


    print(f"检查完成。共 {len(all_model_paths) - len(cpu_todo_paths)} 个任务已完成，还需处理 {len(cpu_todo_paths)} 个任务。")
    if not cpu_todo_paths:
        print("所有CPU任务均已完成或无需处理。")
    else:
        # --- 阶段一: 并行处理CPU任务 ---
        num_workers = args.workers if args.workers is not None else cpu_count()-5
        print(f"使用 {num_workers} 个工作进程。")

        worker_func = partial(run_cpu_tasks_for_mesh, output_base_dir=args.output)

        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(cpu_todo_paths), desc="CPU任务进度") as pbar:
                # 只对过滤后的列表进行处理
                for _ in pool.imap_unordered(worker_func, cpu_todo_paths):
                    pbar.update(1)
        print(f"\nCPU任务阶段完成。")



    output_path = Path(args.output)
    all_processed_markers = sorted(output_path.glob('*/num_parts.json'))
    # all_processed_markers = all_processed_markers[:90000]
    if not all_processed_markers:
        print("未找到任何已处理的标记文件，程序退出。")
        return

    num_processes = cpu_count() - 1 or 1
    print(f"使用 {num_processes} 个进程进行并行处理...")

    worker_func = partial(process_single_marker, input_dir=args.input)

    final_configs = []
    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(worker_func, all_processed_markers)

        for config in tqdm(results_iterator, total=len(all_processed_markers), desc="生成配置"):
            if config:
                final_configs.append(config)

    final_configs.sort(key=lambda item: item['file'])
    configs_path = output_path.parent / 'object_part_configs_new.json'
    print(f"正在保存 {len(final_configs)} 条有效配置到 {configs_path}...")
    with open(configs_path, 'w', encoding='utf-8') as f:
        json.dump(final_configs, f, indent=4, ensure_ascii=False)

    print("全部完成！")

if __name__ == '__main__':
    main()
