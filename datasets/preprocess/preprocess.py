import os
import json
import argparse
import time
from tqdm import tqdm
import pandas as pd
import sys
import numpy as np
import torch
from pathlib import Path
import trimesh
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG
from src.utils.data_utils import scene_to_parts, mesh_to_surface, normalize_mesh
from src.utils.data_utils import normalize_mesh
from src.utils.render_utils import render_single_view
from src.utils.data_utils import normalize_mesh
from src.utils.metric_utils import compute_IoU_for_scene

def process_single_mesh(input_path: str, output_base_dir: str):
    """
    处理单个3D网格文件，对其进行规范化，提取部件，
    并保存处理后的数据和配置文件。

    Args:
        input_path (str): 输入的3D网格文件的路径 (例如, .glb 文件)。
        output_base_dir (str): 输出文件夹的基准目录，处理结果将保存在该目录下的一个与mesh同名的子文件夹中。
    """
    # 1. 验证输入路径
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在于 {input_path}")
        return

    # 2. 设置输出目录
    mesh_name = os.path.basename(input_path).split('.')[0]
    specific_output_path = os.path.join(output_base_dir, mesh_name)
    os.makedirs(specific_output_path, exist_ok=True)
    print(f"正在处理 '{input_path}' -> 保存至 '{specific_output_path}'")

    # 3. 加载并规范化网格
    try:
        mesh = trimesh.load(input_path, process=False)
        mesh = normalize_mesh(mesh)
    except Exception as e:
        print(f"加载或规范化网格 {input_path} 时出错: {e}")
        return

    # 4. 处理部件
    config = {
        "num_parts": len(mesh.geometry)
    }

    if 1 < config["num_parts"] <= 16:
        parts = scene_to_parts(
            mesh,
            return_type="point",
            normalize=False
        )
    else:
        parts = []

    # 5. 处理整个对象
    # 将场景（Scene）转换为单个几何体（Geometry）以便于处理
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    object_surface_data = mesh_to_surface(mesh, return_dict=True)

    # 6. 准备并保存数据
    datas = {
        "object": object_surface_data,
        "parts": parts,
    }

    # 将点数据保存为 .npy 文件
    points_output_path = os.path.join(specific_output_path, 'points.npy')
    np.save(points_output_path, datas)

    # 将配置保存为 .json 文件
    config_output_path = os.path.join(specific_output_path, 'num_parts.json')
    with open(config_output_path, 'w') as f:
        json.dump(config, f, indent=4)



# 为渲染设置默认参数
DEFAULT_RADIUS = 4.0
DEFAULT_IMAGE_SIZE = (2048, 2048)
DEFAULT_LIGHT_INTENSITY = 2.5
DEFAULT_NUM_ENV_LIGHTS = 36

def render_and_save_view(
    input_path: str,
    output_base_dir: str,
    radius: float = DEFAULT_RADIUS,
    image_size: tuple = DEFAULT_IMAGE_SIZE,
    light_intensity: float = DEFAULT_LIGHT_INTENSITY,
    num_env_lights: int = DEFAULT_NUM_ENV_LIGHTS
):
    """
    加载并渲染3D网格的单个视图，并将渲染结果保存为PNG图片。

    Args:
        input_path (str): 输入的3D网格文件的路径。
        output_base_dir (str): 输出文件夹的基准目录。
        radius (float, optional): 渲染相机的半径。默认为 DEFAULT_RADIUS。
        image_size (tuple, optional): 输出图像的分辨率 (宽, 高)。默认为 DEFAULT_IMAGE_SIZE。
        light_intensity (float, optional): 光照强度。默认为 DEFAULT_LIGHT_INTENSITY。
        num_env_lights (int, optional): 环境光数量。默认为 DEFAULT_NUM_ENV_LIGHTS。
    """
    # 1. 验证输入路径
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在于 {input_path}")
        return

    # 2. 设置输出目录
    mesh_name = os.path.basename(input_path).split('.')[0]
    specific_output_path = os.path.join(output_base_dir, mesh_name)
    os.makedirs(specific_output_path, exist_ok=True)

    # 3. 加载、规范化并渲染网格
    try:
        mesh = normalize_mesh(trimesh.load(input_path, process=False))

        # 确保处理的是单个几何体而不是场景
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()

        image = render_single_view(
            mesh,
            radius=radius,
            image_size=image_size,
            light_intensity=light_intensity,
            num_env_lights=num_env_lights,
            return_type='pil'
        )
    except Exception as e:
        print(f"加载或渲染网格 {input_path} 时出错: {e}")
        return

    # 4. 保存渲染图像
    image_output_path = os.path.join(specific_output_path, 'rendering.png')
    image.save(image_output_path)



# ---- 全局只初始化一次 ----
rmbg_weights_dir = "./weight/RMBG-1.4"
device = "cuda" if torch.cuda.is_available() else "cpu"
rmbg_net = None

ROOT = Path("./data/raw")

def run_rmbg(input_path: str, output_root: str):
    """
    对输入图片进行背景移除，并保存结果。

    Args:
        input_path (str): 输入图像路径
        output_root (str): 输出根目录路径

    Returns:
        str: 保存后的图像路径
    """
    global rmbg_net
    if rmbg_net is None:
        rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device).eval()
    assert os.path.exists(input_path), f'{input_path} does not exist'

    # 用父目录名字作为子文件夹名
    mesh_name = os.path.basename(os.path.dirname(input_path))
    output_path = os.path.join(output_root, mesh_name)
    os.makedirs(output_path, exist_ok=True)

    rendering_rmbg = prepare_image(input_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net, device=device)
    rendering_rmbg.save(os.path.join(output_path, f'rendering_rmbg.png'))


def compute_and_save_iou(input_path: str, output_base_dir: str):
    """
    加载3D网格，计算其部件间的IoU，并将结果保存到JSON文件。

    Args:
        input_path (str): 输入的3D网格文件的路径。
        output_base_dir (str): 输出文件夹的基准目录。
    """
    # 1. 验证输入路径
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在于 {input_path}")
        return

    # 2. 设置输出目录
    mesh_name = os.path.basename(input_path).split('.')[0]
    specific_output_path = os.path.join(output_base_dir, mesh_name)
    os.makedirs(specific_output_path, exist_ok=True)
    # print(f"正在为 '{input_path}' 计算 IoU -> 保存至 '{specific_output_path}'")

    # 3. 初始化配置字典
    config = {
        'iou_mean': 0.0,
        'iou_max': 0.0,
        'iou_list': [],
    }

    # 4. 加载网格并计算 IoU
    try:
        mesh = normalize_mesh(trimesh.load(input_path, process=False))
        # 仅当部件数大于1时计算IoU才有意义
        if len(mesh.geometry) > 1:
            iou_list = compute_IoU_for_scene(mesh, return_type='iou_list')
            config['iou_list'] = iou_list
            config['iou_mean'] = np.mean(iou_list) if iou_list else 0.0
            config['iou_max'] = np.max(iou_list) if iou_list else 0.0
        else:
            return

    except Exception as e:
        # print(f"计算 IoU 时发生错误: {e}。将保存默认值。")
        # 确保在出错时，config中的值是默认的空/零值
        config['iou_list'] = []
        config['iou_mean'] = 0.0
        config['iou_max'] = 0.0

    # 5. 保存结果到JSON文件
    iou_output_path = os.path.join(specific_output_path, 'iou.json')
    with open(iou_output_path, 'w') as f:
        json.dump(config, f, indent=4)


# 读取 CSV
# file_path = './data/raw/merged_records/1758779881_downloaded_0.csv'
# df = pd.read_csv(file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/raw')
    parser.add_argument('--output', type=str, default='./data/preprocessed')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # assert os.path.exists(input_path), f'{input_path} does not exist'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # model_dirs = df['local_path'].apply(lambda x: x).unique().tolist()
    model_dirs = [path.as_posix() for path in Path(input_path).rglob("*.glb")]
    valid_meshs = []
    for mesh_name in tqdm(model_dirs):
        mesh_path = os.path.join(input_path, mesh_name)
        # 1. Sample points from mesh surface
        json_path = os.path.join(
        output_path,
        os.path.basename(mesh_name).replace(".glb", ""),
        "num_parts.json"
        )
        if os.path.exists(json_path):
            valid_meshs.append(mesh_name)
            continue
        process_single_mesh(mesh_path,output_path)
        # os.system(f"python datasets/preprocess/mesh_to_point.py --input {mesh_path} --output {output_path}")
        num_parts = json.load(open(os.path.join(output_path, os.path.basename(mesh_name).replace(".glb",""), 'num_parts.json')))['num_parts']
        if num_parts ==1 or num_parts > 20:
            continue
        # 2. Render images
        render_and_save_view(mesh_path,output_path)
        # os.system(f"python datasets/preprocess/render.py --input {mesh_path} --output {output_path}")
        # 3. Remove background for rendered images and resize to 90%
        export_mesh_folder = os.path.join(output_path, os.path.basename(mesh_name).replace('.glb', ''))
        export_rendering_path = os.path.join(export_mesh_folder, 'rendering.png')
        run_rmbg(export_rendering_path,output_path)
        # os.system(f"python datasets/preprocess/rmbg.py --input {export_rendering_path} --output {output_path}")
        # 4. (Optional) Calculate IoU
        compute_and_save_iou(mesh_path,output_path)
        # os.system(f"python datasets/preprocess/calculate_iou.py --input {mesh_path} --output {output_path}")
        valid_meshs.append(mesh_name)

    # generate configs
    configs = []
    for mesh_name in tqdm(valid_meshs):
        mesh_path = os.path.join(output_path, os.path.basename(mesh_name).replace('.glb', ''))
        num_parts_path = os.path.join(mesh_path, 'num_parts.json')
        surface_path = os.path.join(mesh_path, 'points.npy')
        image_path = os.path.join(mesh_path, 'rendering_rmbg.png')
        iou_path = os.path.join(mesh_path, 'iou.json')
        config = {
            "file": mesh_name,
            "num_parts": 0,
            "valid": False,
            "mesh_path": os.path.join(input_path, mesh_name),
            "surface_path": None,
            "image_path": None,
            "iou_mean": 0.0,
            "iou_max": 0.0
        }
        try:
            config["num_parts"] = json.load(open(num_parts_path))['num_parts']
            iou_config = json.load(open(iou_path))
            config['iou_mean'] = iou_config['iou_mean']
            config['iou_max'] = iou_config['iou_max']
            assert os.path.exists(surface_path)
            config['surface_path'] = surface_path
            assert os.path.exists(image_path)
            config['image_path'] = image_path
            config['valid'] = True
            configs.append(config)
        except:
            continue

    configs_path = os.path.join(output_path, 'object_part_configs.json')
    json.dump(configs, open(configs_path, 'w'), indent=4)
