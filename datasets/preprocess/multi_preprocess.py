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
sys.path.append("./")
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG
from src.utils.data_utils import scene_to_parts, mesh_to_surface, normalize_mesh,get_camera,\
get_visible_part_masks,visualize_masks,filter_scene_meshes,mask_to_box,count_non_white_pixels
from src.utils.render_utils import render_single_view,render_cube_views_mesh
from src.utils.metric_utils import compute_IoU_for_scene
DEFAULT_RADIUS = 4.0
DEFAULT_IMAGE_SIZE = (2048, 2048)
DEFAULT_LIGHT_INTENSITY = 2.5
DEFAULT_NUM_ENV_LIGHTS = 36
RMBG_WEIGHTS_DIR = "./pretrained_weights/RMBG-1.4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEBUG_COLOR_PALETTE = [
    (255, 0, 0),   # 1. Bright red
    (0, 255, 0),   # 2. Bright green
    (0, 0, 255),   # 3. Bright blue
    (255, 255, 0), # 4. Bright yellow
    (0, 255, 255), # 5. Bright cyan
    (255, 0, 255), # 6. Bright magenta
    (255, 128, 0), # 7. Orange
    (128, 0, 255), # 8. Purple
    (0, 128, 0),   # 9. Dark green
    (255, 0, 128), # 10. Pink
    (0, 128, 128), # 11. Teal
    (128, 128, 0), # 12. Olive
]

def get_mesh_rotation_matrix(camera_pose: np.ndarray) -> np.ndarray:
    """
    Calculate the rotation matrix needed for Mesh based on camera's c2w pose.
    Logic: Mesh rotation = inverse of camera rotation (Inverse/Transpose)
    """
    R_camera = camera_pose[:3, :3]
    
    R_mesh = R_camera.T
    
    mesh_transform = np.eye(4)
    mesh_transform[:3, :3] = R_mesh
    
    return mesh_transform

def process_mesh_to_data(input_path: str, output_base_dir: str):
    """
    Ultimate simplified version: directly extract OBB data, no OBB image rendering, fully cohesive logic.
    """
    mesh_name = Path(input_path).stem
    output_dir = Path(output_base_dir) / mesh_name
    
    try:
        raw_mesh = trimesh.load(input_path, process=False)
        norm_mesh = normalize_mesh(filter_scene_meshes(raw_mesh))
        
        parts_count = len(norm_mesh.geometry) if isinstance(norm_mesh, trimesh.Scene) else 1
        if not (0 < parts_count <= 20):
            return None

        render_kwargs = {"radius": DEFAULT_RADIUS, "image_size": DEFAULT_IMAGE_SIZE}
        flat_geometry = norm_mesh.to_geometry() if isinstance(norm_mesh, trimesh.Scene) else norm_mesh
        cube_images = render_cube_views_mesh(flat_geometry, return_type='ndarray', **render_kwargs)
        
        pixel_counts = [count_non_white_pixels(img) for img in cube_images]
        best_view_idx = np.argmax(pixel_counts)
        best_render_img = cube_images[best_view_idx]

        camera, poses = get_camera(image_size=DEFAULT_IMAGE_SIZE)
        rotation_matrix = get_mesh_rotation_matrix(poses[best_view_idx])
        norm_mesh.apply_transform(rotation_matrix)
        
        final_flat_mesh = norm_mesh.to_geometry() if isinstance(norm_mesh, trimesh.Scene) else norm_mesh

        
        parts_data_list = scene_to_parts(norm_mesh, return_type="point", normalize=False)
        
        masks = get_visible_part_masks(
            mesh_scene=norm_mesh,
            camera_pose=poses[0],
            camera=camera,
            image_size=DEFAULT_IMAGE_SIZE
        )
        
        valid_data_entries = [] # store (mask_size, part_dict, obb_vector)
        min_mask_size = 50

        for i, part_data in enumerate(parts_data_list):
            current_mask = masks[i]
            mask_size = np.sum(current_mask)
            
            if mask_size < min_mask_size:
                continue

            try:
                points = part_data.get('points')
                if points is None or len(points) < 4:
                    obb_vector = np.zeros(10, dtype=np.float32)
                else:
                    pc = trimesh.PointCloud(points)
                    obb = pc.bounding_box_oriented
                    
                    center = obb.centroid
                    extents = obb.extents
                    transform = obb.transform
                    quaternion = trimesh.transformations.quaternion_from_matrix(transform) # [w, x, y, z]
                    
                    obb_vector = np.concatenate([center, extents, quaternion]).astype(np.float32)
            except Exception as e:
                print(f"OBB calculation warning [{mesh_name} - part {i}]: {e}")
                obb_vector = np.zeros(10, dtype=np.float32)

            part_data['2d_mask'] = current_mask
            part_data['2d_box'] = mask_to_box(current_mask)
            
            valid_data_entries.append({
                "size": mask_size,
                "part": part_data,
                "obb": obb_vector
            })

        valid_data_entries.sort(key=lambda x: x["size"], reverse=True)

        final_parts = [entry["part"] for entry in valid_data_entries]
        final_obbs = np.array([entry["obb"] for entry in valid_data_entries])

        if not final_parts:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(best_render_img).save(output_dir / "rendering.png")
        visualize_masks([p['2d_mask'] for p in final_parts]).save(output_dir / "mask.png")

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
        print(f"Processing failed [{input_path}]: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def run_cpu_tasks_for_mesh(mesh_path: str, output_base_dir: str):
    try:
        render_kwargs = {"radius": DEFAULT_RADIUS, "image_size": DEFAULT_IMAGE_SIZE, "light_intensity": DEFAULT_LIGHT_INTENSITY, "num_env_lights": DEFAULT_NUM_ENV_LIGHTS}
        config = process_mesh_to_data(mesh_path, output_base_dir)
        return config
    except Exception as e:
        print(e)
        return None
    
def validate_obbs(part_obbs) :
    """
    Check if OBB NumPy array (N, 10) is valid (not None, correct shape, and no all-zero rows).
    """
    if part_obbs is None:
        return False # Load failed or key not exist, consider invalid
    if not isinstance(part_obbs, np.ndarray) or part_obbs.ndim != 2 or part_obbs.shape[1] != 10:
        print(f"Warning: OBB format incorrect (shape: {part_obbs.shape if isinstance(part_obbs, np.ndarray) else 'N/A'}), consider invalid.")
        return False
    if part_obbs.shape[0] == 0:
         print(f"Warning: OBB array empty (shape: {part_obbs.shape}), consider invalid.")
         return False

    if np.any(np.all(part_obbs == 0, axis=1)):
        return False
    else:
        return True

def process_single_marker(marker_path, input_dir):
    """
    Process a single marker file, validate and generate corresponding config dict.
    Assume all original files are in the unique input_dir.
    """
    try:
        mesh_output_path = marker_path.parent
        mesh_name = mesh_output_path.name.split('.')[0]
        
        image_path = mesh_output_path / 'rendering.png'
        surface_path = mesh_output_path / 'points.npy'
        obb_image_path = mesh_output_path / 'render_OBB.png'
        mask_path = mesh_output_path / 'mask.png'
        if not all(p.exists() for p in [image_path, surface_path,obb_image_path,mask_path]):
            return None
        base_path = Path(input_dir)
        original_mesh_path = None
        
        path_glb = base_path  / f"{mesh_name}.glb"
        if path_glb.is_file():
            original_mesh_path = str(path_glb)
        else:
            path_obj = base_path / f"{mesh_name}.obj"
            if path_obj.is_file():
                original_mesh_path = str(path_obj)
        
        if not original_mesh_path:
            return None # Original file not found, return None

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
            "file": os.path.relpath(original_mesh_path, start=input_dir), # Directly use input_dir to compute relative path
            "mesh_path": original_mesh_path,
            "surface_path": str(surface_path),
            "image_path": str(image_path),
            "obb_image_path": str(obb_image_path),
            "mask_path":str(mask_path),
            "num_parts": num_parts,
            "valid": True,
        }
        return config
    except Exception:
        return None



def check_single_model(mesh_path_output):
    """Check if single mesh needs reprocessing"""
    mesh_path, output_root = mesh_path_output
    mesh_name = os.path.basename(mesh_path).split('.')[0]

    render_path = os.path.join(output_root, mesh_name, 'rendering.png')
    points_path = os.path.join(output_root, mesh_name, 'points.npy')

    if not os.path.exists(render_path) or not os.path.exists(points_path):
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
                desc="Checking CPU task progress",
                dynamic_ncols=True
            )
        )

    cpu_todo_paths = [r for r in results if r is not None]
    return cpu_todo_paths


def main():
    parser = argparse.ArgumentParser(description="Perform interruptible, parallel preprocessing on 3D model dataset.")
    parser.add_argument('--input', type=str, default='/path/to/your/data', help="Root directory containing .glb files.")
    parser.add_argument('--output', type=str, default='/path/to/your/process', help="Output directory for preprocessed files.")
    parser.add_argument('--workers', type=int, default=1, help="Number of worker processes to use. Defaults to system CPU cores.")
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Finding all .glb files...")
    all_model_paths = [path.as_posix() for path in Path(args.input).rglob("*.glb")]
    print(f"Found {len(all_model_paths)} models in total.")

    cpu_todo_paths = collect_cpu_todo(all_model_paths, args.output)
            
            
    print(f"Check complete. {len(all_model_paths) - len(cpu_todo_paths)} tasks completed, {len(cpu_todo_paths)} tasks remaining.")
    if not cpu_todo_paths:
        print("All CPU tasks completed or not needed.")
    else:
        num_workers = args.workers if args.workers is not None else cpu_count()-5
        print(f"Using {num_workers} worker processes.")
        
        worker_func = partial(run_cpu_tasks_for_mesh, output_base_dir=args.output)
        
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(cpu_todo_paths), desc="CPU task progress") as pbar:
                for _ in pool.imap_unordered(worker_func, cpu_todo_paths):
                    pbar.update(1)
        print(f"\nCPU task phase completed.")


    
    output_path = Path(args.output)
    all_processed_markers = list(output_path.glob('*/num_parts.json'))
    if not all_processed_markers:
        print("No processed marker files found, program exits.")
        return

    num_processes = cpu_count() - 1 or 1 
    print(f"Using {num_processes} processes for parallel processing...")

    worker_func = partial(process_single_marker, input_dir=args.input)
    
    final_configs = []
    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(worker_func, all_processed_markers)
        
        for config in tqdm(results_iterator, total=len(all_processed_markers), desc="Generating configs"):
            if config:
                final_configs.append(config)
                
    configs_path = output_path.parent / 'object_part_configs_new.json'
    print(f"Saving {len(final_configs)} valid configs to {configs_path}...")
    with open(configs_path, 'w', encoding='utf-8') as f:
        json.dump(final_configs, f, indent=4, ensure_ascii=False)
    
    print("All done!")

if __name__ == '__main__':
    main()