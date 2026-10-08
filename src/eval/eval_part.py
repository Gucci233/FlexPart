import os
import argparse
import glob
import json
import sys
import numpy as np
import trimesh
import torch
from multiprocessing import Pool, cpu_count
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from tqdm import tqdm
from typing import *

# ================= Configuration =================
# Ensure the project root is in path for custom imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import metric utilities for Chamfer Distance (CD) calculation
from src.utils.metric_utils import compute_cd_and_f_score

# ================= IoU Utility Functions =================

def compute_box_iou(mesh1, mesh2):
    """
    Calculate Axis-Aligned Bounding Box (AABB) IoU between two meshes.
    """
    # Get AABB boundaries: shape (2, 3) -> [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    box1 = mesh1.bounds
    box2 = mesh2.bounds

    # 1. Calculate intersection volume
    inter_min = np.maximum(box1[0], box2[0])
    inter_max = np.minimum(box1[1], box2[1])

    # Calculate dimensions; if max < min, set to 0
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_dims)

    # 2. Calculate union volume
    vol1 = np.prod(box1[1] - box1[0])
    vol2 = np.prod(box2[1] - box2[0])

    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 0:
        return 0.0

    return inter_vol / union_vol

def compute_voxel_iou(mesh1, mesh2, pitch=0.03, threshold_ratio=1.0, max_points=1000000):
    """
    Calculate Voxel IoU based on surface occupancy.
    Uses Monte Carlo sampling if the grid is too large to maintain efficiency.
    """
    try:
        # 1. Determine common bounding box
        bounds_min = np.minimum(mesh1.bounds[0], mesh2.bounds[0])
        bounds_max = np.maximum(mesh1.bounds[1], mesh2.bounds[1])

        # Expand bounds slightly
        margin = pitch * 2
        bounds_min -= margin
        bounds_max += margin

        # 2. Generate Grid Points
        x_range = np.arange(bounds_min[0], bounds_max[0], pitch)
        y_range = np.arange(bounds_min[1], bounds_max[1], pitch)
        z_range = np.arange(bounds_min[2], bounds_max[2], pitch)

        grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        query_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

        total_points = len(query_points)
        if total_points == 0:
            return 0.0

        # Limit computation: Downsample points if count exceeds max_points
        if total_points > max_points:
            indices = np.random.choice(total_points, max_points, replace=False)
            query_points = query_points[indices]

        # 3. Determine Occupancy
        threshold = pitch * threshold_ratio

        # Sample points on mesh surfaces for KDTree
        area1, area2 = mesh1.area, mesh2.area
        n_sample1 = min(int(area1 / ((pitch/2.0)**2)) + 1000, 200000)
        n_sample2 = min(int(area2 / ((pitch/2.0)**2)) + 1000, 200000)

        p1_surf, _ = trimesh.sample.sample_surface(mesh1, n_sample1)
        p2_surf, _ = trimesh.sample.sample_surface(mesh2, n_sample2)

        tree1 = cKDTree(p1_surf)
        tree2 = cKDTree(p2_surf)

        # Query distances from grid points to surfaces
        dists1, _ = tree1.query(query_points, k=1, workers=-1)
        dists2, _ = tree2.query(query_points, k=1, workers=-1)

        occupancy1 = dists1 <= threshold
        occupancy2 = dists2 <= threshold

        # 4. Calculate IoU
        intersection = np.logical_and(occupancy1, occupancy2).sum()
        union = np.logical_or(occupancy1, occupancy2).sum()

        return intersection / union if union > 0 else 0.0

    except Exception as e:
        print(f"Voxel IoU Error: {e}")
        return 0.0

# ================= Mesh Loading & Processing =================

def load_gt_parts(glb_path):
    """Load Ground Truth parts from a GLB scene."""
    parts_meshes = []
    try:
        scene = trimesh.load(glb_path, force='scene')
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                parts_meshes.append(geom)
    except Exception: pass
    return parts_meshes

def load_gen_parts(object_glb_path):
    """Load Generated parts (either individual GLBs or from an object scene)."""
    parts_meshes = []
    dir_path = os.path.dirname(object_glb_path)
    part_files = sorted(glob.glob(os.path.join(dir_path, "part_*.glb")))

    if len(part_files) > 0:
        for p_file in part_files:
            try:
                mesh = trimesh.load(p_file, force='mesh')
                parts_meshes.append(mesh)
            except Exception: pass
    else:
        try:
            scene_path = sorted(glob.glob(os.path.join(dir_path, "object.glb")))[0]
            scene = trimesh.load(scene_path, force='scene')
            for name, geom in scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    parts_meshes.append(geom)
        except Exception: pass
    return parts_meshes

def normalize_mesh_and_get_transform(mesh, scale=2.0):
    """Normalize mesh to unit scale and return the transform matrix."""
    bbox = mesh.bounding_box
    translation = -bbox.centroid
    s = scale / bbox.primitive.extents.max() if bbox.primitive.extents.max() > 0 else 1.0
    T_trans = np.eye(4); T_trans[:3, 3] = translation
    T_scale = np.eye(4); T_scale[:3, :3] *= s
    final_transform = np.dot(T_scale, T_trans)
    mesh.apply_transform(final_transform)
    return mesh, final_transform

def process_and_align_lists(gt_parts, gen_parts):
    """Normalize and align Generated parts to GT parts using ICP."""
    gt_aligned = [p.copy() for p in gt_parts]
    gen_aligned = [p.copy() for p in gen_parts]

    if not gt_aligned or not gen_aligned:
        return gt_aligned, gen_aligned

    # Normalize globally based on concatenated meshes
    full_gt = trimesh.util.concatenate(gt_aligned)
    full_gen = trimesh.util.concatenate(gen_aligned)

    full_gt, T_gt = normalize_mesh_and_get_transform(full_gt)
    for p in gt_aligned: p.apply_transform(T_gt)

    full_gen, T_gen = normalize_mesh_and_get_transform(full_gen)
    for p in gen_aligned: p.apply_transform(T_gen)

    # Align generated mesh to GT using ICP
    try:
        icp_res = trimesh.registration.mesh_other(full_gen, full_gt, samples=2000, scale=False)
        align_matrix = icp_res[0]
    except Exception:
        align_matrix = np.eye(4)

    for p in gen_aligned: p.apply_transform(align_matrix)
    return gt_aligned, gen_aligned

# ================= Core Metric Calculation =================

def compute_bijective_matching_metrics(gt_parts, gen_parts):
    """
    1. Perform Hungarian Matching based on Chamfer Distance (CD).
    2. Calculate CD, Box IoU, and Voxel IoU for matched pairs.
    """
    num_gt = len(gt_parts)
    num_gen = len(gen_parts)
    cost_matrix = np.zeros((num_gt, num_gen))

    # Build cost matrix using CD
    for i in range(num_gt):
        for j in range(num_gen):
            dist, _ = compute_cd_and_f_score(gt_parts[i], gen_parts[j], threshold=0.1)
            cost_matrix[i, j] = dist.item() if isinstance(dist, torch.Tensor) else dist

    # Hungarian Matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = list(zip(row_ind, col_ind))

    results = {"cd": [], "box_iou": [], "voxel_iou": []}
    for r, c in matches:
        mesh_gt = gt_parts[r]
        mesh_gen = gen_parts[c]

        results["cd"].append(cost_matrix[r, c])
        results["box_iou"].append(compute_box_iou(mesh_gt, mesh_gen))
        results["voxel_iou"].append(compute_voxel_iou(mesh_gt, mesh_gen, pitch=0.03))

    return matches, results

def process_single_item(item):
    """Wrapper function for processing a single object (used for multiprocessing)."""
    try:
        # Resolve paths
        image_path = item['image_path']
        base_dir = os.path.dirname(image_path)
        gt_mesh_path = os.path.join(base_dir, "normalized_rotated_scene.glb")

        # Load and Align
        gt_meshes = load_gt_parts(gt_mesh_path)
        gen_meshes = load_gen_parts(item['pred_mesh_path'])

        if not gt_meshes or not gen_meshes:
            return None

        gt_meshes, gen_meshes = process_and_align_lists(gt_meshes, gen_meshes)

        if not gt_meshes or not gen_meshes:
            return None

        # Calculate metrics
        matches, metrics = compute_bijective_matching_metrics(gt_meshes, gen_meshes)

        return {
            "cd": np.mean(metrics["cd"]) if metrics["cd"] else 0,
            "box_iou": np.mean(metrics["box_iou"]) if metrics["box_iou"] else 0,
            "voxel_iou": np.mean(metrics["voxel_iou"]) if metrics["voxel_iou"] else 0,
            "file": item['file']
        }
    except Exception as e:
        print(f"Error processing {item.get('file', 'unknown')}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate part meshes using metadata with pred_mesh_path entries.")
    parser.add_argument('--json_path', required=True)
    args = parser.parse_args()
    JSON_PATH = args.json_path

    with open(JSON_PATH, 'r') as f:
        data_list = json.load(f)

    all_stats = {"cd": [], "box_iou": [], "voxel_iou": []}

    # Setup multiprocessing
    num_processes = max(1, cpu_count() - 10)
    print(f"Start processing with {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_single_item, data_list), total=len(data_list), desc="Evaluating"))

    # Aggregate results
    valid_count = 0
    for res in results:
        if res is None:
            continue
        valid_count += 1
        all_stats["cd"].append(res["cd"])
        all_stats["box_iou"].append(res["box_iou"])
        all_stats["voxel_iou"].append(res["voxel_iou"])

    # Output Final Results
    if all_stats["cd"]:
        print("=" * 40)
        print("FINAL DATASET METRICS (Average of Matched Parts per Object):")
        print(f"Processed Objects   : {valid_count}/{len(data_list)}")
        print(f"Mean CD             : {np.mean(all_stats['cd']):.6f}")
        print(f"Mean Box IoU        : {np.mean(all_stats['box_iou']):.6f}")
        print(f"Mean Voxel IoU      : {np.mean(all_stats['voxel_iou']):.6f}")
        print("=" * 40)
    else:
        print("No valid results computed.")

    print(f"Source JSON: {JSON_PATH}")

if __name__ == "__main__":
    main()
