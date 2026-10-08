import os
import argparse
import json
import numpy as np
import trimesh
import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= Configuration =================
# Add project root to sys.path for custom utility imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metric_utils import compute_cd_and_f_score
from src.utils.data_utils import normalize_mesh

def align_mesh_with_icp(gen_mesh, gt_mesh):
    """
    Fine-tune alignment using the Iterative Closest Point (ICP) algorithm.
    """
    # Pre-alignment: Center centroids to aid ICP convergence
    gen_mesh.vertices -= gen_mesh.centroid
    gt_mesh.vertices -= gt_mesh.centroid

    try:
        # Compute best transform from gen_mesh (source) to gt_mesh (target)
        matrix, cost = trimesh.registration.mesh_other(
            gen_mesh,
            gt_mesh,
            samples=2000,
            scale=True
        )
        gen_mesh.apply_transform(matrix)
    except Exception as e:
        print(f"ICP Registration Failed: {e}")

    return gen_mesh

def process_single_mesh(args):
    """
    Function to process a single mesh task, designed for multiprocessing.
    """
    gen_path, gt_path, num_samples, fscore_threshold = args
    mesh_name = os.path.basename(os.path.dirname(gen_path)) # Parent folder as ID

    try:
        # 1. Load and Normalize Meshes
        gen_mesh = trimesh.load(gen_path, force='mesh', process=False)
        gen_mesh = normalize_mesh(gen_mesh)

        gt_mesh = trimesh.load(gt_path, force='mesh', process=False)
        gt_mesh = normalize_mesh(gt_mesh)

        # 2. Alignment
        gen_mesh = align_mesh_with_icp(gen_mesh, gt_mesh)

        # Handle cases where trimesh loads files as Scenes
        if isinstance(gen_mesh, trimesh.Scene):
            gen_mesh = trimesh.util.concatenate(gen_mesh.dump(concatenate=True))
        if isinstance(gt_mesh, trimesh.Scene):
            gt_mesh = trimesh.util.concatenate(gt_mesh.dump(concatenate=True))

        # 3. Compute Metrics
        cd, fscore = compute_cd_and_f_score(
            gen_mesh,
            gt_mesh,
            num_samples=num_samples,
            threshold=fscore_threshold
        )

        if np.isnan(fscore):
            fscore = 0.0

        return {
            'name': mesh_name,
            'cd': cd,
            'fscore': fscore,
            'status': 'success'
        }

    except Exception as e:
        return {
            'name': mesh_name,
            'error': str(e),
            'status': 'error'
        }

def evaluate_dataset_multiprocess(
    gen_root_dir: str,
    json_meta_path: str,
    num_samples: int = 10000,
    fscore_threshold: float = 0.01
):
    # Step 1: Parse Meta JSON and build GT lookup dictionary
    print(f"Loading GT Metadata: {json_meta_path} ...")
    with open(json_meta_path, 'r') as f:
        meta_data = json.load(f)

    gt_lookup = {}
    for item in meta_data:
        ref_path = item.get('mesh_path')
        if ref_path:
            mesh_name = Path(ref_path).stem
            # Constructing path to the GT mesh
            gt_lookup[mesh_name] = os.path.join(os.path.dirname(item['image_path']), "normalized_rotated_scene.glb")

    print(f"Indexed {len(gt_lookup)} GT models.")

    # Step 2: Search for generated object.glb files and prepare tasks
    print(f"Searching for generated meshes in: {gen_root_dir} ...")
    gen_files = list(Path(gen_root_dir).rglob("object.glb"))
    print(f"Found {len(gen_files)} generated object.glb files.")

    tasks = []
    for gen_path in gen_files:
        mesh_name = gen_path.parent.name
        if mesh_name in gt_lookup:
            gt_path = gt_lookup[mesh_name]
            if os.path.exists(gt_path):
                tasks.append((str(gen_path), gt_path, num_samples, fscore_threshold))

    print(f"Total valid evaluation tasks: {len(tasks)}")
    if not tasks:
        print("No tasks to execute. Exiting.")
        return

    # Step 3: Multiprocess Execution
    num_workers = max(1, cpu_count() - 10)
    print(f"Starting evaluation with {num_workers} processes...")

    results_list = []
    with Pool(processes=num_workers) as pool:
        results_list = list(tqdm(pool.imap(process_single_mesh, tasks), total=len(tasks), desc="Evaluating"))

    # Step 4: Aggregate and Report Results
    final_metrics = {'cd': [], 'fscore': []}
    error_count = 0

    for res in results_list:
        if res['status'] == 'success':
            final_metrics['cd'].append(res['cd'])
            final_metrics['fscore'].append(res['fscore'])
        else:
            error_count += 1

    print(f"\nProcessing complete. Success: {len(final_metrics['cd'])}, Failed: {error_count}")

    if final_metrics['cd']:
        mean_cd = np.mean(final_metrics['cd'])
        mean_fscore = np.mean(final_metrics['fscore'])

        print("\n" + "="*40)
        print("         EVALUATION RESULTS")
        print("="*40)
        print(f"Evaluated Items : {len(final_metrics['cd'])}")
        print(f"Mean CD         : {mean_cd:.6f}")
        print(f"Mean F-Score    : {mean_fscore:.6f} (threshold={fscore_threshold})")
        print("="*40)
    else:
        print("No valid results computed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated whole meshes against preprocessed ground truth.")
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--gen_root', required=True)
    parser.add_argument('--num_samples', type=int, default=20480)
    parser.add_argument('--fscore_threshold', type=float, default=0.1)
    args = parser.parse_args()
    JSON_PATH = args.json_path
    GEN_ROOT = args.gen_root

    print(f"Input Generation Root: {GEN_ROOT}")

    evaluate_dataset_multiprocess(
        gen_root_dir=GEN_ROOT,
        json_meta_path=JSON_PATH,
        num_samples=args.num_samples,
        fscore_threshold=args.fscore_threshold
    )
