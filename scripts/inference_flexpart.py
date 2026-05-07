import argparse
import os
import sys
import time
import numpy as np
import torch
import trimesh
from PIL import Image
from accelerate.utils import set_seed
from typing import Any, List, Tuple, Optional

sys.path.append("./")
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import (
    render_views_around_mesh, 
    render_normal_views_around_mesh, 
    make_grid_for_images_or_videos, 
    export_renderings
)
from src.pipelines.pipeline_flexpart import FlexPartPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG
from src.models.transformers.flexpart_transformer import FlexPartDiTModel

def load_data(path):
    if path is None or not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def process_prompts(masks, boxes, points, img_size):
    h, w = img_size
    if masks is not None: num_parts = len(masks)
    elif boxes is not None: num_parts = len(boxes)
    elif points is not None: num_parts = len(points)
    else: return [], [], []
    
    final_masks = masks if masks is not None else [None] * num_parts
    final_boxes = []
    final_points = []

    for i in range(num_parts):
        curr_m = masks[i] if masks is not None else None
        curr_b = boxes[i] if boxes is not None else None
        curr_p = points[i] if points is not None else None

        if curr_m is not None:
            y_idx, x_idx = np.where(curr_m)
            if len(y_idx) > 0:
                x1, y1, x2, y2 = x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()
                
                mean_y, mean_x = np.mean(y_idx), np.mean(x_idx)
                dists_sq = (y_idx - mean_y)**2 + (x_idx - mean_x)**2
                best_idx = np.argmin(dists_sq)
                cx, cy = x_idx[best_idx], y_idx[best_idx]
            else:
                x1, y1, x2, y2, cx, cy = 0, 0, 0, 0, 0, 0
        
        elif curr_b is not None:
            x1, y1, x2, y2 = curr_b
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0
            cx, cy = curr_p if curr_p is not None else (0, 0)

        final_boxes.append((
            np.clip((float(x1) + 0.5) / w, 0.0, 1.0),
            np.clip((float(y1) + 0.5) / h, 0.0, 1.0),
            np.clip((float(x2) + 0.5) / w, 0.0, 1.0),
            np.clip((float(y2) + 0.5) / h, 0.0, 1.0)
        ))
        final_points.append((
            np.clip((float(cx) + 0.5) / w, 0.0, 1.0),
            np.clip((float(cy) + 0.5) / h, 0.0, 1.0)
        ))

    return final_masks, final_boxes, final_points

@torch.no_grad()
def run_inference(pipe, image_input, args, prompts):
    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net) if args.rmbg else image_input.convert("RGBA")
    num_parts = len(prompts["masks_prompt"])
    
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=args.num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(args.seed),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        use_flash_decoder=args.use_flash_decoder,
        **prompts
    ).meshes
    return [m if m is not None else trimesh.Trimesh(vertices=[[0,0,0]], faces=[[0,0,0]]) for m in outputs], img_pil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/path/to/your/model/PartObjaverse-Tiny_Test_fix_final/1d97ec6885464f6a9ce7a4cc955bff41/rendering.png")
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--box_path", type=str, default=None)
    parser.add_argument("--point_path", type=str, default=None)
    parser.add_argument("--box3d_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output_single")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--rmbg", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use_flash_decoder", action="store_true")
    args = parser.parse_args()

    device, dtype = "cuda", torch.float16
    set_seed(args.seed)

    rmbg_net = BriaRMBG.from_pretrained("./pretrained_weights/RMBG-1.4").to(device).eval()
    pipe = FlexPartPipeline.from_pretrained("./pretrained_weights/FlexPart")
    pipe = pipe.to(device, dtype)


    raw_img = Image.open(args.image_path)
    surface_data = np.load("/path/to/your/model/PartObjaverse-Tiny_Test_fix_final/1d97ec6885464f6a9ce7a4cc955bff41/points.npy", allow_pickle=True).item()
    masks_in = [mask['2d_mask'] for mask in surface_data['parts']]
    boxes_in = None
    points_in = None
    
    final_masks, final_boxes, final_points = process_prompts(masks_in, boxes_in, points_in, (raw_img.height, raw_img.width))
    num_parts = len(final_masks)

    prompts = {
        "masks_prompt": final_masks,
        "boxes_prompt": final_boxes,
        "points_prompt": final_points,
        "valid_masks": [m is not None for m in final_masks],
        "valid_boxes": [True] * num_parts,
        "valid_points": [True] * num_parts,
    }

    outputs, processed_img = run_inference(pipe, raw_img, args, prompts)
    tag = os.path.basename(args.image_path).split('.')[0]
    export_dir = os.path.join(args.output_dir, tag)
    os.makedirs(export_dir, exist_ok=True)

    for i, m in enumerate(outputs):
        m.export(os.path.join(export_dir, f"part_{i:02}.glb"))
    get_colored_mesh_composition(outputs).export(os.path.join(export_dir, "object.glb"))

    if args.render:
        imgs = render_views_around_mesh(get_colored_mesh_composition(outputs), num_views=36, radius=4)
        export_renderings(imgs, os.path.join(export_dir, "rendering.gif"), fps=18)