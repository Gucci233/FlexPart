import argparse

parser = argparse.ArgumentParser(description="FlexPart interactive point/box demo.")
parser.add_argument('--model_path', default='./weight/FlexPart')
parser.add_argument('--rmbg_model_path', default='./weight/RMBG-1.4')
parser.add_argument('--output_dir', default='./output_gradio')
parser.add_argument('--server_name', default='127.0.0.1')
parser.add_argument('--server_port', type=int, default=7860)
args = parser.parse_args() if __name__ == '__main__' else parser.parse_args([])

import gradio as gr
from gradio_image_prompter import ImagePrompter
import os
import sys
import numpy as np
import torch
import trimesh
from PIL import Image
from accelerate.utils import set_seed
import torch.nn.functional as F
from torchvision import transforms
import time # 引入time用于模拟状态切换的流畅感

from src.utils.data_utils import get_colored_mesh_composition
from src.pipelines.pipeline_flexpart import FlexPartPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG
from kiui.mesh_utils import clean_mesh, decimate_mesh
from typing import Union

DEVICE = "cuda"
DTYPE = torch.float16
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError('The demo requires an NVIDIA GPU with CUDA support.')

rmbg_model_path = args.rmbg_model_path
rmbg_net = BriaRMBG.from_pretrained(rmbg_model_path).to(DEVICE).eval()

flexpart_model_path = args.model_path
pipe = FlexPartPipeline.from_pretrained(flexpart_model_path)
pipe = pipe.to(DEVICE, DTYPE)

def normalize_mesh(mesh: Union[trimesh.Trimesh, trimesh.Scene], scale: float = 2.0):
    bbox = mesh.bounding_box
    translation = -bbox.centroid
    scale = scale / bbox.primitive.extents.max()
    mesh.apply_translation(translation)
    mesh.apply_scale(scale)
    return mesh

def postprocess_mesh(mesh: trimesh.Trimesh, decimate_target=100000):
    mesh = filter_mesh(mesh)
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

    filtered = [m for m in submeshes if len(m.faces) > max_faces*0.2]

    if len(filtered) == 1:
        return filtered[0]

    merged = trimesh.util.concatenate(filtered)
    merged.merge_vertices()
    _ = merged.vertex_normals
    return merged

def preprocess_on_upload(prompter_input):
    if not prompter_input or "image" not in prompter_input or prompter_input["image"] is None:
        gr.Warning("Image load failed or empty.")
        return None
    
    print("🧹 Removing background...")
    original_image = prompter_input["image"]
    
    try:
        pil_image = Image.fromarray(original_image).convert("RGB")
        w, h = pil_image.size
        
        im_np = np.array(pil_image)
        im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        im_tensor = torch.div(im_tensor, 255.0)
        im_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(im_tensor)
        
        im_tensor = F.interpolate(im_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            result = rmbg_net(im_tensor)
        
        mask_tensor = result
        while isinstance(mask_tensor, (list, tuple)):
            mask_tensor = mask_tensor[0]
        
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
            
        mask_tensor = F.interpolate(mask_tensor, size=(h, w), mode='bilinear', align_corners=False)
        
        # BriaRMBG already returns sigmoid probabilities.
        ma = mask_tensor.squeeze().clamp(0, 1).cpu().numpy()
        
        pil_image.putalpha(Image.fromarray((ma * 255).astype(np.uint8)))
        
        new_value = {
            "image": np.array(pil_image),
            "points": [],
            "boxes": []
        }
        print("✅ Background removed")
        return new_value
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Background removal failed: {e}")
        return prompter_input

def parse_prompter_data(prompter_input):
    if not prompter_input or not isinstance(prompter_input, dict):
        return [], []

    raw_list = prompter_input.get("points", [])
    points = []
    boxes = []

    for item in raw_list:
        if len(item) < 3: continue
        marker = item[2] 
        if marker == 1.0:
            points.append([item[0], item[1]])
        elif marker == 2.0:
            if len(item) >= 5:
                x1, y1, x2, y2 = item[0], item[1], item[3], item[4]
                if abs(x2-x1) * abs(y2-y1) > 10:
                    boxes.append([x1, y1, x2, y2])
    return points, boxes

def get_final_prompts(prompter_input, width, height):
    raw_points, raw_boxes = parse_prompter_data(prompter_input)

    points_norm = [{"pixel": p, "norm": (p[0]/width, p[1]/height)} for p in raw_points]
    boxes_norm = [{"pixel": b, "norm": (b[0]/width, b[1]/height, b[2]/width, b[3]/height)} for b in raw_boxes]

    final_boxes_prompt = []
    final_points_prompt = []
    used_pts = set()
    
    for box in boxes_norm:
        bx1, by1, bx2, by2 = box['pixel']
        contained_pt = None
        for idx, pt in enumerate(points_norm):
            px, py = pt['pixel']
            if bx1 <= px <= bx2 and by1 <= py <= by2:
                contained_pt = pt
                used_pts.add(idx)
                break
        
        if contained_pt is None:
            cx = (box['norm'][0] + box['norm'][2]) / 2
            cy = (box['norm'][1] + box['norm'][3]) / 2
            final_points_prompt.append((cx, cy))
        else:
            final_points_prompt.append(contained_pt['norm'])
        final_boxes_prompt.append(box['norm'])

    for idx, pt in enumerate(points_norm):
        if idx not in used_pts:
            final_points_prompt.append(pt['norm'])
            final_boxes_prompt.append(None) 

    return final_boxes_prompt, final_points_prompt

@torch.no_grad()
def generate_3d_mesh_generator(prompter_input, seed, guidance_scale, steps):
    if not prompter_input or "image" not in prompter_input or prompter_input["image"] is None:
        raise gr.Error("Please upload an image first!")
    
    yield gr.update(visible=True, value="<div style='display: flex; justify-content: center; align-items: center; height: 300px; font-size: 24px;'>🚀 Generating 3D Mesh...<br>(Please wait)</div>"), gr.update(visible=False)

    image_input_np = prompter_input["image"] 
    h, w = image_input_np.shape[:2]
    
    boxes_list, points_list = get_final_prompts(prompter_input, w, h)
    num_parts = len(points_list)
    
    if num_parts == 0:
        raise gr.Error("No annotations detected! Please click on the image to add points or boxes.")
    if num_parts > pipe.transformer.config.max_num_parts:
        raise gr.Error("The number of annotated parts exceeds the model capacity.")

    print(f"🚀 Starting inference: {num_parts} Parts")

    processed_boxes = []
    valid_boxes = []
    for b in boxes_list:
        if b is None:
            processed_boxes.append((0.0, 0.0, 1.0, 1.0))
            valid_boxes.append(False)
        else:
            processed_boxes.append(b)
            valid_boxes.append(True)

    prompts = {
        "masks_prompt": None,
        "boxes_prompt": processed_boxes,
        "points_prompt": points_list,
        "valid_masks": [False] * num_parts,
        "valid_boxes": valid_boxes,
        "valid_points": [True] * num_parts,
    }

    pil_image = Image.fromarray(image_input_np)
    
    if pil_image.mode == 'RGBA':
        white_bg = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_bg.paste(pil_image, mask=pil_image.split()[3])
        final_image = white_bg
    else:
        print("⚠️ Warning: Input image has no alpha channel, using original image")
        final_image = pil_image.convert("RGB")

    set_seed(int(seed))
    
    outputs = pipe(
        image=[final_image] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=1024,
        generator=torch.Generator(device=DEVICE).manual_seed(int(seed)),
        num_inference_steps=int(steps),
        guidance_scale=guidance_scale,
        use_flash_decoder=False,
        **prompts
    ).meshes

    if any(m is None or len(m.faces) == 0 for m in outputs):
        raise gr.Error("Mesh extraction returned an empty part. Try another seed or prompt.")
    clean_outputs = [filter_mesh(m) for m in outputs]
    merged_mesh = get_colored_mesh_composition(clean_outputs)
    merged_mesh = normalize_mesh(merged_mesh)
    save_path = os.path.join(OUTPUT_DIR, "latest_result.glb")
    merged_mesh.export(save_path)
    
    print(f"✅ Generation complete, saved to {save_path}")

    yield gr.update(value="<div style='display: flex; justify-content: center; align-items: center; height: 300px; font-size: 24px; color: green;'>✅ Generation Complete!<br>⏳ Loading Viewer...</div>"), gr.update(visible=False)
    
    yield gr.update(visible=False), gr.update(visible=True, value=save_path)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧩 FlexPart 3D")

    with gr.Row():
        with gr.Column(scale=1):
            prompter = ImagePrompter(label="Annotation Area (Auto BG Removal)", interactive=True)
            
            with gr.Accordion("Settings", open=False):
                seed = gr.Number(value=1000, label="Seed", precision=0)
                scale = gr.Slider(1.0, 20.0, value=7.0, label="Guidance Scale")
                steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Steps")

            gen_btn = gr.Button("🚀 Generate 3D Mesh", variant="primary")

        with gr.Column(scale=1):
            # 布局技巧：
            # 1. status_overlay: 用于显示大字文字状态，初始隐藏
            # 2. result_3d: 用于显示模型，初始显示（空）
            # 生成时，二者可见性互换
            
            status_overlay = gr.HTML(
                value="", 
                visible=False, 
                label="Status"
            )
            
            result_3d = gr.Model3D(
                label="3D Preview", 
                clear_color=[1.0, 1.0, 1.0, 1.0], 
                camera_position=(180, 90, 1),
                visible=True
            )

    examples_list = [
        [{"image": "./assets/1.png", "points": []}],
        [{"image": "./assets/2.png", "points": []}],
        [{"image": "./assets/3.png", "points": []}]
    ]
    
    gr.Examples(
        examples=examples_list,
        inputs=[prompter],
        label="Examples (Click to load, then annotate manually)"
    )

    prompter.upload(
        fn=preprocess_on_upload,
        inputs=[prompter],
        outputs=[prompter]
    )

    # 使用 yield 机制的 generator 函数，可以多次更新 outputs
    gen_btn.click(
        fn=generate_3d_mesh_generator,
        inputs=[prompter, seed, scale, steps],
        outputs=[status_overlay, result_3d]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name=args.server_name, server_port=args.server_port)
