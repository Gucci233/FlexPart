import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D parts from an image and pixel-space prompts.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_path", default="./weight/FlexPart")
    parser.add_argument("--rmbg_model_path", default="./weight/RMBG-1.4")
    parser.add_argument("--mask_path")
    parser.add_argument("--box_path")
    parser.add_argument("--point_path")
    parser.add_argument("--output_dir", default="./output_single")
    parser.add_argument("--tag", help="Output subdirectory name; defaults to the image stem.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--rmbg", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use_flash_decoder", action="store_true")
    args = parser.parse_args()
    if args.num_tokens < 1 or args.num_inference_steps < 1 or args.seed < 0:
        parser.error("Tokens and steps must be positive; seed must be nonnegative.")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error("--guidance_scale must be finite and nonnegative.")
    if not any((args.mask_path, args.box_path, args.point_path)):
        parser.error("Supply at least one prompt file: --mask_path, --box_path, or --point_path.")
    if args.tag and (args.tag in (".", "..") or "/" in args.tag or "\\" in args.tag):
        parser.error("--tag must be a single directory name.")
    return args


def main():
    args = parse_args()
    import torch
    from PIL import Image
    from accelerate.utils import set_seed
    from src.utils.prompt_utils import load_prompt, process_prompts
    from src.utils.data_utils import get_colored_mesh_composition
    from src.utils.render_utils import render_views_around_mesh, export_renderings
    from src.pipelines.pipeline_flexpart import FlexPartPipeline

    with Image.open(args.image_path) as image:
        raw_img = image.convert("RGBA")
    masks, boxes, points = (load_prompt(path) for path in
                            (args.mask_path, args.box_path, args.point_path))
    final_masks, final_boxes, final_points = process_prompts(
        masks, boxes, points, (raw_img.height, raw_img.width))
    num_parts = len(final_points)
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires an NVIDIA GPU with CUDA support.")
    set_seed(args.seed)
    pipe = FlexPartPipeline.from_pretrained(args.model_path).to("cuda", torch.float16)
    if num_parts > pipe.transformer.config.max_num_parts:
        raise ValueError("The number of prompt parts exceeds the model capacity.")
    # Preserve image coordinates: background removal must not crop or pad the image.
    if args.rmbg:
        from src.models.briarmbg import BriaRMBG
        from src.utils.image_utils import remove_background_preserve_size
        rmbg_net = BriaRMBG.from_pretrained(args.rmbg_model_path).to("cuda").eval()
        raw_img = remove_background_preserve_size(raw_img, rmbg_net)
    white = Image.new("RGBA", raw_img.size, (255, 255, 255, 255))
    image_input = Image.alpha_composite(white, raw_img).convert("RGB")
    prompts = {
        "masks_prompt": final_masks,
        "boxes_prompt": final_boxes,
        "points_prompt": final_points,
        "valid_masks": [masks is not None] * num_parts,
        "valid_boxes": [masks is not None or boxes is not None] * num_parts,
        "valid_points": [True] * num_parts,
    }
    with torch.inference_mode():
        outputs = pipe(
            image=[image_input] * num_parts,
            attention_kwargs={"num_parts": num_parts},
            num_tokens=args.num_tokens,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            use_flash_decoder=args.use_flash_decoder,
            **prompts,
        ).meshes
    if any(mesh is None or len(mesh.faces) == 0 for mesh in outputs):
        raise RuntimeError("Mesh extraction returned an empty part. Try another seed or prompt.")
    export_dir = Path(args.output_dir) / (args.tag or Path(args.image_path).stem)
    export_dir.mkdir(parents=True, exist_ok=True)
    for i, mesh in enumerate(outputs):
        mesh.export(export_dir / f"part_{i:02d}.glb")
    composition = get_colored_mesh_composition(outputs)
    composition.export(export_dir / "object.glb")
    if args.render:
        images = render_views_around_mesh(composition, num_views=36, radius=4)
        export_renderings(images, str(export_dir / "rendering.gif"), fps=18)
    print(f"Saved {num_parts} parts to {export_dir}")


if __name__ == "__main__":
    main()
