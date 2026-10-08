"""Assemble the released FlexPart transformer and TripoSG components into a pipeline."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COMPONENTS = ("vae", "image_encoder_dinov2", "feature_extractor_dinov2", "scheduler")


def validate_checkpoint(weight_path, config):
    # Check all tensor names and shapes without allocating the 6 GB transformer.
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from src.models.transformers.flexpart_transformer import FlexPartDiTModel
    with init_empty_weights(include_buffers=True):
        model = FlexPartDiTModel.from_config(config)
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    with safe_open(str(weight_path), framework="pt", device="cpu") as checkpoint:
        actual = {key: tuple(checkpoint.get_slice(key).get_shape()) for key in checkpoint.keys()}
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    mismatched = {key: (expected[key], actual[key]) for key in expected.keys() & actual.keys()
                  if expected[key] != actual[key]}
    if missing or unexpected or mismatched:
        raise ValueError(f"Checkpoint/config mismatch: missing={missing}, unexpected={unexpected}, shapes={mismatched}")
    print(f"Validated {len(actual)} checkpoint tensors against the inference configuration.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", help="Use a local released checkpoint instead of downloading it.")
    parser.add_argument("--checkpoint_repo", default="gucci233/FlexPart")
    parser.add_argument("--checkpoint_revision", help="Optional Hugging Face commit or tag.")
    parser.add_argument("--base_model", default="VAST-AI/TripoSG", help="Hugging Face repo ID or local TripoSG directory.")
    parser.add_argument("--output_dir", default="./weight/FlexPart")
    parser.add_argument("--transformer_config", default=str(ROOT / "configs/flexpart_inference.json"))
    args = parser.parse_args()
    output = Path(args.output_dir)
    if (output / "model_index.json").exists():
        parser.error(f"Output already contains a pipeline: {output}. Choose a new --output_dir.")
    from huggingface_hub import hf_hub_download, snapshot_download
    checkpoint = (Path(args.checkpoint_dir) / "diffusion_pytorch_model.safetensors"
                  if args.checkpoint_dir else Path(hf_hub_download(
                      args.checkpoint_repo, "diffusion_pytorch_model.safetensors", revision=args.checkpoint_revision)))
    config = json.loads(Path(args.transformer_config).read_text(encoding="utf-8"))
    validate_checkpoint(checkpoint, config)
    base = Path(args.base_model)
    if not base.is_dir():
        base = Path(snapshot_download(args.base_model, allow_patterns=[f"{name}/*" for name in COMPONENTS]))
    for name in COMPONENTS:
        if not (base / name / ("scheduler_config.json" if name == "scheduler" else
                              "preprocessor_config.json" if name == "feature_extractor_dinov2" else "config.json")).is_file():
            raise FileNotFoundError(f"Missing required TripoSG component: {base / name}")
        if name in ("vae", "image_encoder_dinov2") and not any(
                list((base / name).glob('*.safetensors')) + list((base / name).glob('*.bin'))):
            raise FileNotFoundError(f"Missing model weights for TripoSG component: {base / name}")
    for name in COMPONENTS:
        shutil.copytree(base / name, output / name, dirs_exist_ok=True)
    transformer_dir = output / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint.resolve() != (transformer_dir / checkpoint.name).resolve():
        shutil.copy2(checkpoint, transformer_dir / checkpoint.name)
    (transformer_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    index = {
        "_class_name": "FlexPartPipeline",
        "feature_extractor_dinov2": ["transformers", "BitImageProcessor"],
        "image_encoder_dinov2": ["transformers", "Dinov2Model"],
        "scheduler": ["src.schedulers.scheduling_rectified_flow", "RectifiedFlowScheduler"],
        "transformer": ["src.models.transformers.flexpart_transformer", "FlexPartDiTModel"],
        "vae": ["src.models.autoencoders.autoencoder_kl_triposg", "TripoSGVAEModel"],
    }
    (output / "model_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared pipeline at {output}")


if __name__ == "__main__":
    main()
