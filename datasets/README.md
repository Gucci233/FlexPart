# Dataset Preparation

The main FlexPart preprocessing entry point is `datasets/preprocess/multi_preprocess.py`. Run commands from the code repository root.

## Raw Meshes

Place raw `.glb` objects under `./data/raw`. Subdirectories are supported, but every GLB filename stem must be unique because processed objects are stored by stem.

```text
data/raw/
├── chair.glb
└── nested/
    └── lamp.glb
```

The pipeline processes GLB files. It filters invalid geometry and objects outside its current 1–20-part range, selects a view from six rendered directions, normalizes and rotates the scene, samples object and part surfaces, and derives visible part masks, 2D boxes, and 3D oriented boxes. Parts with fewer than 50 visible mask pixels are filtered.

For Objaverse sources, see [PartCrafter's data instructions](https://github.com/wgsxm/PartCrafter/blob/main/datasets/README.md). For PartVerse-XL, see [FullPart](https://github.com/hkdsc/fullpart). Follow the source datasets' access and license terms.

## Run Preprocessing

Install the repository dependencies and rendering libraries first. For headless rendering, configure EGL as needed:

```bash
PYOPENGL_PLATFORM=egl python datasets/preprocess/multi_preprocess.py \
  --input ./data/raw \
  --output ./data/preprocessed \
  --workers 4
```

`--workers` controls mesh processing workers and defaults to 1. Metadata collection uses a separate CPU process pool. The main pipeline does not require RMBG weights; rendered views already supply the training images.

Completed objects with all required output files are skipped when rerunning. Files with incomplete output are processed again. Validated metadata is saved to the parent of the output directory:

```text
data/
├── object_part_configs_new.json
└── preprocessed/
    └── chair/
        ├── rendering.png
        ├── mask.png
        ├── points.npy
        ├── normalized_rotated_scene.glb
        └── num_parts.json
```

The pipeline does not generate `render_OBB.png`; this optional visualization is not required for training metadata.

## Output Files

| File | Contents |
| --- | --- |
| `rendering.png` | Selected object view |
| `mask.png` | Visualization of visible part masks |
| `points.npy` | Object surface data, per-part surface data and 2D prompts, and `part_obbs` |
| `normalized_rotated_scene.glb` | Normalized, view-aligned source scene for evaluation |
| `num_parts.json` | Number of retained parts |

`points.npy` contains a pickled NumPy dictionary:

```python
{
    "object": {...},
    "parts": [{..., "2d_mask": ..., "2d_box": ...}, ...],
    "part_obbs": ...  # Array with shape (N, 10)
}
```

Only load trusted preprocessed data with `allow_pickle=True`. These training dictionaries are different from the numeric prompt arrays accepted by command-line inference.

Each oriented bounding box has the layout:

```text
[center_x, center_y, center_z,
 extent_x, extent_y, extent_z,
 quaternion_w, quaternion_x, quaternion_y, quaternion_z]
```

Objects with invalid OBB arrays are omitted from training metadata. The exported source scene retains its source geometry; visible-part filtering affects the training part annotations. Match this policy to the intended evaluation protocol.

## Training Metadata

`object_part_configs_new.json` is a list of entries containing `file`, `mesh_path`, `surface_path`, `image_path`, `mask_path`, `num_parts`, and `valid`. `obb_image_path` is null when no optional OBB visualization exists.

Paths follow the input/output paths supplied to preprocessing. Run training from the repository root when using relative paths. The default `dataset.config` in both training YAML files points to `./data/object_part_configs_new.json`; update it if you choose another output location.

For part evaluation, create a copy of the metadata and add `pred_mesh_path` to each entry, pointing to the corresponding generated `object.glb`.

## Other Scripts

`preprocess.py`, `render.py`, and `rmbg.py` are auxiliary preprocessing tools inherited from the base project. Use `multi_preprocess.py` for the current FlexPart part-aware training format. The RMBG tool uses `./weight/RMBG-1.4` by default.
