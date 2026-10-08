"""Validate pixel-space prompts and convert them to pipeline conditions."""
import numpy as np


def load_prompt(path):
    return None if path is None else np.load(path, allow_pickle=False)


def process_prompts(masks, boxes, points, img_size):
    h, w = img_size
    supplied = [x for x in (masks, boxes, points) if x is not None]
    if not supplied:
        raise ValueError("Supply at least one of --mask_path, --box_path, --point_path.")
    if any(not isinstance(x, np.ndarray) or x.ndim == 0 or x.dtype.kind not in 'buif' for x in supplied):
        raise ValueError("Prompts must be numeric NumPy arrays with a part dimension.")
    n = len(supplied[0])
    if n == 0 or any(len(x) != n for x in supplied):
        raise ValueError("Prompt arrays must contain the same nonzero number of parts.")
    for name, value, shape in (("masks", masks, (n, h, w)),
                               ("boxes", boxes, (n, 4)), ("points", points, (n, 2))):
        if value is not None and (value.shape != shape or not np.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite numeric array with shape {shape}.")
    if masks is not None and any(not np.any(m) for m in masks):
        raise ValueError("Every part mask must contain at least one foreground pixel.")
    if boxes is not None:
        if (np.any(boxes[:, :2] > boxes[:, 2:]) or np.any(boxes < 0)
                or np.any(boxes[:, [0, 2]] >= w) or np.any(boxes[:, [1, 3]] >= h)):
            raise ValueError("Boxes must be ordered [x1, y1, x2, y2] within the image.")
    if points is not None and (np.any(points < 0) or np.any(points[:, 0] >= w)
                               or np.any(points[:, 1] >= h)):
        raise ValueError("Points must be [x, y] coordinates within the image.")
    final_boxes, final_points = [], []
    for i in range(n):
        if masks is not None:
            ys, xs = np.where(masks[i] != 0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            index = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
            cx, cy = xs[index], ys[index]
        elif boxes is not None:
            x1, y1, x2, y2 = boxes[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        else:
            x1, y1, x2, y2 = 0, 0, w - 1, h - 1
            cx, cy = points[i]
        final_boxes.append(tuple(float(np.clip((float(v) + 0.5) / size, 0, 1))
                                 for v, size in zip((x1, y1, x2, y2), (w, h, w, h))))
        final_points.append((float(np.clip((float(cx) + 0.5) / w, 0, 1)),
                             float(np.clip((float(cy) + 0.5) / h, 0, 1))))
    return (None if masks is None else [m.astype(bool) for m in masks],
            final_boxes, final_points)
