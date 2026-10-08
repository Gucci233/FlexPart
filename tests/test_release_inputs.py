import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.prompt_utils import load_prompt, process_prompts


class PromptTests(unittest.TestCase):
    def test_point_only_leaves_mask_absent(self):
        masks, boxes, points = process_prompts(None, None, np.array([[0, 0], [19, 9]]), (10, 20))
        self.assertIsNone(masks)
        np.testing.assert_allclose(points, [[0.025, 0.05], [0.975, 0.95]])
        self.assertEqual(len(boxes), 2)

    def test_box_center_is_derived_in_pixel_coordinates(self):
        _, _, points = process_prompts(None, np.array([[2, 4, 8, 6]]), None, (10, 20))
        np.testing.assert_allclose(points, [[5.5 / 20, 5.5 / 10]])

    def test_mask_centroid_stays_in_foreground(self):
        mask = np.zeros((1, 10, 20), dtype=bool)
        mask[0, 1:8, 2] = True
        mask[0, 7, 2:12] = True
        masks, boxes, points = process_prompts(mask, None, None, (10, 20))
        px, py = points[0]
        self.assertTrue(mask[0, int(py * 10), int(px * 20)])
        np.testing.assert_allclose(boxes[0], [2.5 / 20, 1.5 / 10, 11.5 / 20, 7.5 / 10])
        self.assertEqual(masks[0].dtype, np.bool_)

    def test_invalid_prompts_fail_before_model_loading(self):
        cases = [
            (None, None, None),
            (None, None, np.empty((0, 2))),
            (None, None, np.array([[np.nan, 1]])),
            (None, None, np.array([[20, 1]])),
            (None, np.array([[8, 2, 3, 5]]), None),
            (np.zeros((1, 10, 20)), None, None),
            (np.ones((1, 9, 20)), None, None),
            (None, np.ones((2, 4)), np.ones((1, 2))),
            (None, None, np.array(1)),
        ]
        for masks, boxes, points in cases:
            with self.subTest(shape=str((masks, boxes, points))):
                with self.assertRaises(ValueError):
                    process_prompts(masks, boxes, points, (10, 20))

    def test_load_rejects_pickled_objects_and_missing_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / 'object.npy'
            np.save(path, {'arbitrary': 'object'})
            with self.assertRaises(ValueError):
                load_prompt(path)
            with self.assertRaises(FileNotFoundError):
                load_prompt(Path(tmp) / 'missing.npy')


class PreprocessingMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise metadata generation without importing CUDA/rendering dependencies.
        path = ROOT / 'datasets/preprocess/multi_preprocess.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = {'validate_obbs', 'process_single_marker', 'check_single_model'}
        module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                  and node.name in names], type_ignores=[])
        cls.scope = {'np': np, 'Path': Path, 'os': os, 'json': json}
        exec(compile(module, str(path), 'exec'), cls.scope)

    def test_annotation_without_obb_image_and_nested_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            raw = Path(tmp) / 'raw/nested'
            output = Path(tmp) / 'processed/chair.v2'
            raw.mkdir(parents=True)
            output.mkdir(parents=True)
            source = raw / 'chair.v2.glb'
            source.touch()
            for name in ('rendering.png', 'mask.png', 'normalized_rotated_scene.glb'):
                (output / name).touch()
            (output / 'num_parts.json').write_text('{"num_parts": 1}')
            np.save(output / 'points.npy', {'parts': [{}], 'part_obbs': np.ones((1, 10))})
            config = self.scope['process_single_marker'](output / 'num_parts.json', raw.parent)
            self.assertIsNotNone(config)
            self.assertEqual(config['file'], os.path.join('nested', 'chair.v2.glb'))
            self.assertEqual(config['num_parts'], 1)
            self.assertIsNone(config['obb_image_path'])
            self.assertIsNone(self.scope['check_single_model']((str(source), str(output.parent))))
            (output / 'normalized_rotated_scene.glb').unlink()
            self.assertEqual(self.scope['check_single_model']((str(source), str(output.parent))), str(source))
            self.assertIsNone(self.scope['process_single_marker'](output / 'num_parts.json', raw.parent))


class EntryPointTests(unittest.TestCase):
    def test_help_without_ml_dependencies(self):
        for script in ('app.py', 'scripts/inference_flexpart.py', 'scripts/prepare_weights.py'):
            result = subprocess.run([sys.executable, str(ROOT / script), '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)

    def test_cli_requires_a_prompt(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/inference_flexpart.py'),
                                 '--image_path', 'example.png'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Supply at least one prompt file', result.stderr)


if __name__ == '__main__':
    unittest.main()
