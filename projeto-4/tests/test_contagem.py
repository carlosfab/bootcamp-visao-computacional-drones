"""Testes offline: python -m unittest discover -s tests -p 'test_contagem.py'."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suporte.contagem import (count_metrics, evaluate_predictions, iou_matrix,
                              match_detections, nms, predict_image, predict_tiled,
                              tile_windows)


class FakeModel:
    """Uma resposta por imagem, sem torch, rede ou download de pesos."""

    def __init__(self, predictions, names=None):
        self.predictions = iter(predictions)
        self.names = names or {19: "cow", 7: "truck", 42: "cattle"}
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        boxes, scores, classes = next(self.predictions)
        result = SimpleNamespace(names=self.names, boxes=SimpleNamespace(
            xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
            conf=np.asarray(scores, dtype=float), cls=np.asarray(classes)))
        return [result]


class MetricsTests(unittest.TestCase):
    def test_iou_geometry_empty_and_zero_area(self):
        np.testing.assert_allclose(iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]]), [[1 / 3]])
        self.assertEqual(iou_matrix([], [[0, 0, 1, 1]]).shape, (0, 1))
        self.assertEqual(iou_matrix([[0, 0, 1, 1]], []).shape, (1, 0))
        self.assertEqual(iou_matrix([[0, 0, 0, 0]], [[0, 0, 0, 0]]).item(), 0)

    def test_no_ground_truth_or_predictions(self):
        box = [[0, 0, 10, 10]]
        for gt, pred, scores, expected in [([], [], [], (0, 0, 0)),
                                         (box, [], [], (0, 0, 1)),
                                         ([], box, [.8], (0, 1, 0))]:
            result = match_detections(gt, pred, scores)
            self.assertEqual(tuple(result[key] for key in ("tp", "fp", "fn")), expected)

    def test_duplicate_predictions_score_priority_and_ties(self):
        gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
        pred = [gt[0], gt[0], gt[1]]
        result = match_detections(gt, pred, [.3, .8, .8])
        self.assertEqual(result["tp_indices"], [1, 2])
        self.assertEqual(result["fp_indices"], [0])
        tied = match_detections([gt[0]], [gt[0], gt[0]], [.8, .8])
        self.assertEqual(tied["tp_indices"], [0])
        gt_tie = match_detections([gt[0], gt[0]], [gt[0]], [.9])
        self.assertEqual(gt_tie["matches"][0]["gt_index"], 0)

    def test_fp_fn_can_cancel_count_error(self):
        rows = [{"image_id": "cancelamento", "gt_boxes": [[0, 0, 10, 10]],
                 "pred_boxes": [[20, 20, 30, 30]], "scores": [.9]}]
        result = evaluate_predictions(rows)
        summary = result["summary"]
        self.assertEqual((summary["mae"], summary["exact_accuracy"]), (0, 1))
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"]), (0, 1, 1))
        self.assertEqual(summary["f1"], 0)
        json.dumps(result, allow_nan=False)

    def test_counting_bias_and_image_average(self):
        result = count_metrics([1, 3, 2], [3, 2, 2])
        self.assertAlmostEqual(result["mae"], 1)
        self.assertAlmostEqual(result["rmse"], np.sqrt(5 / 3))
        self.assertAlmostEqual(result["bias"], 1 / 3)
        self.assertEqual(result["exact_match_count"], 1)
        self.assertEqual(count_metrics([], [])["mae"], None)

    def test_thresholding_includes_empty_images_and_micro_metrics(self):
        box = [[0, 0, 10, 10]]
        rows = [{"image_id": "gado", "gt_boxes": box, "pred_boxes": box, "scores": [.6]},
                {"image_id": "vazio", "gt_boxes": [], "pred_boxes": box, "scores": [.2]}]
        summary = evaluate_predictions(rows, conf_threshold=.5)["summary"]
        self.assertEqual((summary["n_images"], summary["tp"], summary["fp"], summary["mae"]), (2, 1, 0, 0))
        self.assertEqual(evaluate_predictions([])["summary"]["n_images"], 0)

    def test_invalid_inputs_do_not_silently_change_metrics(self):
        invalid = [lambda: iou_matrix([[10, 0, 0, 10]], []),
                   lambda: match_detections([], [[0, 0, 1, 1]], []),
                   lambda: match_detections([], [], [], iou_threshold=0),
                   lambda: count_metrics([1], [1.5]),
                   lambda: count_metrics([1], [-1]),
                   lambda: count_metrics([1], [float("nan")]),
                   lambda: nms([], [], iou_threshold=2)]
        for operation in invalid:
            with self.assertRaises(ValueError):
                operation()
        row = {"image_id": "same", "gt_boxes": [], "pred_boxes": [], "scores": []}
        with self.assertRaises(ValueError):
            evaluate_predictions([row, row])

    def test_nms_stable_indices_and_empty(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10]]
        self.assertEqual(nms(boxes, [.9, .9, .8]).tolist(), [0, 2])
        self.assertEqual(nms(boxes, [.7, .9, .8]).tolist(), [1, 2])
        self.assertEqual(nms([], []).tolist(), [])


class InferenceTests(unittest.TestCase):
    def test_tiles_cover_borders_without_duplicate_windows(self):
        for width, height, size in [(15, 13, 6), (12, 12, 6), (3, 4, 6), (6, 6, 6), (1, 1, 6)]:
            windows = tile_windows(width, height, size, .2)
            self.assertEqual(len(windows), len(set(windows)))
            covered = np.zeros((height, width), dtype=bool)
            for x1, y1, x2, y2 in windows:
                self.assertTrue(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
                covered[y1:y2, x1:x2] = True
            self.assertTrue(covered.all())
        self.assertEqual(tile_windows(1280, 640, 640, 0), [(0, 0, 640, 640), (640, 0, 1280, 640)])
        with self.assertRaises(ValueError):
            tile_windows(20, 20, overlap=1)

    def test_names_confidence_clipping_class_remap_and_rgb(self):
        model = FakeModel([([[-3, -4, 8, 7], [5, 5, 30, 20], [0, 0, 2, 2],
                            [0, 0, 2, 2], [30, 30, 40, 40]],
                            [.9, .8, .99, .01, .9], [19, 42, 7, 19, 19])])
        rgb = np.zeros((10, 20, 3), dtype=np.uint8)
        rgb[0, 0] = [10, 20, 30]
        result = predict_image(model, rgb, conf=.05)
        np.testing.assert_allclose(result["boxes"], [[0, 0, 8, 7], [5, 5, 20, 10]])
        np.testing.assert_allclose(result["scores"], [.9, .8])
        self.assertEqual(result["class_ids"].tolist(), [0, 0])
        self.assertEqual(model.calls[0]["source"][0, 0].tolist(), [30, 20, 10])
        self.assertEqual(model.calls[0]["device"], "cpu")

    def test_missing_class_is_explicit_and_empty_predictions_are_valid(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        model = FakeModel([([], [], [])], {0: "cow"})
        self.assertEqual(predict_image(model, rgb)["boxes"].shape, (0, 4))
        model = FakeModel([([], [], [])], {0: "sheep"})
        with self.assertRaises(ValueError):
            predict_image(model, rgb)

    def test_tiling_global_coordinates_dedup_and_corner_clipping(self):
        # Two 10-pixel tiles start at x=0 and x=8; both see the same cow.
        model = FakeModel([
            ([[8, 2, 10, 6]], [.9], [19]),
            ([[0, 2, 2, 6], [8, 8, 15, 15]], [.8, .7], [19, 42]),
        ])
        result = predict_tiled(model, np.zeros((10, 18, 3), dtype=np.uint8),
                               tile_size=10, overlap=.2)
        np.testing.assert_allclose(result["boxes"], [[8, 2, 10, 6], [16, 8, 18, 10]])
        np.testing.assert_allclose(result["scores"], [.9, .7])
        self.assertEqual(len(model.calls), 2)

    def test_tiling_with_no_detections(self):
        model = FakeModel([([], [], [])])
        prediction = predict_tiled(model, np.zeros((4, 3, 3), dtype=np.uint8), tile_size=10)
        self.assertEqual(prediction["boxes"].shape, (0, 4))
        self.assertEqual(prediction["class_ids"].shape, (0,))


if __name__ == "__main__":
    unittest.main()
