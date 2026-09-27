"""Smoke offline do fluxo: imagens/labels sintéticos, sem pesos ou rede.

Estes fixtures testam o software; não medem desempenho de detecção real.
Execute: python -m unittest discover -s tests -p 'test_*.py'.
"""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suporte.experimento import (THRESHOLDS, calibrar, desenhar, inferir,
                                inventario, ler_caixas, salvar_resultado)


class FakeModel:
    """Respostas controladas para verificar encadeamento, não acurácia de YOLO."""

    names = {0: "cow"}

    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        boxes, scores = next(self.predictions)
        return [SimpleNamespace(names=self.names, boxes=SimpleNamespace(
            xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
            conf=np.asarray(scores, dtype=float), cls=np.zeros(len(scores))))]


class DatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for split in ("train", "val", "test"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)

    def add_image(self, split, name="foto.png", annotation="0 .5 .5 .2 .4\n"):
        path = self.root / "images" / split / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 50), (120, 30, 60)).save(path)
        label = self.root / "labels" / split / Path(name).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        if annotation is not None:
            label.write_text(annotation, encoding="utf-8")
        return path, label


class LabelAndInventoryTests(DatasetFixture):
    def test_yolo_normalization_and_empty_label(self):
        _, label = self.add_image("train")
        np.testing.assert_allclose(ler_caixas(label, 100, 50), [[40, 15, 60, 35]])
        label.write_text("\n \n", encoding="utf-8")
        self.assertEqual(ler_caixas(label, 100, 50).shape, (0, 4))

    def test_missing_label_is_not_a_negative(self):
        _, label = self.add_image("val", annotation=None)
        with self.assertRaises(FileNotFoundError):
            ler_caixas(label, 100, 50)
        with self.assertRaises(FileNotFoundError):
            inventario(self.root)

    def test_invalid_yolo_records_raise(self):
        _, label = self.add_image("train")
        invalid = ["1 .5 .5 .2 .2", "0 .5 .5 0 .2", "0 50 25 10 10",
                   "0 .01 .5 .2 .2", "0 .5 .5 nan .2", "0 .5 .5 .2 inf",
                   "cow .5 .5 .2 .2", "0 .5 .5 .2", "0 .5 .5 .2 .2 9"]
        for line in invalid:
            with self.subTest(line=line):
                label.write_text(line, encoding="utf-8")
                with self.assertRaises(ValueError):
                    ler_caixas(label, 100, 50)
        for width, height in [(0, 50), (100, -1), (100.5, 50), (True, 50)]:
            with self.assertRaises(ValueError):
                ler_caixas(label, width, height)

    def test_border_roundoff_is_clipped_but_larger_overflow_is_rejected(self):
        _, label = self.add_image("train", annotation="0 .4999999 .5 1 1")
        np.testing.assert_allclose(ler_caixas(label, 100, 50), [[0, 0, 99.99999, 50]])
        label.write_text("0 .499 .5 1 1", encoding="utf-8")
        with self.assertRaises(ValueError):
            ler_caixas(label, 100, 50)

    def test_inventory_split_ids_counts_and_negative_scale(self):
        self.add_image("train")
        self.add_image("val", "nested/foto.png", annotation="")
        self.add_image("test")
        table = inventario(self.root)
        self.assertEqual(table["image_id"].tolist(),
                         ["train/foto.png", "val/nested/foto.png", "test/foto.png"])
        self.assertEqual(table["count"].tolist(), [1, 0, 1])
        self.assertAlmostEqual(table.iloc[0]["median_box_side"], 20)
        self.assertTrue(np.isnan(table.iloc[1]["median_box_side"]))
        self.assertTrue(all(Path(path).is_absolute() for path in table["image"]))

    def test_inventory_rejects_same_stem_different_extensions_and_missing_split(self):
        self.add_image("train", "same.png")
        self.add_image("train", "same.jpg")
        with self.assertRaises(ValueError):
            inventario(self.root)
        with tempfile.TemporaryDirectory() as missing:
            with self.assertRaises(FileNotFoundError):
                inventario(missing)


class CalibrationTests(unittest.TestCase):
    def rows(self):
        box = [0, 0, 10, 10]
        return [{"image_id": "val/positivo", "split": "val", "gt_boxes": [box],
                 "pred_boxes": [box], "scores": [.7]},
                {"image_id": "val/negativo", "split": "val", "gt_boxes": [],
                 "pred_boxes": [box], "scores": [.2]}]

    def test_calibration_maximizes_f1_then_prefers_highest_tied_threshold(self):
        rows = self.rows()
        original = copy.deepcopy(rows)
        threshold, table = calibrar(rows)
        self.assertEqual(threshold, .7)
        self.assertEqual(table["threshold"].tolist(), list(THRESHOLDS))
        self.assertEqual(table.loc[table["threshold"] == .7, "f1"].item(), 1)
        self.assertEqual(table.loc[table["threshold"] == .8, "fn"].item(), 1)
        self.assertEqual(rows, original)
        self.assertNotIn("map", table.columns)
        self.assertNotIn("ap", table.columns)

    def test_calibration_rejects_test_train_and_unknown_provenance(self):
        for split in ("test", "train", None):
            rows = self.rows()
            if split is None:
                rows[1].pop("split")
            else:
                rows[1]["split"] = split
            with self.subTest(split=split), self.assertRaises(ValueError):
                calibrar(rows)
        with self.assertRaises(ValueError):
            calibrar([])


class ExperimentFlowTests(DatasetFixture):
    def test_smoke_inventory_inference_calibration_and_saved_schema(self):
        self.add_image("train")
        self.add_image("val", "positivo.png")
        self.add_image("val", "negativo.png", annotation="")
        self.add_image("test", "positivo.png")
        self.add_image("test", "negativo.png", annotation="")
        table = inventario(self.root)
        val = table.loc[table["split"] == "val"]
        # Inventory sorts negativo before positivo; validation has one removable FP.
        model = FakeModel([([[0, 0, 10, 10]], [.2]), ([[40, 15, 60, 35]], [.7])])
        val_rows = inferir(model, val)
        threshold, _ = calibrar(val_rows)
        self.assertEqual(threshold, .7)
        self.assertTrue(all(call["conf"] == .05 for call in model.calls))
        test = table.loc[table["split"] == "test"]
        model = FakeModel([([[0, 0, 10, 10]], [.9]), ([[0, 0, 10, 10]], [.9])])
        rows = inferir(model, test)
        output = self.root / "results"
        summary = salvar_resultado(output, rows, threshold)
        self.assertEqual(summary, json.loads((output / "resumo.json").read_text()))
        self.assertEqual(rows, json.loads((output / "predictions.json").read_text()))
        by_presence = summary["by_presence"]
        self.assertEqual(by_presence["positive"]["mae"], 0)
        self.assertEqual(by_presence["positive"]["f1"], 0)
        self.assertEqual(by_presence["negative"]["false_positive_image_rate"], 1)
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"]), (0, 2, 1))
        self.assertEqual(summary["mae"], .5)
        self.assertEqual(summary["timing"]["n_timed_images"], 2)
        self.assertIn("não é AP/mAP", summary["metric_scope"])
        per_image = pd.read_csv(output / "per_image.csv")
        self.assertEqual(per_image["split"].tolist(), ["test", "test"])
        self.assertEqual(per_image["count_error"].tolist(), [1, 0])

    def test_inference_checks_inventory_and_image_dimensions(self):
        self.add_image("test")
        table = inventario(self.root)
        for malformed in (table.drop(columns=["label"]), pd.concat([table, table])):
            with self.assertRaises(ValueError):
                inferir(FakeModel([]), malformed)
        table.loc[:, "width"] = 101
        with self.assertRaises(ValueError):
            inferir(FakeModel([]), table)

    def test_tiled_smoke_on_image_smaller_than_one_tile(self):
        self.add_image("test")
        table = inventario(self.root)
        model = FakeModel([([[40, 15, 60, 35]], [.8])])
        rows = inferir(model, table, tiled=True)
        self.assertEqual(rows[0]["pred_boxes"], [[40, 15, 60, 35]])
        self.assertEqual(rows[0]["split"], "test")
        self.assertEqual(model.calls[0]["device"], "cpu")

    def test_empty_result_schema_and_missing_timing_are_explicit(self):
        summary = salvar_resultado(self.root / "empty", [], .25)
        self.assertEqual(summary["n_images"], 0)
        self.assertIsNone(summary["mae"])
        self.assertIsNone(summary["by_presence"]["negative"]["false_positive_image_rate"])
        self.assertEqual(summary["timing"]["n_timed_images"], 0)
        self.assertIsNone(summary["timing"]["mean_seconds"])
        self.assertTrue(pd.read_csv(self.root / "empty" / "per_image.csv").empty)
        row = {"image_id": "empty", "gt_boxes": [], "pred_boxes": [], "scores": []}
        summary = salvar_resultado(self.root / "negative", [row], .25)
        self.assertEqual(summary["by_presence"]["negative"]["false_positive_image_rate"], 0)
        self.assertEqual(summary["by_presence"]["positive"]["n_images"], 0)

    def test_invalid_timing_or_nonfinite_extra_does_not_create_result_files(self):
        row = {"image_id": "empty", "gt_boxes": [], "pred_boxes": [], "scores": []}
        for extra in ({"seconds": -1}, {"seconds": float("nan")}, {"extra": float("inf")}):
            output = self.root / "invalid"
            with self.assertRaises(ValueError):
                salvar_resultado(output, [{**row, **extra}], .25)
            self.assertFalse(output.exists())

    def test_plot_returns_sans_serif_figure_without_mutating_image(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        original = rgb.copy()
        figure = desenhar(rgb, [[0, 0, 10, 10]], [[15, 5, 25, 15]], [.8], .5)
        self.addCleanup(plt.close, figure)
        self.assertEqual(figure.axes[0].title.get_fontfamily(), ["DejaVu Sans"])
        self.assertEqual(len(figure.axes[0].patches), 2)
        np.testing.assert_array_equal(rgb, original)
        figure.savefig(self.root / "plot.png")
        self.assertGreater((self.root / "plot.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
