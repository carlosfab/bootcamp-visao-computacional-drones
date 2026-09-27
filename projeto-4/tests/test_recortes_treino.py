"""Fixtures sintéticos verificam recortes; não medem qualidade de um modelo."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suporte.experimento import ler_caixas
from suporte.recortes_treino import preparar_recortes


def sha(content):
    return hashlib.sha256(content).hexdigest()


class RecortesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.samples = []

    def add_parent(self, split, name, boxes=(), width=800, height=640):
        image = self.base / "images" / split / f"{name}.jpg"
        label = self.base / "labels" / split / f"{name}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (width, height), (30, 120, 40)).save(image, quality=95)
        lines = []
        for x1, y1, x2, y2 in boxes:
            values = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                      (x2 - x1) / width, (y2 - y1) / height]
            lines.append("0 " + " ".join(f"{value:.17g}" for value in values))
        label.write_text("\n".join(lines) + ("\n" if lines else ""))
        self.samples.append({"id": name, "split": split,
                             "image": image.relative_to(self.base).as_posix(),
                             "label": label.relative_to(self.base).as_posix(),
                             "width": width, "height": height,
                             "derived_sha256": sha(image.read_bytes()),
                             "label_sha256": sha(label.read_bytes()),
                             "farm": "farm", "flight": f"flight-{split}",
                             "date": "2026-01-01" if split == "train" else "2026-02-01"})

    def write_manifest(self):
        (self.base / "manifesto.json").write_text(
            json.dumps({"dataset": "synthetic-unit-fixture", "samples": self.samples},
                       sort_keys=True, indent=2) + "\n")

    def build(self):
        self.write_manifest()
        destination = preparar_recortes(self.base)
        return destination, json.loads((destination / "manifesto.json").read_text())

    def test_half_visible_border_box_is_clipped_and_normalized_to_tile(self):
        self.add_parent("train", "edge", [[620, 40, 660, 80]])
        self.add_parent("val", "empty")
        destination, manifest = self.build()
        first = next(row for row in manifest["samples"] if row["id"] == "edge__x00000_y00000")
        self.assertEqual(first["window"], [0, 0, 640, 640])
        self.assertEqual((first["width"], first["height"]), (640, 640))
        np.testing.assert_allclose(ler_caixas(destination / first["label"], 640, 640),
                                   [[620, 40, 640, 80]], atol=1e-6)
        with Image.open(destination / first["image"]) as image:
            self.assertEqual(image.size, (640, 640))
        self.assertEqual(first["parent_id"], "edge")

    def test_ambiguous_fragment_discards_entire_tile_including_other_good_box(self):
        self.add_parent("train", "ambiguous", [[100, 100, 120, 120], [635, 40, 655, 80]])
        self.add_parent("val", "empty")
        _, manifest = self.build()
        train = [row for row in manifest["samples"] if row["split"] == "train"]
        self.assertEqual(len(train), 1)
        self.assertEqual(train[0]["window"], [160, 0, 800, 640])
        self.assertEqual(train[0]["count"], 1)
        self.assertEqual(manifest["summary"]["train"]["discarded_ambiguous_tiles"], 1)

    def test_one_pixel_fragment_is_discarded_even_when_half_visible(self):
        self.add_parent("train", "thin", [[639, 40, 641, 80]])
        self.add_parent("val", "empty")
        _, manifest = self.build()
        self.assertNotIn("thin__x00000_y00000", {row["id"] for row in manifest["samples"]})
        self.assertEqual(manifest["summary"]["train"]["discarded_ambiguous_tiles"], 1)

    def test_negative_selection_is_deterministic_and_validation_is_not_sampled(self):
        self.add_parent("train", "source", [[100, 100, 140, 140]], width=1280)
        self.add_parent("val", "empty", width=1280)
        destination, manifest = self.build()
        self.assertEqual(manifest["summary"]["train"]["positive"], 1)
        self.assertEqual(manifest["summary"]["train"]["negative"], 1)
        self.assertEqual(manifest["summary"]["train"]["discarded_negative_tiles"], 1)
        self.assertEqual(manifest["summary"]["val"]["negative"], 3)
        negative_ids = [f"source__x{x:05d}_y00000" for x in (512, 640)]
        expected = min(negative_ids, key=lambda name: (sha(f"gado-recortes-v1\n{name}".encode()), name))
        actual = next(row["id"] for row in manifest["samples"]
                      if row["split"] == "train" and row["count"] == 0)
        self.assertEqual(actual, expected)
        original_manifest = (destination / "manifesto.json").read_bytes()
        original_mtime = (destination / "manifesto.json").stat().st_mtime_ns
        self.assertEqual(preparar_recortes(self.base), destination)
        self.assertEqual((destination / "manifesto.json").read_bytes(), original_manifest)
        self.assertEqual((destination / "manifesto.json").stat().st_mtime_ns, original_mtime)
        for row in manifest["samples"]:
            self.assertEqual(row["flight"], f"flight-{row['split']}")

    def test_never_reads_test_files_and_yaml_excludes_test(self):
        self.add_parent("train", "source", [[100, 100, 140, 140]])
        self.add_parent("val", "empty")
        # No corresponding files exist. Accessing them would fail the build.
        self.samples.append({"split": "test", "id": "DO_NOT_READ",
                             "image": "images/test/missing.jpg", "label": "labels/test/missing.txt"})
        destination, manifest = self.build()
        self.assertEqual({row["split"] for row in manifest["samples"]}, {"train", "val"})
        self.assertNotIn("test:", (destination / "data.yaml").read_text())
        self.assertFalse((destination / "images" / "test").exists())

    def test_small_image_uses_its_actual_crop_dimensions_without_resize(self):
        self.add_parent("train", "small", [[10, 20, 30, 40]], width=100, height=80)
        self.add_parent("val", "empty", width=100, height=80)
        destination, manifest = self.build()
        row = next(row for row in manifest["samples"] if row["split"] == "train")
        self.assertEqual(row["window"], [0, 0, 100, 80])
        with Image.open(destination / row["image"]) as image:
            self.assertEqual(image.size, (100, 80))
        np.testing.assert_allclose(ler_caixas(destination / row["label"], 100, 80), [[10, 20, 30, 40]])

    def test_changed_source_configuration_or_output_is_rejected_without_overwrite(self):
        self.add_parent("train", "source", [[100, 100, 140, 140]])
        self.add_parent("val", "empty")
        destination, manifest = self.build()
        original = (destination / "manifesto.json").read_bytes()
        source_manifest = self.base / "manifesto.json"
        source_bytes = source_manifest.read_bytes()
        source_manifest.write_bytes(source_bytes + b" ")
        with self.assertRaises(ValueError):
            preparar_recortes(self.base)
        source_manifest.write_bytes(source_bytes)
        config = destination / "configuracao.json"
        config_bytes = config.read_bytes()
        config.write_bytes(config_bytes + b" ")
        with self.assertRaises(ValueError):
            preparar_recortes(self.base)
        config.write_bytes(config_bytes)
        label = destination / manifest["samples"][0]["label"]
        label.write_text("alterado")
        with self.assertRaises(ValueError):
            preparar_recortes(self.base)
        self.assertEqual(label.read_text(), "alterado")
        self.assertEqual((destination / "manifesto.json").read_bytes(), original)

    def test_capture_group_cannot_cross_train_and_validation(self):
        self.add_parent("train", "source", [[100, 100, 140, 140]])
        self.add_parent("val", "empty")
        self.samples[1]["flight"] = self.samples[0]["flight"]
        self.write_manifest()
        with self.assertRaises(ValueError):
            preparar_recortes(self.base)
        self.assertFalse((self.base.parent / "gado-recortes").exists())


if __name__ == "__main__":
    unittest.main()
