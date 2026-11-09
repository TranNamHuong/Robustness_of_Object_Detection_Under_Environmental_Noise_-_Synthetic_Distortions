from __future__ import annotations

import unittest

from server.robustness_api.main import (
    MODEL_INFO, _build_pipeline, _validated_condition,
    conditions, models, health, app, recommendations, gallery,
)
from server.robustness_api.benchmark_repository import get_experiments, get_records


class ApiContractTests(unittest.TestCase):
    def test_health_and_model_readiness_shape(self):
        status = health()
        self.assertIn(status["device"], {"cpu", "cuda:0"})
        self.assertIn("yolov8s", status["models"])
        self.assertEqual({item["id"] for item in models()}, set(MODEL_INFO))
        self.assertTrue(all(item["pretrained"] for item in models()))

    def test_conditions_publish_documented_levels(self):
        by_id = {item["id"]: item for item in conditions()}
        self.assertEqual(by_id["gaussian"]["levels"], [0.01, 0.03, 0.06])
        self.assertIn("gaussian_nlm", by_id)
        self.assertEqual(by_id["clean"]["levels"], [])

    def test_only_measured_levels_are_accepted(self):
        self.assertEqual(_validated_condition("gaussian_nlm", 0.06), ("gaussian", 0.06, True))
        with self.assertRaisesRegex(ValueError, "was not measured"):
            _validated_condition("gaussian", 0.02)

    def test_seeded_corruption_is_repeatable_and_recovery_same_image(self):
        import numpy as np

        image = np.full((32, 40, 3), 128, dtype=np.uint8)
        first = _build_pipeline(image, "gaussian_nlm", 0.03, 7, "auto")
        second = _build_pipeline(image, "gaussian_nlm", 0.03, 7, "auto")
        np.testing.assert_array_equal(first[3], second[3])
        self.assertIsNotNone(first[4])
        self.assertEqual(first[2], "nlmeans")

    def test_invalid_upload_is_rejected_without_inference(self):
        from server.robustness_api.inference import validate_image_bytes

        with self.assertRaisesRegex(ValueError, "not a decodable image"):
            validate_image_bytes(b"not an image", "not-image.jpg")

    def test_unlabeled_inference_contract_has_no_accuracy_metrics(self):
        paths = {route.path for route in app.routes}
        self.assertTrue({"/api/health", "/api/models", "/api/conditions", "/api/detect", "/api/compare"}.issubset(paths))
        # The live response payload is built without GT fields in phase P2;
        # benchmark counts remain isolated in the dataset-backed API.
        self.assertIn("inference", app.description.lower())

    def test_benchmark_adapter_retains_protocol_and_counts(self):
        records = get_records("coco_normalized_320")
        self.assertEqual(len(records), 18)
        self.assertEqual({row["images"] for row in records}, {5000})
        self.assertEqual({row["input_size"] for row in records}, {320})
        self.assertEqual({row["protocol_id"] for row in records}, {"coco_normalized_320"})
        restored = next(row for row in records if row["model"] == "YOLOv8n" and row["condition"] == "salt_pepper_0.06" and row["stage"] == "restored")
        self.assertAlmostEqual(restored["mAP50"], 0.3951112450, places=7)
        self.assertEqual(restored["source_file"], "runs/normalized_detector_comparison_fullval/metrics.csv")

    def test_experiment_catalog_reports_real_data_coverage(self):
        catalog = {item["id"]: item for item in get_experiments()}
        self.assertEqual(catalog["coco_normalized_320"]["record_count"], 18)
        self.assertEqual(catalog["coco_normalized_320"]["model_count"], 2)
        self.assertEqual(catalog["open_images_300"]["record_count"], 1)
        self.assertEqual(catalog["open_images_300"]["map50_95_count"], 0)
        self.assertTrue(catalog["user_reference_unverified"]["reference_only"])

    def test_three_subset_aggregate_keeps_mean_and_std(self):
        records = get_records("coco_3x500_subsets")
        clean = next(row for row in records if row["condition"] == "clean")
        self.assertEqual(clean["subset_count"], 3)
        self.assertAlmostEqual(clean["mAP50"], 0.5412654296, places=8)
        self.assertAlmostEqual(clean["mAP50_std"], 0.0256415726, places=8)

    def test_external_sets_are_separate_and_keep_their_class_support(self):
        pilot = get_records("open_images_100")
        expansion = get_records("open_images_300")
        self.assertEqual({row["images"] for row in pilot}, {100})
        self.assertEqual({row["classes_scored"] for row in pilot}, {8})
        self.assertEqual({row["images"] for row in expansion}, {300})
        self.assertEqual({row["classes_scored"] for row in expansion}, {7})
        self.assertTrue(all(row["mAP50_95"] is None for row in pilot + expansion))

    def test_supplied_historical_table_is_reference_only(self):
        reference = get_records("user_reference_unverified")
        self.assertEqual(len(reference), 3)
        self.assertTrue(all(row.get("reference_only") for row in reference))
        self.assertTrue(all(row["images"] is None for row in reference))
        self.assertTrue(all("condition" in row["protocol"].lower() for row in reference))

    def test_recommendations_rank_only_same_protocol_and_block_reference(self):
        best = recommendations("coco_normalized_320", "map", "clean")
        self.assertEqual(best["status"], "ok")
        self.assertGreaterEqual(len(best["candidates"]), 2)
        self.assertEqual(best["winner"]["metric"], "mAP50_95")
        robust = recommendations("coco_normalized_320", "robustness", "gaussian_0.06")
        self.assertEqual(robust["status"], "ok")
        self.assertEqual(robust["winner"]["metric"], "relative_mAP_drop_pct")
        reference = recommendations("user_reference_unverified", "map", "clean")
        self.assertEqual(reference["status"], "insufficient_evidence")
        with self.assertRaisesRegex(Exception, "Select a measured noisy condition"):
            recommendations("coco_normalized_320", "robustness", None)
        for criterion in ("precision", "recall"):
            self.assertEqual(recommendations("coco_normalized_320", criterion, "clean")["status"], "ok")
        self.assertEqual(recommendations("coco_yolov8_clean_640", "realtime", "clean")["status"], "ok")
        self.assertEqual(recommendations("coco_normalized_320", "recovery", "gaussian_0.06")["status"], "ok")
        self.assertEqual(recommendations("open_images_100", "map", "clean")["status"], "insufficient_evidence")

    def test_gallery_keeps_ground_truth_provenance_and_confidence_filter(self):
        result = gallery(page=1, page_size=2, min_score=0.25)
        self.assertEqual(result["total"], 300)
        self.assertEqual(len(result["items"]), 2)
        first = result["items"][0]
        self.assertTrue(first["image_url"].startswith("/data/open-images-300/"))
        self.assertTrue(first["license"].startswith("https://"))
        self.assertTrue(first["ground_truth"])
        self.assertGreater(first["image_size"]["width"], 0)
        self.assertTrue(all(box["confidence"] >= 0.25 for box in first["predictions"]))


if __name__ == "__main__":
    unittest.main()
