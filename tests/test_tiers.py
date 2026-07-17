from __future__ import annotations

from pathlib import Path
import unittest

from benchmark.manifest import load_manifest
from client.catalog import TIERS, resolve_tier_dataset


ROOT = Path(__file__).resolve().parents[1]


class TierCatalogTests(unittest.TestCase):
    def test_catalog_has_expected_budgets_quotas_and_datasets(self) -> None:
        self.assertEqual([tier.id for tier in TIERS], ["easy", "medium", "hard"])
        self.assertEqual([tier.training_seconds for tier in TIERS], [60, 600, 3600])
        self.assertEqual([tier.evaluation_seconds for tier in TIERS], [30, 300, 1800])
        self.assertEqual([tier.daily_attempts for tier in TIERS], [60, 6, 1])
        self.assertEqual([len(tier.datasets) for tier in TIERS], [5, 5, 1])

    def test_dataset_selection_is_tier_scoped(self) -> None:
        tier, dataset = resolve_tier_dataset("hard", None)
        self.assertEqual((tier.id, dataset.id), ("hard", "h1"))
        with self.assertRaisesRegex(ValueError, "dataset for Easy"):
            resolve_tier_dataset("easy", "m1")
        with self.assertRaisesRegex(ValueError, "unknown tier"):
            resolve_tier_dataset("extreme", "e1")

    def test_hard_public_label_does_not_disclose_dataset_contents(self) -> None:
        hard = next(tier for tier in TIERS if tier.id == "hard")
        self.assertEqual(
            [(dataset.id, dataset.label) for dataset in hard.datasets],
            [("h1", "H1 · Hidden evaluation")],
        )
        self.assertIsNone(hard.datasets[0].manifest_filename)

    def test_all_public_catalog_manifests_match_tier_runtime(self) -> None:
        for tier in TIERS:
            for dataset in tier.datasets:
                if dataset.manifest_filename is None:
                    continue
                with self.subTest(tier=tier.id, dataset=dataset.id):
                    manifest = load_manifest(
                        ROOT / "benchmark" / "manifests" / dataset.manifest_filename
                    )
                    self.assertEqual(
                        manifest.runtime.total_training_time_seconds,
                        tier.training_seconds,
                    )
                    self.assertEqual(manifest.runtime.seeds, (74,))
                    self.assertEqual(manifest.data.batch_size, 512)
                    self.assertEqual(manifest.data.eval_batch_size, 512)
                    self.assertEqual(manifest.model_state.maximum_elements, 500_000_000)
                    self.assertIsNotNone(manifest.data.data_root)


if __name__ == "__main__":
    unittest.main()
