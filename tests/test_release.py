from __future__ import annotations

import json
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseSurfaceTests(unittest.TestCase):
    def test_default_install_is_cli_only(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["requires-python"], "==3.13.5")
        self.assertEqual(project["project"]["dependencies"], ["httpx>=0.28,<1"])
        self.assertEqual(
            set(project["project"]["optional-dependencies"]["benchmark"]),
            {"jsonargparse==4.49.0", "numpy==2.5.0", "torch==2.12.1"},
        )
        self.assertEqual(
            project["tool"]["setuptools"]["packages"],
            ["benchmark", "benchmark.manifests", "client", "data"],
        )

    def test_operator_only_files_are_absent(self) -> None:
        for relative in (
            "service/app.py",
            "service/db.py",
            "service/tiers.py",
            "modal_runner.py",
            "Dockerfile",
            "Dockerfile.modal-deploy",
            "docker-compose.yml",
            "PROTOTYPE.md",
            ".env.example",
            "shells/deploy_modal.sh",
        ):
            with self.subTest(path=relative):
                self.assertFalse((ROOT / relative).exists())

    def test_generator_covers_public_accelerator_manifests_only(self) -> None:
        generator = (ROOT / "scripts" / "generate_datasets.sh").read_text(
            encoding="utf-8"
        )
        manifest_dir = ROOT / "benchmark" / "manifests"
        public_manifests = sorted(manifest_dir.glob("h100_easy_*.json")) + sorted(
            manifest_dir.glob("h100_medium_*.json")
        )
        for manifest_path in public_manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data_root = manifest["data"]["data_root"]
            with self.subTest(manifest=manifest_path.name):
                self.assertIn(f"--output_dir {data_root}", generator)

        self.assertEqual(list(manifest_dir.glob("h100_hard_*.json")), [])
        self.assertNotIn("hard", generator.lower())

    def test_open_source_documents_are_present(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertIn("uv tool install git+https://github.com/", readme)
        self.assertIn("Apache License 2.0", readme)


if __name__ == "__main__":
    unittest.main()
