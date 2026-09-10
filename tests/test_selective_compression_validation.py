import json
import sys
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "selective_compression_validation"
sys.path.insert(0, str(MODEL_DIR))

from selective_compression_validation import (  # noqa: E402
    hierarchical_matrix,
    circulating_matrix,
    metastable_matrix,
    run_analysis,
    run_hierarchical,
    run_metastable,
    run_circulating,
)


class SelectiveCompressionValidationTests(unittest.TestCase):
    def assert_nested_close(self, frozen, current, path="root", tolerance=1e-12):
        """Compare registered numerics semantically across BLAS/LAPACK builds."""
        if isinstance(frozen, dict):
            self.assertIsInstance(current, dict, path)
            self.assertEqual(set(frozen), set(current), path)
            for key in frozen:
                self.assert_nested_close(frozen[key], current[key], f"{path}.{key}", tolerance)
            return
        if isinstance(frozen, list):
            self.assertIsInstance(current, list, path)
            self.assertEqual(len(frozen), len(current), path)
            for index, (left, right) in enumerate(zip(frozen, current)):
                self.assert_nested_close(left, right, f"{path}[{index}]", tolerance)
            return
        if isinstance(frozen, (int, float)) and not isinstance(frozen, bool):
            self.assertIsInstance(current, (int, float), path)
            self.assertAlmostEqual(float(frozen), float(current), delta=tolerance, msg=path)
            return
        self.assertEqual(frozen, current, path)

    def test_registered_matrices_are_stochastic(self):
        for matrix in (metastable_matrix(), hierarchical_matrix(), circulating_matrix()):
            self.assertTrue((matrix >= 0).all())
            for row in matrix:
                self.assertAlmostEqual(float(row.sum()), 1.0)

    def test_metastable_partition_is_exact_and_aligned(self):
        result = run_metastable()
        self.assertLessEqual(result["closure_defect"], 1e-12)
        self.assertLessEqual(result["memory_residual"], 1e-12)
        self.assertLessEqual(result["response_intra_distortion"], 1e-12)
        self.assertGreaterEqual(result["dynamical_response_alignment"], 1 - 1e-12)

    def test_asymmetry_breaks_metastable_interface(self):
        result = run_metastable(asymmetry=0.4)
        self.assertGreater(result["closure_defect"], 0)
        self.assertGreater(result["memory_residual"], 0)
        self.assertGreater(result["response_intra_distortion"], 0)

    def test_hierarchical_macro_interface_is_exact(self):
        result = run_hierarchical()
        self.assertLessEqual(result["closure_defect"], 1e-12)
        self.assertLessEqual(result["memory_residual"], 1e-12)
        self.assertLessEqual(result["response_intra_distortion"], 1e-12)
        self.assertGreaterEqual(result["dynamical_response_alignment"], 1 - 1e-12)
        self.assertEqual(result["response_rg_alignment"], 1.0)

    def test_registered_nuisance_grid_forms_exact_plateau(self):
        analysis = run_analysis()
        for variation in analysis["plateau_variation"].values():
            self.assertLessEqual(variation, analysis["registered_tolerance"])

    def test_circulating_family_is_nonreversible(self):
        result = run_circulating()
        self.assertGreater(result["detailed_balance_defect"], 0.0)
        self.assertLessEqual(result["closure_defect"], 1e-12)

    def test_circulating_heterogeneity_breaks_joint_interface(self):
        exact = run_circulating(0.0)
        broken = run_circulating(0.8)
        self.assertGreater(broken["closure_defect"], exact["closure_defect"])
        self.assertGreater(broken["memory_residual"], exact["memory_residual"])
        self.assertGreater(broken["response_intra_distortion"], exact["response_intra_distortion"])

    def test_nonreversible_platform_has_registered_endpoint(self):
        analysis = run_analysis()
        endpoint = analysis["circulating_platform_endpoint"]
        self.assertIsNotNone(endpoint)
        self.assertGreaterEqual(endpoint, 0.0)
        self.assertLess(endpoint, 0.8)

    def test_frozen_results_match_current_code(self):
        frozen = json.loads((MODEL_DIR / "results.json").read_text(encoding="utf-8"))
        self.assert_nested_close(frozen, run_analysis())

    def test_machine_contracts_exist(self):
        for name in ("README.md", "model-card.yml", "preregistration.yml", "results.json"):
            self.assertTrue((MODEL_DIR / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
