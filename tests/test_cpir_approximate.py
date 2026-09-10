import json
import sys
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "cpir_approximate"
sys.path.insert(0, str(MODEL_DIR))
from cpir_approximate import run_all  # noqa: E402


class CPIRApproximateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_all()

    def test_contraction_and_defect(self):
        case = self.result["contractive"]
        self.assertAlmostEqual(case["alpha"], 0.5)
        self.assertAlmostEqual(case["one_step_defect"], 0.02)

    def test_geometric_bound_all_horizons(self):
        for row in self.result["contractive"]["horizons"]:
            self.assertLessEqual(row["error"], row["bound"] + 1e-12)

    def test_bound_starts_at_zero(self):
        row = self.result["contractive"]["horizons"][0]
        self.assertEqual(row["error"], 0.0)
        self.assertEqual(row["bound"], 0.0)

    def test_long_horizon_bound_saturates_below_delta_over_gap(self):
        case = self.result["contractive"]
        self.assertLess(case["horizons"][-1]["bound"], case["one_step_defect"] / (1 - case["alpha"]) + 1e-12)

    def test_macro_measurable_target(self):
        case = self.result["contractive"]
        self.assertLessEqual(case["bounded_target_error"], case["target_bound"] + 1e-12)

    def test_noncontractive_linear_bound(self):
        case = self.result["noncontractive"]
        self.assertEqual(case["alpha"], 1.0)
        self.assertAlmostEqual(case["linear_bound"], case["horizon"] * case["defect"])

    def test_hidden_target_counterexample(self):
        case = self.result["hidden_target"]
        self.assertEqual(case["macro_distribution_error"], 0.0)
        self.assertGreater(case["hidden_target_error"], 0.0)

    def test_frozen_results_match_current_code(self):
        frozen = json.loads((MODEL_DIR / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(frozen, json.loads(json.dumps(self.result)))

    def test_machine_contracts_exist(self):
        for name in ("README.md", "model-card.yml", "preregistration.yml", "results.json"):
            self.assertTrue((MODEL_DIR / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
