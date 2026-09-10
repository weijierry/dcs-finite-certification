import json
import sys
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "cpir_exact"
sys.path.insert(0, str(MODEL_DIR))
from cpir_exact import run_all  # noqa: E402


class CPIRExactRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_all()

    def test_positive_exact_quotient(self):
        positive = self.result["positive"]
        self.assertEqual(positive["lumpability_defect"], 0.0)
        self.assertEqual(positive["target_measurability_defect"], 0.0)
        self.assertEqual(positive["persistence_measurability_defect"], 0.0)
        self.assertTrue(positive["implementation_invariant"])
        self.assertEqual(positive["macro_kernel"], [[0.8, 0.2], [0.3, 0.7]])

    def test_x1(self):
        self.assertTrue(self.result["CPIR-X1"]["macro_cycle"])

    def test_x2(self):
        self.assertEqual(self.result["CPIR-X2"]["repaired_block_count"], 1)
        self.assertTrue(self.result["CPIR-X2"]["requires_recertification"])

    def test_x3(self):
        self.assertGreater(self.result["CPIR-X3"]["lumpability_defect"], 0.0)

    def test_x4(self):
        self.assertFalse(self.result["CPIR-X4"]["implementation_invariant"])

    def test_x5(self):
        self.assertEqual(self.result["CPIR-X5"]["lumpability_defect"], 0.0)
        self.assertGreater(self.result["CPIR-X5"]["target_measurability_defect"], 0.0)

    def test_x6(self):
        self.assertTrue(self.result["CPIR-X6"]["one_step_equal"])
        self.assertFalse(self.result["CPIR-X6"]["two_step_equal"])

    def test_x7(self):
        self.assertGreater(self.result["CPIR-X7"]["persistence_measurability_defect"], 0.0)

    def test_x8(self):
        self.assertGreater(self.result["CPIR-X8"]["compression_ratio"], 1.0)
        self.assertFalse(self.result["CPIR-X8"]["new_reachable_capability"])

    def test_all_registered_cases_exist(self):
        self.assertEqual(set(self.result), {"positive"} | {f"CPIR-X{i}" for i in range(1, 9)})

    def test_frozen_results_match_current_code(self):
        frozen = json.loads((MODEL_DIR / "results.json").read_text(encoding="utf-8"))
        normalized = json.loads(json.dumps(self.result))
        self.assertEqual(frozen, normalized)

    def test_machine_contracts_exist(self):
        for name in ("README.md", "model-card.yml", "preregistration.yml", "results.json"):
            self.assertTrue((MODEL_DIR / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
