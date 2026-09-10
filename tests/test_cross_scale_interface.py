import sys
import unittest
import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "cross_scale_interface"
sys.path.insert(0, str(MODEL_DIR))

from cross_scale_interface import (  # noqa: E402
    ACTIONS,
    STATES,
    controlled_transition_matrices,
    exact_interface_frontier,
    pareto_frontier,
    partition_certificate,
    partition_label,
    registered_partition,
    run_analysis,
    set_partitions,
    with_failure_block,
)


class CrossScaleInterfaceTests(unittest.TestCase):
    def test_all_live_partitions_are_enumerated(self):
        self.assertEqual(len(set_partitions()), 15)

    def test_controlled_kernels_are_stochastic(self):
        matrices = controlled_transition_matrices()
        self.assertEqual(set(matrices), set(ACTIONS))
        for matrix in matrices.values():
            self.assertEqual(len(matrix), len(STATES))
            for row in matrix:
                self.assertAlmostEqual(sum(row), 1.0)
                self.assertTrue(all(value >= 0.0 for value in row))

    def test_registered_partition_has_joint_exact_certificate(self):
        certificate = partition_certificate(registered_partition())
        self.assertAlmostEqual(certificate["response_intra_distortion"], 0.0)
        self.assertGreater(certificate["response_inter_separation"], 0.0)
        self.assertAlmostEqual(certificate["closure_defect"], 0.0)
        self.assertAlmostEqual(certificate["memory_residual"], 0.0)
        self.assertAlmostEqual(certificate["intervention_exchange_error"], 0.0)
        self.assertAlmostEqual(certificate["persistence_distortion"], 0.0)
        self.assertEqual(certificate["causal_response_rank"], 1)

    def test_heterogeneity_breaks_response_and_closure(self):
        matrices = controlled_transition_matrices(0.25)
        certificate = partition_certificate(registered_partition(), matrices)
        self.assertGreater(certificate["response_intra_distortion"], 0.0)
        self.assertGreater(certificate["closure_defect"], 0.0)
        self.assertGreater(certificate["memory_residual"], 0.0)

    def test_registered_partition_is_on_symmetric_pareto_frontier(self):
        result = run_analysis()
        label = partition_label(with_failure_block(registered_partition()))
        self.assertIn(label, result["pareto_labels"])

    def test_scientific_gate_excludes_trivial_pareto_extremes(self):
        certificates = [partition_certificate(partition) for partition in set_partitions()]
        labels = {item["label"] for item in exact_interface_frontier(certificates)}
        registered = partition_label(with_failure_block(registered_partition()))
        self.assertEqual(labels, {registered})

    def test_heterogeneous_model_has_no_exact_nontrivial_interface(self):
        result = run_analysis(0.25)
        self.assertEqual(result["exact_interface_labels"], [])

    def test_identity_partition_is_dominated_by_reusable_interface(self):
        certificates = [partition_certificate(partition) for partition in set_partitions()]
        frontier_labels = {item["label"] for item in pareto_frontier(certificates)}
        identity_label = partition_label(with_failure_block(tuple((state,) for state in STATES[:-1])))
        self.assertNotIn(identity_label, frontier_labels)

    def test_full_analysis_reports_every_certificate(self):
        result = run_analysis()
        self.assertEqual(result["partition_count"], 15)
        self.assertEqual(len(result["certificates"]), 15)
        self.assertGreaterEqual(result["pareto_count"], 1)

    def test_frozen_results_match_current_code(self):
        result_path = MODEL_DIR / "results.json"
        frozen = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(frozen["symmetric"], run_analysis(0.0))
        self.assertEqual(frozen["heterogeneous"], run_analysis(0.25))

    def test_machine_contracts_are_present(self):
        for name in ("model-card.yml", "preregistration.yml", "results.json", "README.md"):
            self.assertTrue((MODEL_DIR / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
