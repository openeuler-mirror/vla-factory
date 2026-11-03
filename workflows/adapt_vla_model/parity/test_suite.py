"""Regression checks for Judge-owned coverage and all-case verdicts."""
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from adapt_vla_model import judge
from adapt_vla_model.parity.parity_io import compare_loss


class SuiteTests(unittest.TestCase):
    def test_one_bad_case_cannot_be_hidden_by_other_passes(self):
        for bad_status in ("FAIL", "INCOMPLETE", "PASS"):
            with self.subTest(status=bad_status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                score = {"PASS": 0.5, "FAIL": 2.0, "INCOMPLETE": float("inf")}[bad_status]
                results = [{"status": "PASS", "normalized_error": 0.5} for _ in range(5)] + [
                    {"status": bad_status, "normalized_error": score}
                ]
                with patch.object(judge, "_run_parity_case", side_effect=results) as run:
                    result = judge._run_parity(root, None, root, root, {}, {}, root, "rev")
                self.assertEqual(bad_status, result["status"])
                self.assertEqual(6, len(result["cases"]))
                self.assertEqual(6, len({case["case_id"] for case in result["cases"]}))
                self.assertEqual([(s, 42) for s in range(6)],
                                 [(c.kwargs["sample_seed"], c.kwargs["seed"]) for c in run.call_args_list])
                self.assertTrue(Path(result["report"]).is_file())
                self.assertEqual("NOT_EVALUATED", result["coverage"]["independent_upstream_validation"])

    def test_gate_uses_fixed_reference_tolerance_and_matches_normalized_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            factory = root / "factory.json"
            report = root / "report.json"
            reference.write_text('{"loss": 1.0}', encoding="utf-8")

            # math.isclose(1, 2, rel_tol=.5) passes because the candidate value
            # enlarges its tolerance. Reference-normalized parity must fail.
            factory.write_text('{"loss": 2.0}', encoding="utf-8")
            self.assertFalse(compare_loss(reference, factory, report, atol=0.0, rtol=0.5))
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(0.5, result["reference_tolerance"])
            self.assertEqual(2.0, result["normalized_error"])

            factory.write_text('{"loss": 1.5}', encoding="utf-8")
            self.assertTrue(compare_loss(reference, factory, report, atol=0.0, rtol=0.5))
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual("PASS", result["status"])
            self.assertEqual(1.0, result["normalized_error"])

    def test_nonfinite_losses_fail_even_when_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("NaN", "Infinity", "-Infinity"):
                (root / "loss.json").write_text('{"loss": ' + value + '}')
                self.assertFalse(compare_loss(root / "loss.json", root / "loss.json", root / "report.json"))
