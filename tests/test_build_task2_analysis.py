import unittest
from pathlib import Path

import pandas as pd

from src.build_task2_analysis import build_report, select_failure_case


class BuildTask2AnalysisTests(unittest.TestCase):
    def test_select_failure_case_uses_lowest_overall_quality(self) -> None:
        per_query = pd.DataFrame(
            [
                {"Query": "success", "bm25": 0.8, "tfidf": 0.4, "winner": "bm25"},
                {"Query": "mixed", "bm25": 0.8, "tfidf": 0.2, "winner": "bm25"},
                {"Query": "failure", "bm25": 0.4, "tfidf": 0.2, "winner": "bm25"},
            ]
        )

        failure_case = select_failure_case(per_query)

        self.assertEqual(failure_case["Query"], "failure")

    def test_build_report_mentions_success_and_failure_case(self) -> None:
        summary_row = pd.Series(
            {
                "evaluated_queries": 2,
                "avg_tfidf_p_at_5": 0.3,
                "avg_bm25_p_at_5": 0.6,
                "baseline_wins": 0,
                "improved_wins": 2,
                "ties": 0,
            }
        )
        per_query = pd.DataFrame(
            [
                {"Query": "success query", "bm25": 0.8, "tfidf": 0.4, "winner": "bm25"},
                {"Query": "failure query", "bm25": 0.4, "tfidf": 0.2, "winner": "bm25"},
            ]
        )

        report = build_report(summary_row, per_query, Path("/nonexistent"))

        self.assertIn("### Success Case", report)
        self.assertIn("### Failure Case", report)
        self.assertIn("Failure case (lowest overall retrieval quality)", report)


if __name__ == "__main__":
    unittest.main()
