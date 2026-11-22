"""Kiểm thử đơn vị tất định cho toàn bộ module & kết quả Full Scope (Commits 1–7)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_19datasets_full import bayesian_sign_test_dirichlet
from experiments.extract_full_alarms_and_figs import match_alarms_greedy
from experiments.run_sensitivity_rq3 import evaluate_delayed_sparse_supervised_pht
from src.baselines import evaluate_supervised_pht
from src.data_loader import ALL_19_DATASETS, _load_arabic_ts_fallback, generate_abrupt_drift_stream
from src.models import PAPER_TABLE2_MODELS, build_model


class TestFullScopeAnalytics(unittest.TestCase):
    """Bộ kiểm thử xác minh data loader hợp nhất, Bayesian Sign Test, Table 5, RQ3 và W&B."""

    def test_unified_19_dataset_registry_and_arabic_ts_fallback(self) -> None:
        self.assertEqual(len(ALL_19_DATASETS), 19)
        self.assertEqual(len(set(ALL_19_DATASETS)), 19)
        for ds in ALL_19_DATASETS:
            key = ds.lower().replace("-", "").replace("_", "")
            self.assertIn(key, PAPER_TABLE2_MODELS, f"Missing Table 2 entry for {ds}")

        manifest = json.loads((REPO_ROOT / "data" / "hosted_sources.json").read_text(encoding="utf-8"))
        for arab in ("Arabic", "ArabicShuffled"):
            spec = manifest["datasets"][arab]
            self.assertIn("url", spec)
            self.assertIn("kaggle_relpath", spec)
            self.assertIn("local_ts_dir", spec)

        X_fb, y_fb = _load_arabic_ts_fallback(str(REPO_ROOT / "data" / "dataset_huggingface" / "Arabic"), shuffled=True)
        self.assertEqual(X_fb.shape, (8798, 27))
        self.assertEqual(len(np.unique(y_fb)), 10)
        self.assertTrue(np.isfinite(X_fb).all())

    def test_bayesian_sign_test_rope_and_simplex_properties(self) -> None:
        win_stats = bayesian_sign_test_dirichlet(
            np.full(19, 0.85), np.full(19, 0.50), higher_is_better=True, rope=0.05, seed=42
        )
        self.assertEqual((win_stats["n_win"], win_stats["n_rope"], win_stats["n_lose"]), (19, 0, 0))
        self.assertGreater(win_stats["Simplex_P_Win"], 0.99)
        self.assertAlmostEqual(
            win_stats["Simplex_P_Win"] + win_stats["Simplex_P_ROPE"] + win_stats["Simplex_P_Lose"],
            1.0,
            places=5,
        )

        tie_stats = bayesian_sign_test_dirichlet(
            np.zeros(19), np.zeros(19), higher_is_better=True, rope=0.05, seed=42
        )
        self.assertEqual(tie_stats["n_rope"], 19)
        self.assertGreater(tie_stats["Simplex_P_ROPE"], 0.99)

        cost_stats = bayesian_sign_test_dirichlet(
            np.full(19, 0.10), np.full(19, 0.50), higher_is_better=False, rope=0.05, seed=42
        )
        self.assertEqual(cost_stats["n_win"], 19)
        self.assertGreater(cost_stats["Simplex_P_Win"], 0.99)

    def test_greedy_alarm_matcher_and_table5_csv_schema(self) -> None:
        tp, fp, fn, delays = match_alarms_greedy([100, 500], [120, 145, 900], window_size=50)
        self.assertEqual((tp, fp, fn), (1, 2, 1))
        self.assertEqual(delays, [20])

        t5_path = REPO_ROOT / "results" / "table5_alarm_analysis_19datasets.csv"
        self.assertTrue(t5_path.is_file(), f"Missing {t5_path}")
        df_t5 = pd.read_csv(t5_path)
        self.assertEqual(list(df_t5["Method"]), ["STUDD", "DSS", "WS", "DWS", "OS", "OF", "FF"])
        self.assertTrue(((df_t5["Our_MDR"] >= 0.0) & (df_t5["Our_MDR"] <= 1.0)).all())
        self.assertTrue(((df_t5["Our_FAR"] >= 0.0) & (df_t5["Our_FAR"] <= 1.0)).all())

    def test_delayed_sparse_pht_matches_ss_when_unconstrained(self) -> None:
        X_syn, y_syn, _ = generate_abrupt_drift_stream(
            n_samples=3000, n_features=6, drift_points=[1200], random_state=42
        )
        teacher = build_model("RF", random_state=42, n_jobs=1)
        ref_ss = evaluate_supervised_pht(
            X_syn,
            y_syn,
            base_model=teacher,
            window_size=400,
            delta=0.001,
            threshold=25.0,
            l_access=1.0,
            l_delay=0,
            method_name="SS",
            random_state=42,
        )
        rq3_ss = evaluate_delayed_sparse_supervised_pht(
            X_syn,
            y_syn,
            teacher,
            window_size=400,
            l_access=1.0,
            l_delay=0,
            delta=0.001,
            threshold=25.0,
            random_state=42,
        )
        self.assertEqual(rq3_ss["alarm_indices"], ref_ss["alarm_indices"])
        self.assertAlmostEqual(rq3_ss["kappa"], ref_ss["kappa"], places=10)
        self.assertAlmostEqual(rq3_ss["accuracy"], ref_ss["accuracy"], places=10)

        rq3_path = REPO_ROOT / "results" / "sensitivity_rq3_summary.csv"
        self.assertTrue(rq3_path.is_file(), f"Missing {rq3_path}")
        df_rq3 = pd.read_csv(rq3_path)
        self.assertEqual(len(df_rq3), 26)

    def test_fullscope_wandb_manifest_canonical_urls(self) -> None:
        manifest_path = REPO_ROOT / "results" / "fullscope_wandb_runs.json"
        self.assertTrue(manifest_path.is_file(), f"Missing {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs = payload.get("runs", {})
        expected_keys = (
            "table2_selection",
            "bayesian_efficiency",
            "table5_alarms",
            "sensitivity_rq3",
            "protocol_ablation",
        )
        for key in expected_keys:
            self.assertIn(key, runs, f"Missing W&B run entry '{key}'")
            url = str(runs[key].get("url", ""))
            self.assertTrue(url.startswith("https://wandb.ai/lgkai-hcmus/STUDD/runs/"), url)
            self.assertNotIn("forge.coreweave.com", url)


if __name__ == "__main__":
    unittest.main()
