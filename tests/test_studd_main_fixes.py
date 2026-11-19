"""Deterministic standard-library unit tests for STUDD main branch fixes.

Run with:
    python3 -m unittest discover -s tests -v
"""

import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_METHODS = {
    "BL-st",
    "BL-ret",
    "STUDD",
    "SS",
    "DSS",
    "WS",
    "DWS",
    "OS",
    "OF",
    "FF",
}

# Cerqueira et al. (2023) Table 3 (Cohen's Kappa) & Table 4 (Labeling Cost) for STUDD.
EXPECTED_STUDD_PAPER_METRICS = {
    "Arabic": (0.71, 0.23),
    "ArabicShuffled": (0.88, 0.11),
    "Bike": (0.32, 0.06),
    "CoverType": (0.52, 0.09),
    "Electricity": (0.29, 0.43),
    "GMSC": (0.32, 0.01),
    "Gas": (0.48, 0.86),
    "Insects": (0.81, 0.09),
    "InsectsAbrupt": (0.73, 0.28),
    "Keystroke": (0.84, 0.31),
    "Luxembourg": (1.00, 0.53),
    "NOAA": (0.39, 0.06),
    "Outdoor": (0.36, 0.25),
    "Ozone": (0.13, 0.39),
    "Poker": (0.48, 0.01),
    "Posture": (0.50, 0.02),
    "Powersupply": (0.10, 0.07),
    "Rialto": (0.38, 0.21),
    "Sensors": (0.59, 0.16),
}


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStuddMainFixes(unittest.TestCase):
    """Comprehensive deterministic verification suite for Workstreams 1-3."""

    def test_all_python_files_compile(self) -> None:
        """Test 1: Every .py file under src/ and experiments/ compiles cleanly."""
        py_files = sorted(
            list((REPO_ROOT / "src").rglob("*.py"))
            + list((REPO_ROOT / "experiments").rglob("*.py"))
        )
        self.assertGreater(len(py_files), 0, "Expected Python files under src/ and experiments/.")

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, py_file in enumerate(py_files):
                cfile = os.path.join(tmpdir, f"{idx}_{py_file.stem}.pyc")
                py_compile.compile(str(py_file), cfile=cfile, doraise=True)

    def test_page_hinkley_no_false_alarm_after_loss_drop(self) -> None:
        """Test 2: Fixed Page-Hinkley (both copies identical) yields 0 false alarms after loss drop."""
        main_det_path = REPO_ROOT / "src" / "detector.py"
        khoa_det_path = REPO_ROOT / "src" / "khoa" / "detector.py"

        self.assertEqual(
            main_det_path.read_bytes(),
            khoa_det_path.read_bytes(),
            "src/detector.py and src/khoa/detector.py must be byte-identical.",
        )

        for label, det_path in (("src.detector", main_det_path), ("src.khoa.detector", khoa_det_path)):
            with self.subTest(module=label):
                mod = _load_module_from_path(f"_test_{label.replace('.', '_')}", det_path)
                det = mod.PageHinkley(delta=0.001, threshold=50.0, alpha=0.9999, min_instances=30)

                alarms = 0
                for _ in range(200):
                    drift, stat = det.update(0.5)
                    self.assertGreaterEqual(stat, 0.0)
                    self.assertEqual(det.min_sum_deviation, 0.0)
                    if drift:
                        alarms += 1

                for _ in range(25_000):
                    drift, stat = det.update(0.05)
                    self.assertGreaterEqual(stat, 0.0)
                    self.assertGreaterEqual(det.sum_deviation, 0.0)
                    self.assertEqual(det.min_sum_deviation, 0.0)
                    if drift:
                        alarms += 1

                self.assertEqual(
                    alarms,
                    0,
                    f"{label}.PageHinkley fired {alarms} spurious false alarm(s) on stationary post-drop stream.",
                )

    def test_page_hinkley_detects_abrupt_drift(self) -> None:
        """Test 3: Fixed Page-Hinkley detects abrupt drift and resets internal state cleanly."""
        det_paths = (
            ("src.detector", REPO_ROOT / "src" / "detector.py"),
            ("src.khoa.detector", REPO_ROOT / "src" / "khoa" / "detector.py"),
        )
        for label, det_path in det_paths:
            with self.subTest(module=label):
                mod = _load_module_from_path(f"_test_drift_{label.replace('.', '_')}", det_path)
                det = mod.PageHinkley(delta=0.001, threshold=50.0, alpha=0.9999, min_instances=30)

                for step in range(200):
                    drift, stat = det.update(0.0)
                    self.assertFalse(drift, f"Unexpected alarm during stationary phase at step {step}")
                    self.assertEqual(stat, 0.0)

                detected = False
                alarm_stat = 0.0
                for _ in range(200):
                    drift, stat = det.update(1.0)
                    if drift:
                        detected = True
                        alarm_stat = stat
                        break

                self.assertTrue(detected, f"{label}.PageHinkley failed to detect abrupt drift.")
                self.assertGreater(alarm_stat, 50.0)
                self.assertEqual(det.sample_count, 0)
                self.assertEqual(det.sum_deviation, 0.0)
                self.assertEqual(det.running_mean, 0.0)
                self.assertEqual(det.min_sum_deviation, 0.0)

    def test_khoa_runner_and_notebooks_separate_teacher_student_seeds(self) -> None:
        """Test 4: Khoa runner and all 6 notebooks pass distinct seeds (seed vs seed + 100)."""
        runner_path = REPO_ROOT / "experiments" / "reproduce_khoa_6datasets.py"
        runner_src = runner_path.read_text(encoding="utf-8")
        runner_tree = ast.parse(runner_src, filename=str(runner_path))

        eval_fn = next(
            (
                node
                for node in runner_tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "evaluate_dataset"
            ),
            None,
        )
        self.assertIsNotNone(eval_fn, "evaluate_dataset not found in reproduce_khoa_6datasets.py")

        learner_seeds = []
        for node in ast.walk(eval_fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "create_learner"
            ):
                seed_kw = next((kw for kw in node.keywords if kw.arg == "random_state"), None)
                self.assertIsNotNone(seed_kw, "create_learner call missing random_state keyword")
                learner_seeds.append(ast.unparse(seed_kw.value))

        self.assertEqual(
            learner_seeds,
            ["seed", "seed + 100"],
            f"Expected distinct Teacher/Student seeds ['seed', 'seed + 100'], got {learner_seeds}",
        )

        notebook_paths = sorted((REPO_ROOT / "notebooks" / "khoa").glob("0[1-6]-*.ipynb"))
        self.assertEqual(len(notebook_paths), 6, f"Expected 6 Khoa notebooks, found {len(notebook_paths)}")

        for nb_path in notebook_paths:
            with self.subTest(notebook=nb_path.name):
                nb = json.loads(nb_path.read_text(encoding="utf-8"))
                code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
                self.assertGreaterEqual(len(code_cells), 6)

                # Validate Python syntax of every code cell (ignoring IPython line magics).
                for idx, cell in enumerate(code_cells):
                    lines = [
                        line
                        for line in "".join(cell["source"]).splitlines(keepends=True)
                        if not line.lstrip().startswith("%")
                    ]
                    ast.parse("".join(lines), filename=f"{nb_path.name}:cell{idx}")

                cell4_src = "".join(nb["cells"][4]["source"])
                cell4_tree = ast.parse(cell4_src, filename=f"{nb_path.name}:cell4")

                fit_pred_fn = next(
                    (
                        n
                        for n in cell4_tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "fit_predictor"
                    ),
                    None,
                )
                self.assertIsNotNone(fit_pred_fn, f"fit_predictor missing in {nb_path.name}")
                arg_names = [a.arg for a in fit_pred_fn.args.args]
                self.assertIn("seed", arg_names, f"fit_predictor missing seed parameter in {nb_path.name}")

                fit_pair_fn = next(
                    (
                        n
                        for n in cell4_tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "fit_pair"
                    ),
                    None,
                )
                self.assertIsNotNone(fit_pair_fn, f"fit_pair missing in {nb_path.name}")
                pair_seeds = []
                for node in ast.walk(fit_pair_fn):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "fit_predictor"
                    ):
                        seed_kw = next((kw for kw in node.keywords if kw.arg == "seed"), None)
                        self.assertIsNotNone(
                            seed_kw, f"fit_predictor call in fit_pair missing seed= in {nb_path.name}"
                        )
                        pair_seeds.append(ast.unparse(seed_kw.value))
                self.assertEqual(
                    pair_seeds,
                    ["SEED", "SEED + 100"],
                    f"Expected ['SEED', 'SEED + 100'] in fit_pair of {nb_path.name}, got {pair_seeds}",
                )

                cell6_src = "".join(nb["cells"][6]["source"])
                self.assertIn('"student_seed"', cell6_src, f"Cell 6 missing student_seed in {nb_path.name}")
                self.assertIn("SEED + 100", cell6_src, f"Cell 6 missing SEED + 100 in {nb_path.name}")

    def test_data_loader_bike_and_insects_parser_source(self) -> None:
        """Test 5: _parse_hosted_dataframe handles insects_zip (header=None) and bike_hourly_csv without leakage."""
        loader_path = REPO_ROOT / "src" / "data_loader.py"
        loader_src = loader_path.read_text(encoding="utf-8")
        tree = ast.parse(loader_src, filename=str(loader_path))

        parser_fn = next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_parse_hosted_dataframe"
            ),
            None,
        )
        self.assertIsNotNone(parser_fn, "_parse_hosted_dataframe not found in src/data_loader.py")

        insects_if = None
        bike_if = None
        for stmt in parser_fn.body:
            if isinstance(stmt, ast.If):
                test_str = ast.unparse(stmt.test)
                self.assertNotIn(
                    '{"insects_zip", "bike_hourly_csv"}',
                    test_str,
                    "insects_zip and bike_hourly_csv must not share a combined branch.",
                )
                if 'fmt == "insects_zip"' in test_str or "fmt == 'insects_zip'" in test_str:
                    insects_if = stmt
                elif 'fmt == "bike_hourly_csv"' in test_str or "fmt == 'bike_hourly_csv'" in test_str:
                    bike_if = stmt

        self.assertIsNotNone(insects_if, "Separate `if fmt == 'insects_zip':` branch missing.")
        self.assertIsNotNone(bike_if, "Separate `if fmt == 'bike_hourly_csv':` branch missing.")

        insects_src = ast.get_source_segment(loader_src, insects_if) or ""
        self.assertIn("header=None", insects_src)
        self.assertIn("AbruptInsects.data", insects_src)
        self.assertIn("ZipFile", insects_src)

        # Verify every pd.read_csv call in insects_zip passes header=None.
        insects_read_csv_calls = [
            node
            for node in ast.walk(insects_if)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_csv"
        ]
        self.assertGreaterEqual(len(insects_read_csv_calls), 2)
        for call in insects_read_csv_calls:
            header_kw = next((kw for kw in call.keywords if kw.arg == "header"), None)
            self.assertIsNotNone(header_kw, "read_csv in insects_zip branch missing header=None")
            self.assertEqual(ast.unparse(header_kw.value), "None")

        bike_src = ast.get_source_segment(loader_src, bike_if) or ""
        for col in ("temp", "atemp", "hum", "windspeed"):
            self.assertIn(f'"{col}"', bike_src)
        self.assertIn("142", bike_src)
        self.assertIn('"cnt"', bike_src)
        self.assertIn('"target"', bike_src)

        # Ensure neither 'casual' nor 'registered' is referenced as a string literal in bike_hourly_csv AST.
        bike_string_literals = {
            node.value
            for node in ast.walk(bike_if)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertNotIn("casual", bike_string_literals)
        self.assertNotIn("registered", bike_string_literals)

    def test_integrated_190_csv_schema_and_paper_tables(self) -> None:
        """Test 6: results/studd_19datasets_summary.csv has 190 valid rows and canonical paper values."""
        csv_path = REPO_ROOT / "results" / "studd_19datasets_summary.csv"
        self.assertTrue(csv_path.is_file(), f"Missing {csv_path}")

        with csv_path.open(encoding="utf-8-sig", newline="") as fp:
            rows = list(csv.DictReader(fp))

        self.assertEqual(len(rows), 190, f"Expected 190 rows, found {len(rows)}")
        pairs = {(r["Dataset"], r["Method"]) for r in rows}
        self.assertEqual(len(pairs), 190, "Duplicate (Dataset, Method) pairs detected.")

        datasets = {r["Dataset"] for r in rows}
        methods = {r["Method"] for r in rows}
        self.assertEqual(len(datasets), 19, f"Expected 19 datasets, found {len(datasets)}")
        self.assertEqual(datasets, set(EXPECTED_STUDD_PAPER_METRICS.keys()))
        self.assertEqual(methods, EXPECTED_METHODS)

        studd_rows = {}
        for row in rows:
            wandb_url = row["WandB_Run"].strip()
            self.assertTrue(
                wandb_url.startswith("https://wandb.ai/"),
                f"Non-canonical WandB_Run URL for {row['Dataset']}/{row['Method']}: {wandb_url}",
            )
            if row["Method"] == "BL-ret":
                self.assertEqual(
                    row["Alarms"].strip(),
                    "0",
                    f"BL-ret Alarms must be '0' for {row['Dataset']}, got {row['Alarms']!r}",
                )
                self.assertTrue(
                    row["Retrains"].strip().isdigit() and int(row["Retrains"].strip()) > 0,
                    f"BL-ret Retrains must be a non-empty positive integer for {row['Dataset']}, got {row['Retrains']!r}",
                )
            elif row["Method"] == "STUDD":
                studd_rows[row["Dataset"]] = (float(row["Paper_Kappa"]), float(row["Paper_Cost"]))

        self.assertEqual(len(studd_rows), 19)
        for dataset, (exp_kappa, exp_cost) in EXPECTED_STUDD_PAPER_METRICS.items():
            got_kappa, got_cost = studd_rows[dataset]
            self.assertAlmostEqual(
                got_kappa,
                exp_kappa,
                places=2,
                msg=f"STUDD Paper_Kappa mismatch for {dataset}: got {got_kappa}, expected {exp_kappa}",
            )
            self.assertAlmostEqual(
                got_cost,
                exp_cost,
                places=2,
                msg=f"STUDD Paper_Cost mismatch for {dataset}: got {got_cost}, expected {exp_cost}",
            )

    def test_docs_accuracy_and_artifacts(self) -> None:
        """Test 7: README accuracy, experiments/README.md cleanup, and rank plot PNG validity."""
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn(
            "online learner",
            readme_text.lower(),
            "README.md must not describe STUDD's Student as an 'online learner'.",
        )

        exp_readme_text = (REPO_ROOT / "experiments" / "README.md").read_text(encoding="utf-8")
        self.assertNotIn(
            "prepare_data.py",
            exp_readme_text,
            "experiments/README.md must not reference missing prepare_data.py.",
        )

        rank_plot_path = REPO_ROOT / "results" / "fig_19datasets_rank_plot.png"
        self.assertTrue(rank_plot_path.is_file(), f"Missing {rank_plot_path}")
        raw_header = rank_plot_path.read_bytes()[:8]
        self.assertEqual(
            raw_header,
            b"\x89PNG\r\n\x1a\n",
            f"{rank_plot_path} does not have valid PNG magic bytes.",
        )


if __name__ == "__main__":
    unittest.main()
