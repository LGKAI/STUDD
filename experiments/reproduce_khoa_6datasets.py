"""Evaluate the six Khoa streams using the ten-method runner from dat-6datasets.

The Kaggle notebooks supply checksum-verified arrays and the Table 2 configuration.
Run this file with --publish to upload the exported metrics from an authenticated machine.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import sleep

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.khoa.baselines import ALL_METHODS, evaluate_all_10_methods
from src.khoa.studd import create_learner

PAPER_ORDER = ("STUDD", "BL-st", "BL-ret", "SS", "DSS", "WS", "DWS", "OS", "OF", "FF")
PAPER_TABLES = {
    "InsectsAbrupt": (
        (0.73, 0.59, 0.74, 0.78, 0.78, 0.79, 0.73, 0.79, 0.80, 0.77),
        (0.28, 0.09, 0.94, 0.19, 0.19, 0.19, 0.28, 0.19, 0.19, 0.66),
    ),
    "Insects": (
        (0.81, 0.81, 0.77, 0.74, 0.74, 0.79, 0.81, 0.71, 0.70, 0.77),
        (0.09, 0.09, 0.94, 0.28, 0.28, 0.09, 0.09, 0.47, 0.66, 0.94),
    ),
    "Keystroke": (
        (0.84, 0.84, 0.93, 0.88, 0.84, 0.84, 0.84, 0.84, 0.84, 0.93),
        (0.31, 0.31, 0.94, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.63),
    ),
    "Arabic": (
        (0.71, 0.63, 0.72, 0.78, 0.76, 0.68, 0.75, 0.73, 0.75, 0.72),
        (0.23, 0.11, 0.91, 0.34, 0.23, 0.80, 0.57, 0.34, 0.23, 0.80),
    ),
    "ArabicShuffled": (
        (0.88,) * 10,
        (0.11, 0.11, 0.91, 0.11, 0.11, 0.11, 0.11, 0.23, 0.11, 0.23),
    ),
    "Posture": (
        (0.50, 0.34, 0.51, 0.40, 0.48, 0.46, 0.50, 0.49, 0.47, 0.49),
        (0.02, 0.01, 0.99, 0.08, 0.05, 0.11, 0.11, 0.02, 0.02, 0.74),
    ),
}
BASELINE_REVISION = "00834c380c32ff41ed7746d41e949babe3aca658"
DELTA = 0.001
PH_THRESHOLD = 50.0
KS_ALPHA = 0.001


def evaluate_dataset(X, y, *, dataset, slug, teacher, student, window, seed, source, checksum, output):
    """Evaluate a verified stream and export comparable metrics and history."""
    # OS/OF in the shared runner may compare numeric class codes. The
    # mapping is used only as a representation of labels, never as a feature.
    _, labels = np.unique(y, return_inverse=True)
    models = [create_learner(code, random_state=seed, n_jobs=2) for code in (teacher, student)]
    results = evaluate_all_10_methods(
        X, labels, *models, window_size=window, random_state=seed,
        delta=DELTA, ph_threshold=PH_THRESHOLD, ks_alpha=KS_ALPHA,
    )
    paper_kappa, paper_cost = (dict(zip(PAPER_ORDER, values)) for values in PAPER_TABLES[dataset])
    rows, timeline, alarms = [], {}, []
    n_eval = len(X) - window
    stride = max(25, n_eval // 1000)
    sample_indices = np.unique(np.r_[np.arange(stride-1, n_eval, stride), n_eval-1])
    timeline["stream_step"] = sample_indices + 1

    for method in ALL_METHODS:
        res = results[method]
        assert len(res["y_pred_online"]) == n_eval
        kappa, cost = res["kappa"], res["labeling_cost"]
        delta_kappa = kappa - paper_kappa[method]
        rows.append({
            "Dataset": dataset, "Method": method, "Teacher": teacher,
            "Student": student if method == "STUDD" else "",
            "Samples": len(X), "Window_Size": window,
            "Our_Accuracy": res["accuracy"], "Our_Kappa": kappa,
            "Paper_Kappa": paper_kappa[method], "Kappa_Diff": delta_kappa,
            "Kappa_Rel_Diff_Pct": 100 * delta_kappa / paper_kappa[method],
            "Our_Cost": cost, "Paper_Cost": paper_cost[method],
            "Cost_Diff": cost - paper_cost[method],
            "Alarms": res["num_alarms"], "Retrains": res["num_retrains"],
        })
        correct = res["y_true_online"] == res["y_pred_online"]
        rolling = pd.Series(correct).rolling(200).mean().to_numpy()
        timeline[f"{method}/accuracy_200"] = rolling[sample_indices]
        alarms.extend({"Method": method, "Source_Row_Zero_Based": int(row)}
                      for row in res["alarm_indices"])
    output.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / f"{slug}_reproduction.csv", index=False)
    pd.DataFrame(timeline).to_csv(output / f"{slug}_history.csv", index=False)
    pd.DataFrame(alarms, columns=["Method", "Source_Row_Zero_Based"]).to_csv(
        output / f"{slug}_alarms.csv", index=False,
    )
    config = {
        "dataset": dataset, "teacher": teacher, "student": student, "seed": seed,
        "samples": len(X), "stream_size": n_eval, "window_size": window,
        "source_url": source, "source_sha256": checksum,
        "detector": "dat-6datasets PageHinkley", "delta": DELTA, "ph_threshold": PH_THRESHOLD,
        "ks_alpha": KS_ALPHA, "baseline_revision": BASELINE_REVISION,
        "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")},
    }
    (output / f"{slug}_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(comparison[["Method", "Our_Kappa", "Paper_Kappa", "Alarms"]].to_string(index=False))
    return comparison


def publish_dataset(slug, output, *, entity, project, existing_run=None):
    """Publish one Kaggle dataset result with ten series and per-method summaries."""
    import wandb

    frame = pd.read_csv(output / f"{slug}_reproduction.csv", keep_default_na=False)
    assert tuple(frame["Method"]) == ALL_METHODS
    config = json.loads((output / f"{slug}_config.json").read_text())
    history = pd.read_csv(output / f"{slug}_history.csv")
    alarms = pd.read_csv(output / f"{slug}_alarms.csv")
    metrics = {
        f"{r.Method}/{key}": getattr(r, column)
        for r in frame.itertuples(index=False)
        for key, column in {
            "kappa": "Our_Kappa", "paper_kappa": "Paper_Kappa", "accuracy": "Our_Accuracy",
            "cost": "Our_Cost", "paper_cost": "Paper_Cost", "alarms": "Alarms",
            "retrains": "Retrains",
        }.items()
    }
    assert len(frame) == 10
    group = "khoa-6datasets-10methods"
    name = f"khoa-{slug}-10-methods-seed{config['seed']}"
    api = wandb.Api()
    existing = [] if existing_run else list(api.runs(
        f"{entity}/{project}", filters={"displayName": name, "group": group}
    ))
    if len(existing) > 1:
        raise ValueError(f"Multiple W&B runs match {name}; select the correct run before publishing.")

    if existing_run or existing:
        run_id = existing_run or existing[0].id
    else:
        with wandb.init(
            entity=entity, project=project, name=name, group=group,
            tags=["khoa", "10-methods", "kaggle"], config=config,
        ) as run:
            run.define_metric("stream_step")
            for method in ALL_METHODS:
                run.define_metric(f"{method}/accuracy_200", step_metric="stream_step")
            for row in history.to_dict("records"):
                run.log({key: float(value) for key, value in row.items() if pd.notna(value)})
            run.log({
                "comparison_table": wandb.Table(dataframe=frame),
                "alarm_positions": wandb.Table(dataframe=alarms),
            })
            run.summary.update(metrics)
            run.summary["method_count"] = len(frame)
            run_id = run.id

    for _ in range(15):
        api.flush()
        remote = api.run(f"{entity}/{project}/{run_id}")
        if remote.state == "finished" and remote.history(
            samples=10, keys=["stream_step", "STUDD/accuracy_200"], pandas=False
        ):
            break
        sleep(2)
    else:
        raise RuntimeError(f"W&B run {run_id} has not published the full history.")

    assert remote.name == name
    assert remote.config["source_sha256"] == config["source_sha256"]
    assert remote.config["detector"] == config["detector"]
    desired = {"method_count": len(frame), **metrics}
    missing = {key: value for key, value in desired.items() if key not in remote.summary}
    if missing:
        remote.summary.update(missing)
        api.flush()
        remote = api.run(f"{entity}/{project}/{run_id}")
    assert all(np.isclose(remote.summary[key], value) for key, value in desired.items())
    url = remote.url
    (output / f"{slug}_wandb.json").write_text(json.dumps({"run_id": run_id, "url": url}) + "\n")
    print(f"Verified {slug}: {url}")
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", metavar="DATASET", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity", default="lgkai-hcmus")
    parser.add_argument("--project", default="Testing")
    parser.add_argument("--existing-run")
    args = parser.parse_args()
    publish_dataset(args.publish, args.output, entity=args.entity, project=args.project,
                    existing_run=args.existing_run)
