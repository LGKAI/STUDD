---
title: "Experimental Reproduction"
short_title: "Reproduction"
label: intro-ml-lab1-reproduction
description: "Reproduction lab: rebuild the paper's experiments, track every run, and report how close the rebuild lands to the published results."
---

## Objectives

Reproduction checks how faithful a rebuild can be. The team rebuilds the
paper's experiments from what the paper reports, runs them for real, and
compares its measured numbers against the published ones. The point is not to
match the numbers perfectly. The point is to say precisely how far the rebuild
landed and why. That deviation analysis is where a deep understanding of the
method shows, and it feeds the results section of the deck submitted with
{ref}`intro-ml-m2-reading`.

## Choosing the scope

```{important}
The team picks **one of two scopes** and states the choice and its reasoning in
the deck:

- **Full scope**: all main experiments reported in the paper, on the same or
  equivalent datasets.
- **Core scope**: the two most important experiments at minimum, including the
  headline comparison of the paper, with the rest replaced by one additional
  analysis the team designs itself, such as a sensitivity sweep over the
  configuration the paper varies.
```

The scope is a judgment call the team owns. A sprawling paper with heavy
experiments justifies the core scope, and the additional analysis then carries
the weight of the missed experiments.

## The rules of implementation

- Reimplement from the **paper's description only**. If the paper ships a
  public implementation, that code exists for verification and comparison: the
  team may read it, run it, and compare outputs against it, but never copy
  from it into its own code, and never submit it as its own work.
- Reuse is allowed for everything around the method: existing libraries,
  standard implementations, and public tooling may serve the auxiliary parts
  of the pipeline, such as data loaders, metrics, and standard model blocks.
  The paper's core contribution stays implemented and understood by the team,
  and every reuse is documented and cited in the README.
- Python 3.9 or newer, with the standard scientific stack: `numpy`, `scipy`,
  `scikit-learn`, and `torch`, `tensorflow` or `jax` where the paper needs deep
  learning. Anything outside the standard stack gets justified in the README.
- Fix the random seeds, document them, and keep the runs repeatable, so any
  run can be redone.
- Every important function carries a docstring stating purpose, parameters,
  and return value. The README explains how to set up and run everything.

## Running and tracking the experiments

```{note}
Experiments run on **the platform the team chooses**: cloud instances, hosted
notebooks, or the team's own hardware. What counts is that the runs are real,
finished, and documented, and that the **tracked repository** carries
everything.
```

- Every experiment lands in one public **Weights & Biases** project, the
  experiment tracking platform at <https://wandb.ai>. The platform records the
  configuration, the metrics, and the plots of every run, so the deck can cite
  the team's numbers run by run rather than as bare claims. W&B is the short
  name from here on.
- The repository README carries the mapping table: for each experiment, the
  notebook or script that produced it, the W&B link to the run, and one line
  on what the run shows.
- **Datasets load programmatically from hosted sources.** The experiments
  read their data during the run from **Hugging Face** or **Kaggle**, and the
  dataset-preparation code in the notebook or script downloads and
  reconstructs the working structures itself. This replaces manual
  preparation, so a run works end to end from one place. Prefer the datasets
  that are already public on those two platforms over fresh uploads. When no
  public copy of what the paper uses exists, the team prepares the dataset
  itself and uploads it to its own Hugging Face or Kaggle dataset, so the
  programmatic workflow still runs end to end.

## What the results must contain

- **The comparison against the paper.** Choose whichever figure form fits the
  experiment: a table, a chart or a curve, or a qualitative input-output pair.
  The paper's number and the team's number stand side by side, and the
  relative deviation appears wherever numbers apply.
- **The deviation analysis.** Every gap worth noticing gets explained:
  configuration the paper left unstated, library and version differences
  between the paper's environment and the team's, the effect of random
  initialization, and hardware realities. This is the part that separates a
  run from an understanding.
- **The original-code comparison.** When the paper ships code, the team runs
  it, compares its outputs against the team's rebuild, and localizes where the
  two pipelines diverge. Such a comparison often surfaces bugs and problems in
  the original implementation that the team identified and fixed in its own
  rebuild, and a deviation explained by one of these findings is usually the
  most insightful part of the reproduction.

## Where it lands

The reproduction lands in two places. The results section of the shared deck
from {ref}`intro-ml-m2-reading` presents the comparison against the paper and
the deviation analysis, with each team number citing its W&B run. The tracked
repository holds the implementation, the notebooks, and the branch history
that produced them. Commit early and often: the history is on show, and it is
the record of how the work happened.
