---
title: "Empirical Demonstration"
short_title: "Empirics"
label: intro-ml-lab2-empirics
description: "Empirical lab: design experiments that demonstrate the weaknesses found in the critique, and compress the whole critique into one poster."
---

## Objectives

The team designs and runs experiments in which the weaknesses found in
{ref}`intro-ml-m3-analysis` become measurable facts, and then compresses the
whole critique, theoretical and empirical, into the single poster that the
poster workshop grades.

## Designing the demonstration experiments

Pick the families that fit the weaknesses on the table. The menu below
suggests the shapes that usually land:

- **Assumption-violating data.** Build or pick inputs that break the
  assumptions the critique found, and measure what the method's quality does
  as the assumption bends. A degradation curve in front of a stated assumption
  says more than pages of prose.
- **Adversarial and out-of-distribution settings.** Push the method past the
  regimes its evaluation covered, and show where its behavior changes.
- **Component ablations.** Remove or degrade the pieces the theory credits the
  method's properties to, and watch which properties survive.
- **Replacement baselines.** Swap a credited component for a plainer
  alternative and check whether the theory's claim of necessity holds up.
- **Scaling runs.** Confront the complexity claims with wall-clock reality at
  sizes the theory says something about.
- **Sensitivity sweeps.** Vary one configuration knob at a time to expose how
  brittle the method is to a factor the paper treated as settled.

The implementation and tracking rules of {ref}`intro-ml-lab1-reproduction`
hold here in full: reimplement from the description, fixed documented seeds,
runs on the platform the team chooses, every run tracked. Use the same public
W&B project and tag the runs of this lab separately, so the two labs stay
distinguishable at a glance.

## The poster

```{important}
- The poster is **one `pptx` file containing a single slide** in the form
  factor of **portrait A2**.
- It illustrates the issues the team found. The theoretical analysis sits next
  to its empirical demonstration, wired to it: an argued weakness with its
  measured evidence, issue by issue.
- When more than two weaknesses end up proven both theoretically and
  empirically, the team selects the **two most notable** ones for the poster.
  The remaining proven weaknesses are described in the `README.md` of the
  tracked repository, next to their evidence.
- The team's empirical numbers cite their W&B runs. Claims that trace to
  nothing visible count as unbacked.
- The poster leaves through **Moodle** as a single file.
- At the workshop, the final oral exam, the team presents directly from its
  poster and answers the examiners' questions there.
```

The single page is the discipline: the poster must fit on one portrait A2
page, and it carries the issues that matter with the evidence that shows
them. The deck already told the full story. The poster is the distilled
argument, and the workshop is where the team defends it in the open.
