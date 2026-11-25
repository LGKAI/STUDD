---
title: "Critical Analysis"
short_title: "Critique"
label: intro-ml-m3-analysis
description: "Critique milestone: question the theory the paper stands on. Demonstrating the found weaknesses empirically belongs to the demonstration lab."
---

## Objectives

After reading and rebuilding, the team knows how the method works and what its
experiments show. Now the team separates the claims from the evidence. This
milestone critiques **the theory only**: the assumptions the method stands on,
the soundness of its claims, the formulation of its problem, and the reality of
its complexity story. Where the critique finds a weakness, the job of
demonstrating it empirically belongs to {ref}`intro-ml-lab2-empirics`.

## What deserves scrutiny

### Hidden assumptions

Every method assumes something about the world it runs in. Find them: the
shape of the data distribution the theory expects, independence or structure
assumptions over features or nodes, the availability of full data or labels at
training time, the smoothness the optimization silently requires, the
computation model that hides cost the hardware eventually pays. For each
assumption, ask how often the real data satisfies it and what happens when it
does not.

A few textbook assumptions show what to look for. **Naive Bayes** assumes the
features are conditionally independent given the class, and its long service
in document classification shows how far that assumption carries and where it
stops. **Ordinary least squares** assumes the errors share one constant
variance and stay uncorrelated with the features, which is the Gauss-Markov
ground the closed-form solution and its guarantees stand on. **Empirical risk
minimization** assumes the data arrives independently from the same
distribution the model will meet in use, so a shift between training data and
serving data breaks the promise that the measured risk estimates the real
risk. Each of these is a falsifiable claim about the world, which is what
makes an assumption worth naming and testing.

### Soundness of the claims

- Take the lemmas, theorems, and stated properties from
  {ref}`intro-ml-m2-reading` and test them against their proof. Do they hold as
  written? Are the conclusions conditioned on hypotheses the paper mentions
  only in passing, somewhere in a remark?
- Distinguish the claims the experiments actually support from the claims they
  merely accompany. A strong benchmark table does not prove every sentence in
  the introduction.

### The formulation and the objective

- Would the objective capture the goal the paper advertises? Methods optimize
  what is formalized, and the formalization can drift away from the intent.
  Look for shortcuts: surrogates standing in for the real target, regularizers
  whose side effects the theory ignores, losses that admit degenerate
  solutions.

### Complexity and generality

- The complexity story retold in {ref}`intro-ml-m2-reading` claims growth rates
  under assumptions. Question them: are the claimed rates competitive in the
  regimes the experiments actually occupy, or only in the regime where the
  method wins?
- Question the generalization the paper sells: how far does the method stretch
  beyond the settings it was evaluated on, and what in the theory supports
  that stretch?

## The shape of the analysis

The analysis exists as content for two artifacts. Its theoretical arguments go
to the theoretical side of the poster built with {ref}`intro-ml-lab2-empirics`,
and its substance must satisfy three floors:

- at least **three strengths** of the method, each argued with evidence from
  the reading or from the reproduction rather than asserted.
- at least **two limitations** the paper leaves unanswered, each argued, each
  a theoretical limitation, and each stated so plainly that an experiment
  could be designed against it.
- at least **two future directions** the paper opens, each tied closely to one
  of the two limitations above, so each direction reads as the answer to a
  named weakness.

A limitation argued here and demonstrated there makes the poster's strongest
axis, and the critique keeps its value even for papers the team likes: praise
argued with evidence scores the same as criticism argued with evidence. When
the critique ends up with more weaknesses proven than a poster can hold, the
rule in {ref}`intro-ml-lab2-empirics` decides which two reach the poster and
where the remaining ones are described.
