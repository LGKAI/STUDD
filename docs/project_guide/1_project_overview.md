---
title: "Term Project: Overview"
short_title: "Project Overview"
label: intro-ml-project-index
description: "Overview of the term project: each team adopts one published machine learning paper and carries it through selection, reading, reproduction, theory critique, and empirical stress testing."
---

The term project of **Introduction to Machine Learning** asks each team to adopt
**one published machine learning paper** and spend the semester with it. The
team reads the paper until it can retell it in its own words, rebuilds the
experiments to see how faithful such a rebuild can be, questions the theory
where it is thin, and demonstrates the found weaknesses with experiments of its
own design. The work runs as three milestones and two labs, each with its own
page, and the course closes it at the final oral exam.

## The final oral exam

```{note}
The project ends in a final oral exam in the form of a **poster workshop**,
rather than a seminar or a direct question answering session. Each team hangs
its poster, presents the semester's work directly from it, and answers the
examiners' questions standing next to the work the poster summarizes.
```

## The route

The checkpoints run in the order below, each with a dedicated page. The
milestones set up the knowledge, and the labs turn it into running experiments:

1. **{ref}`intro-ml-m1-selection`**: the team registers and claims its paper by
   submitting the selection form published on Moodle.
2. **{ref}`intro-ml-m2-reading`**: the team internalizes the paper and hands the
   retelling in as a slide deck.
3. **{ref}`intro-ml-lab1-reproduction`**: the team reruns the paper's
   experiments and measures how close its rebuild lands to the published
   numbers.
4. **{ref}`intro-ml-m3-analysis`**: the team critiques the theory of the paper,
   from its assumptions to its complexity story.
5. **{ref}`intro-ml-lab2-empirics`**: the team demonstrates the found
   weaknesses empirically and compresses the whole critique into one poster.

The checkpoints pair up by deliverable: the reading milestone and the
reproduction lab submit together as one slide deck, and the critique milestone
and the empirical-demonstration lab submit together as one poster.

## Logistics

- The project is a **team** effort. The selection form captures every member,
  and the confirmed paper binds its team for the rest of the semester.
- Each team keeps **one public code repository** for the entire semester. Code,
  notebooks, and documentation accumulate in it from the first week, and the
  commit history documents the progress. Everything the deck and the poster
  report must trace back to this repository.
- Submissions leave through **Moodle**: the selection form at the first
  milestone, the slide deck file and the repository link at the deck
  checkpoint, and the poster file at the poster checkpoint. Deadlines follow
  the Moodle announcements of the course.
- Experiments are tracked run by run on a public
  [Weights & Biases](https://wandb.ai) project so that every reported number in
  the deck or the poster links to its run. The tracking rules live in
  {ref}`intro-ml-lab1-reproduction` and apply to both labs.

## Grading

| Part | Weight |
|---|---|
| The selection form | pass gate, no points of its own |
| **The slide deck**, the reading milestone and the reproduction lab together | **45%** |
| **The poster**, the critique milestone and the empirical-demonstration lab together | **50%** |
| Tracked repository quality, W&B discipline, and artifact formatting | **5%** |
| **Total** | **100%** |

The selection form carries no points of its own and gates the semester: a team
that has not registered its paper earns no grade for the project. The final
rubric row scores the working discipline around the artifacts rather than the
artifacts themselves, so keep the repository readable, the experiments
tracked, and the formatting of both artifacts tight.

An excellent deck retells the paper precisely in the team's own words,
visualizes the method faithfully, and pins the reproduction results against the
published ones with an account of every deviation, to the point that the
examiners could reconstruct what the paper does without opening it. A deck
that retells correctly but visualizes thinly, or reports deviations without
explaining them, reads as unfinished work. An excellent poster puts a real
weakness on the table, argues it theoretically, and confirms it with an
experiment designed exactly for that weakness. A poster that asserts
weaknesses without theory, or shows experiments disconnected from any claimed
weakness, reduces the semester to unbacked opinions.

## Integrity and the use of AI

Reimplement the proposed method from the paper's description. If the paper
ships a public implementation, that code serves verification and comparison
only. Copying the paper's text or its figures, copying code from the original
implementation, or copying the work of another team zeros the project.

As usual, AI tools are **not restricted**: use them wisely. They may support
the writing and the coding, but every member stands behind the whole artifact
at the poster workshop and explains any part of the work there. The lab
instructor reserves the right to conduct additional assessments on some of the
teams, should the work of a team make them needed.

Each checkpoint page specifies its own deliverable: the selection form in
{ref}`intro-ml-m1-selection`, the slide deck in {ref}`intro-ml-m2-reading`,
and the poster in {ref}`intro-ml-lab2-empirics`.
