---
title: "Reading the Paper"
short_title: "Paper Reading"
label: intro-ml-m2-reading
description: "Reading milestone: internalize the paper and hand the retelling in as a slide deck, together with the reproduction results."
---

## Objectives

Reading a paper is more than parsing its sentences. By the end of this
milestone, the team holds a working understanding it can defend without the PDF
open: what problem the paper attacks, what gap it fills, what is genuinely new
in its method, how the method works part by part, and how the authors evaluate
it. This milestone shares its deliverable with {ref}`intro-ml-lab1-reproduction`:
the two submit together as one slide deck, where the reading supplies the
retelling and the lab supplies the running experiments.

## The reading guide

Work through the paper in the order below. Every element gets retold in the
team's own words, and the retelling must survive in front of an examiner who
has not read the paper. Verbatim sentences out of the paper do not count as an
answer. The point of the milestone is the conversion.

### A fast pass before the close reading

Read the paper in the recommended order: the **abstract**, then the
**introduction**, then the **conclusion**, then the **background and related
work**, then the **experiments**, and the **method** last. The first three
tell you what the paper promises and where it lands, and that is enough to
judge early whether the paper is worth the rest of the semester. The
background situates the claim among its neighbors, the experiments show
whether the evidence holds up before you pay for the details, and the method,
the heaviest part, reads fastest once you already know what to look for. A
team that cannot say what the paper promises and where it lands after this
fast pass should go back to {ref}`intro-ml-m1-selection` and reconsider.

### Context and motivation

- The problem the paper attacks, stated plainly, with a concrete example of
  your own that shows why the problem matters in the real world.
- At least two predecessor approaches the paper builds on. For each one,
  state the core idea, the limitation that remains, and the practical
  consequence of that limitation. A limitation without its consequence is an
  afterthought, not an analysis.

### Contributions

List the contributions in the paper's own terms, then reorganize them by kind:
theoretical contributions, technical contributions, and empirical
contributions. Novelty is judged against the predecessors above: state for
each contribution what exactly was not possible or not attempted before.

### The method

This is the technical heart of the deck. The retelling covers five layers:

- **Notation and the formal problem.** Restate every symbol the paper uses in
  one consistent notation of the team's own, and restate the problem the
  method solves, formally where the paper is formal.
- **The overall pipeline.** Draw the method end to end in a diagram of the
  team's own making and walk the reader through the data from input to output.
- **The components, one by one.** Each component answers three questions: what
  it does, why the method needs it, and how it works precisely. Key formulas
  appear with every variable explained, and nothing on a slide floats without
  its meaning in the speaker notes.
- **The theoretical properties.** Restate every lemma, theorem, or proof sketch
  the paper claims and explain what it means in intuitive terms. A full
  reproduction of the proofs is not required. A working understanding of what
  the properties promise and where their limits sit is.
- **Complexity.** The time and space story of the method, and how it compares
  with the predecessors named above.

## The combined deliverable: the slide deck

The reading milestone and {ref}`intro-ml-lab1-reproduction` submit as **one deck**.

- One `pptx` file of **at least 20 slides and at most 40 slides**, counting
  the main story and any appendix slides alike. Fewer slides usually mean the
  team skipped the analysis. More slides usually mean the team copied the
  paper section by section.
- The deck is a visual artifact. It re-describes the paper visually:
  contributions, novelty, the gap it fills, the method, and the methodology of
  datasets, metrics, and protocol, each rendered as drawn figures, diagrams,
  and tables rather than paragraphs.
- Every slide carries **speaker notes of at most 500 words**. The notes hold
  the verbal analysis, the reasoning, and the connective tissue that the
  visuals cannot fit. Notes are read as submitted. There is no oral
  presentation, so the deck plus its notes are the complete artifact.
- Citations render in **IEEE style, numbered, at the bottom of each slide that
  cites them**. There is **no separate references slide**: a claim stands next
  to its source on the same page.
- All figures belong to the team. Redraw or recreate whatever the paper shows.
  Pasting the paper's figures counts as copying under the integrity rules.
- The deck closes the story with the reproduction: the team's results placed
  against the paper's, each team number citing its W&B run, as told by
  {ref}`intro-ml-lab1-reproduction`.

```{important}
The deck leaves through **Moodle as a single `pptx` file under 20 MB**,
speaker notes included. The deck file is uploaded on its own, and the **link
to the team's public tracked repository is submitted separately on Moodle**
as its own entry. A deck above the size limit or a missing repository link
renders the checkpoint unsubmitted until it is fixed.
```

## How the deck gets read

The examiners open the deck, read it slide by slide against the speaker notes,
and check the retelling against the paper where the retelling makes claims. A
deck that restates the paper earns little. A deck that shows the team
understood, drew it, and tested it earns the milestone. Slides that dump raw
source code fail the visual requirement, and slides without speaker notes
count as incomplete.
