# Cover letter — Journal of Systems and Software

Dear Editors-in-Chief,

We submit the manuscript "Glaukopis: Layered Instruction Fine-Tuning for Cyber Threat Intelligence LLM Training" for consideration as a research paper in the *Journal of Systems and Software*.

**Fit with the journal.** The article is a software-engineering study of a machine-learning system, the topic the journal lists as *Software Engineering for AI systems*. Its subject is the pipeline that produces a domain-specialized language model:

- a generator that compiles a template language into typed queries over a knowledge graph of twelve public cyber-threat-intelligence (CTI) catalogs, so that every training target resolves to the facts that produced it;
- the build-time checks applied to the generated data, two of which were added after earlier releases failed silently;
- a four-stage training pipeline whose intermediate checkpoints are retained and evaluated, so that a capability regression can be localized to the stage and the data shard that introduced it.

The evaluation covers three CTI benchmarks and a general-reasoning diagnostic. It compares the model against the base model, a security-specialized open model, six hosted frontier models, and the single-stage model produced by the pipeline's own first stage, with item-level bootstrap intervals and response-level error taxonomies.

The article reports negative results alongside positive ones, which the journal's aims explicitly welcome:

- gains are confined to the benchmark whose schemas the data targets;
- general reasoning falls by 11.6 points;
- one training stage collapses actor attribution and biases vulnerability-severity judgments;
- measured against the single-stage baseline, the later stages are a trade: they add 3.2 points to the three-benchmark mean but cost 2.3 points on the aligned benchmark.

It closes with seven lessons for engineering traceable, regression-aware domain specialization.

**Prior history.** No version of this work has been published, and no preprint has been posted. We submitted it three times before, and each round of reviews changed the manuscript.

1. ***IEEE ICTAI 2026*** **(submission 236), an abridged conference version. Not accepted** (one weak accept, one weak reject, two rejects). The reviewers asked:
   - whether the gains came from the staged schedule or from more task-aligned data;
   - whether alignment between the training data and the benchmark explained the gains;
   - for confidence intervals;
   - for corrected cost accounting.

   In response, we:
   - re-derived every number from the run records, correcting an aggregation inconsistency (the final checkpoint's three-benchmark mean is 65.0, not the 66.3 reported in that version) and a double count in the cost ledger;
   - added paired item-level bootstrap intervals;
   - added an overlap audit (13-gram, entity, and gold-fact) with scores on overlapping and non-overlapping items;
   - added response-level taxonomies of the attribution and severity errors.
2. ***Computers & Security*** **(COSE-D-26-05507), the full journal version. Rejected after two reviews** (October 2026).
   - Reviewer 1 asked for:
     - self-contained exposition and background;
     - justified design choices;
     - experimental baselines;
     - precise descriptions of the data sources;
     - exact graph sizes;
     - an interpretation of every result and example;
     - a durable alternative to dollar costs.
   - Reviewer 2 raised:
     - the lack of a single-stage control;
     - hyperparameters that change together at stage boundaries;
     - the attribution and severity regressions;
     - the loss of general reasoning and the absence of gains on independent benchmarks;
     - the developer–evaluator overlap;
     - missing citations for the hosted models.

   In response, we:
   - rewrote the abstract and introduction, and added a system overview with a glossary;
   - added background on CTI catalogs, instruction tuning, and benchmarks, and a table of exactly what is loaded from each source;
   - added a baselines section: base model, security-specialized open model, six cited hosted models;
   - explained both regressions mechanistically from the content of the training shards;
   - added a subject-level analysis of the general-reasoning loss and a section on evaluation independence;
   - reported inference efficiency in physical units.
3. ***International Journal of Information Security*** **(Springer), an expanded version.** It added the full description of the template engine, the source-selection criteria, research directions, and appendices on the benchmarks, the contamination screen, and example training records. **Not accepted**; the editor suggested the *Journal of Systems and Software*.

**Revisions for this submission.** We reframed the manuscript as a software-engineering study for this journal's readership. It now includes:

- five research questions;
- a section on engineering machine-learning systems and their training data;
- a table of the data pipeline's quality checks, including the silent failures that led to them;
- the first training stage's checkpoint, presented explicitly as a single-stage baseline, with the later stages measured against it by paired item-level differences (answering the single-stage concern of the earlier reviews);
- seven lessons learned;
- threats to validity organized by construct, internal, external, and conclusion validity, and reliability;
- exact node and relationship counts for the knowledge graph (answering Reviewer 1 of *Computers & Security*);
- the appendices, moved to supplementary material.

We would be glad to provide the earlier reviews and our responses on request.

**Length.** The article runs to 27 two-column pages, of which about 24 are main text and the rest declarations and references; this exceeds the 18 pages the guide encourages. The appendices have already been moved to a supplementary file. The remaining length answers the earlier reviews, which asked for:

- background that a reader new to cyber threat intelligence can follow;
- a stated rationale for every design choice;
- an interpretation of every result;
- mechanistic analyses of both regressions;
- full documentation of the baselines.

We would welcome the editors' guidance on further condensation.

**Declarations.** The work was funded by Athena Software Group Inc., which develops the system described. The corresponding author is its founder, chief executive officer, and board chairman, the co-author is its Chief Scientist, and the corresponding author is also a co-author of one of the three benchmarks used. The manuscript discloses this developer–evaluator overlap and analyzes it (Sections 1, 7.3 and 7.5). No human participants or personal data were involved. Large language models were used only as writing and coding aids, as stated in Section 7.1 and in the generative-AI declaration. The software, templates, launchers, evaluation harness, results workbook, and analysis scripts are in the project's public, source-available GitHub repository. The evaluation benchmarks are public and cited. The training shards and per-item result files are too large for the repository and are available on request. The manuscript is not under consideration elsewhere, and both authors have approved it.

**Suggested reviewers.** [Optional: three or four independent researchers in software engineering for machine learning, LLM evaluation, or CTI benchmarking, with institutional e-mail addresses; none affiliated with the authors, Athena Software Group Inc., Florida Atlantic University EECS, or the AthenaBench team.]

Thank you for considering our work.

Sincerely,

Peter J. Worth Jr. (corresponding author)
Department of Electrical Engineering and Computer Science, Florida Atlantic University, Boca Raton, FL, USA
pworthjr@athenasecuritygrp.com
on behalf of Ionut Cardei
