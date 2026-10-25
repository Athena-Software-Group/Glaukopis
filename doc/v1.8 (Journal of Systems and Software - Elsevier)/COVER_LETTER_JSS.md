# Cover letter — Journal of Systems and Software

Dear Editors-in-Chief,

We submit the manuscript "Glaukopis: Traceable Instruction Data and Checkpoint-Level Diagnosis of Capability Regressions in a Cyber Threat Intelligence Language Model" for consideration as a research paper in the *Journal of Systems and Software*.

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

**Prior history.** [Confirm and edit:] A full version of this work was submitted to *Computers & Security* and an expanded version to the *International Journal of Information Security*; neither was accepted, and the editor of the latter suggested the *Journal of Systems and Software* as a better fit. An abridged version was submitted to IEEE ICTAI 2026 and not accepted. None of these versions was published, and no preprint has been posted. The present manuscript has been rewritten for a software-engineering readership and addresses the earlier reviews. It:

- makes the exposition self-contained;
- justifies each design choice;
- adds a baselines section;
- measures the later training stages against a single-stage baseline with paired item-level statistics;
- explains both regressions mechanistically from the training data;
- reports inference efficiency in durable physical units.

We would be glad to provide the earlier reviews and our responses on request.

**Length.** The article runs to 27 two-column pages, of which about 24 are main text and the rest declarations and references; this exceeds the 18 pages the guide encourages. The appendices have already been moved to a supplementary file. The remaining length answers the earlier reviews, which asked for:

- background that a reader new to cyber threat intelligence can follow;
- a stated rationale for every design choice;
- an interpretation of every result;
- mechanistic analyses of both regressions;
- full documentation of the baselines.

We would welcome the editors' guidance on further condensation.

**Declarations.** The work was funded by Athena Software Group Inc., which develops the system described. The corresponding author is its founder and chief executive officer, the co-author is its Chief Scientist, and the corresponding author is also a co-author of one of the three benchmarks used. The manuscript discloses this developer–evaluator overlap and analyzes it (Sections 1, 7.3 and 7.5). No human participants or personal data were involved. Large language models were used only as writing and coding aids, as stated in Section 7.1 and in the generative-AI declaration. The software, templates, launchers, evaluation harness, results workbook, and analysis scripts are in the project's public, source-available GitHub repository. The evaluation benchmarks are public and cited. The training shards and per-item result files are too large for the repository and are available on request. The manuscript is not under consideration elsewhere, and both authors have approved it.

**Suggested reviewers.** [Optional: three or four independent researchers in software engineering for machine learning, LLM evaluation, or CTI benchmarking, with institutional e-mail addresses; none affiliated with the authors, Athena Software Group Inc., Florida Atlantic University EECS, or the AthenaBench team.]

Thank you for considering our work.

Sincerely,

Peter J. Worth Jr. (corresponding author)
Department of Electrical Engineering and Computer Science, Florida Atlantic University, Boca Raton, FL, USA
pworthjr@athenasecuritygrp.com
on behalf of Ionut Cardei
