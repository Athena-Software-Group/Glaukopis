# Analysis scripts

These scripts produce the tables and paired statistics in the Glaukopis papers. They are snapshots, copied unchanged from the manuscript working tree, which is not part of this repository.

| Folder | Script | Produces |
|---|---|---|
| `computers_security_v1.5/` | `make_tables.py` | Result tables as CSV |
| | `bootstrap_v21_32b.py` | Paired item-bootstrap intervals for the v21 32B chain |
| | `overlap_subsets.py`, `entity_overlap.py` | Benchmark overlap and entity exposure |
| `v1.6/` | `v16_additions.py` | Numbers added in v1.6 |
| `jss_v1.8/` | `v18_additions.py` | Numbers added in v1.8 (Journal of Systems and Software) |
| `aaiml_2027/` | `overlap_three_way.py` | Three-way split of VSP exposure |
| | `posthoc_screen.py` | Verbatim 13-gram screen of the general and CSE data |
| | `stage_tokens.py` | Examples and tokens per training stage |
| | `bootstrap_new_runs.py` | Paired intervals including the control runs |

**To run them.** Most scripts read the repository location from `GLAUKOPIS_ROOT`, which defaults to the authors' path, so set it to your clone. The four scripts in `computers_security_v1.5/` have that path as a constant near the top of the file; edit it there.

**Inputs.** The training shards and the per-item prediction and score files are too large to host here and are available from the corresponding author on reasonable request. The scripts expect the per-item files under `SFT/test/responses/<alias>/`. Output goes to a `data/` folder beside the script's folder, except `bootstrap_v21_32b.py`, which writes next to itself.
