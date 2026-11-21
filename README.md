# Multiuser Active RIS–NOMA Covert Communication

Research code and reproducible benchmark records for power allocation with one covert receiver, multiple public receivers, active-RIS noise, hardware distortion, public-rate constraints, Bob's SIC/outage constraint, and a minimum detection-error requirement at Willie.

The main experiment is the six-method benchmark in `experiments/ris_noma_step1/step7h/`. The repository retains the complete development and final-test output directories, including the input snapshots and selected checkpoints needed to verify the saved results and rerun the final comparison.

## Experimental scope

- GNN and pooled MLP are trained on K=2, 3, 4, 5 public users, using 32 training layouts and a separate 16-layout validation set.
- The final test uses 40 new layouts, IDs 5000–5039, with eight power/DEP cases per layout and three paired seed replicates.
- K=2–5 is the test within the trained user-count range. K=6 is an unseen user count and is reported separately. K=6 has 960 allocation trials per method, not a single test scenario.
- The six methods are Full PSO, random short PSO, direct pooled MLP, direct GNN, MLP + short PSO, and GNN + short PSO.
- Full PSO uses 1,800 objective evaluations; the validation-selected short budget is 150. The counts include the initial 30-particle population, giving 59 and 4 updates respectively. Each budget has its own complete inertia schedule.
- RIS has 32 ULA elements, Rician factor 1, and hardware-distortion coefficient 0.1. The public-user count K is distinct from the Rician factor. The adopted scalar RIS-noise model and Bob-aligned phase rule are retained.

Passed percentages are physical feasibility-assessment rates, not accuracy against PSO labels. Failed allocations contribute zero to the gated-rate average and remain in its denominator. Full PSO is a large-budget numerical reference, not a global-optimality proof.

## Contents

| Path | Purpose |
|---|---|
| `experiments/ris_noma_step1/` | Source, configurations, dependencies and tests |
| `results/step7h_development32/` | Complete validation comparison, fixed budget selection, diagnostic folds, protocol and model/input snapshots |
| `results/step7h_test40/` | Complete final-test records, physical rows, figures, six checkpoints and independent-assessment evidence |
| `results/step7h_test40/inputs/train.jsonl` | Original 32-layout training graph/label input snapshot |
| `results/step7h_test40/inputs/validation.jsonl` | Separate validation graph/label input snapshot |
| `results/step7h_test40/inputs/` | Corpus, training, geometry and physical configuration snapshots |
| `results/step7h_test40/models/` | Six selected deployment checkpoints |

To verify or rerun the saved benchmark, keep both result directories complete: `results/step7h_development32/` and `results/step7h_test40/`. Their contents serve different purposes:

- `inputs/` and `models/` contain the saved input data, settings and trained model weights.
- `search/` records the allocations returned by the methods; `audit/` records their evaluation on independent fading samples.
- `source_snapshot/` records the source used for the run. The development directory also contains `cv/`, with the diagnostic cross-validation records.
- Receipt files and `completion.json` let the checker detect missing or changed files and confirm that the recorded run is complete.

The CSV summaries are convenient for reading the results. The other records are also needed by the supplied verification and rerun commands, so retain the full directories when sharing this benchmark.

## Clone the repository

This repository is public. Anyone can download it using Git over HTTPS; no GitHub account or access invitation is required for cloning.

```bash
git clone https://github.com/JannovHoang/active-ris-noma-covert.git
cd active-ris-noma-covert
```

Run these commands from the folder where you want the downloaded project to be created. The remaining instructions start from the repository root, `active-ris-noma-covert`, unless a different working directory is stated. Create your own Python environment as described below; the code does not depend on the author's local drive paths.

## Environment

Python 3.12 and the CPU backend are the reference setup. On Windows PowerShell, from the repository root:

```powershell
python -m venv .venv
$pyRepo = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $pyRepo -m pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu
& $pyRepo -m pip install -r experiments/ris_noma_step1/requirements_step7h.txt
```

On Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r experiments/ris_noma_step1/requirements_step7h.txt
```

The recorded Windows environment uses Python 3.12.7, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, PyTorch 2.15.0+cpu and filelock 3.32.7. The source requirements declare supported ranges. The original version metadata is preserved in the result plans. Runtime and small floating-point differences can depend on the environment; compare feasibility and numerical results separately from timing.

## Verify the complete saved experiment

These commands check the committed evidence without performing the PSO searches again. From the repository root in PowerShell:

```powershell
Push-Location experiments/ris_noma_step1
& $pyRepo -X utf8 run_step7h.py check --output "..\..\results\step7h_development32"
& $pyRepo -X utf8 run_step7h.py check --output "..\..\results\step7h_test40"
Pop-Location
```

Both commands must report `"complete": true`. The final check covers all six methods, 40 layouts, 200 layout/K groups and 28,800 physical rows. It verifies source/checkpoint hashes, train/validation separation, normalization lineage, evaluation counts, search/audit commitments, saved statistical decisions and aggregate consistency. It is a consistency audit of the recorded study, not a proof that overfitting is impossible.

On Linux, from the repository root:

```bash
cd experiments/ris_noma_step1
../../.venv/bin/python run_step7h.py check --output ../../results/step7h_development32
../../.venv/bin/python run_step7h.py check --output ../../results/step7h_test40
cd ../..
```

## Rerun the final K=2–5 / K=6 comparison

The saved development protocol and its full supporting records are sufficient to launch the original final-test runner. A new label-generation or main-model training run is not required.

```powershell
Push-Location experiments/ris_noma_step1
& $pyRepo -X utf8 run_step7h.py test --protocol "..\..\results\step7h_development32\protocol.json" --output "..\..\results\step7h_reproduced" --workers 2
& $pyRepo -X utf8 run_step7h.py check --output "..\..\results\step7h_reproduced"
Pop-Location
```

This reruns the 40-layout comparison with six methods for K=2–6, using the original selected checkpoints and parameter schedule. It performs 10.8 million objective evaluations and the corresponding independent fading assessments, so it is a substantial CPU experiment. It is a rerun of the already observed test, not a newly untouched holdout.

Use a fresh output directory. If the same run was interrupted, repeat the `test` command with `--resume`, keeping the same protocol, worker count and environment. Do not use the original result directory as the output of a new run. New output directories are ignored by Git until explicitly selected for publication.

## Tests

From PowerShell at the repository root:

```powershell
Push-Location experiments/ris_noma_step1
& $pyRepo -X utf8 -m unittest discover -s tests_step7h -v
& $pyRepo -X utf8 -m unittest discover -s tests -p test_pso.py -v
& $pyRepo -X utf8 -m unittest discover -s tests -p test_dep.py -v
Pop-Location
```

## Source map

The paths below are relative to `experiments/ris_noma_step1/`.

| Component | File |
|---|---|
| Main command-line entry | `run_step7h.py` |
| Development, protocol sealing and final rerun | `step7h/driver.py` |
| Six-method search, initialization and independent audit | `step7h/benchmark.py` |
| Role-specific channel streams and allocation correction | `step7h/physics.py` |
| Verification of the saved study | `step7h/checks.py` |
| Summaries, paired comparisons and layout-cluster bootstrap | `step7h/analysis.py` |
| PSO implementation and feasibility-first comparator used by the 7h runner | `step7h/frozen_runtime/ris_noma/pso.py` |
| Allocation decoder and cached physical objective | `step7h/frozen_runtime/ris_noma/multik_core.py` |
| Search margins | `step7h/frozen_runtime/ris_noma/continuous_data.py` |
| Channel generation and geometric LoS | `step7h/frozen_runtime/ris_noma/channels.py`, `geometry_los.py` |
| SINR, SIC, rate, outage and assessment | `step7h/frozen_runtime/ris_noma/stable_assessment.py` |
| Gamma/empirical detection error | `step7h/frozen_runtime/ris_noma/dep.py` |
| Finite-sample assessment bounds | `step7h/frozen_runtime/ris_noma/validation.py` |
| Models and supervised training | `step7h/frozen_runtime/ris_noma/curve_models.py`, `curve_fit.py`, `multik_models.py`, `multik_learning_models.py` |

## PSO source, label generation and predictions

The repository includes the PSO implementation, the code that uses it to generate training labels, and the code that runs PSO during the benchmark. The paths below are relative to `experiments/ris_noma_step1/`; click a file to open its source.

| File | Role |
|---|---|
| [ris_noma/pso.py](experiments/ris_noma_step1/ris_noma/pso.py) | Core PSO implementation. `optimize(...)` initializes the swarm, updates particle velocities and positions, and tracks personal and global best solutions. |
| [ris_noma/multik_benchmark_core.py](experiments/ris_noma_step1/ris_noma/multik_benchmark_core.py) | `search_pool(...)` calls PSO and retains a bounded pool of candidate allocations for label selection. |
| [ris_noma/continuous_workers.py](experiments/ris_noma_step1/ris_noma/continuous_workers.py) | Runs the label-generation search, selects labels on a separate channel bank, and audits the chosen labels on another independent bank. |
| [ris_noma/curve_data.py](experiments/ris_noma_step1/ris_noma/curve_data.py) | Builds the nested training-layout corpus used in 7g, reusing the label-generation workers. |
| [run_step7e.py](experiments/ris_noma_step1/run_step7e.py), [run_step7g.py](experiments/ris_noma_step1/run_step7g.py) | Command-line entry points for the earlier label-generation and learning-curve stages. Their original data dependencies are described below. |
| [step7h/benchmark.py](experiments/ris_noma_step1/step7h/benchmark.py) | Runs direct GNN/MLP predictions, standalone PSO and PSO initialized by a network prediction; records evaluation counts and timing. |
| [step7h/frozen_runtime/ris_noma/pso.py](experiments/ris_noma_step1/step7h/frozen_runtime/ris_noma/pso.py) | The preserved PSO implementation actually imported by the 7h runner. |

The roles are different at each stage:

1. **Generate labels offline.** PSO searches for reference power allocations. Candidate selection and an independent audit record feasibility; a label is a numerical reference, not a guaranteed global optimum.
2. **Train and predict with GNN/MLP.** The networks learn from the training inputs and PSO-generated labels. A trained network predicts an allocation for a new input without running PSO.
3. **Optionally refine the prediction.** In GNN/MLP + short PSO, the corrected network prediction supplies one initial particle and the other 29 particles are random. Short PSO then searches for an improved allocation. Full PSO and random short PSO use random initialization.

`step7h/__init__.py` places the bundled `frozen_runtime` first on the import path. Editing the top-level `ris_noma/pso.py` therefore does not change the PSO used by `run_step7h.py`. Keep the bundled runtime as the reference for the published results; develop an alternative optimizer in a separate module/runner or separately named runtime and protocol, as described below.

## Further development and alternative optimizers

Use `main` for the reviewed reference and `develop` for ongoing work. For an optimizer experiment:

```bash
git switch develop
git switch -c feature/qpso
```

After making and testing a change, review and publish the branch:

```bash
git status --short
git add .
git diff --cached --stat
git commit -m "Add experimental optimizer changes"
git push -u origin feature/qpso
```

Open a pull request when the change is ready for review. The reviewed `main` branch remains the reference for the recorded study. If a new result directory should be published, explicitly allow that directory in `.gitignore` and review its contents before committing.

QPSO is not implemented in the reference. An alternative optimizer can reuse the allocation decoder, physical objective and independent assessment. A candidate has K+1 normalized coordinates in [0,1]; the objective returns `Outcome(score, violation, feasible, metrics)`. The standard implementation uses `better` to prefer feasibility, then maximize rate, and otherwise minimize normalized violations. Count every objective call, including initialization and auxiliary trials.

The archived runtime and sealed protocol identify the exact code associated with the original results. Keep that reference intact. Add an experimental module/runner or create a separately named runtime and protocol for a changed algorithm. A hash mismatch indicates that the code no longer matches the original sealed protocol; use a new experimental plan and output for a changed method.

For fair comparisons, declare scenario/layout lists, evaluation budgets, random streams, margins, checkpoint policy and assessment rules before examining new test results. Reuse the same search inputs for paired methods and keep the assessment bank separate from optimization.

The included training and validation snapshots contain graph inputs and labels used by the selected models. The repository does not include the entire historical label-generation corpus, every previous learning-curve stage or all 7g diagnostic artifacts. Repeating all earlier development stages through their original command-line interfaces requires those additional data packages. They are not needed for the saved checks and final benchmark rerun above.

## Read the results

Start with `results/step7h_test40/metrics.csv`, `pso_metrics.csv`, `direct_metrics.csv`, `by_k.csv`, `by_cv.csv` and `paired_contrasts.csv`. Detailed physical rows and failure rows are retained for inspection.

![Performance by public-user count](results/step7h_test40/figures/by_k.png)

The results support conclusions within the stated channel, layout, SIC and detector assumptions. GNN and MLP are close within the trained K range; extrapolation to K=6 still has a gap to Full PSO. High feasibility on the observed cases is not a guarantee for every future layout.

The inherited active-RIS model is described by P. Chen et al., *Active RIS-NOMA-Aided Covert Communication With Hardware Impairments*, IEEE Wireless Communications Letters, 2023.
