# Revised milestone notebooks: dataset, preprocessing and EDA

These new notebooks preserve the earlier notebooks in ../milestone_roles/. Names remain unassigned. Implement the tasks yourselves; no solution code is included.

- [Role 1 - Data acquisition, provenance, and raw-data EDA](01_raw_data_provenance_and_eda.ipynb)
- [Role 2 - Waveform simulation, injection, and label EDA](02_simulations_labels_and_selection.ipynb)
- [Role 3 - Preprocessing and signal validation](03_preprocessing_and_quality_checks.ipynb)
- [Role 4 - Dataset audit, EDA synthesis, and methodology plan](04_dataset_audit_eda_and_methodology_plan.ipynb)

## What this milestone covers

The project question remains whether four-parameter inference generalizes from O3a H1 noise to O4a H1 noise. This checkpoint concentrates on preparing and understanding the dataset: source provenance, physical labels, preprocessing, actual split counts, EDA, plots, and limitations. Model training, model performance tables and calibration conclusions are outside this notebook set's required work.

The methodology section must still describe three algorithms because the rubric asks for a baseline and two improvements: linear regression, its small regularized extension Ridge, and SNPE with a masked autoregressive flow. Describe them as the next common comparison; do not invent new training results. Feature importance is explicitly requested by the rubric. Input-target association is useful EDA but does not fully substitute for model-based importance; Role 4 must state this gap honestly if importance is deferred.

**Existing evidence versus your work:** The repository already contains saved noise, injections and processed data. The revised PDF uses those artifacts, not results from these new notebooks. Your task is a separately identified manual pilot. Do not claim your pilot has the old dataset's 800-per-run counts until you actually generate and check them. The earlier split was a 750-row O3a development pool, 50 O3a held-out examples and 800 O4a evaluation examples. A 600/150 subdivision of the development pool is a proposal, not a completed split.

**Implementation rule:** Write your own code in the empty cells. You may call standard scientific libraries and consult their documentation, but do not import ligo_pipeline, project utils, config.py, another notebook's implementation, or prepared training/model checkpoints. Trusted raw strain with source metadata is allowed. You may write your own small functions after explaining their steps. No solution code is supplied here.

**Ownership:** Assign one person per notebook; names are unassigned. Review each other's outputs, but let each owner make their own notebook changes. Record actual contributions and dates. Every cell has a sequential cell_number and every markdown cell displays its number; the preceding task labels the empty code cell. Renumber all cells from 1 when inserting new cells. Execution counters are not stable cell numbering.

## Common contract and parallel work

Agree on one small experiment before scaling up: H1; O3a development and within-run evaluation; O4a cross-run evaluation; 4096 Hz; four-second final windows; a 35-350 Hz preprocessing band; target order chirp_mass, mass_ratio, chi_eff, distance. Keep any fixed-mass, distance-only exercise separate from this four-target pilot.

Use a new experiment_id and store files under artifacts/manual_preprocessing_milestone/<experiment_id>/. The role-owned subfolders are data/, simulation/, preprocessing/ and analysis/. Put experiment.json in data/ and repeat its ID and relevant settings in each output's metadata. Store raw segment duration, sample rate, waveform assumptions, parameter ranges, seeds, intended counts, actual counts and units. Do not overwrite the existing artifacts or the older role notebooks.

Role 1 supplies raw arrays, noise_id/group_id, source GPS intervals and source split assignments. Role 3 supplies noise-only PSD files early. Role 2 uses those PSDs for SNR selection, then supplies padded injected buffers, merger indices and labels. Role 3 supplies processed arrays and row metadata. Role 4 audits and combines evidence, counts and EDA; this role does not train the final models.

**Source-group separation:** Assign groups before simulation. Overlapping raw GPS intervals are one group even if filenames differ. Use train, validation, O3a_test and O4a_test split names; keep all O4a out of fitting and tuning. A recommended small design has at least three O3a training groups, one O3a validation group, one O3a test group and two O4a groups. Choose feasible durations and target counts together; these are planning suggestions, not completed allocations. If your pilot cannot support this design, document the limitation. Track nonoverlapping final crop windows as well as group identities.

**Handoff format:** Use .npy/.npz numeric arrays and .csv/.json metadata with repo-relative paths. Join per-example records by unique sample_id; join noise/PSD records by noise_id. Preserve run, split, group and units. Never assume two files have matching row order. Save rejection ledgers, shapes, dtypes, versions and seeds. Changes to schema/settings require all consumers to check compatibility.

**Start in parallel:** Role 1 can inspect one raw segment, Role 2 can inspect clean waveforms, Role 3 can validate sinusoids/FFT/filtering, and Role 4 can design the counting ledger, plot checklist and split audit. First integrate one real example end to end through data preparation. Then expand the pilot; no model fit is needed to check these handoffs.

## Milestone completion and handoff

Before marking your role complete, show the teammate consuming your artifacts one real record and its provenance. Give them a settings file, retained/rejected counts and a short interpretation, not just an array. Check sample_id, noise_id, group_id, run, split and shape at the handoff. Save outputs under your role-owned folder for the same experiment_id.

Deliver your completed notebook, a contribution log, 100-150 words of findings, figures with descriptive titles/axes/units/legends/captions and an explicit list of unfinished checks. Do not copy another person's implementation into your notebook. Preserve the older notebooks and existing PDFs.

**Stop at this checkpoint:** dataset preparation, validated preprocessing and EDA, exact allocation tables and the three-algorithm methodology plan. Full linear/Ridge/SNPE training, final model ranking, posterior calibration results and 800-per-run manual regeneration are not mandatory tasks in this set. Feature importance remains an acknowledged rubric gap if deferred; the team must decide how to complete that requirement before submission rather than asserting full compliance.

**Write a final reflection:** What did you personally implement? What evidence supports it? Which decisions need another owner's agreement? Which findings belong to the new pilot rather than the historical project? What remains before submission?
