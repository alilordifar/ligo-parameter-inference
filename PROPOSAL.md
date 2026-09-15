# Project Proposal

DSC 207 (UC Berkeley) course project. This is a scoped course deliverable —
a proof-of-concept demonstrating a pipeline and an honestly-reported
finding, not a publishable research result. See `BACKGROUND.md` for
plain-language science context and `TUTORIAL.md` for the full technical
detail behind every claim below.

## Research question

**Does a normalizing-flow neural posterior estimator (NF-PE), trained to
infer BBH source parameters from one LIGO observing run's noise, silently
lose calibration when evaluated on a different run's noise?**

"Silently" is the operative word: a model can look fine by ordinary point-
estimate metrics while its *uncertainty quantification* — the part that
matters for treating its output as a trustworthy probabilistic answer — is
systematically wrong. Simulation-Based Calibration (SBC) is the tool this
project uses to check for that specifically, rather than for accuracy.

## Why O3a vs. O4a, over other run pairs 

Three pairs were considered:

- **O1 vs. O2** — rejected: minimal hardware change between these runs, so
  a null result would be uninformative (no strong reason to expect drift in
  the first place) and a positive result would be hard to attribute
  confidently to noise-distribution shift vs. other factors.
- **O2 vs. O4a** — rejected: spans *two* separate major upgrade cycles,
  confounding attribution — a finding either way couldn't be pinned to a
  single, well-characterized cause.
- **O3a vs. O4a (chosen)** — corresponds to one well-documented upgrade
  cycle: increased circulating laser power and a transition from
  frequency-independent to frequency-dependent squeezed-light injection. A
  real, specific, citable hardware change with a known noise-shape impact.
  This is a well-motivated run-to-run shift, not a proof that the upgrade is
  the only difference between the two data populations.

## Data, methods, and model

- **Data:** GWOSC public H1 strain releases. 8 quality-vetted, event-free,
  1024-second noise segments per run (O3a, O4a), selected and DQ-vetted per
  `TUTORIAL.md` Stage 2.
- **Signal model:** simulated BBH injections, `IMRPhenomD` waveforms, and a
  project-defined prior based on Bilby conventions (component masses
  Uniform(5,100) solar masses, aligned spins Uniform(0,0.8), distance
  uniform-in-comoving-volume over [100,2000] Mpc),
  matched-filter SNR ≥ 8, 800 accepted injections per run.
- **Inference model:** `sbi`'s SNPE with a Masked Autoregressive Flow (MAF)
  density estimator — single simulation round, trained on 750 O3a
  injections (50 held out for the baseline SBC test).
- **Evaluation:** Simulation-Based Calibration, KS-test against uniform
  rank statistics (`TUTORIAL.md` Stage 6).

### What the model actually sees and produces

Each injection is converted into one fixed-length, 4-second H1 strain window
(16,384 samples at 4096 Hz) centered on the simulated merger. The window is
whitened using the corresponding noise PSD and bandpassed to 35-350 Hz. The
model input is therefore a numerical waveform array, not an image and not a
handwritten list of summary statistics.

The matching target is a four-number vector:

```text
[chirp mass, mass ratio, effective spin, luminosity distance]
```

After training, the model does not return only four point predictions. Given
one new waveform, it generates many four-number samples from an estimated
posterior distribution, `p(parameters | waveform)`. Those samples can be
summarized as medians and credible intervals, while retaining correlations
between parameters. The project evaluates whether this probability
distribution is calibrated, not merely whether its median is accurate.

The 750 training examples are O3a waveform/known-parameter pairs. The model
is then evaluated on 50 held-out O3a injections and 800 O4a injections. The
O4a examples never train the model. Because the current implementation splits
injections after generating them from a small set of noise segments, the O3a
baseline should be described as a test on held-out injections under the same
run's noise conditions, not as a fully independent detector-noise test.

## Target parameters, and why 3 are excluded

Four parameters are estimated: **chirp mass**, **mass ratio**, **effective
spin (χ_eff)**, **luminosity distance**. **Sky location, inclination, and
polarization angle are deliberately excluded.** This isn't a simplification
of convenience — it's a direct consequence of using a **single detector
(H1 only)**. Those three parameters are recovered in real LIGO analyses by
comparing arrival time, amplitude, and phase across *multiple* geographically
separated detectors (triangulation); with one detector, they are not
identifiable from the strain alone, so including them in the target set
would mean asking the model to learn something the data structurally cannot
support.

## The two baselines

- **Baseline #1 — within-distribution (O3a).** SBC on O3a's own held-out
  test set (never used in training). This is the calibration reference
  point: "how well-calibrated is this model on data that looks like what it
  trained on?" All comparisons against O4a are measured relative to this.
- **Baseline #2 — retraining-strategy comparison (future work, not
  implemented).** A natural follow-up question this project does *not*
  answer: given evidence of drift, is it better to retrain only when a
  drift-detection signal fires, or to simply always train on the largest
  available pooled window (O3a+O4a together)? Answering this requires
  training a **second** model on pooled data and comparing its calibration
  and cost profile against the drift-triggered approach — a genuinely
  separate, larger experiment, intentionally out of scope here. Natural
  next step after this proof-of-concept.

## Lightweight PSD-based drift-detection check

Before the expensive SBC evaluation, this project also demonstrates a much
cheaper, complementary monitoring signal: comparing the O3a and O4a average
noise ASD curves (already computed for the ASD plot) via a two-sample
KS-test and a median relative-magnitude difference, restricted to the
35–350 Hz analysis band (`ligo_pipeline.evaluation.detect_psd_drift`, full
math in `TUTORIAL.md` Stage 2b). The motivating question: **could a signal
this cheap — no trained model, no posterior sampling, just two PSD curves —
have flagged "retraining/recalibration warranted" before ever running SBC
at all?**

This is included specifically because it's nearly free given work already
done elsewhere in the pipeline (reuses `average_psd`'s output, adds a few
cells, no new data or training) — not as a claim that PSD comparison is a
substitute for full calibration testing. The actual flag this check
produces for this run's specific noise segments is printed and discussed
directly in `main.ipynb` Section 2, alongside a discussion of what
agreement or disagreement with the SBC finding below would mean. The
thresholds used (`psd_drift_ks_alpha`, `psd_drift_relative_threshold` in
`config.py`) are illustrative defaults for this demonstration, **not
values tuned or validated against known-good/known-drifted noise pairs** —
a real monitoring pipeline would need that calibration work first. In
particular, the KS calculation treats frequency-bin ASD values as empirical
samples even though neighboring frequency bins are correlated, so its
p-value is a heuristic monitoring score, not a formally calibrated
independent-sample hypothesis test.

## Finding: no strong evidence of calibration degradation, with a caveat

Across all 4 target parameters, O3a's (baseline) KS statistic falls inside,
or very close to, the range spanned by 10 independent O4a subsamples of
matched size — see `main.ipynb` Section 7 for the full comparison table and
plots. In plain terms: **this model's calibration does not show a clear,
consistent degradation from O3a's noise to O4a's noise**, despite O4a
representing a real, documented hardware/sensitivity change the model never
trained on.

**Important caveat — statistical power.** O3a's held-out set is small
(n=50), which is exactly why the repeated-subsample comparison exists
rather than a single KS statistic — a single value at n=50 is noisy enough
that a real but modest calibration gap could easily be indistinguishable
from sampling variation at this scale. This finding should be read as **"no
degradation detected at this sample size,"** not **"proven absent."** A
larger held-out set is the most direct way to increase this test's power,
and is a natural (if unglamorous) extension.

This negative-but-legitimate result is still a reportable answer to the
research question: it demonstrates the cross-run SBC methodology working
end-to-end on a real, documented hardware shift, and reports honestly that
it did not find the effect it was built to detect — a more valuable
contribution here than the specific direction of the result.

## Secondary finding: distance is poorly calibrated on both runs

Distance calibration is the worst of the 4 parameters on *both* O3a and
O4a — consistently, not just on the cross-run test. Because this shows up
identically on the training-distribution baseline, it is **consistent with a
structural limitation of single-detector parameter estimation** (the
well-known distance/inclination degeneracy — with inclination excluded
from this single-detector target set, its correlation with distance may show
up as extra posterior width or miscalibration on distance itself) rather
than clearly attributable to the O3a→O4a noise shift. This experiment does
not isolate that mechanism from prior choice, waveform assumptions, or the
small calibration sample.

## Open, unresolved anomaly

A small number of held-out examples (3 of 800, in the original run this
pipeline is based on) show pathological posterior-sampling collapse —
near-zero rejection-sampling acceptance — and are excluded from the
calibration statistics (tracked explicitly via `failed_indices`, never
silently zero-filled; see `TUTORIAL.md` Stage 6). These are failures of the
posterior sampling procedure, not by themselves proof that the learned
posterior is wrong; the effective SBC sample size must be reported after
their exclusion. Two candidate explanations were tested and **both ruled out**: the failures are not
concentrated in a rare corner of the training prior (~9% of O3a training
examples share the same high-mass/high-spin region as 2 of the 3
failures — not actually rare), and the noise segment shared by those same
2 failures has an unremarkable broadband std relative to the other 7 O4a
segments. The third failure shares neither trait. **Status: unresolved,
reported as a characterized-but-undiagnosed anomaly** (`main.ipynb`
Section 8) — a plausible but unconfirmed hypothesis (very high-SNR
posteriors producing narrower, more fragile proposal distributions) doesn't
explain every case, and the sample size is too small for a firm conclusion.

## Future work

- **Drift-triggered retrain vs. pooled training (Baseline #2, above).** The
  natural next experiment: train a second model on pooled O3a+O4a data and
  compare its calibration/cost tradeoff against retraining only when the
  drift-detection check (or a real SBC-based check) fires.
- Systematic diagnosis of the posterior-sampling failure mode — e.g.
  deliberately constructed high-SNR/edge-of-prior test cases with direct
  inspection of proposal-distribution acceptance rates.
- A larger O3a held-out set, to give the calibration comparison in Section
  7 more statistical power.

Everything above beyond one sentence each is intentionally *not* pursued in
this deliverable — see the notebook's Section 8 and Methods summary for
where the line was drawn and why.

## Related work

- **Wildberger et al.**, *"Adapting to noise distribution shifts in
  flow-based gravitational-wave inference,"* Phys. Rev. D 107, 084046
  (2023), [arXiv:2211.08801](https://arxiv.org/abs/2211.08801) — the
  closest prior work. Trains a normalizing-flow NPE (DINGO) on O2 noise
  plus a single O3 PSD, and addresses inference-quality drift through O3
  via PSD *forecasting*, rather than the SBC-based calibration monitoring
  or drift-triggered retraining framing used here. Primary point of
  comparison and contrast for this project.
- **BilbyFlow**, [arXiv:2609.00766](https://arxiv.org/abs/2609.00766) —
  shows NPE calibration (via pp-plots, closely related to the SBC approach
  used here) degrades under real off-source noise compared to idealized
  Gaussian noise — supports the premise that noise *realism* specifically
  affects calibration, distinct from this project's run-to-run comparison.
- **AresGW**, [arXiv:2407.07820](https://arxiv.org/abs/2407.07820) — a CBC
  *detection* (not parameter-estimation) network evaluated for
  generalization across O1–O4; establishes "train on one run, test on
  others" as a recognized pattern in GW ML, applied here to a different
  task (detection rather than PE) and a different failure mode
  (sensitivity rather than calibration).
