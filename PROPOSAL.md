# Testing Calibration Drift in Gravitational-Wave Parameter Estimation Across LIGO Observing Runs

**Ali Lordifar** (ali_lordifar@berkeley.edu), **Kennedy Goliman** (kennedyg@berkeley.edu), **Noah Fonck** (noahfonck@berkeley.edu), **Aaron Rassiq** (aaronrassiq@berkeley.edu)

## Motivation

Gravitational-wave (GW) astronomy increasingly relies on machine learning for rapid parameter estimation (PE), inferring the physical properties (mass, spin, distance) of a merging black hole binary directly from detector strain data. Neural Posterior Estimation (NPE), using normalizing flow models, offers large speedups over classical Bayesian samplers by learning to output a full posterior distribution in a single forward pass rather than through expensive iterative sampling. However, these models are trained on simulated signals injected into noise from one specific observing run, while LIGO's detector noise characteristics change measurably between runs due to hardware upgrades (e.g., increased laser power, quantum squeezed-light injection). This raises an operationally important question: does an NPE model's *calibration*, whether its stated confidence intervals are actually correct, not just whether its point estimates are accurate, degrade silently when applied to a different observing run's noise than it was trained on? A model that remains accurate on average but becomes secretly overconfident poses a real risk for downstream scientific conclusions. We investigate this using O3a (2019) and O4a (2023), two observing runs spanning a documented hardware upgrade cycle, and evaluate calibration using Simulation-Based Calibration (SBC), a standard diagnostic for testing whether a model's posterior ranks are statistically uniform.

## Data

We use open, unauthenticated strain data from LIGO's Hanford (H1) detector, provided by the Gravitational Wave Open Science Center (GWOSC): https://gwosc.org/data/. We fetch real, event-free, quality-checked noise segments from O3a and O4a using GWOSC's public data-quality flags to exclude segments containing glitches or gaps. Since real confirmed mergers are too sparse to train a neural network on, we simulate several hundred binary black hole waveforms (via the **IMRPhenomD** approximant, using the **pycbc** package) from specified priors over component masses, spins, and distance, and inject them into the real noise segments, producing realistic, ground-truth-labeled training and test data for both observing runs.

## Related Work

The closest prior work, Wildberger et al. [1], trains a normalizing-flow NPE model (DINGO) across the O2–O3 transition and mitigates degradation by forecasting future noise power spectral densities, evaluated via Jensen-Shannon divergence against reference posteriors. We build on this by instead testing calibration directly via SBC, a more standard uncertainty-quantification diagnostic, and by explicitly comparing within-run versus cross-run performance. Related work on NPE calibration under realistic detector noise [2] shows calibration is sensitive to noise realism, and AresGW [3] evaluates GW *detection* (not parameter estimation) networks across multiple observing runs, establishing cross-run generalization testing as a recognized pattern in this space.

## References

[1] J. Wildberger, M. Dax, S. R. Green, J. Gair, M. Pürrer, J. H. Macke, A. Buonanno, and B. Schölkopf, "Adapting to Noise Distribution Shifts in Flow-Based Gravitational-Wave Inference," arXiv:[2211.08801](https://arxiv.org/abs/2211.08801) (2022).

[2] L. Pinchbeck, E. Thrane, C. Balazs, and P. Lasky, "BilbyFlow: User-Friendly Neural Posterior Estimation for Gravitational-Wave Astronomy," arXiv:[2609.00766](https://arxiv.org/abs/2609.00766) (2026).

[3] A. E. Koloniari, E. C. Koursoumpa, P. Nousi, P. Lampropoulos, N. Passalis, A. Tefas, and N. Stergioulas, "New Gravitational Wave Discoveries Enabled by Machine Learning," *Machine Learning: Science and Technology* 6, 015054 (2025); arXiv:[2407.07820](https://arxiv.org/abs/2407.07820).

## GitHub Repository

https://github.com/alilordifar/ligo-parameter-inference