# Linear Regression Parameter Guide

This guide explains what the first regression model is meant to estimate, how to interpret the parameter values, and what each training example contains.

## What the model estimates

The input is a short, noisy gravitational-wave strain segment around a simulated binary-black-hole merger. The target is a vector of four source parameters:

1. **Chirp mass** (`chirp_mass`, solar masses)
2. **Mass ratio** (`mass_ratio`, dimensionless)
3. **Effective spin** (`chi_eff`, dimensionless)
4. **Distance** (`distance`, megaparsecs)

These are derived from the component masses and spins used to generate each waveform. The model does not directly predict the two component masses or the two individual spins.

## Interpreting the parameters

### Chirp mass

The chirp mass is a combination of the two black-hole masses that strongly controls how quickly the waveform's frequency rises during inspiral:

```text
M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
```

It is not the mass of either black hole by itself. For example, two equal 30-solar-mass black holes have a chirp mass of about 26 solar masses. Component masses in this project are drawn between 5 and 100 solar masses, which gives a possible chirp-mass range of roughly 4.35 to 87 solar masses. The accepted examples need not fill that range evenly.

### Mass ratio

The project defines mass ratio as the smaller component mass divided by the larger one:

```text
q = m2 / m1, where m1 >= m2
```

It is unitless and lies between 0 and 1. A value of `1` means equal masses; `0.5` means the smaller black hole is half the mass of the larger one; a value near `0.05` means the pair is very unequal. Given the component-mass bounds, the possible ratio is approximately 0.05 to 1.

### Effective spin

Effective spin (`chi_eff`) is a mass-weighted summary of the components' spins along the orbital axis:

```text
chi_eff = (m1 * chi1 + m2 * chi2) / (m1 + m2)
```

It is dimensionless. Zero means no net aligned spin; positive values indicate spins aligned with the orbit, and negative values indicate anti-aligned spins. **Important dataset detail:** the current injection generator draws each component spin from `[0, 0.8]`, so the generated effective spins are nonnegative. The configured model bounds, `[-0.8, 0.8]`, allow negative values that are not represented in these training injections.

### Distance

Distance is measured in megaparsecs (Mpc). One Mpc is about 3.26 million light-years. The configured range is 100 to 2,000 Mpc. A larger distance generally makes the observed signal fainter, all else being equal.

Distances are drawn uniformly in volume, which favors larger distances over a uniform-in-distance draw. Injections are also accepted only when their optimal signal-to-noise ratio is at least 8. Consequently, the examples that reach training are not a uniform sample of the configured parameter ranges.

## What “750 examples” means

The examples are **simulated injections**, not 750 real astrophysical observations. Each is made by generating a waveform from a random parameter draw, adding it to a segment of detector noise, and retaining its known parameter values as the training target.

The data split is:

| Use | Run | Number of injections |
|---|---|---:|
| Training | O3a | 750 |
| Held-out O3a evaluation | O3a | 50 |
| Cross-run evaluation | O4a | 800 |
| **Total** |  | **1,600** |

Each example is centered on the waveform's merger and surrounded by noise. The project uses a pool of noise segments for each run, so the examples are not necessarily based on 1,600 independent noise realizations.

The preprocessing whitens each injected segment using the power spectral density (PSD) estimated from its original noise segment, then applies a 35–350 Hz bandpass and crops around merger.

### Crop length detail

The configuration calls the crop setting `crop_window_sec = 2`. In the preprocessing code, that value is used for **both sides** of merger: two seconds before and two seconds after. The resulting input is therefore 4 seconds long, or 16,384 samples at 4,096 Hz. The tutorial describes it as a 2-second window, which does not match the current code's interpretation.

## What “small” and “large” mean

There is no single scale shared by all four outputs. Interpret each value against that parameter's physical meaning and the data distribution:

- Chirp mass is in solar masses; tens of solar masses are typical of this project's broad component-mass bounds.
- Mass ratio is a fraction; values near 1 are equal-mass systems, while values near 0 are increasingly unequal.
- Effective spin is a signed, unitless quantity; in this dataset it is nonnegative because of the current spin draw.
- Distance is in Mpc; larger values mean more distant sources, but the accepted data are affected by the volume draw and SNR threshold.

The configured prior bounds are useful guardrails, but they should not be mistaken for the exact observed range or distribution of the training targets.

## Implication for a linear-regression baseline

The four targets have very different numerical scales: distance can be in the thousands, chirp mass in the tens, and mass ratio and effective spin are around 0 to 1. Scale the input features and each target before fitting a multi-output linear regression, then convert predictions back to the original units for interpretation. Treat this as a simple point-estimate baseline; it will not by itself express the uncertainty in a parameter estimate.

## Where these details come from

- Parameter bounds and preprocessing settings: [`config.py`](../config.py)
- Injection draws and SNR threshold: [`ligo_pipeline/injections.py`](../ligo_pipeline/injections.py)
- Derived targets and centered crop: [`ligo_pipeline/preprocessing.py`](../ligo_pipeline/preprocessing.py)
- Pipeline explanation and split: [`TUTORIAL.md`](../TUTORIAL.md)