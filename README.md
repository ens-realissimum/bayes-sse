# BayesSSE

**Bayesian State-Space Estimation**

## Overview

`BayesSSE` is a Python library for Bayesian recursive estimation/filtration in stochastic state-space models. It recursively estimates hidden states of dynamical systems from noisy observations while accounting for process noise, observation noise, and known exogenous inputs, such as control inputs.

## Installation

This project uses `pyproject.toml` for Python packaging. The PyPI distribution is `bayes-sse`, and the Python import package is `bayes_sse`.

### Running Simulations

The package installs the `bayes-sse simulate` command. Plotting options such as `--draw` and `--save-plots` require the `simulations` extra.

Run a simulation without opening plots:

```bash
bayes-sse simulate gamma-process-gaussian-observation --runs 10 --points 50 --filter pf
```

where `gamma-process-gaussian-observation` is the simulation scenario to run.

Open live matplotlib windows:

```bash
bayes-sse simulate gamma-process-gaussian-observation --runs 10 --points 50 --filter cdkf --draw
```

`--draw` keeps Python alive until you close the plot windows.

To keep editable plot windows open and still get your shell prompt back, run the simulation as a background job:

```bash
bayes-sse simulate gamma-process-gaussian-observation --runs 200 --points 200 --filter ukf --draw &
```

Then run another filter while the first plot windows stay open:

```bash
bayes-sse simulate gamma-process-gaussian-observation --runs 200 --points 200 --filter ckf --draw &
```

For non-interactive reports, save plots as PNG files instead:

```bash
bayes-sse simulate gamma-process-gaussian-observation --runs 10 --points 50 --filter pf --save-plots plots
```

## Quick Reference

| Use Case | Command |
|----------|---------|
| **Basic install** | `pip install .` |
| **Editable (dev)** | `pip install -e .` |
| **With simulations** | `pip install -e ".[simulations]"` |
| **With dev tools** | `pip install -e ".[dev]"` |
| **From GitHub** | `pip install git+https://github.com/ens-realissimum/bayes-sse.git` |
| **Upgrade** | `pip install --upgrade bayes-sse` |
| **Uninstall** | `pip uninstall bayes-sse` |

