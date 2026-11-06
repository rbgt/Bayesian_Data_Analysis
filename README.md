# Bayesian Data Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyMC3](https://img.shields.io/badge/PyMC3-1F4E79?style=flat-square)
![Swift](https://img.shields.io/badge/Swift_for_TensorFlow-F05138?style=flat-square&logo=swift&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

Personal work on **Bayesian statistics** and **probabilistic modelling**: forecasting with uncertainty in PyMC3, and core inference algorithms (MCMC, Expectation-Maximisation) **implemented from scratch** in Swift.

## Notebooks

### 📈 Bayesian time series forecasting with PyMC3
[`PyMC3 - Probabilistic Time series forecasting/`](PyMC3%20-%20Probabilistic%20Time%20series%20forecasting/)

- Models asset prices (ETH/USD, then GOOG stock 2014–2016) as a **Gaussian random walk** with priors on drift and volatility.
- Samples the posterior with **NUTS** and produces **posterior predictive** forecasts on a held-out test set.
- Reports **95 % credible intervals** around the forecast.

### 🎲 MCMC from scratch: Metropolis-Hastings in Swift
[`[Swift]MCMC_MetropolisHastings_from_scratch/`](%5BSwift%5DMCMC_MetropolisHastings_from_scratch/)

- Implements the **Metropolis-Hastings** algorithm: transition model, prior, log-likelihood and acceptance rule.
- Infers the standard deviation of a population from 1,000 observations, with trace plots, burn-in and posterior histograms.
- Uses posterior estimates to regenerate the full population and compares it to the original.

### 🧩 Expectation-Maximisation for Gaussian Mixture Models in Swift
[`[Swift]EM_algorithm_GaussianMixtureModel/`](%5BSwift%5DEM_algorithm_GaussianMixtureModel/)

- Implements the **EM algorithm** for a GMM from scratch, with custom probability distributions (Bernoulli, Normal).
- Recovers the parameters of 2- and 3-component mixtures and plots the fitted densities against the generative ones.

## Running the notebooks

- **PyMC3 notebook**: Python with `pymc3`, `theano`, `pandas`, `statsmodels` and `matplotlib`.
- **Swift notebooks**: written for **Swift for TensorFlow** in Google Colab (each notebook has an *Open in Colab* badge). Swift for TensorFlow is now archived, so they are kept for reference.
