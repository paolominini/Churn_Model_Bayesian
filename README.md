# Customer Churn Prediction via Bayesian Probit Model

**Paolo Minini & Silvia Guidi** — Bayesian Analysis, Università degli Studi di Milano

---

## Overview

This project develops a Bayesian Probit model to predict customer churn using transactional data from a UK-based online retailer. In a non-contractual retail setting, where customers can silently drift away without formal notice, the model infers latent behavioral signals from purchase records to estimate the probability of churn at the individual customer level.

Inference is conducted via the **No-U-Turn Sampler (NUTS)**, an adaptive variant of Hamiltonian Monte Carlo (HMC), implemented in PyMC. The model achieves an out-of-sample **AUC of 0.77** and **72.4% accuracy** on the held-out test set.

---

## Dataset

**Online Retail II** — publicly available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

The dataset contains all transactions from a UK-based online gift-ware retailer between 2009 and 2011. Key fields include:

| Field | Description |
|---|---|
| `InvoiceNo` | Unique transaction ID (prefix `C` = cancellation) |
| `StockCode` | Unique product code |
| `Description` | Product name |
| `Quantity` | Items per transaction (negative = return) |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice` | Price per unit in GBP |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country of residence |

---

## Feature Engineering

Raw invoice-level records are aggregated into a single cross-sectional row per customer. The following predictors are constructed:

| Variable | Description |
|---|---|
| **Frequency** | Number of unique invoices per customer |
| **Monetary Value** | Total spend (quantity × unit price), log-transformed via log(1+x) |
| **Cancellation Rate** | Ratio of cancelled invoices to total invoices |
| **StockCode Diversity** | Number of distinct products purchased |
| **Return Propensity** | Ratio of returned quantity to purchased quantity |
| **Avg Time Between Purchases** | Standard deviation of inter-purchase day gaps |
| **Country (UK dummy)** | Binary indicator for UK-based customers |

> **Note:** Recency was excluded from the model due to mechanical collinearity with the churn label — both are derived from the date of the customer's last purchase.

### Churn Label Construction

The churn window is derived empirically from the distribution of inter-purchase intervals across the customer base:

- **Churn window:** 79 days (median of customer-level mean inter-purchase gaps)
- **Global cutoff:** 21 September 2011
- A customer is classified as **churned (y = 1)** if their last purchase date precedes the cutoff; **active (y = 0)** otherwise.

| Class | Count | Share |
|---|---|---|
| Churn (y = 1) | 3,119 | 53% |
| Active (y = 0) | 2,762 | 47% |

---

## Model

The model is specified as a **Bayesian Probit**:

```
P(yᵢ = 1) = Φ(Xᵢᵀ β)
```

where `Φ` is the standard normal CDF (the probit link function) and `β` is the coefficient vector.

**Prior:** `β ~ N(0, 100·I)` — a non-informative multivariate normal centered at zero with large variance, leaving the posterior essentially determined by the likelihood.

**Sampler settings:**

- Algorithm: NUTS (No-U-Turn Sampler) via PyMC
- Chains: 8 parallel chains
- Warm-up: 1,000 iterations per chain
- Post-warmup draws: 1,000 per chain (8,000 total)
- Target acceptance probability: 0.95

---

## Results

### Convergence Diagnostics

All chains converged to the same stationary distribution. Key diagnostics:

- **R̂ (Gelman-Rubin):** 1.00 for all coefficients (max 1.0007, well below the 1.01 threshold)
- **ESS (bulk and tail):** ranging from ~5,500 to ~8,900 across all parameters
- **BFMI:** 0.97–1.11 across all 8 chains (warning threshold: 0.3)

### Posterior Estimates

| Coefficient | Mean | 95% HDI | ESS bulk | ESS tail | R̂ |
|---|---|---|---|---|---|
| Frequency | −0.363* | [−0.474, −0.261] | 6,446 | 5,755 | 1.0 |
| Monetary Value | −0.274* | [−0.343, −0.206] | 5,579 | 5,571 | 1.0 |
| Cancellation Rate | +0.058* | [+0.013, +0.101] | 6,785 | 5,758 | 1.0 |
| StockCode Diversity | −0.389* | [−0.471, −0.312] | 6,679 | 6,080 | 1.0 |
| Return Propensity | +0.035 | [−0.014, +0.085] | 6,949 | 5,516 | 1.0 |
| Avg Time Between Purchases | −0.157* | [−0.195, −0.116] | 7,612 | 5,852 | 1.0 |
| Country (UK dummy) | +0.015 | [−0.024, +0.055] | 8,937 | 6,107 | 1.0 |

*\* HDI does not include zero — statistically significant.*

### Economic Interpretation

- **Frequency and Monetary Value** (negative): higher engagement and spending reduce churn probability, consistent with habit formation and switching costs.
- **StockCode Diversity** (negative, strongest effect): customers purchasing across many product categories are harder to replace from a single alternative supplier.
- **Cancellation Rate** (positive): higher cancellation rates proxy for transactional friction and accumulated dissatisfaction.
- **Return Propensity** (not significant): in a wholesale context, processing returns is standard operational behavior, not a signal of disengagement.
- **Avg Time Between Purchases** (negative): irregular purchasing intervals reflect restock-driven wholesale buying patterns rather than customer drift.
- **Country (UK dummy)** (not significant): geographic location offers no predictive power beyond behavioral variables.

### Predictive Performance (held-out test set, 80/20 split)

| Metric | Value |
|---|---|
| Accuracy | 72.4% |
| ROC-AUC | 0.77 |
| Recall (churners) | 80% |
| Specificity (active) | 64% |

Predictions are generated by averaging over all 8,000 posterior draws of β, explicitly accounting for parameter uncertainty in the final churn probability estimate.

