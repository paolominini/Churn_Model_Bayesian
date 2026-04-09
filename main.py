# Dependencies
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, classification_report)

# ---- Data Loading ----
X = np.load('data/X.npy')
y = np.load('data/Y.npy')

N, K = X.shape
print(f"Observations: {N}, Variables: {K}")

# Data Check
print("y dtype:", y.dtype)
print("y valori unici:", np.unique(y))
print("y shape:", y.shape)
print("X shape:", X.shape)
print("X contiene NaN:", np.isnan(X).any())
print("y contiene NaN:", np.isnan(y).any())

print("NaN per colonna:", np.isnan(X).sum(axis=0))
print("Righe con almeno un NaN:", np.isnan(X).any(axis=1).sum())

# --- BAYESIAN PROBIT MODEL ---
with pm.Model() as probit_model:
    
    # Non informative Prior
    beta = pm.MvNormal(
        'beta',
        mu = np.zeros(K),
        cov = 100*np.eye(K),
        shape = K
    )

    # Likelihood probit
    mu = pm.math.dot(X, beta)
    p = pm.math.invprobit(mu)
    y_obs = pm.Bernoulli('y_obs', p=p, observed = y)

# --- NUTS ---
print("Beginning the training...")

with probit_model:
    trace = pm.sample(
        draws = 1000, # samples after warmup
        tune = 1000, # warmup 
        chains = 4, # calibrate with Rhat !!!!!!!!
        cores = 4, # CPU cores
        target_accept = 0.95, # acceptance probability of target
        random_seed = 42, 
        return_inferencedata= True
    )

# --- Diagnostic ---
print(az.summary(trace, var_names = ['beta']))
print("Max R-hat:", float(az.rhat(trace)['beta'].max()))

az.plot_trace(trace, var_names = ['beta'])
az.plot_energy(trace)
