"""Cross-activation universality test: does T-E dissociation survive different activations?"""
import csv, numpy as np

def load(path, tick='1000'):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r['tick'] == tick]

def corr(x, y):
    x, y = np.array(x, float), np.array(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return np.corrcoef(x, y)[0, 1]

# Load all datasets
tanh_data = load('sweep_size48.csv')  # tanh, size 48, all 5 topos
relu_data = load('sweep_relu.csv')    # relu, size 24, vn/moore/random
linear_data = load('sweep_linear.csv')  # linear, size 24, vn/moore/random

print('=' * 80)
print('UNIVERSALITY TEST: T-E DISSOCIATION ACROSS ACTIVATION FUNCTIONS')
print('=' * 80)
print()

datasets = [
    ('tanh (size 48)', tanh_data),
    ('relu (size 24)', relu_data),
    ('linear (size 24)', linear_data),
]

topos = ['von_neumann', 'moore', 'random']

for label, data in datasets:
    print(f'--- {label} ---')
    header = f"{'Topology':>15s}   r(T,E)    r(E,self)  r(T,self)  mean_T    mean_E    mean_self"
    print(header)
    for topo in topos:
        subset = [r for r in data if r['topology'] == topo]
        if not subset:
            continue
        T = [float(r['mean_T']) for r in subset]
        E = [float(r['mean_E']) for r in subset]
        S = [float(r['mean_self']) for r in subset]
        rTE = corr(T, E)
        rES = corr(E, S)
        rTS = corr(T, S)
        print(f'{topo:>15s}   {rTE:+.3f}     {rES:+.3f}      {rTS:+.3f}     '
              f'{np.mean(T):.4f}    {np.mean(E):+.4f}    {np.mean(S):.4f}')
    print()

# Summary comparison table
print('=' * 80)
print('KEY COMPARISON: MOORE r(T,E) ACROSS ACTIVATIONS')
print('=' * 80)
print()
print(f"{'Activation':>12s}  {'Size':>4s}   r(T,E)    r(E,self)  r(T,self)")
for label, data, act, size in [
    ('tanh', tanh_data, 'tanh', '48'),
    ('relu', relu_data, 'relu', '24'),
    ('linear', linear_data, 'linear', '24'),
]:
    subset = [r for r in data if r['topology'] == 'moore']
    T = [float(r['mean_T']) for r in subset]
    E = [float(r['mean_E']) for r in subset]
    S = [float(r['mean_self']) for r in subset]
    rTE = corr(T, E)
    rES = corr(E, S)
    rTS = corr(T, S)
    print(f'{act:>12s}  {size:>4s}   {rTE:+.4f}    {rES:+.4f}     {rTS:+.4f}')

print()
print('KEY COMPARISON: VON NEUMANN r(T,E) ACROSS ACTIVATIONS (should be ~0)')
for label, data, act, size in [
    ('tanh', tanh_data, 'tanh', '48'),
    ('relu', relu_data, 'relu', '24'),
    ('linear', linear_data, 'linear', '24'),
]:
    subset = [r for r in data if r['topology'] == 'von_neumann']
    T = [float(r['mean_T']) for r in subset]
    E = [float(r['mean_E']) for r in subset]
    rTE = corr(T, E)
    print(f'{act:>12s}  {size:>4s}   r(T,E)={rTE:+.4f}')

print()
print('KEY COMPARISON: RANDOM r(T,E) ACROSS ACTIVATIONS (should be ~0 or positive)')
for label, data, act, size in [
    ('tanh', tanh_data, 'tanh', '48'),
    ('relu', relu_data, 'relu', '24'),
    ('linear', linear_data, 'linear', '24'),
]:
    subset = [r for r in data if r['topology'] == 'random']
    T = [float(r['mean_T']) for r in subset]
    E = [float(r['mean_E']) for r in subset]
    rTE = corr(T, E)
    print(f'{act:>12s}  {size:>4s}   r(T,E)={rTE:+.4f}')

# PCA per activation on moore
print()
print('=' * 80)
print('DIMENSIONALITY COLLAPSE: PCA ON MOORE ACROSS ACTIVATIONS')
print('=' * 80)
from numpy.linalg import eigh
metrics = ['mean_self', 'mean_phi', 'mean_R', 'mean_T', 'mean_E']

for label, data, act in [
    ('tanh (48)', tanh_data, 'tanh'),
    ('relu (24)', relu_data, 'relu'),
    ('linear (24)', linear_data, 'linear'),
]:
    subset = [r for r in data if r['topology'] == 'moore']
    X = np.array([[float(r[m]) for m in metrics] for r in subset])
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    Z = (X - mu) / (std + 1e-12)
    C = np.cov(Z.T)
    eigvals, eigvecs = eigh(C)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    var_explained = eigvals / eigvals.sum()
    cum = np.cumsum(var_explained)
    n95 = int(np.searchsorted(cum, 0.95)) + 1
    print(f'  {act:>8s}: PC1={var_explained[0]:.3f}  PC2={var_explained[1]:.3f}  '
          f'cum2={cum[1]:.3f}  dims_95%={n95}')
    # PC1 loadings
    parts = [f'{metrics[j]}={eigvecs[j,0]:+.3f}' for j in range(5)]
    print(f'           PC1: {", ".join(parts)}')
