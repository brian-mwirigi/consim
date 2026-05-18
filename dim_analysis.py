"""Dimensionality analysis: do the 5 MCH metrics collapse to fewer dimensions?"""
import csv, os, sys, numpy as np
from numpy.linalg import eigh

REQUIRED_FILES = ['sweep_size48.csv', 'sweep_null.csv']


def load(path, tick='1000'):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r['tick'] == tick]


def main():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        sys.exit(
            f"Error: missing required CSV file(s): {', '.join(missing)}\n"
            f"Generate them with:\n"
            f"  python run.py --sweep --sweep-seeds 1-20 "
            f"--sweep-topos von_neumann,moore,hex,random,small_world "
            f"--size 48 --ticks 1000 --sweep-csv sweep_size48.csv\n"
            f"  python run.py --sweep --sweep-seeds 1-20 "
            f"--sweep-topos von_neumann,moore,hex,random,small_world "
            f"--size 48 --ticks 1000 --lr 0 --sweep-csv sweep_null.csv"
        )

    data = load('sweep_size48.csv')
    null = load('sweep_null.csv')
    topos = ['von_neumann', 'moore', 'hex', 'random', 'small_world']
    metrics = ['mean_self', 'mean_phi', 'mean_R', 'mean_T', 'mean_E']

    print('=' * 80)
    print('DIMENSIONALITY ANALYSIS: Are the 5 MCH-related metrics independent?')
    print('=' * 80)

    for topo in topos + ['ALL']:
        subset = data if topo == 'ALL' else [r for r in data if r['topology'] == topo]
        X = np.array([[float(r[m]) for m in metrics] for r in subset])

        mu = X.mean(axis=0)
        std = X.std(axis=0)
        Z = (X - mu) / (std + 1e-12)

        C = np.cov(Z.T)
        eigvals, eigvecs = eigh(C)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        var_explained = eigvals / eigvals.sum()
        cum_var = np.cumsum(var_explained)

        n95 = int(np.searchsorted(cum_var, 0.95)) + 1
        n90 = int(np.searchsorted(cum_var, 0.90)) + 1

        print()
        print(f'--- {topo} (n={len(subset)}) ---')
        for i in range(5):
            print(f'  PC{i+1}: {var_explained[i]:.4f} (cum {cum_var[i]:.4f})')
        print(f'  Components for 90%: {n90}   for 95%: {n95}')

        # Loadings
        for pc in range(min(2, n95)):
            parts = []
            for j in range(5):
                parts.append(f'{metrics[j]}={eigvecs[j, pc]:+.3f}')
            print(f'  PC{pc+1} loadings: {", ".join(parts)}')

    # Full correlation matrix per topology
    print()
    print('=' * 80)
    print('CORRELATION MATRICES (5 metrics)')
    print('=' * 80)

    for topo in topos:
        subset = [r for r in data if r['topology'] == topo]
        X = np.array([[float(r[m]) for m in metrics] for r in subset])
        C = np.corrcoef(X.T)
        print(f'\n--- {topo} ---')
        header = '             ' + '  '.join(f'{m:>9s}' for m in metrics)
        print(header)
        for i in range(5):
            row = f'{metrics[i]:>12s} '
            row += '  '.join(f'{C[i,j]:+.3f}    ' for j in range(5))
            print(row)

    # KEY: conditional analysis — within each noise level on moore
    print()
    print('=' * 80)
    print('WITHIN-NOISE PCA ON MOORE (controls for noise as confound)')
    print('=' * 80)
    noises = ['0.04', '0.08', '0.12', '0.20', '0.30']
    for noise in noises:
        subset = [r for r in data if r['topology'] == 'moore' and r['noise'] == noise]
        X = np.array([[float(r[m]) for m in metrics] for r in subset])
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        Z = (X - mu) / (std + 1e-12)
        C = np.cov(Z.T)
        eigvals, _ = eigh(C)
        eigvals = eigvals[::-1]
        var_explained = eigvals / eigvals.sum()
        cum_var = np.cumsum(var_explained)
        n95 = int(np.searchsorted(cum_var, 0.95)) + 1
        print(f'  noise={noise} (n={len(subset)}): '
              f'PC1={var_explained[0]:.3f} PC2={var_explained[1]:.3f} '
              f'cum2={cum_var[1]:.3f}  dims_95%={n95}')

    # Compare learning vs null dimensionality
    print()
    print('=' * 80)
    print('DIMENSIONALITY: LEARNING vs NULL on MOORE')
    print('=' * 80)
    for label, dataset in [('Learning', data), ('Null', null)]:
        subset = [r for r in dataset if r['topology'] == 'moore']
        X = np.array([[float(r[m]) for m in metrics] for r in subset])
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        Z = (X - mu) / (std + 1e-12)
        C = np.cov(Z.T)
        eigvals, eigvecs = eigh(C)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        var_explained = eigvals / eigvals.sum()
        cum_var = np.cumsum(var_explained)
        n95 = int(np.searchsorted(cum_var, 0.95)) + 1
        print(f'  {label:10s}: PC1={var_explained[0]:.4f} PC2={var_explained[1]:.4f} '
              f'cum2={cum_var[1]:.4f} dims_95%={n95}')
        for pc in range(2):
            parts = []
            for j in range(5):
                parts.append(f'{metrics[j]}={eigvecs[j, pc]:+.3f}')
            print(f'    PC{pc+1}: {", ".join(parts)}')


if __name__ == '__main__':
    main()
