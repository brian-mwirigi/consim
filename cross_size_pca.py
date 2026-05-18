"""Cross-size PCA: does dimensionality collapse replicate at sizes 12, 24, 48?"""
import csv, os, sys, numpy as np
from numpy.linalg import eigh

METRICS = ['mean_self', 'mean_phi', 'mean_R', 'mean_T', 'mean_E']
TOPOS = ['von_neumann', 'moore', 'hex', 'random', 'small_world']

FILES = [
    ('Size 12 (144 agents)',    'sweep_size12_full.csv'),
    ('Size 18 (324 agents)',    'sweep_size18_full.csv'),
    ('Size 24 (576 agents)',    'sweep_size24_full.csv'),
    ('Size 36 (1296 agents)',   'sweep_size36_full.csv'),
    ('Size 48 (2304 agents)',   'sweep_size48.csv'),
    ('Size 96 (9216 agents)',   'sweep_size96.csv'),
]
REQUIRED_FILES = [path for _, path in FILES]


def load(path, tick='1000'):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r['tick'] == tick]

def pca_report(data, topo):
    subset = [r for r in data if r['topology'] == topo]
    if len(subset) < 3:
        return None, None, None, len(subset)
    X = np.array([[float(r[m]) for m in METRICS] for r in subset])
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    C = np.cov(Z.T)
    eigvals, eigvecs = eigh(C)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    ve = eigvals / eigvals.sum()
    cum = np.cumsum(ve)
    n95 = int(np.searchsorted(cum, 0.95)) + 1
    return ve, eigvecs, n95, len(subset)

def main():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        sys.exit(
            f"Error: missing required CSV file(s): {', '.join(missing)}\n"
            f"Generate them with, e.g.:\n"
            f"  python run.py --sweep --sweep-seeds 1-20 "
            f"--sweep-topos von_neumann,moore,hex,random,small_world "
            f"--size 48 --ticks 1000 --sweep-csv sweep_size48.csv"
        )

    print('=' * 80)
    print('CROSS-SIZE DIMENSIONALITY COLLAPSE: PC1 variance explained on Moore')
    print('=' * 80)
    header = '{:>22s}  {:>6s}  {:>6s}  {:>6s}  {:>8s}'.format(
        'Size', 'PC1', 'PC2', 'Cum2', 'dims_95%')
    print(header)
    print('-' * 56)

    for label, path in FILES:
        data = load(path)
        ve, vecs, n95, n = pca_report(data, 'moore')
        if ve is not None:
            print('{:>22s}  {:.4f}  {:.4f}  {:.4f}  {:>8d}'.format(
                label, ve[0], ve[1], ve[0]+ve[1], n95))

    print()
    print('=' * 80)
    print('PC1 LOADINGS ON MOORE (by size)')
    print('=' * 80)
    for label, path in FILES:
        data = load(path)
        ve, vecs, n95, n = pca_report(data, 'moore')
        if ve is not None:
            parts = ['{}={:+.3f}'.format(METRICS[j], vecs[j, 0]) for j in range(5)]
            print('  {}: {}'.format(label, '  '.join(parts)))

    print()
    print('=' * 80)
    print('FULL TOPOLOGY COMPARISON (by size)')
    print('=' * 80)
    for label, path in FILES:
        data = load(path)
        print('\n  {}:'.format(label))
        print('    {:>14s}  {:>6s}  {:>6s}  {:>6s}  {:>8s}  {:>4s}'.format(
            'Topology', 'PC1', 'PC2', 'Cum2', 'dims_95%', 'n'))
        print('    ' + '-' * 58)
        for topo in TOPOS:
            ve, vecs, n95, n = pca_report(data, topo)
            if ve is not None:
                print('    {:>14s}  {:.4f}  {:.4f}  {:.4f}  {:>8d}  {:>4d}'.format(
                    topo, ve[0], ve[1], ve[0]+ve[1], n95, n))

    print()
    print('=' * 80)
    print('r(T,E) ON MOORE (cross-size check)')
    print('=' * 80)
    for label, path in FILES:
        data = load(path)
        subset = [r for r in data if r['topology'] == 'moore']
        T = np.array([float(r['mean_T']) for r in subset])
        E = np.array([float(r['mean_E']) for r in subset])
        r_te = np.corrcoef(T, E)[0, 1]
        print('  {}: r(T,E) = {:+.4f}  (n={})'.format(label, r_te, len(subset)))

    # Scaling law fit: PC1 vs N (number of agents) on moore
    print()
    print('=' * 80)
    print('SCALING LAW FIT: PC1(moore) vs N agents')
    print('=' * 80)
    sizes = []
    pc1_moore = []
    pc1_random = []
    rte_moore = []
    for label, path in FILES:
        data = load(path)
        ve_m, _, _, _ = pca_report(data, 'moore')
        ve_r, _, _, _ = pca_report(data, 'random')
        subset = [r for r in data if r['topology'] == 'moore']
        T = np.array([float(r['mean_T']) for r in subset])
        E = np.array([float(r['mean_E']) for r in subset])
        N = int(label.split('(')[1].split()[0])
        sizes.append(N)
        pc1_moore.append(ve_m[0])
        pc1_random.append(ve_r[0])
        rte_moore.append(np.corrcoef(T, E)[0, 1])

    sizes = np.array(sizes)
    pc1_moore = np.array(pc1_moore)
    pc1_random = np.array(pc1_random)
    rte_moore = np.array(rte_moore)

    # Log-log fit: PC1 = a * N^b
    log_N = np.log(sizes)
    log_PC1 = np.log(pc1_moore)
    b, log_a = np.polyfit(log_N, log_PC1, 1)
    a = np.exp(log_a)
    print('  Power law fit: PC1 = {:.4f} * N^{:.4f}'.format(a, b))
    print('  R^2 = {:.4f}'.format(1 - np.sum((log_PC1 - (b * log_N + log_a))**2) / np.sum((log_PC1 - log_PC1.mean())**2)))

    # Log fit: PC1 = a + b*log(N)
    b2, a2 = np.polyfit(np.log(sizes), pc1_moore, 1)
    pred_log = a2 + b2 * np.log(sizes)
    ss_res = np.sum((pc1_moore - pred_log)**2)
    ss_tot = np.sum((pc1_moore - pc1_moore.mean())**2)
    print('  Log fit: PC1 = {:.4f} + {:.4f} * ln(N)'.format(a2, b2))
    print('  R^2 = {:.4f}'.format(1 - ss_res / ss_tot))

    print()
    print('  {:>6s}  {:>10s}  {:>10s}  {:>10s}'.format('N', 'PC1_moore', 'PC1_rand', 'r(T,E)'))
    print('  ' + '-' * 42)
    for i in range(len(sizes)):
        print('  {:>6d}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
            sizes[i], pc1_moore[i], pc1_random[i], rte_moore[i]))

    # Incremental gains to find inflection
    print()
    print('  Incremental PC1 gains (moore):')
    for i in range(1, len(sizes)):
        dPC1 = pc1_moore[i] - pc1_moore[i-1]
        dN = sizes[i] - sizes[i-1]
        print('    N {} -> {}: dPC1 = {:+.4f}  (dPC1/dN = {:.6f})'.format(
            sizes[i-1], sizes[i], dPC1, dPC1/dN))

if __name__ == '__main__':
    main()
