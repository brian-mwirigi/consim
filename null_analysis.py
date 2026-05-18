"""Compare T-E dissociation between null model (lr=0) and learning model."""
import csv, os, sys
import numpy as np

REQUIRED_FILES = ['sweep_null.csv', 'sweep_size48.csv']


def load_csv(path, tick='1000'):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r['tick'] == tick]
    return rows


def corr(x, y):
    x, y = np.array(x, float), np.array(y, float)
    if len(x) < 3:
        return float('nan')
    return np.corrcoef(x, y)[0, 1]


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

    null = load_csv('sweep_null.csv')
    learn = load_csv('sweep_size48.csv')

    topos = ['von_neumann', 'moore', 'hex', 'random', 'small_world']

    print('=' * 80)
    print('T-E DISSOCIATION: NULL MODEL (lr=0) vs LEARNING MODEL (lr=0.01)')
    print('=' * 80)
    print()

    for label, data in [('LEARNING (lr=0.01)', learn), ('NULL (lr=0)', null)]:
        print(f'--- {label} ---')
        header = f"{'Topology':>15s}  r(T,E)   r(E,self) r(T,self) mean_T   mean_E   mean_self"
        print(header)
        for topo in topos:
            subset = [r for r in data if r['topology'] == topo]
            T = [float(r['mean_T']) for r in subset]
            E = [float(r['mean_E']) for r in subset]
            S = [float(r['mean_self']) for r in subset]
            rTE = corr(T, E)
            rES = corr(E, S)
            rTS = corr(T, S)
            mT = np.mean(T)
            mE = np.mean(E)
            mS = np.mean(S)
            print(f'{topo:>15s}  {rTE:+.3f}   {rES:+.3f}    {rTS:+.3f}    {mT:.4f}   {mE:+.4f}   {mS:.4f}')

        structured = [r for r in data if r['topology'] != 'random']
        T = [float(r['mean_T']) for r in structured]
        E = [float(r['mean_E']) for r in structured]
        S = [float(r['mean_self']) for r in structured]
        rTE = corr(T, E)
        rES = corr(E, S)
        rTS = corr(T, S)
        print(f"{'ALL STRUCTURED':>15s}  {rTE:+.3f}   {rES:+.3f}    {rTS:+.3f}")

        all_T = [float(r['mean_T']) for r in data]
        all_E = [float(r['mean_E']) for r in data]
        all_S = [float(r['mean_self']) for r in data]
        rTE = corr(all_T, all_E)
        rES = corr(all_E, all_S)
        rTS = corr(all_T, all_S)
        print(f"{'ALL TOPOS':>15s}  {rTE:+.3f}   {rES:+.3f}    {rTS:+.3f}")
        print()

    # Variance comparison
    print('=' * 80)
    print('VARIANCE COMPARISON (measures whether lr=0 collapses variation)')
    print('=' * 80)
    print()
    header = f"{'Topology':>15s}  {'':8s} std(T)   std(E)   std(self)"
    print(header)
    for topo in topos:
        null_sub = [r for r in null if r['topology'] == topo]
        learn_sub = [r for r in learn if r['topology'] == topo]
        for label, sub in [('learn', learn_sub), ('null', null_sub)]:
            T = [float(r['mean_T']) for r in sub]
            E = [float(r['mean_E']) for r in sub]
            S = [float(r['mean_self']) for r in sub]
            print(f'{topo:>15s}  {label:8s} {np.std(T):.6f} {np.std(E):.6f} {np.std(S):.6f}')
        print()

    # Key question: does the T-E anti-correlation on moore disappear?
    print('=' * 80)
    print('KEY COMPARISON: MOORE GRID')
    print('=' * 80)
    moore_learn = [r for r in learn if r['topology'] == 'moore']
    moore_null = [r for r in null if r['topology'] == 'moore']

    for label, sub in [('Learning', moore_learn), ('Null', moore_null)]:
        T = [float(r['mean_T']) for r in sub]
        E = [float(r['mean_E']) for r in sub]
        S = [float(r['mean_self']) for r in sub]
        rTE = corr(T, E)
        rES = corr(E, S)
        rTS = corr(T, S)
        print(f'{label:10s}: r(T,E)={rTE:+.4f}  r(E,self)={rES:+.4f}  r(T,self)={rTS:+.4f}')
        print(f'            mean_T={np.mean(T):.4f}  mean_E={np.mean(E):+.4f}  mean_self={np.mean(S):.4f}')
        print(f'            std_T={np.std(T):.6f}  std_E={np.std(E):.6f}  std_self={np.std(S):.6f}')
        print()


if __name__ == '__main__':
    main()
