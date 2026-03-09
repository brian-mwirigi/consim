"""Does the crossover replicate at size 12 and size 48?"""
import csv
import numpy as np

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    final = [r for r in rows if int(r['tick']) == 1000]
    lookup = {}
    for r in final:
        lookup[(r['topology'], float(r['noise']), int(r['seed']))] = r
    # detect available seeds from the data
    available_seeds = sorted(set(int(r['seed']) for r in final))
    return final, lookup, available_seeds

topos = ['von_neumann', 'moore', 'hex', 'random', 'small_world']
noises = [0.04, 0.08, 0.12, 0.20, 0.30]

# The 10 noise-beneficiary seeds identified at size 24
ben_seeds_24 = [4, 38, 39, 50, 5, 40, 3, 9, 16, 45]
vic_seeds_24 = [22]

for label, path in [("SIZE 12 (144 agents)", "sweep_size12.csv"),
                     ("SIZE 24 (576 agents)", "sweep_1250.csv"),
                     ("SIZE 48 (2304 agents)", "sweep_size48.csv")]:
    print("=" * 80)
    print(label)
    print("=" * 80)
    
    final, lookup, available_seeds = load(path)
    n_seeds = len(available_seeds)
    
    # 1. Recompute: which seeds benefit from noise in ALL topologies at THIS size?
    universal_ben = []
    universal_vic = []
    for seed in available_seeds:
        benefits_all = True
        hurts_all = True
        total_delta = 0
        for topo in topos:
            lo = float(lookup[(topo, 0.04, seed)]['mean_self'])
            hi = float(lookup[(topo, 0.30, seed)]['mean_self'])
            if hi <= lo:
                benefits_all = False
            if hi >= lo:
                hurts_all = False
            total_delta += (hi - lo)
        if benefits_all:
            universal_ben.append((seed, total_delta))
        if hurts_all:
            universal_vic.append((seed, total_delta))
    
    universal_ben.sort(key=lambda x: -x[1])
    universal_vic.sort(key=lambda x: x[1])
    
    print(f"\nSeeds benefiting from noise in ALL 5 topologies: {len(universal_ben)}/{n_seeds}")
    for s, d in universal_ben[:15]:
        print(f"  seed={s:3d}  total delta: {d:+.4f}")
    print(f"\nSeeds hurt by noise in ALL 5 topologies: {len(universal_vic)}/{n_seeds}")
    for s, d in universal_vic[:10]:
        print(f"  seed={s:3d}  total delta: {d:+.4f}")
    
    # 2. Check the SIZE-24 beneficiary seeds at THIS size (only those present)
    ben_present = [s for s in ben_seeds_24 if s in available_seeds]
    vic_present = [s for s in vic_seeds_24 if s in available_seeds]
    neutral_seeds = [s for s in available_seeds if s not in ben_seeds_24 and s not in vic_seeds_24]
    
    if ben_present:
        print(f"\n--- Size-24 beneficiary seeds ({ben_present}) at this size ---")
        vic_label = f"victim(s{vic_present[0]})" if vic_present else "victim(n/a)"
        print(f"\n{'noise':>6s}  {'beneficiaries':>14s}  {'neutral':>10s}  {vic_label:>12s}  {'gap(b-v)':>10s}")
        for noise in noises:
            ben_avg = np.mean([float(lookup[(topo, noise, s)]['mean_self'])
                              for s in ben_present for topo in topos])
            neu_avg = np.mean([float(lookup[(topo, noise, s)]['mean_self'])
                              for s in neutral_seeds for topo in topos]) if neutral_seeds else float('nan')
            if vic_present:
                vic_avg = np.mean([float(lookup[(topo, noise, s)]['mean_self'])
                                  for s in vic_present for topo in topos])
                print(f"{noise:6.2f}  {ben_avg:14.4f}  {neu_avg:10.4f}  {vic_avg:12.4f}  {ben_avg - vic_avg:+10.4f}")
            else:
                print(f"{noise:6.2f}  {ben_avg:14.4f}  {neu_avg:10.4f}  {'n/a':>12s}  {'n/a':>10s}")
    else:
        print(f"\n--- Size-24 beneficiary seeds: none present in this dataset (seeds {available_seeds[0]}-{available_seeds[-1]}) ---")
    
    # 3. Does seed 4 specifically still benefit everywhere?
    if 4 in available_seeds:
        print(f"\n--- Seed 4 noise curve ---")
        print(f"{'noise':>6s}", end="")
        for topo in topos:
            print(f"  {topo:>14s}", end="")
        print()
        for noise in noises:
            print(f"{noise:6.2f}", end="")
            for topo in topos:
                val = float(lookup[(topo, noise, 4)]['mean_self'])
                print(f"  {val:14.4f}", end="")
            print()
    
    # 4. Does seed 22 specifically still get hurt everywhere?
    if 22 in available_seeds:
        print(f"\n--- Seed 22 noise curve ---")
        print(f"{'noise':>6s}", end="")
        for topo in topos:
            print(f"  {topo:>14s}", end="")
        print()
        for noise in noises:
            print(f"{noise:6.2f}", end="")
            for topo in topos:
                val = float(lookup[(topo, noise, 22)]['mean_self'])
                print(f"  {val:14.4f}", end="")
            print()
    
    # 5. Noise-helps-moore effect
    print(f"\n--- Moore noise effect ---")
    for noise in noises:
        subset = [r for r in final if r['topology'] == 'moore' and abs(float(r['noise']) - noise) < 0.001]
        ms = np.mean([float(r['mean_self']) for r in subset])
        print(f"  noise={noise:.2f}  moore mean_self={ms:.4f}")
    
    # 6. Random fixed point check
    print(f"\n--- Random noise immunity check ---")
    for noise in noises:
        subset = [r for r in final if r['topology'] == 'random' and abs(float(r['noise']) - noise) < 0.001]
        ms = np.mean([float(r['mean_self']) for r in subset])
        mp = np.mean([float(r['mean_phi']) for r in subset])
        print(f"  noise={noise:.2f}  random: self={ms:.4f}  phi={mp:.4f}")
    
    # 7. Small_world Phi boost
    print(f"\n--- Small_world Phi boost vs von_neumann ---")
    for noise in noises:
        sw = np.mean([float(lookup[('small_world', noise, s)]['mean_phi']) for s in available_seeds])
        vn = np.mean([float(lookup[('von_neumann', noise, s)]['mean_phi']) for s in available_seeds])
        sw_s = np.mean([float(lookup[('small_world', noise, s)]['mean_self']) for s in available_seeds])
        vn_s = np.mean([float(lookup[('von_neumann', noise, s)]['mean_self']) for s in available_seeds])
        print(f"  noise={noise:.2f}  phi_boost={sw-vn:+.4f}  self_diff={sw_s-vn_s:+.4f}")
    
    # 8. How many of the size-24 beneficiary seeds ALSO benefit here?
    overlap = [s for s, _ in universal_ben if s in ben_seeds_24]
    print(f"\n--- Overlap: size-24 beneficiaries that are also beneficiaries here ---")
    print(f"  {len(overlap)}/{len(ben_seeds_24)} seeds overlap: {overlap}")
    
    print()
