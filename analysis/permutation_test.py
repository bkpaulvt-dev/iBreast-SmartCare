#!/usr/bin/env python3
"""
permutation_test.py
===================
Selection-aware label-permutation test for the single-loop model search.

Null hypothesis: the condition labels are unrelated to the sensor features.
Statistic      : the best F1 score over ALL 3,762 configurations
                 (627 feature subsets x 6 algorithms) under leave-one-donor-out
                 cross-validation, i.e. the same search that produced the
                 single-loop results.
p value        : proportion of label permutations whose best F1 score equals
                 or exceeds the best F1 score observed with the true labels.

The test is exact with respect to the search: every configuration is
considered for every permutation. For speed, a configuration is abandoned as
soon as its leave-one-out errors make the observed score unreachable, and a
permutation is finished as soon as one configuration reaches the observed
score. Neither shortcut changes the outcome (exceeds / does not exceed).

Sequential stopping (Besag & Clifford 1991): permutations stop once `--hits`
exceedances have been seen (p = hits / permutations run) or `--max-perm`
permutations have been run (p = (exceedances + 1) / (permutations + 1)).

Usage:
  python permutation_test.py --lab NewMilkResults.xlsx --q Questionnaire.csv \
         --task Inflammation --observed 0.9091 --max-perm 200 --jobs 8
"""
import argparse, itertools, json, os, sys, time, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nested_cv_permutation as N

ORDER = ['KNN (k=3)', 'Naive Bayes', 'Logistic Reg', 'SVM (RBF)', 'Grad Boost', 'Rand Forest']
SUBSETS = [s for k in N.SIZES for s in itertools.combinations(range(len(N.FEATURES)), k)]
_X = None


def f1_of(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def reaches(X, y, algo, sub, thr):
    """True if this configuration's LOO F1 >= thr. Stops as soon as thr is unreachable."""
    n = len(y); npos = int(y.sum())
    Xs = X[:, list(sub)]
    order = [i for i in range(n) if y[i] == 1] + [i for i in range(n) if y[i] == 0]
    tp = fp = fn = 0
    for i in order:
        tr = np.arange(n) != i
        pred = int(N.pipe(algo).fit(Xs[tr], y[tr]).predict(Xs[i:i + 1])[0])
        if y[i] == 1:
            tp += pred == 1; fn += pred == 0
        else:
            fp += pred == 1
        if f1_of(npos - fn, fp, fn) < thr - 5e-4:      # best case from here is below thr
            return False
    return f1_of(tp, fp, fn) >= thr - 5e-4


def search_exceeds(y, thr):
    """True if ANY of the 3,762 configurations reaches thr for labels y."""
    for algo in ORDER:
        for sub in SUBSETS:
            if reaches(_X, y, algo, sub, thr):
                return True, algo, [N.FEATURES[k] for k in sub]
    return False, None, None


def _one(args):
    b, y, thr, seed = args
    yp = np.random.RandomState(seed + b).permutation(y)
    t0 = time.time()
    hit, algo, feats = search_exceeds(yp, thr)
    return b, bool(hit), algo, feats, time.time() - t0, yp.tolist()


def _init(X):
    global _X
    _X = X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lab', required=True); ap.add_argument('--q', required=True)
    ap.add_argument('--task', required=True, choices=list(N.TASKS))
    ap.add_argument('--observed', type=float, required=True, help='best F1 with the true labels')
    ap.add_argument('--max-perm', type=int, default=200)
    ap.add_argument('--hits', type=int, default=10)
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--seed', type=int, default=20261115)
    ap.add_argument('--out', default='perm_results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = N.load(a.lab, a.q)
    X = d[N.FEATURES].values.astype(float); y = d[a.task].values.astype(int)
    _init(X)
    path = f'{a.out}/perm_{a.task}.jsonl'
    done = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); done[r['b']] = r
    # sanity check: the true labels must reach the observed score
    if 'true' not in done:
        t0 = time.time(); hit, algo, feats = search_exceeds(y, a.observed)
        assert hit, 'the observed score is not reached with the true labels'
        r = {'b': 'true', 'hit': True, 'algo': algo, 'features': feats, 'sec': time.time() - t0}
        open(path, 'a').write(json.dumps(r) + '\n'); done['true'] = r
        print(f'true labels reach F1 >= {a.observed}: {algo} {feats} ({r["sec"]:.0f}s)', flush=True)

    from multiprocessing import Pool
    todo = [b for b in range(a.max_perm) if b not in done]
    hits = sum(1 for k, r in done.items() if k != 'true' and r['hit'])
    nrun = len([k for k in done if k != 'true'])
    with Pool(a.jobs, initializer=_init, initargs=(X,)) as pool:
        for b, hit, algo, feats, sec, yp in pool.imap(_one, [(b, y, a.observed, a.seed) for b in todo]):
            r = {'b': b, 'hit': hit, 'algo': algo, 'features': feats, 'sec': sec, 'labels': yp}
            open(path, 'a').write(json.dumps(r) + '\n')
            nrun += 1; hits += hit
            print(f'perm {b}: {"EXCEEDS" if hit else "below"} ({sec:.0f}s)  running: {hits}/{nrun}', flush=True)
            if hits >= a.hits:
                pool.terminate(); break
    p = hits / nrun if hits >= a.hits else (hits + 1) / (nrun + 1)
    res = {'task': a.task, 'observed_best_F1': a.observed, 'permutations': nrun, 'exceedances': hits,
           'p': p, 'stopped_early': hits >= a.hits, 'seed': a.seed}
    json.dump(res, open(f'{a.out}/summary_{a.task}.json', 'w'), indent=2)
    print('RESULT', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
