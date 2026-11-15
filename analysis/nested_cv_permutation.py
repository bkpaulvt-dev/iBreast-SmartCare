#!/usr/bin/env python3
"""
nested_cv_permutation.py
========================
Analyses requested by Reviewer comment 1 for the iBreast manuscript.

  A. FLAT search   : 627 feature subsets x 6 algorithms under LOO-CV
                     (the procedure behind Table 1; selection and evaluation
                     use the same folds, so the best score is optimistic).
  B. FIXED models  : the three Table 1 models, with exact Clopper-Pearson
                     95% CIs (Acc, Se, Sp), bootstrap 95% CIs (F1, AUC) and an
                     exact label-permutation test. NOT selection-aware.
  C. NESTED LOO-CV : outer leave-one-donor-out; inside every outer fold the
                     feature subset AND algorithm are re-selected by an inner
                     LOO-CV on the 9 training donors only. Unbiased w.r.t.
                     model selection. Reported with CIs and a binomial test
                     against chance.
  D. SEARCH-LEVEL PERMUTATION (optional, --perm-search B):
                     the whole flat search is repeated on B label
                     permutations; p = P(best permuted score >= best observed).
                     This is the selection-aware permutation test for Table 1.

Model settings are identical to MULTIVARIATE_LAB_2to5.py.

Usage
-----
  python nested_cv_permutation.py --lab LabledMilkSamples_FabricatedLabSensor.xlsx \
         --q Questionnaire.csv --out results --jobs 32 --perm-search 200

  --algos fast   restricts to LR, SVM, NB, KNN (minutes instead of hours)
  --skip-nested / --skip-flat to run parts only
"""
import argparse, itertools, json, math, os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

from joblib import Parallel, delayed
from scipy.stats import beta, binomtest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ['K_mV', 'Na_mV', 'Cl_mV', 'EC_Ohm',
            'Na_K_Ratio', 'Na_Cl_Ratio', 'K_Cl_Ratio',
            'Na_EC_Ratio', 'K_EC_Ratio', 'Cl_EC_Ratio']
SIZES = [2, 3, 4, 5]
TASKS = {'Inflammation': 'Breast inflammation',
         'Mastitis_Diagnosed': 'Mastitis',
         'Blocked_Ducts': 'Blocked ducts'}
TABLE1 = {'Inflammation': ('SVM (RBF)', ('EC_Ohm', 'Na_K_Ratio')),
          'Mastitis_Diagnosed': ('Naive Bayes', ('Cl_mV', 'Na_Cl_Ratio')),
          'Blocked_Ducts': ('Rand Forest', ('Cl_mV', 'Na_K_Ratio'))}
ALGO_ORDER = ['Logistic Reg', 'SVM (RBF)', 'Rand Forest', 'Grad Boost',
              'Naive Bayes', 'KNN (k=3)']
FAST = ['Logistic Reg', 'SVM (RBF)', 'Naive Bayes', 'KNN (k=3)']


def make_model(name):
    if name == 'Logistic Reg':
        return LogisticRegression(max_iter=1000, random_state=42)
    if name == 'SVM (RBF)':
        return SVC(kernel='rbf', probability=True, gamma='scale', random_state=42)
    if name == 'Rand Forest':
        return RandomForestClassifier(n_estimators=100, max_depth=3,
                                      random_state=42, n_jobs=1)
    if name == 'Grad Boost':
        return GradientBoostingClassifier(n_estimators=50, max_depth=2,
                                          random_state=42)
    if name == 'Naive Bayes':
        return GaussianNB()
    if name == 'KNN (k=3)':
        return KNeighborsClassifier(n_neighbors=3)
    raise ValueError(name)


def pipe(name):
    return Pipeline([('sc', StandardScaler()), ('clf', make_model(name))])


# ───────────────────────── data ─────────────────────────
def load(lab_path, q_path):
    lab = pd.read_excel(lab_path, sheet_name='FabricatedSensor')
    lab.columns = [str(c).strip() for c in lab.columns]
    ren = {}
    for c in lab.columns:
        lc = c.lower()
        if 'mother' in lc or 'id' == lc: ren[c] = 'Sample_ID'
        elif lc.startswith('ec'): ren[c] = 'EC_Ohm'
        elif lc.startswith('k'): ren[c] = 'K_mV'
        elif lc.startswith('na'): ren[c] = 'Na_mV'
        elif lc.startswith('cl'): ren[c] = 'Cl_mV'
    lab = lab.rename(columns=ren)
    for c in ['K_mV', 'Na_mV', 'Cl_mV', 'EC_Ohm']:
        lab[c] = pd.to_numeric(lab[c])
    lab['Na_K_Ratio'] = lab.Na_mV / lab.K_mV
    lab['Na_Cl_Ratio'] = lab.Na_mV / lab.Cl_mV
    lab['K_Cl_Ratio'] = lab.K_mV / lab.Cl_mV
    lab['Na_EC_Ratio'] = lab.Na_mV / lab.EC_Ohm
    lab['K_EC_Ratio'] = lab.K_mV / lab.EC_Ohm
    lab['Cl_EC_Ratio'] = lab.Cl_mV / lab.EC_Ohm

    q = pd.read_csv(q_path, encoding='latin-1')
    def col(key):
        return [c for c in q.columns if key in c.lower()][0]
    out = pd.DataFrame({
        'Sample_ID': q[col('5 character code')].astype(str).str.strip(),
        'Inflammation': (q[col('breast inflammation')].str.strip() == 'Yes').astype(int),
        'Mastitis_Diagnosed': (q[col('diagnosed with mastitis')].str.strip() == 'Yes').astype(int),
        'Blocked_Ducts': (q[col('blocked milk ducts')].str.strip() == 'Yes').astype(int),
    })
    d = out.merge(lab[['Sample_ID'] + FEATURES], on='Sample_ID').sort_values('Sample_ID')
    assert len(d) == 10, f"expected 10 donors, got {len(d)}"
    return d.reset_index(drop=True)


def configs(algos):
    out = []
    for k in SIZES:
        for sub in itertools.combinations(range(len(FEATURES)), k):
            for a in algos:
                out.append((a, sub))
    return out


# ───────────────────────── metrics ─────────────────────────
def cm(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    return tp, tn, fp, fn


def f1(y, p):
    tp, tn, fp, fn = cm(y, p)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def auc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float('nan')


def cp(k, n, a=0.05):
    """Exact Clopper-Pearson interval."""
    if n == 0: return (float('nan'), float('nan'))
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return (float(lo), float(hi))


def boot_ci(y, pred, prob, B=5000, seed=0):
    """Stratified percentile bootstrap over donors for F1 and AUC."""
    r = np.random.RandomState(seed)
    y = np.asarray(y); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    f, a = [], []
    for _ in range(B):
        idx = np.r_[r.choice(pos, len(pos)), r.choice(neg, len(neg))]
        f.append(f1(y[idx], pred[idx])); a.append(auc(y[idx], prob[idx]))
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return q(f), q(a)


def summarise(y, pred, prob):
    tp, tn, fp, fn = cm(y, pred); n = len(y)
    f_ci, a_ci = boot_ci(np.asarray(y), np.asarray(pred), np.asarray(prob))
    k = tp + tn
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Acc': k / n, 'Acc_CI': cp(k, n),
        'Se': tp / (tp + fn) if tp + fn else float('nan'), 'Se_CI': cp(tp, tp + fn),
        'Sp': tn / (tn + fp) if tn + fp else float('nan'), 'Sp_CI': cp(tn, tn + fp),
        'F1': f1(y, pred), 'F1_CI': f_ci,
        'AUC': auc(y, prob), 'AUC_CI': a_ci,
        'majority_rate': max(np.mean(y), 1 - np.mean(y)),
        'p_binom_vs_0.5': float(binomtest(k, n, 0.5, alternative='greater').pvalue),
        'p_binom_vs_majority': float(binomtest(
            k, n, max(np.mean(y), 1 - np.mean(y)), alternative='greater').pvalue),
    }


# ───────────────────────── core fits ─────────────────────────
def loo_fixed(X, y, algo, sub, need_prob=True):
    """LOO predictions (label, prob) for one fixed (algorithm, subset)."""
    n = len(y); Xs = X[:, list(sub)]
    pred = np.zeros(n, int); prob = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        if len(np.unique(y[tr])) < 2:
            pred[i] = int(y[tr][0]); prob[i] = float(y[tr][0]); continue
        m = pipe(algo).fit(Xs[tr], y[tr])
        pred[i] = int(m.predict(Xs[i:i + 1])[0])
        if need_prob:
            prob[i] = float(m.predict_proba(Xs[i:i + 1])[0, 1])
    return pred, prob


def _flat_chunk(X, y, chunk):
    out = []
    for algo, sub in chunk:
        pred, _ = loo_fixed(X, y, algo, sub, need_prob=False)
        out.append((f1(y, pred), acc(y, pred)))
    return out


def flat_search(X, y, cfgs, jobs):
    chunks = [cfgs[i::max(1, jobs * 4)] for i in range(max(1, jobs * 4))]
    res = Parallel(n_jobs=jobs)(delayed(_flat_chunk)(X, y, c) for c in chunks)
    f = np.zeros(len(cfgs)); a = np.zeros(len(cfgs))
    for ci, r in enumerate(res):
        idx = list(range(len(cfgs)))[ci::max(1, jobs * 4)]
        for j, (ff, aa) in zip(idx, r):
            f[j] = ff; a[j] = aa
    return f, a


def _pair_chunk(X, y, chunk, pairs):
    """For every config: fit once per left-out PAIR {i,j}, predict both i and j.
    P[c, i, j] = label predicted for donor j by the model trained without i and j."""
    n = len(y); out = np.full((len(chunk), n, n), -1, np.int8)
    for c, (algo, sub) in enumerate(chunk):
        Xs = X[:, list(sub)]
        for i, j in pairs:
            tr = np.ones(n, bool); tr[[i, j]] = False
            if len(np.unique(y[tr])) < 2:
                out[c, i, j] = out[c, j, i] = int(y[tr][0]); continue
            m = pipe(algo).fit(Xs[tr], y[tr])
            p = m.predict(Xs[[i, j]])
            out[c, j, i] = int(p[0])   # trained w/o i,j -> predict i  (outer fold j)
            out[c, i, j] = int(p[1])   # trained w/o i,j -> predict j  (outer fold i)
    return out


def select(f, a, cfgs):
    """Deterministic selection rule: max F1, then max accuracy, then fewest
    features, then first in the fixed enumeration order."""
    order = sorted(range(len(cfgs)),
                   key=lambda c: (-round(f[c], 10), -round(a[c], 10), len(cfgs[c][1]), c))
    return order[0]


def nested_loo(X, y, cfgs, jobs):
    n = len(y); pairs = list(itertools.combinations(range(n), 2))
    nchunk = max(1, jobs * 8)
    chunks = [cfgs[i::nchunk] for i in range(nchunk)]
    res = Parallel(n_jobs=jobs)(delayed(_pair_chunk)(X, y, c, pairs) for c in chunks)
    P = np.zeros((len(cfgs), n, n), np.int8)
    for ci, r in enumerate(res):
        idx = list(range(len(cfgs)))[ci::nchunk]
        P[idx] = r
    pred = np.zeros(n, int); prob = np.zeros(n); chosen = []
    for i in range(n):                       # outer fold: donor i held out
        inner = [j for j in range(n) if j != i]
        yi = y[inner]
        f = np.array([f1(yi, P[c, i, inner]) for c in range(len(cfgs))])
        a = np.array([acc(yi, P[c, i, inner]) for c in range(len(cfgs))])
        best = select(f, a, cfgs)
        algo, sub = cfgs[best]
        tr = np.arange(n) != i
        m = pipe(algo).fit(X[tr][:, list(sub)], y[tr])
        pred[i] = int(m.predict(X[i:i + 1, list(sub)])[0])
        prob[i] = float(m.predict_proba(X[i:i + 1, list(sub)])[0, 1])
        chosen.append({'held_out': int(i), 'algorithm': algo,
                       'features': [FEATURES[k] for k in sub],
                       'inner_F1': float(f[best]), 'inner_Acc': float(a[best]),
                       'n_tied_at_best_inner_F1': int((np.round(f, 10) == round(f[best], 10)).sum())})
    return pred, prob, chosen


def exact_perm_fixed(X, y, algo, sub, jobs, max_enum=5000, seed=0):
    """Exact permutation test for ONE pre-specified model (all distinct
    labelings with the same class sizes). Not selection-aware."""
    n = len(y); k = int(y.sum())
    combos = list(itertools.combinations(range(n), k))
    if len(combos) > max_enum:
        r = np.random.RandomState(seed)
        combos = [tuple(r.choice(n, k, replace=False)) for _ in range(max_enum)]
    def one(pos):
        yp = np.zeros(n, int); yp[list(pos)] = 1
        pred, prob = loo_fixed(X, yp, algo, sub)
        return acc(yp, pred), f1(yp, pred), auc(yp, prob)
    res = np.array(Parallel(n_jobs=jobs)(delayed(one)(c) for c in combos))
    pred, prob = loo_fixed(X, y, algo, sub)
    obs = np.array([acc(y, pred), f1(y, pred), auc(y, prob)])
    p = [(np.nan_to_num(res[:, m], nan=-1) >= obs[m] - 1e-12).mean() for m in range(3)]
    return {'n_labelings': len(combos), 'p_Acc': float(p[0]), 'p_F1': float(p[1]),
            'p_AUC': float(p[2]), 'null_mean_Acc': float(res[:, 0].mean())}, pred, prob


def perm_search(X, y, cfgs, jobs, B, seed=0):
    """Selection-aware permutation test of the FLAT search (max statistic)."""
    f, a = flat_search(X, y, cfgs, jobs)
    obs_f, obs_a = f.max(), a.max()
    r = np.random.RandomState(seed); nf = []; na = []
    for b in range(B):
        yp = r.permutation(y)
        fb, ab = flat_search(X, yp, cfgs, jobs)
        nf.append(fb.max()); na.append(ab.max())
        print(f"    perm {b + 1}/{B}: best F1={fb.max():.3f} best Acc={ab.max():.2f}", flush=True)
    nf = np.array(nf); na = np.array(na)
    return {'B': B, 'obs_best_F1': float(obs_f), 'obs_best_Acc': float(obs_a),
            'p_best_F1': float((1 + (nf >= obs_f - 1e-12).sum()) / (B + 1)),
            'p_best_Acc': float((1 + (na >= obs_a - 1e-12).sum()) / (B + 1)),
            'null_best_F1_mean': float(nf.mean()), 'null_best_Acc_mean': float(na.mean()),
            'null_best_F1': nf.tolist(), 'null_best_Acc': na.tolist()}


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lab', required=True); ap.add_argument('--q', required=True)
    ap.add_argument('--out', default='results'); ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--algos', choices=['all', 'fast'], default='all')
    ap.add_argument('--tasks', nargs='*', default=list(TASKS))
    ap.add_argument('--skip-flat', action='store_true')
    ap.add_argument('--skip-nested', action='store_true')
    ap.add_argument('--skip-fixed', action='store_true')
    ap.add_argument('--perm-search', type=int, default=0)
    a = ap.parse_args()
    jobs = os.cpu_count() if a.jobs == -1 else a.jobs
    os.makedirs(a.out, exist_ok=True)
    d = load(a.lab, a.q)
    X = d[FEATURES].values.astype(float)
    algos = ALGO_ORDER if a.algos == 'all' else FAST
    cfgs = configs(algos)
    print(f"donors={len(d)}  configs={len(cfgs)}  algos={algos}  jobs={jobs}", flush=True)
    R = {'settings': {'algos': algos, 'n_configs': len(cfgs), 'features': FEATURES}}
    for t in a.tasks:
        y = d[t].values.astype(int); r = {'pos': int(y.sum()), 'neg': int((1 - y).sum())}
        print(f"\n=== {TASKS[t]}  ({r['pos']} pos / {r['neg']} neg) ===", flush=True)
        if not a.skip_fixed:
            algo, feats = TABLE1[t]; sub = tuple(FEATURES.index(f) for f in feats)
            t0 = time.time()
            pt, pred, prob = exact_perm_fixed(X, y, algo, sub, jobs)
            s = summarise(y, pred, prob); s['exact_permutation_fixed_model'] = pt
            s['algorithm'] = algo; s['features'] = list(feats)
            r['fixed_table1_model'] = s
            print(f"  [B fixed] {algo} {feats}: Acc={s['Acc']:.2f} F1={s['F1']:.3f} "
                  f"AUC={s['AUC']:.3f} | perm p(Acc)={pt['p_Acc']:.3f} "
                  f"p(AUC)={pt['p_AUC']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if not a.skip_flat:
            t0 = time.time(); f, ac = flat_search(X, y, cfgs, jobs)
            b = select(f, ac, cfgs)
            r['flat_search'] = {
                'best_F1': float(f.max()), 'best_Acc': float(ac.max()),
                'n_configs_at_best_F1': int((np.round(f, 10) == round(f.max(), 10)).sum()),
                'n_configs_Acc_ge_0.8': int((ac >= 0.8 - 1e-12).sum()),
                'median_Acc_all_configs': float(np.median(ac)),
                'selected': {'algorithm': cfgs[b][0], 'features': [FEATURES[k] for k in cfgs[b][1]]}}
            pd.DataFrame({'algorithm': [c[0] for c in cfgs],
                          'features': ['|'.join(FEATURES[k] for k in c[1]) for c in cfgs],
                          'F1': f, 'Acc': ac}).to_csv(f"{a.out}/flat_{t}.csv", index=False)
            print(f"  [A flat ] best F1={f.max():.3f} Acc={ac.max():.2f}; "
                  f"{r['flat_search']['n_configs_at_best_F1']} configs tie at best F1; "
                  f"median Acc over all configs={np.median(ac):.2f}  ({time.time() - t0:.0f}s)", flush=True)
        if not a.skip_nested:
            t0 = time.time(); pred, prob, chosen = nested_loo(X, y, cfgs, jobs)
            s = summarise(y, pred, prob); s['selected_per_outer_fold'] = chosen
            s['n_distinct_models_selected'] = len({(c['algorithm'], tuple(c['features'])) for c in chosen})
            s['predictions'] = pred.tolist(); s['probabilities'] = prob.tolist()
            r['nested_loo'] = s
            print(f"  [C nested] Acc={s['Acc']:.2f} [{s['Acc_CI'][0]:.2f},{s['Acc_CI'][1]:.2f}] "
                  f"F1={s['F1']:.3f} Se={s['Se']:.2f} Sp={s['Sp']:.2f} AUC={s['AUC']:.3f} "
                  f"p(binom vs 0.5)={s['p_binom_vs_0.5']:.3f}; "
                  f"{s['n_distinct_models_selected']} distinct models over 10 folds  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if a.perm_search:
            r['perm_search'] = perm_search(X, y, cfgs, jobs, a.perm_search)
            print(f"  [D perm-search] p(best F1)={r['perm_search']['p_best_F1']:.3f}", flush=True)
        R[t] = r
        json.dump(R, open(f"{a.out}/results.json", 'w'), indent=2)
    print("\nDONE ->", a.out)


if __name__ == '__main__':
    main()
