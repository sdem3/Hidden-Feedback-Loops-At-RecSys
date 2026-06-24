# theorem: loop-amplifier (Thm 2) — is the closed_loop KL-gap a drift_alpha artifact?
"""
Decisive ablation: does the closed_loop − static gap in KL(P_T‖P_0) survive
when the exogenous GMM-center drift `drift_alpha` is switched OFF?

Motivation
----------
In H7/H8 the closed_loop runs used drift_alpha=0.02 (active ONLY in closed_loop),
which directly translates GMM component_means toward consumed content. Because
`kl_from_initial` is a Gaussian KL whose mean-shift term `diff·Σ0⁻¹·diff` is fed
by that translated mean, the measured closed_loop KL-gap could be a pure artifact
of drift_alpha rather than a consequence of the retraining loop.

This script reproduces the H7 warm-MLP configuration exactly and runs a 2×2 design:
    {drift_alpha = 0.02, 0.0} × {closed_loop, static}
with N_SEEDS seeds, then reports the bootstrap CI of the KL-gap closed−static in
each drift_alpha setting. We also decompose KL into its mean-shift and dispersion
parts so we can see WHICH component the loop moves.

If the gap survives at drift_alpha=0 → the loop (retraining on own logs) is the
real cause and drift_alpha can be removed from the theory. If it collapses to
zero → the original H7 gap was a drift_alpha artifact, and the theory must be
rewritten accordingly. Either way the result is recorded honestly.
"""

import sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../code
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import product

from sim.user_generator import GMMUserGenerator
from sim.environment    import SimulationEnvironment, ExperimentDataset
from sim.click_model    import ClickModel
from models.rec_models  import RecModel
from models.serving     import ServingPolicy

RESULTS_DIR = ROOT / 'experiments' / 'results' / 'diagnostic'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── H7 configuration (verbatim) ─────────────────────────────────────
EMB_DIM, N_USERS, N_ITEMS, K_REC = 8, 300, 300, 10
T, T_RET, REPLACE = 100, 10, 0.05
ADHERENCE, USER_DRIFT = 0.7, 0.005
INTER_DIST, SIGMA_K, K_GMM = 5.0, 0.8, 3
N_PRETRAIN_EPOCHS, PRETRAIN_LR, PRETRAIN_SUBSAMPLE = 25, 3e-3, 6000

SMOKE = '--smoke' in sys.argv
N_SEEDS = 3 if SMOKE else 30

PARAMS = dict(N_USERS=N_USERS, N_ITEMS=N_ITEMS, EMB_DIM=EMB_DIM,
              K_GMM=K_GMM, K_REC=K_REC, T=T, T_RET=T_RET,
              REPLACE=REPLACE, ADHERENCE=ADHERENCE, USER_DRIFT=USER_DRIFT,
              INTER_DIST=INTER_DIST, SIGMA_K=SIGMA_K)


# ── helpers copied from H7 ──────────────────────────────────────────
def make_gmm_params(K, dim, inter_dist, sigma):
    means, covs, weights = [], [], []
    for k in range(K):
        angle = 2 * np.pi * k / K
        m = np.zeros(dim)
        m[0] = inter_dist * np.cos(angle)
        m[1] = inter_dist * np.sin(angle)
        means.append(m); covs.append(sigma**2 * np.eye(dim)); weights.append(1.0 / K)
    return means, covs, weights


def make_items(N_items, K, means, sigma, dim, rng):
    parts = []
    for k in range(K):
        n_k = N_items // K if k < K - 1 else N_items - (N_items // K) * (K - 1)
        parts.append(rng.multivariate_normal(means[k], sigma**2 * np.eye(dim), n_k))
    return np.vstack(parts)


def make_true_pref(users, items, tau=2.0):
    return 1 / (1 + np.exp(-(users @ items.T / tau)))


class _PairBCEDataset(Dataset):
    def __init__(self, users_emb, items_emb, target_matrix, subsample, rng):
        n_u, n_i = target_matrix.shape
        all_pairs = np.array([(u, i) for u in range(n_u) for i in range(n_i)])
        if subsample is not None and subsample < len(all_pairs):
            idx = rng.choice(len(all_pairs), subsample, replace=False)
            all_pairs = all_pairs[idx]
        self.pairs     = all_pairs.astype(np.int64)
        self.targets   = target_matrix[self.pairs[:, 0], self.pairs[:, 1]].astype(np.float32)
        self.users_emb = users_emb.astype(np.float32)
        self.items_emb = items_emb.astype(np.float32)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        return (torch.from_numpy(self.users_emb[u]),
                torch.from_numpy(self.items_emb[i]),
                torch.tensor(self.targets[idx]))


def pretrain_model(model, user_emb, item_emb, target_matrix, rng, device):
    ds = _PairBCEDataset(user_emb, item_emb, target_matrix, PRETRAIN_SUBSAMPLE, rng)
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    crit = nn.BCELoss()
    model.train()
    for _ in range(N_PRETRAIN_EPOCHS):
        for u_b, i_b, t_b in loader:
            u_b, i_b, t_b = u_b.to(device), i_b.to(device), t_b.to(device)
            opt.zero_grad()
            loss = crit(model(u_b, i_b), t_b)
            loss.backward(); opt.step()


def build_env(mode, seed, drift_alpha):
    """Warm-start MLP, H7 config, with explicit drift_alpha."""
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    means, covs, weights = make_gmm_params(K_GMM, EMB_DIM, INTER_DIST, SIGMA_K)
    gen = GMMUserGenerator(component_means=means, component_covs=covs,
                           component_weights=weights,
                           replacement_rate=REPLACE, memory_effect=6)
    np.random.seed(seed)
    user_emb, _ = gen.initialize(N_USERS)
    item_emb = make_items(N_ITEMS, K_GMM, means, SIGMA_K, EMB_DIM, rng)
    true_pref = make_true_pref(user_emb, item_emb)
    matrix = np.full((N_USERS, N_ITEMS), np.nan)
    dataset = ExperimentDataset(user_emb.copy(), item_emb.copy(), matrix)

    device = torch.device('cpu')
    model = RecModel(EMB_DIM, EMB_DIM, hidden_size=64).to(device)
    pretrain_model(model, user_emb, item_emb, true_pref, rng, device)  # warm

    policy = ServingPolicy('top_k')
    alpha_c = 0.0 if mode == 'no_influence' else ADHERENCE
    click = ClickModel(adherence=alpha_c, usage_rate=0.8, noise_level=0.05)
    d_alpha = drift_alpha if mode == 'closed_loop' else 0.0
    env = SimulationEnvironment(
        dataset=dataset, rec_model=model, user_generator=gen, click_model=click,
        serving_policy=policy, mode=mode, true_preference_matrix=true_pref,
        retrain_period=T_RET, K=K_REC, device=device, seen_filter=True,
        user_drift_beta=USER_DRIFT, drift_alpha=d_alpha)
    return env, true_pref


def kl_decompose(U, mean0, cov0):
    """Return (kl_total, kl_mean_shift, kl_dispersion) of N(mean_t,cov_t)‖N(mean0,cov0)."""
    mean_t = np.mean(U, axis=0); cov_t = np.cov(U, rowvar=False)
    mean0 = np.atleast_1d(mean0); cov0 = np.atleast_2d(cov0)
    cov_t = np.atleast_2d(cov_t); d = mean0.shape[0]
    cov0_inv = np.linalg.pinv(cov0)
    s0, ld0 = np.linalg.slogdet(cov0); s1, ld1 = np.linalg.slogdet(cov_t)
    if s0 <= 0 or s1 <= 0:
        return 0.0, 0.0, 0.0
    diff = mean0 - mean_t
    kl_mean = 0.5 * float(diff @ cov0_inv @ diff)
    kl_disp = 0.5 * float(np.trace(cov0_inv @ cov_t) - d + ld0 - ld1)
    return max(0.0, kl_mean + kl_disp), max(0.0, kl_mean), kl_disp


# ── run 2×2 ─────────────────────────────────────────────────────────
def run():
    rows = []
    designs = list(product([0.02, 0.0], ['closed_loop', 'static']))
    t_start = time.time()
    for d_alpha, mode in designs:
        for seed in range(N_SEEDS):
            env, _ = build_env(mode, seed, d_alpha)
            for t in range(T):
                env.step(t)
            U = env.dataset.users_embeddings
            kl_t, kl_m, kl_d = kl_decompose(
                U, env.metrics.initial_mean, env.metrics.initial_cov)
            df = env.metrics.get_dataframe()
            rows.append(dict(
                drift_alpha=d_alpha, mode=mode, seed=seed,
                kl_total=kl_t, kl_mean_shift=kl_m, kl_dispersion=kl_d,
                kl_metric=float(df['kl_from_initial'].iloc[-1]),
                trace_T=float(df['trace_sigma'].iloc[-1]),
                trace_0=float(df['trace_sigma'].iloc[0]),
                true_quality_T=float(df['true_quality'].iloc[-1]),
                true_quality_0=float(df['true_quality'].iloc[0]),
                observed_quality_T=float(df['observed_quality'].iloc[-1]),
            ))
        print(f'  done drift_alpha={d_alpha} mode={mode} '
              f'({time.time()-t_start:.0f}s)')
    return pd.DataFrame(rows)


def bootstrap_gap(cl, st, n_boot=5000, rng=None):
    rng = rng or np.random.default_rng(0)
    cl = np.asarray(cl); st = np.asarray(st)
    idx_cl = rng.integers(0, len(cl), size=(n_boot, len(cl)))
    idx_st = rng.integers(0, len(st), size=(n_boot, len(st)))
    diffs = cl[idx_cl].mean(1) - st[idx_st].mean(1)
    return float(diffs.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


if __name__ == '__main__':
    print(f'Running drift_alpha ablation (warm-MLP, H7 config), '
          f'N_SEEDS={N_SEEDS}, SMOKE={SMOKE}')
    df = run()
    out = RESULTS_DIR / 'drift_alpha_ablation.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved raw → {out}')

    rng = np.random.default_rng(42)
    summary = []
    print('\n=== KL-gap closed_loop − static, by drift_alpha ===')
    for d_alpha in [0.02, 0.0]:
        sub = df[df['drift_alpha'] == d_alpha]
        for col in ['kl_metric', 'kl_total', 'kl_mean_shift', 'kl_dispersion', 'trace_T']:
            cl = sub[sub['mode'] == 'closed_loop'][col].values
            st = sub[sub['mode'] == 'static'][col].values
            g, lo, hi = bootstrap_gap(cl, st, rng=rng)
            sig = '' if (lo <= 0 <= hi) else '  *SIGNIF*'
            summary.append(dict(drift_alpha=d_alpha, metric=col,
                                cl_mean=float(np.mean(cl)), st_mean=float(np.mean(st)),
                                gap=g, lo=lo, hi=hi, significant=not (lo <= 0 <= hi)))
            print(f'  da={d_alpha:<4}  {col:14s}  '
                  f'cl={np.mean(cl):7.3f}  st={np.mean(st):7.3f}  '
                  f'gap={g:+7.3f}  [{lo:+.3f},{hi:+.3f}]{sig}')
    sdf = pd.DataFrame(summary)
    sout = RESULTS_DIR / 'drift_alpha_ablation_summary.csv'
    sdf.to_csv(sout, index=False)
    print(f'\nSaved summary → {sout}')
