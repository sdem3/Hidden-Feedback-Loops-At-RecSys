"""
Декомпозиция причин коллапса tr(Sigma) по режимам.

Эксперименты:
  A — baseline 4 режима с расширенными метриками (Gini, entropy, coverage);
  B — beta-sweep по всем режимам;
  C — alpha-sweep в режиме static (без переобучения);
  D — факторный (mode, alpha, beta).

См. соответствующий раздел отчёта по экспериментам.
"""

from __future__ import annotations
import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from itertools import product

from sim.user_generator import GMMUserGenerator
from sim.environment    import SimulationEnvironment, ExperimentDataset
from sim.click_model    import ClickModel
from sim.metrics        import MetricsTracker
from models.rec_models  import RecModel
from models.serving     import ServingPolicy


FIG_DIR = Path(__file__).resolve().parents[3] / 'paper' / 'figures' / 'diagnostic'
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'diagnostic'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Подготовка
# ---------------------------------------------------------------------------

def make_gmm_params(K, dim, inter_dist, sigma):
    means, covs, weights = [], [], []
    for k in range(K):
        angle = 2 * np.pi * k / K
        m = np.zeros(dim)
        m[0] = inter_dist * np.cos(angle)
        m[1] = inter_dist * np.sin(angle)
        means.append(m)
        covs.append(sigma**2 * np.eye(dim))
        weights.append(1.0 / K)
    return means, covs, weights


def make_items(N_items, K, means, sigma, dim, rng):
    parts = []
    for k in range(K):
        n_k = N_items // K if k < K - 1 else N_items - (N_items // K) * (K - 1)
        parts.append(rng.multivariate_normal(means[k], sigma**2 * np.eye(dim), n_k))
    return np.vstack(parts)


def make_true_pref(users, items, tau=2.0):
    scores = users @ items.T / tau
    return 1 / (1 + np.exp(-scores))


def build_env(mode, seed, params, override_alpha=None, override_beta=None):
    """Строит окружение. override_* позволяет принудительно зануление α или β."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)

    means, covs, weights = make_gmm_params(
        params['K_GMM'], params['EMB_DIM'],
        params['INTER_DIST'], params['SIGMA_K'])

    gen = GMMUserGenerator(
        component_means=means, component_covs=covs, component_weights=weights,
        replacement_rate=params['REPLACE'], memory_effect=6)

    np.random.seed(seed)
    user_emb, _ = gen.initialize(params['N_USERS'])
    item_emb = make_items(params['N_ITEMS'], params['K_GMM'], means,
                          params['SIGMA_K'], params['EMB_DIM'], rng)
    true_pref = make_true_pref(user_emb, item_emb)
    matrix    = np.full((params['N_USERS'], params['N_ITEMS']), np.nan)
    dataset   = ExperimentDataset(user_emb.copy(), item_emb.copy(), matrix)

    model  = RecModel(params['EMB_DIM'], params['EMB_DIM'], hidden_size=64)
    policy = ServingPolicy(params.get('POLICY', 'top_k'),
                           epsilon=params.get('EPS', 0.1),
                           temperature=params.get('TEMP', 1.0),
                           diversity_lambda=params.get('DIV_LAMBDA', 0.5))

    if override_alpha is not None:
        alpha_c = float(override_alpha)
    else:
        alpha_c = 0.0 if mode == 'no_influence' else params['ADHERENCE']

    click  = ClickModel(adherence=alpha_c, usage_rate=params['USAGE_RATE'],
                        novelty_preference=0.0,
                        noise_level=params.get('NOISE', 0.05))

    beta = params['USER_DRIFT'] if override_beta is None else float(override_beta)
    d_alpha = params['DRIFT_ALPHA'] if mode == 'closed_loop' else 0.0

    env = SimulationEnvironment(
        dataset=dataset, rec_model=model,
        user_generator=gen, click_model=click,
        serving_policy=policy, mode=mode,
        true_preference_matrix=true_pref,
        retrain_period=params['T_RET'], K=params['K_REC'],
        seen_filter=True,
        user_drift_beta=beta,
        drift_alpha=d_alpha)
    return env


def run_one(mode, seed, params, override_alpha=None, override_beta=None):
    env = build_env(mode, seed, params, override_alpha, override_beta)
    extra_logs = {'unique_recs_per_step': [], 'n_interactions': []}
    for t in range(params['T']):
        env.step(t)
        recs = []
        if env.metrics.history:
            # Извлекаем дополнительные дешёвые статистики из шага
            pass
    df = env.metrics.get_dataframe()
    return df


# ---------------------------------------------------------------------------
# Эксперимент A: воспроизведение базового эффекта + диагностические переменные
# ---------------------------------------------------------------------------

def expA_baseline(params, modes, seeds):
    """Базовый прогон: tr(Σ), KL, gini, entropy, true_quality для всех режимов."""
    rows = []
    for mode in modes:
        for seed in seeds:
            t0 = time.time()
            df = run_one(mode, seed, params)
            elapsed = time.time() - t0
            print(f'  A | {mode:14s} seed={seed:2d}  trΣ:{df.trace_sigma.iloc[0]:.1f}'
                  f'->{df.trace_sigma.iloc[-1]:.2f}'
                  f'  KL_T:{df.kl_from_initial.iloc[-1]:.3f}'
                  f'  G:{df.gini_exposure.iloc[-1]:.3f}'
                  f'  H:{df.exposure_entropy.iloc[-1]:.2f}'
                  f'  q:{df.true_quality.mean():.3f}  ({elapsed:.1f}s)')
            for _, row in df.iterrows():
                rows.append(dict(
                    mode=mode, seed=seed, t=int(row.t),
                    trace_sigma=float(row.trace_sigma),
                    kl_from_initial=float(row.kl_from_initial),
                    leading_eigenvalue=float(row.leading_eigenvalue),
                    gini_exposure=float(row.gini_exposure),
                    exposure_entropy=float(row.exposure_entropy),
                    catalog_coverage=float(row.catalog_coverage),
                    intra_list_diversity=float(row.intra_list_diversity),
                    observed_quality=float(row.observed_quality),
                    true_quality=float(row.true_quality),
                    intra_jaccard=float(row.intra_jaccard),
                    inter_cluster_distance=float(row.inter_cluster_distance),
                ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Эксперимент B: эффект β при фиксированных модах
# ---------------------------------------------------------------------------

def expB_beta_sweep(params, modes, seeds, betas):
    rows = []
    for mode in modes:
        for beta in betas:
            for seed in seeds:
                df = run_one(mode, seed, params, override_beta=beta)
                rows.append(dict(
                    mode=mode, seed=seed, beta=beta,
                    trace_sigma_T=float(df.trace_sigma.iloc[-1]),
                    trace_sigma_0=float(df.trace_sigma.iloc[0]),
                    kl_T=float(df.kl_from_initial.iloc[-1]),
                    gini_T=float(df.gini_exposure.iloc[-1]),
                    entropy_T=float(df.exposure_entropy.iloc[-1]),
                ))
                print(f'  B | mode={mode:14s} β={beta:.4f} seed={seed:2d} '
                      f'trΣ->{df.trace_sigma.iloc[-1]:.2f}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Эксперимент C: эффект α (adherence) — отделяем «эффект петли» от «эффект β»
# ---------------------------------------------------------------------------

def expC_alpha_sweep(params, seeds, alphas, mode='static'):
    """В режиме static варьируем α от 0 до 1 — если коллапс растёт с α,
    значит причина в β-дрейфе через α (а не в петле как таковой)."""
    rows = []
    for alpha in alphas:
        for seed in seeds:
            df = run_one(mode, seed, params, override_alpha=alpha)
            rows.append(dict(
                mode=mode, seed=seed, alpha=alpha,
                trace_sigma_T=float(df.trace_sigma.iloc[-1]),
                trace_sigma_0=float(df.trace_sigma.iloc[0]),
                kl_T=float(df.kl_from_initial.iloc[-1]),
                gini_T=float(df.gini_exposure.iloc[-1]),
                entropy_T=float(df.exposure_entropy.iloc[-1]),
            ))
            print(f'  C | α={alpha:.2f} seed={seed:2d} '
                  f'trΣ:{df.trace_sigma.iloc[0]:.1f}->{df.trace_sigma.iloc[-1]:.2f}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Эксперимент D: 2D-факторный β × mode + measurement расходимостей
# ---------------------------------------------------------------------------

def expD_factorial(params, modes, seeds, betas, alphas):
    rows = []
    for mode, beta, alpha in product(modes, betas, alphas):
        if mode == 'no_influence' and alpha != 0:
            continue
        for seed in seeds:
            df = run_one(mode, seed, params,
                         override_alpha=alpha if mode != 'no_influence' else 0.0,
                         override_beta=beta)
            rows.append(dict(
                mode=mode, beta=beta, alpha=alpha, seed=seed,
                trace_sigma_T=float(df.trace_sigma.iloc[-1]),
                trace_sigma_0=float(df.trace_sigma.iloc[0]),
                kl_T=float(df.kl_from_initial.iloc[-1]),
                gini_T=float(df.gini_exposure.iloc[-1]),
                entropy_T=float(df.exposure_entropy.iloc[-1]),
                drop_pct=100 * (1 - df.trace_sigma.iloc[-1] / df.trace_sigma.iloc[0]),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {'closed_loop': '#d62728', 'static': '#ff7f0e',
          'fresh_oracle': '#2ca02c', 'no_influence': '#1f77b4'}


def plot_A(dfA, params, fname):
    """Множественные диагностические метрики по режимам."""
    metrics = [
        ('trace_sigma',      r'$\mathrm{tr}(\hat{\Sigma}_t^u)$'),
        ('kl_from_initial',  r'$KL(P_t\|P_0)$'),
        ('gini_exposure',    r'Gini показов'),
        ('exposure_entropy', r'$H(\mathrm{exposure})$'),
        ('catalog_coverage', 'catalog coverage'),
        ('observed_quality', r'обс. качество (CTR)'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric, ylab) in zip(axes.flat, metrics):
        for mode in dfA['mode'].unique():
            d = dfA[dfA['mode'] == mode]
            g = d.groupby('t')[metric].agg(['mean', 'std']).reset_index()
            ax.plot(g.t, g['mean'], label=mode, color=COLORS.get(mode, 'k'), lw=1.8)
            ax.fill_between(g.t, g['mean'] - g['std'], g['mean'] + g['std'],
                            color=COLORS.get(mode, 'k'), alpha=0.15)
        ax.set_xlabel('t')
        ax.set_ylabel(ylab)
        ax.grid(True, ls='--', alpha=0.4)
        if metric == 'trace_sigma':
            ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Диагностика коллапса (N={params["N_USERS"]}, T={params["T"]}, '
                 f'seeds={dfA.seed.nunique()})', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches='tight')
    plt.close(fig)
    print(f'    saved {fname}')


def plot_B(dfB, fname):
    """β-sweep: tr(Σ_T) vs β для каждого режима."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode in dfB['mode'].unique():
        d = dfB[dfB['mode'] == mode]
        g = d.groupby('beta')[['trace_sigma_T', 'kl_T']].agg(['mean', 'std']).reset_index()
        ts_m, ts_s = g['trace_sigma_T']['mean'], g['trace_sigma_T']['std']
        kl_m, kl_s = g['kl_T']['mean'], g['kl_T']['std']
        b = g['beta']
        axes[0].errorbar(b, ts_m, yerr=ts_s, label=mode, color=COLORS.get(mode, 'k'),
                          marker='o', capsize=3)
        axes[1].errorbar(b, kl_m, yerr=kl_s, label=mode, color=COLORS.get(mode, 'k'),
                          marker='o', capsize=3)
    for ax, ylab in zip(axes, [r'$\mathrm{tr}(\hat{\Sigma}_T)$', r'$KL(P_T\|P_0)$']):
        ax.set_xlabel(r'$\beta$ (user drift)')
        ax.set_ylabel(ylab)
        ax.set_xscale('log')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9)
    fig.suptitle(r'Эксперимент B: эффект $\beta$ во всех режимах')
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches='tight')
    plt.close(fig)
    print(f'    saved {fname}')


def plot_C(dfC, fname):
    """α-sweep в static."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    g = dfC.groupby('alpha')[['trace_sigma_T', 'kl_T', 'gini_T']].agg(['mean', 'std']).reset_index()
    a = g['alpha']
    axes[0].errorbar(a, g['trace_sigma_T']['mean'], yerr=g['trace_sigma_T']['std'],
                      marker='o', capsize=3, color='#ff7f0e')
    axes[0].set_xlabel(r'$\alpha$ (adherence в static)')
    axes[0].set_ylabel(r'$\mathrm{tr}(\hat{\Sigma}_T)$')

    axes[1].errorbar(a, g['gini_T']['mean'], yerr=g['gini_T']['std'],
                      marker='o', capsize=3, color='#ff7f0e', label='Gini')
    axes[1].set_xlabel(r'$\alpha$')
    axes[1].set_ylabel('Gini показов')
    for ax in axes:
        ax.grid(True, ls='--', alpha=0.4)
    fig.suptitle(r'Эксперимент C: коллапс в static растёт с $\alpha$ (а петли нет!)')
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches='tight')
    plt.close(fig)
    print(f'    saved {fname}')


def plot_D(dfD, fname):
    """Heatmap: β × α по режимам."""
    modes = ['closed_loop', 'static', 'fresh_oracle']
    fig, axes = plt.subplots(1, len(modes), figsize=(15, 4.5))
    for ax, mode in zip(axes, modes):
        d = dfD[dfD['mode'] == mode]
        if d.empty:
            continue
        pivot = d.groupby(['alpha', 'beta'])['drop_pct'].mean().unstack()
        im = ax.imshow(pivot.values, origin='lower', aspect='auto', cmap='RdYlBu_r',
                       vmin=0, vmax=100,
                       extent=[pivot.columns.min(), pivot.columns.max(),
                               pivot.index.min(), pivot.index.max()])
        ax.set_xlabel(r'$\beta$'); ax.set_ylabel(r'$\alpha$')
        ax.set_title(mode)
        plt.colorbar(im, ax=ax, label='drop tr(Σ), %')
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches='tight')
    plt.close(fig)
    print(f'    saved {fname}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    BASE = dict(
        N_USERS=300, N_ITEMS=300, EMB_DIM=8,
        K_GMM=3, K_REC=10, T=100, T_RET=10,
        REPLACE=0.05, ADHERENCE=0.7,
        USER_DRIFT=0.005, DRIFT_ALPHA=0.02,
        INTER_DIST=5.0, SIGMA_K=0.8,
        USAGE_RATE=0.8, NOISE=0.05,
        POLICY='top_k')

    MODES = ['closed_loop', 'static', 'fresh_oracle', 'no_influence']

    # ----- A: baseline + diagnostics -----
    print('\n[A] Baseline diagnostic — все режимы, 8 seeds')
    t0 = time.time()
    seeds_A = list(range(8))
    dfA = expA_baseline(BASE, MODES, seeds_A)
    dfA.to_csv(OUT_DIR / 'A_baseline.csv', index=False)
    plot_A(dfA, BASE, 'DIAG_A_baseline_metrics.pdf')
    print(f'  A done in {time.time()-t0:.1f}s')

    # ----- B: β-sweep по всем режимам -----
    print('\n[B] β-sweep (отключение β-дрейфа), 5 seeds × 5 β × 4 modes')
    t0 = time.time()
    seeds_B = list(range(5))
    betas   = [0.0, 0.001, 0.005, 0.02, 0.05]
    dfB = expB_beta_sweep(BASE, MODES, seeds_B, betas)
    dfB.to_csv(OUT_DIR / 'B_beta_sweep.csv', index=False)
    plot_B(dfB, 'DIAG_B_beta_sweep.pdf')
    print(f'  B done in {time.time()-t0:.1f}s')

    # ----- C: α-sweep в static -----
    print('\n[C] α-sweep в static (без переобучения!), 5 seeds × 6 α')
    t0 = time.time()
    seeds_C = list(range(5))
    alphas  = [0.0, 0.2, 0.4, 0.5, 0.7, 0.9]
    dfC = expC_alpha_sweep(BASE, seeds_C, alphas, mode='static')
    dfC.to_csv(OUT_DIR / 'C_alpha_sweep_static.csv', index=False)
    plot_C(dfC, 'DIAG_C_alpha_sweep_static.pdf')
    print(f'  C done in {time.time()-t0:.1f}s')

    # ----- D: факторный β × α по 3 режимам -----
    print('\n[D] Факторный β × α (3 modes × 4β × 4α × 3 seeds)')
    t0 = time.time()
    seeds_D = list(range(3))
    betas_D = [0.0, 0.005, 0.02, 0.05]
    alphas_D = [0.0, 0.3, 0.7, 0.9]
    dfD = expD_factorial(BASE, ['closed_loop', 'static', 'fresh_oracle'],
                         seeds_D, betas_D, alphas_D)
    dfD.to_csv(OUT_DIR / 'D_factorial.csv', index=False)
    plot_D(dfD, 'DIAG_D_beta_alpha_factorial.pdf')
    print(f'  D done in {time.time()-t0:.1f}s')

    # ----- Сводка -----
    print('\n=== Сводка по эксперименту B (β=0) ===')
    b0 = dfB[dfB['beta'] == 0.0].groupby('mode').agg(
        trace_sigma_T_mean=('trace_sigma_T', 'mean'),
        trace_sigma_T_std=('trace_sigma_T', 'std'),
        kl_T_mean=('kl_T', 'mean'),
        gini_T_mean=('gini_T', 'mean'),
    )
    print(b0.round(3).to_string())

    print('\n=== Сводка по эксперименту C (static, α-sweep) ===')
    g = dfC.groupby('alpha').agg(
        trace_sigma_T_mean=('trace_sigma_T', 'mean'),
        trace_sigma_T_std=('trace_sigma_T', 'std'),
        gini_T_mean=('gini_T', 'mean'),
    )
    print(g.round(3).to_string())

    print('\nDone.')


if __name__ == '__main__':
    main()
