"""
Ablation-эксперимент по компонентам коллапса tr(Sigma).

Двенадцать конфигов: основные 4 режима + контроли с alpha=0, beta=0
и случайной выдачей. Назначение — отделить три источника:
beta-дрейф, петлю переобучения, концентрацию выдачи.

Дополнительно: PCA-снапшоты, per-user displacement, concentration of recs.
"""

from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.decomposition import PCA

from sim.user_generator import GMMUserGenerator
from sim.environment    import SimulationEnvironment, ExperimentDataset
from sim.click_model    import ClickModel
from models.rec_models  import RecModel
from models.serving     import ServingPolicy


FIG_DIR = Path(__file__).resolve().parents[3] / 'paper' / 'figures' / 'diagnostic'
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'diagnostic'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Базовая настройка
# ---------------------------------------------------------------------------

def make_gmm_params(K, dim, inter_dist, sigma):
    means, covs, weights = [], [], []
    for k in range(K):
        angle = 2 * np.pi * k / K
        m = np.zeros(dim)
        m[0] = inter_dist * np.cos(angle)
        m[1] = inter_dist * np.sin(angle)
        means.append(m); covs.append(sigma**2 * np.eye(dim)); weights.append(1.0/K)
    return means, covs, weights


def make_items(N_items, K, means, sigma, dim, rng):
    parts = []
    for k in range(K):
        n_k = N_items // K if k < K-1 else N_items - (N_items//K)*(K-1)
        parts.append(rng.multivariate_normal(means[k], sigma**2*np.eye(dim), n_k))
    return np.vstack(parts)


def make_pref(u, i, tau=2.0):
    return 1/(1+np.exp(-(u@i.T)/tau))


CONFIGS = {
    # name: (mode, override_alpha, override_beta, policy)
    'closed_loop':           ('closed_loop',  None, None,  'top_k'),
    'static':                ('static',       None, None,  'top_k'),
    'fresh_oracle':          ('fresh_oracle', None, None,  'top_k'),
    'no_influence':          ('no_influence', None, None,  'top_k'),
    'closed_loop_alpha0':    ('closed_loop',  0.0,  None,  'top_k'),
    'static_alpha0':         ('static',       0.0,  None,  'top_k'),
    'closed_loop_beta0':     ('closed_loop',  None, 0.0,   'top_k'),
    'static_beta0':          ('static',       None, 0.0,   'top_k'),
    'fresh_oracle_beta0':    ('fresh_oracle', None, 0.0,   'top_k'),
    'closed_loop_random':    ('closed_loop',  None, None,  'softmax'),  # T очень большая
    'static_random':         ('static',       None, None,  'softmax'),
    'no_influence_beta0':    ('no_influence', None, 0.0,   'top_k'),
}


PARAMS = dict(
    N_USERS=300, N_ITEMS=300, EMB_DIM=8,
    K_GMM=3, K_REC=10, T=100, T_RET=10,
    REPLACE=0.05, ADHERENCE=0.7,
    USER_DRIFT=0.005, DRIFT_ALPHA=0.02,
    INTER_DIST=5.0, SIGMA_K=0.8,
    USAGE_RATE=0.8, NOISE=0.05)


def build_env(cfg_name, seed):
    mode, ov_a, ov_b, policy_name = CONFIGS[cfg_name]
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)

    means, covs, w = make_gmm_params(PARAMS['K_GMM'], PARAMS['EMB_DIM'],
                                     PARAMS['INTER_DIST'], PARAMS['SIGMA_K'])
    gen = GMMUserGenerator(means, covs, w,
                           replacement_rate=PARAMS['REPLACE'], memory_effect=6)
    np.random.seed(seed)
    u_emb, _ = gen.initialize(PARAMS['N_USERS'])
    i_emb = make_items(PARAMS['N_ITEMS'], PARAMS['K_GMM'], means,
                       PARAMS['SIGMA_K'], PARAMS['EMB_DIM'], rng)
    pref = make_pref(u_emb, i_emb)
    matrix = np.full((PARAMS['N_USERS'], PARAMS['N_ITEMS']), np.nan)
    ds = ExperimentDataset(u_emb.copy(), i_emb.copy(), matrix)

    model = RecModel(PARAMS['EMB_DIM'], PARAMS['EMB_DIM'], hidden_size=64)
    # Если 'softmax' — это random serving с очень большой температурой
    if policy_name == 'softmax':
        pol = ServingPolicy('softmax', temperature=100.0)
    else:
        pol = ServingPolicy(policy_name)

    if ov_a is not None:
        alpha = float(ov_a)
    else:
        alpha = 0.0 if mode == 'no_influence' else PARAMS['ADHERENCE']
    click = ClickModel(adherence=alpha, usage_rate=PARAMS['USAGE_RATE'],
                       novelty_preference=0.0, noise_level=PARAMS['NOISE'])

    beta = PARAMS['USER_DRIFT'] if ov_b is None else float(ov_b)
    d_alpha = PARAMS['DRIFT_ALPHA'] if mode == 'closed_loop' else 0.0

    env = SimulationEnvironment(
        dataset=ds, rec_model=model, user_generator=gen, click_model=click,
        serving_policy=pol, mode=mode, true_preference_matrix=pref,
        retrain_period=PARAMS['T_RET'], K=PARAMS['K_REC'],
        seen_filter=True, user_drift_beta=beta, drift_alpha=d_alpha)
    return env, u_emb.copy(), i_emb.copy()


def run(cfg_name, seed, save_trajectories=False):
    env, u0, i_emb = build_env(cfg_name, seed)
    traj = [env.dataset.users_embeddings.copy()] if save_trajectories else None
    for t in range(PARAMS['T']):
        env.step(t)
        if save_trajectories and (t+1) in (1, 10, 25, 50, 75, 99):
            traj.append(env.dataset.users_embeddings.copy())
    df = env.metrics.get_dataframe()
    uT = env.dataset.users_embeddings.copy()
    return df, u0, uT, i_emb, traj


# ---------------------------------------------------------------------------
# Run all configs
# ---------------------------------------------------------------------------

def main():
    configs = list(CONFIGS.keys())
    n_seeds = 5
    rows = []
    pca_snapshots = {}

    for cfg in configs:
        print(f'\n=== {cfg} ===')
        for seed in range(n_seeds):
            t0 = time.time()
            save_traj = (seed == 0)  # сохраняем траектории только для seed=0
            df, u0, uT, i_emb, traj = run(cfg, seed, save_trajectories=save_traj)
            dt = time.time() - t0
            disp = np.mean(np.linalg.norm(uT - u0, axis=1))
            cov0_norm = np.linalg.norm(np.cov(u0, rowvar=False), 'fro')
            covT_norm = np.linalg.norm(np.cov(uT, rowvar=False), 'fro')
            print(f'  seed={seed} trΣ:{df.trace_sigma.iloc[0]:.2f}->{df.trace_sigma.iloc[-1]:.2f}'
                  f' KL_T:{df.kl_from_initial.iloc[-1]:.2f}'
                  f' Gini:{df.gini_exposure.iloc[-1]:.3f}'
                  f' disp:{disp:.3f}'
                  f' ({dt:.1f}s)')
            for _, r in df.iterrows():
                rows.append(dict(
                    cfg=cfg, seed=seed, t=int(r.t),
                    trace_sigma=float(r.trace_sigma),
                    kl_from_initial=float(r.kl_from_initial),
                    gini_exposure=float(r.gini_exposure),
                    exposure_entropy=float(r.exposure_entropy),
                    catalog_coverage=float(r.catalog_coverage),
                    intra_list_diversity=float(r.intra_list_diversity),
                    observed_quality=float(r.observed_quality),
                    true_quality=float(r.true_quality),
                    intra_cluster_variance=float(r.intra_cluster_variance),
                    inter_cluster_distance=float(r.inter_cluster_distance),
                ))
            if save_traj:
                pca_snapshots[cfg] = (u0, traj, i_emb)

    df_all = pd.DataFrame(rows)
    df_all.to_csv(OUT_DIR / 'mechanism.csv', index=False)
    print(f'\nSaved {len(df_all)} rows to mechanism.csv')

    # --- сводка ---
    print('\n=== Финальные метрики (среднее по seed) ===')
    summ = df_all[df_all['t'] == df_all['t'].max()].groupby('cfg').agg(
        trace_sigma=('trace_sigma', 'mean'),
        kl_T=('kl_from_initial', 'mean'),
        gini=('gini_exposure', 'mean'),
        coverage=('catalog_coverage', 'mean'),
        true_quality=('true_quality', 'mean'),
        intra_cluster_var=('intra_cluster_variance', 'mean'),
    )
    trΣ0 = df_all[df_all['t'] == 0].groupby('cfg')['trace_sigma'].mean()
    summ['trΣ_0'] = trΣ0
    summ['drop_pct'] = 100 * (1 - summ['trace_sigma'] / summ['trΣ_0'])
    print(summ.round(3).to_string())
    summ.to_csv(OUT_DIR / 'mechanism_summary.csv')

    # --- основной диагностический рисунок ---
    metrics = [('trace_sigma', r'$\mathrm{tr}(\hat{\Sigma}_t^u)$'),
               ('kl_from_initial', r'$KL(P_t\|P_0)$'),
               ('gini_exposure', 'Gini показов'),
               ('catalog_coverage', 'coverage'),
               ('intra_cluster_variance', 'intra-cluster variance'),
               ('true_quality', 'NDCG@K')]
    groups = {
        'main 4 modes':   ['closed_loop', 'static', 'fresh_oracle', 'no_influence'],
        'b0 controls':    ['closed_loop_beta0', 'static_beta0', 'fresh_oracle_beta0', 'no_influence_beta0'],
        'a0 controls':    ['closed_loop', 'closed_loop_alpha0', 'static_alpha0', 'no_influence'],
        'random serving': ['closed_loop', 'static', 'closed_loop_random', 'static_random'],
    }

    palette = plt.cm.tab10.colors

    for gname, glist in groups.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (mkey, ylab) in zip(axes.flat, metrics):
            for idx, cfg in enumerate(glist):
                d = df_all[df_all['cfg'] == cfg]
                if d.empty:
                    continue
                g = d.groupby('t')[mkey].agg(['mean', 'std']).reset_index()
                ax.plot(g.t, g['mean'], label=cfg, color=palette[idx % 10], lw=1.6)
                ax.fill_between(g.t, g['mean']-g['std'], g['mean']+g['std'],
                                color=palette[idx % 10], alpha=0.12)
            ax.set_xlabel('t'); ax.set_ylabel(ylab)
            ax.grid(True, ls='--', alpha=0.4)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'Mechanism diagnostic: {gname}', y=1.01)
        plt.tight_layout()
        slug = gname.replace(' ', '_').replace('=', '').replace('β','b').replace('α','a')
        fig.savefig(FIG_DIR / f'DIAG_mech_{slug}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'  saved DIAG_mech_{slug}.pdf')

    # --- PCA snapshots для нескольких ключевых режимов ---
    key_cfgs = ['closed_loop', 'static', 'fresh_oracle', 'no_influence',
                'static_beta0', 'closed_loop_alpha0']
    n_show = sum(1 for c in key_cfgs if c in pca_snapshots)
    if n_show > 0:
        fig, axes = plt.subplots(n_show, 4, figsize=(16, 3.5*n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)
        for ri, cfg in enumerate([c for c in key_cfgs if c in pca_snapshots]):
            u0, traj, i_emb = pca_snapshots[cfg]
            pca = PCA(n_components=2)
            pca.fit(np.vstack([u0, i_emb]))
            snaps_t = [0, 25, 50, 99]
            traj_t_idx = [0, 2, 3, 5]  # соответствует (1,10,25,50,75,99) - выбираем близкие
            # уточняем: traj содержит [t=0, t=1, t=10, t=25, t=50, t=75, t=99]
            traj_map = {0:0, 25:3, 50:4, 99:6}
            for ci, t in enumerate(snaps_t):
                ax = axes[ri, ci]
                idx = traj_map.get(t, 0)
                if idx >= len(traj):
                    idx = len(traj) - 1
                u_t = traj[idx]
                u_proj = pca.transform(u_t)
                i_proj = pca.transform(i_emb)
                ax.scatter(i_proj[:,0], i_proj[:,1], s=3, c='lightgray', alpha=0.5, label='items')
                ax.scatter(u_proj[:,0], u_proj[:,1], s=8, c='C0', alpha=0.6, label='users')
                ax.set_title(f'{cfg}, t={t}', fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                # одинаковый extent
                all_pts = np.vstack([u_proj, i_proj])
                xmin, xmax = all_pts[:,0].min(), all_pts[:,0].max()
                ymin, ymax = all_pts[:,1].min(), all_pts[:,1].max()
                ax.set_xlim(xmin-0.5, xmax+0.5); ax.set_ylim(ymin-0.5, ymax+0.5)
        plt.tight_layout()
        fig.savefig(FIG_DIR / 'DIAG_mech_pca_snapshots.pdf', bbox_inches='tight')
        plt.close(fig)
        print('  saved DIAG_mech_pca_snapshots.pdf')

    print('\nDone.')


if __name__ == '__main__':
    main()
