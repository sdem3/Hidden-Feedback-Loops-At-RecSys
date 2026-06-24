"""
exp_2026-06-12_display_covariance.py
=====================================
Прямая проверка предсказаний СПЕЦИАЛЬНОЙ ГАУССОВОЙ МОДЕЛИ петлевого канала
(THEOREM_2.md, §6, Теорема 2).

Теория утверждает: переобучение как улучшение политики делает ВЫДАЧУ резче —
матрица резкости A растёт (6.1), ковариация ПОКАЗАННОГО контента
C = (Psi^-1 + A)^-1 сужается, и в closed_loop она по Лёвнеру меньше, чем в
static при совпадающем старте (Теорема 2):  C^closed ⪯ C^stat.
Через шумовой пол β-коллапса (Теорема 1) это углубляет коллапс вкусов.

Раньше это была НЕДОКАЗАННАЯ предпосылка (бывшее B3'), проверявшаяся лишь
КОСВЕННО (через дисперсионную часть KL вкусов). Здесь мы измеряем матрицу
ковариации ПОКАЗАННЫХ айтемов Σ_ξ напрямую — то, на чём построен §6.

Что измеряется (на матрице показанных top-K айтемов, по кластерам):
  - tr Σ_ξ(t)            — объём разброса выдачи (предсказание: closed < static);
  - logdet Σ_ξ(t)        — объём (volume);
  - λ_max Σ_ξ(t)         — ведущее направление;
  - Loewner-проверка     — доля (seed, кластер), где Σ_ξ^stat − Σ_ξ^closed ⪰ 0;
  - дисперсионное расхождение 𝒟(Σ_ξ(T) ‖ Σ_ξ(0)) — насколько у́же стало.

Части:
  A. closed vs static, базовый конфиг, ±seen_filter, N_SEEDS_A seeds  (ядро: Теорема 2).
  B. α-свип {0,0.3,0.7,0.9}: разрыв сужения растёт с α            (§6.5, скорость 1+κα).
  C. σ-свип {0.3,0.8,2.0}: зависимость от ширины кластера          (§6.6, порог).

Запуск:
  python exp_2026-06-12_display_covariance.py --smoke   # быстрый прогон (2 seeds)
  python exp_2026-06-12_display_covariance.py           # полный

API изучен из exp_2026-06-07_drift_alpha_ablation.py; сигнатуры sim/ и models/
НЕ меняются — захват показанных айтемов сделан через подкласс с переопределением
_make_recommendations (стэшит recs + метки на момент рекомендации).
"""

import sys
import time
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
from sim.environment import SimulationEnvironment, ExperimentDataset
from sim.click_model import ClickModel
from models.rec_models import RecModel
from models.serving import ServingPolicy

RESULTS_DIR = ROOT / 'experiments' / 'results' / 'diagnostic'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(__file__).resolve().parents[3] / 'paper' / 'figures' / 'diagnostic'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- #
# Конфигурация (как в H7 / §21-абляции)
# ----------------------------------------------------------------------------- #
EMB_DIM = 8
N_USERS = 300
N_ITEMS = 300
K_REC = 10
T = 100
T_RET = 10
REPLACE = 0.05
ADHERENCE = 0.7
USER_DRIFT = 0.005
INTER_DIST = 5.0
SIGMA_K = 0.8
K_GMM = 3

N_PRETRAIN_EPOCHS = 25
PRETRAIN_LR = 3e-3
PRETRAIN_SUBSAMPLE = 6000

SMOKE = '--smoke' in sys.argv
N_SEEDS_A = 2 if SMOKE else 15      # Part A
N_SEEDS_B = 2 if SMOKE else 8       # Part B (α-sweep)
N_SEEDS_C = 2 if SMOKE else 6       # Part C (σ-sweep)
if SMOKE:
    T = 30


# ----------------------------------------------------------------------------- #
# Хелперы (verbatim из exp_2026-06-07)
# ----------------------------------------------------------------------------- #
def make_gmm_params(K, dim, inter_dist, sigma):
    means, covs, weights = [], [], []
    for k in range(K):
        angle = 2 * np.pi * k / K
        m = np.zeros(dim)
        m[0] = inter_dist * np.cos(angle)
        m[1] = inter_dist * np.sin(angle)
        means.append(m)
        covs.append(sigma ** 2 * np.eye(dim))
        weights.append(1.0 / K)
    return means, covs, weights


def make_items(N_items, K, means, sigma, dim, rng):
    parts = []
    for k in range(K):
        n_k = N_items // K if k < K - 1 else N_items - (N_items // K) * (K - 1)
        parts.append(rng.multivariate_normal(means[k], sigma ** 2 * np.eye(dim), n_k))
    return np.vstack(parts)


def make_true_pref(users, items, tau=2.0):
    return 1.0 / (1.0 + np.exp(-(users @ items.T) / tau))


class _PairBCEDataset(Dataset):
    def __init__(self, users_emb, items_emb, target_matrix, subsample, rng):
        n_u, n_i = target_matrix.shape
        all_pairs = np.array([(u, i) for u in range(n_u) for i in range(n_i)])
        if subsample is not None and subsample < len(all_pairs):
            idx = rng.choice(len(all_pairs), subsample, replace=False)
            all_pairs = all_pairs[idx]
        self.pairs = all_pairs.astype(np.int64)
        self.targets = target_matrix[self.pairs[:, 0], self.pairs[:, 1]].astype(np.float32)
        self.users_emb = users_emb.astype(np.float32)
        self.items_emb = items_emb.astype(np.float32)

    def __len__(self):
        return len(self.pairs)

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
            loss.backward()
            opt.step()


# ----------------------------------------------------------------------------- #
# Подкласс окружения: стэшит показанные айтемы + метки на момент рекомендации.
# Сигнатуры sim/ не трогаем — только переопределяем _make_recommendations.
# ----------------------------------------------------------------------------- #
class RecordingEnv(SimulationEnvironment):
    def _make_recommendations(self, U, I, n_users, n_items):
        recs = super()._make_recommendations(U, I, n_users, n_items)
        # метки кластеров выровнены с U (pre-drift, pre-churn) на этот момент
        self._last_recs = recs
        self._last_labels = np.asarray(self.user_gen.cluster_labels).copy()
        self._last_items = I
        return recs


def build_env(mode, seed, adherence=ADHERENCE, sigma_k=SIGMA_K, seen_filter=True):
    """Строит RecordingEnv с warm-start MLP. closed_loop и static при общем seed
    делят идентичные стартовые популяцию/айтемы/true_pref (парный разрыв)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    means, covs, weights = make_gmm_params(K_GMM, EMB_DIM, INTER_DIST, sigma_k)
    gen = GMMUserGenerator(component_means=means, component_covs=covs,
                           component_weights=weights,
                           replacement_rate=REPLACE, memory_effect=6)
    np.random.seed(seed)
    user_emb, _ = gen.initialize(N_USERS)
    item_emb = make_items(N_ITEMS, K_GMM, means, sigma_k, EMB_DIM, rng)
    true_pref = make_true_pref(user_emb, item_emb)
    matrix = np.full((N_USERS, N_ITEMS), np.nan)
    dataset = ExperimentDataset(user_emb.copy(), item_emb.copy(), matrix)

    device = torch.device('cpu')
    model = RecModel(EMB_DIM, EMB_DIM, hidden_size=64).to(device)
    pretrain_model(model, user_emb, item_emb, true_pref, rng, device)  # warm-start

    policy = ServingPolicy('top_k')
    alpha_c = 0.0 if mode == 'no_influence' else adherence
    click = ClickModel(adherence=alpha_c, usage_rate=0.8, noise_level=0.05)
    env = RecordingEnv(
        dataset=dataset, rec_model=model, user_generator=gen, click_model=click,
        serving_policy=policy, mode=mode, true_preference_matrix=true_pref,
        retrain_period=T_RET, K=K_REC, device=device, seen_filter=seen_filter,
        user_drift_beta=USER_DRIFT, drift_alpha=0.0)  # drift_alpha — мёртвый, =0
    return env


# ----------------------------------------------------------------------------- #
# Ковариация показанного Σ_ξ по кластерам
# ----------------------------------------------------------------------------- #
def display_cov_per_cluster(recs, labels, items, K):
    """Возвращает dict k -> Σ_ξ (d,d) — ковариация показанных top-K векторов
    среди пользователей кластера k (на момент рекомендации)."""
    out = {}
    for k in range(K):
        users_k = [u for u in recs if labels[u] == k]
        if len(users_k) < 2:
            continue
        V = np.vstack([items[recs[u]] for u in users_k])   # (sum_k * K_REC, d)
        if V.shape[0] < 2:
            continue
        out[k] = np.atleast_2d(np.cov(V, rowvar=False))
    return out


def cov_scalars(Sig, eps=1e-9):
    d = Sig.shape[0]
    Sig = Sig + eps * np.eye(d)
    tr = float(np.trace(Sig))
    ev = np.linalg.eigvalsh(Sig)
    s, ld = np.linalg.slogdet(Sig)
    logdet = float(ld) if s > 0 else -200.0
    return tr, logdet, float(ev[-1]), float(ev[0])


def dispersion_div(Sig_T, Sig_0, eps=1e-9):
    """𝒟(Σ_T ‖ Σ_0) = 0.5(tr(Σ0^-1 Σ_T) - d + logdet Σ0 - logdet Σ_T) — §6.7."""
    d = Sig_0.shape[0]
    Sig_0 = Sig_0 + eps * np.eye(d)
    Sig_T = Sig_T + eps * np.eye(d)
    inv0 = np.linalg.pinv(Sig_0)
    s0, ld0 = np.linalg.slogdet(Sig_0)
    sT, ldT = np.linalg.slogdet(Sig_T)
    if s0 <= 0 or sT <= 0:
        return np.nan
    return 0.5 * float(np.trace(inv0 @ Sig_T) - d + ld0 - ldT)


def loewner_psd_frac(Sig_stat, Sig_closed, rtol=1e-6):
    """1 если Σ_stat - Σ_closed ⪰ 0 (с относит. допуском), иначе 0; плюс min-eig."""
    D = Sig_stat - Sig_closed
    ev = np.linalg.eigvalsh(0.5 * (D + D.T))
    scale = max(np.trace(Sig_stat), np.trace(Sig_closed), 1e-9) / Sig_stat.shape[0]
    psd = float(ev[0] >= -rtol * scale)
    return psd, float(ev[0])


def bootstrap_gap(cl, st, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    cl = np.asarray(cl, dtype=float)
    st = np.asarray(st, dtype=float)
    cl = cl[~np.isnan(cl)]
    st = st[~np.isnan(st)]
    if len(cl) == 0 or len(st) == 0:
        return np.nan, np.nan, np.nan
    idx_cl = rng.integers(0, len(cl), size=(n_boot, len(cl)))
    idx_st = rng.integers(0, len(st), size=(n_boot, len(st)))
    diffs = cl[idx_cl].mean(1) - st[idx_st].mean(1)
    return float(diffs.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


# ----------------------------------------------------------------------------- #
# Один прогон: возвращает по-шаговые скаляры Σ_ξ (среднее по кластерам) и
# матрицы Σ_ξ по кластерам на t=0 и t=T-1 (для Loewner-проверки и 𝒟).
# ----------------------------------------------------------------------------- #
def run_one(mode, seed, adherence=ADHERENCE, sigma_k=SIGMA_K, seen_filter=True, T_run=None):
    T_run = T_run or T
    env = build_env(mode, seed, adherence, sigma_k, seen_filter)
    per_step = []        # (t, mean_tr, mean_logdet, mean_topeig)
    mats0 = None
    matsT = None
    for t in range(T_run):
        env.step(t)
        covs = display_cov_per_cluster(env._last_recs, env._last_labels,
                                       env._last_items, K_GMM)
        if covs:
            trs, lds, tops = [], [], []
            for Sig in covs.values():
                tr, ld, top, _ = cov_scalars(Sig)
                trs.append(tr); lds.append(ld); tops.append(top)
            per_step.append((t, np.mean(trs), np.mean(lds), np.mean(tops)))
        if t == 0:
            mats0 = covs
        if t == T_run - 1:
            matsT = covs
    return per_step, mats0, matsT


# ----------------------------------------------------------------------------- #
# Part A: closed vs static, ±seen_filter
# ----------------------------------------------------------------------------- #
def part_A():
    print('\n=== Part A: display-covariance sharpening (closed vs static) ===')
    raw = []          # по-шаговые средние
    endpoint = []     # t=T-1 по (seed, seen_filter): tr/logdet по режимам + Loewner + 𝒟
    for seen_filter in (True, False):
        for seed in range(N_SEEDS_A):
            res = {}
            for mode in ('closed_loop', 'static'):
                per_step, m0, mT = run_one(mode, seed, seen_filter=seen_filter)
                res[mode] = (per_step, m0, mT)
                for (t, tr, ld, top) in per_step:
                    raw.append(dict(part='A', seen_filter=seen_filter, seed=seed,
                                    mode=mode, t=t, tr_xi=tr, logdet_xi=ld, topeig_xi=top))
            # endpoint per-cluster Loewner + 𝒟, paired closed vs static
            _, m0_cl, mT_cl = res['closed_loop']
            _, m0_st, mT_st = res['static']
            for k in range(K_GMM):
                if k in mT_cl and k in mT_st:
                    tr_cl = cov_scalars(mT_cl[k])[0]
                    tr_st = cov_scalars(mT_st[k])[0]
                    psd, mineig = loewner_psd_frac(mT_st[k], mT_cl[k])
                    disp_cl = dispersion_div(mT_cl[k], m0_cl[k]) if (m0_cl and k in m0_cl) else np.nan
                    disp_st = dispersion_div(mT_st[k], m0_st[k]) if (m0_st and k in m0_st) else np.nan
                    endpoint.append(dict(seen_filter=seen_filter, seed=seed, cluster=k,
                                         tr_closed=tr_cl, tr_static=tr_st,
                                         tr_gap=tr_cl - tr_st,
                                         loewner_psd=psd, min_eig_diff=mineig,
                                         disp_closed=disp_cl, disp_static=disp_st,
                                         disp_gap=disp_cl - disp_st))
        print(f'  seen_filter={seen_filter}: done {N_SEEDS_A} seeds')
    return pd.DataFrame(raw), pd.DataFrame(endpoint)


# ----------------------------------------------------------------------------- #
# Part B: α-sweep — разрыв сужения растёт с adherence (§6.5)
# ----------------------------------------------------------------------------- #
def part_B():
    print('\n=== Part B: α-sweep (self-reinforcement, §6.5) ===')
    rows = []
    for alpha in (0.0, 0.3, 0.7, 0.9):
        for seed in range(N_SEEDS_B):
            trs = {}
            for mode in ('closed_loop', 'static'):
                _, _, mT = run_one(mode, seed, adherence=alpha, seen_filter=False)
                vals = [cov_scalars(mT[k])[0] for k in mT] if mT else []
                trs[mode] = float(np.mean(vals)) if vals else np.nan
            rows.append(dict(part='B', alpha=alpha, seed=seed,
                             tr_closed=trs['closed_loop'], tr_static=trs['static'],
                             tr_gap=trs['closed_loop'] - trs['static']))
        print(f'  alpha={alpha}: done {N_SEEDS_B} seeds')
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- #
# Part C: σ-sweep — зависимость сужения от ширины кластера (§6.6)
# ----------------------------------------------------------------------------- #
def part_C():
    print('\n=== Part C: σ-sweep (cluster width / threshold, §6.6) ===')
    rows = []
    for sig in (0.3, 0.8, 2.0):
        for seed in range(N_SEEDS_C):
            trs = {}
            for mode in ('closed_loop', 'static'):
                _, _, mT = run_one(mode, seed, sigma_k=sig, seen_filter=False)
                vals = [cov_scalars(mT[k])[0] for k in mT] if mT else []
                trs[mode] = float(np.mean(vals)) if vals else np.nan
            rows.append(dict(part='C', sigma_k=sig, seed=seed,
                             tr_closed=trs['closed_loop'], tr_static=trs['static'],
                             tr_gap=trs['closed_loop'] - trs['static']))
        print(f'  sigma_k={sig}: done {N_SEEDS_C} seeds')
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- #
# Анализ + печать сводки
# ----------------------------------------------------------------------------- #
def summarize(rawA, endA, rawB, rawC):
    summary = []

    # Part A: endpoint gaps per seen_filter
    print('\n----- SUMMARY: Part A (closed - static, t=T) -----')
    for sf in (True, False):
        sub = endA[endA['seen_filter'] == sf]
        if len(sub) == 0:
            continue
        m, lo, hi = bootstrap_gap(sub['tr_closed'], sub['tr_static'])
        loew = float(sub['loewner_psd'].mean())
        dgap_m, dgap_lo, dgap_hi = bootstrap_gap(sub['disp_closed'], sub['disp_static'])
        tr_cl = float(sub['tr_closed'].mean()); tr_st = float(sub['tr_static'].mean())
        sig = 'SIGNIF' if not (lo <= 0 <= hi) else 'n.s.'
        sigd = 'SIGNIF' if not (dgap_lo <= 0 <= dgap_hi) else 'n.s.'
        print(f'  seen_filter={sf}:  tr Σξ closed={tr_cl:.4f} static={tr_st:.4f}  '
              f'gap={m:+.4f} [{lo:+.4f};{hi:+.4f}] {sig}')
        print(f'                    Loewner(Σξ^stat ⪰ Σξ^closed) holds in {loew*100:.0f}% (seed,cluster)')
        print(f'                    dispersion 𝒟 gap closed-static={dgap_m:+.4f} '
              f'[{dgap_lo:+.4f};{dgap_hi:+.4f}] {sigd}')
        summary.append(dict(part='A', axis='seen_filter', level=sf,
                            tr_closed=tr_cl, tr_static=tr_st,
                            tr_gap=m, tr_gap_lo=lo, tr_gap_hi=hi, tr_gap_sig=(sig == 'SIGNIF'),
                            loewner_frac=loew,
                            disp_gap=dgap_m, disp_gap_lo=dgap_lo, disp_gap_hi=dgap_hi,
                            disp_gap_sig=(sigd == 'SIGNIF')))

    # Part B: α-sweep
    print('\n----- SUMMARY: Part B (tr Σξ gap closed-static by α) -----')
    for alpha in sorted(rawB['alpha'].unique()):
        sub = rawB[rawB['alpha'] == alpha]
        m, lo, hi = bootstrap_gap(sub['tr_closed'], sub['tr_static'])
        sig = 'SIGNIF' if not (lo <= 0 <= hi) else 'n.s.'
        print(f'  α={alpha}:  gap={m:+.4f} [{lo:+.4f};{hi:+.4f}] {sig}')
        summary.append(dict(part='B', axis='alpha', level=alpha,
                            tr_gap=m, tr_gap_lo=lo, tr_gap_hi=hi, tr_gap_sig=(sig == 'SIGNIF')))

    # Part C: σ-sweep
    print('\n----- SUMMARY: Part C (tr Σξ gap closed-static by σ) -----')
    for sig_k in sorted(rawC['sigma_k'].unique()):
        sub = rawC[rawC['sigma_k'] == sig_k]
        m, lo, hi = bootstrap_gap(sub['tr_closed'], sub['tr_static'])
        sg = 'SIGNIF' if not (lo <= 0 <= hi) else 'n.s.'
        print(f'  σ={sig_k}:  gap={m:+.4f} [{lo:+.4f};{hi:+.4f}] {sg}')
        summary.append(dict(part='C', axis='sigma_k', level=sig_k,
                            tr_gap=m, tr_gap_lo=lo, tr_gap_hi=hi, tr_gap_sig=(sg == 'SIGNIF')))

    return pd.DataFrame(summary)


def make_figures(rawA, rawB, rawC):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Fig 1: trajectory of tr Σξ, closed vs static, seen_filter=False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sf in zip(axes, (False, True)):
        sub = rawA[rawA['seen_filter'] == sf]
        for mode, color in (('closed_loop', 'C3'), ('static', 'C0')):
            g = sub[sub['mode'] == mode].groupby('t')['tr_xi']
            mean = g.mean()
            sd = g.std()
            ax.plot(mean.index, mean.values, color=color, label=mode)
            ax.fill_between(mean.index, mean - sd, mean + sd, color=color, alpha=0.15)
        ax.set_title(f'tr Σξ (показанное) — seen_filter={sf}')
        ax.set_xlabel('шаг t')
        ax.set_ylabel('tr Σξ (средн. по кластерам)')
        ax.legend()
    fig.suptitle('Заострение выдачи переобучением: ковариация показанного Σξ, closed vs static')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'DIAG_display_cov_trajectory.pdf', bbox_inches='tight')
    plt.close(fig)

    # Fig 2: gap by α and by σ
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alpha in sorted(rawB['alpha'].unique()):
        sub = rawB[rawB['alpha'] == alpha]
        axes[0].bar(str(alpha), sub['tr_gap'].mean(),
                    yerr=sub['tr_gap'].std() / max(1, np.sqrt(len(sub))), color='C2')
    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].set_title('Part B: разрыв tr Σξ (closed−static) по α')
    axes[0].set_xlabel('adherence α'); axes[0].set_ylabel('tr Σξ gap')
    for sig_k in sorted(rawC['sigma_k'].unique()):
        sub = rawC[rawC['sigma_k'] == sig_k]
        axes[1].bar(str(sig_k), sub['tr_gap'].mean(),
                    yerr=sub['tr_gap'].std() / max(1, np.sqrt(len(sub))), color='C4')
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_title('Part C: разрыв tr Σξ (closed−static) по σ')
    axes[1].set_xlabel('ширина кластера σ'); axes[1].set_ylabel('tr Σξ gap')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'DIAG_display_cov_gaps.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigures -> {FIG_DIR}/DIAG_display_cov_trajectory.pdf, DIAG_display_cov_gaps.pdf')


if __name__ == '__main__':
    t0 = time.time()
    print(f'[display-covariance] SMOKE={SMOKE}  T={T}  '
          f'seeds A/B/C={N_SEEDS_A}/{N_SEEDS_B}/{N_SEEDS_C}')
    rawA, endA = part_A()
    rawB = part_B()
    rawC = part_C()

    rawA.to_csv(RESULTS_DIR / 'display_cov_trajectory.csv', index=False)
    endA.to_csv(RESULTS_DIR / 'display_cov_endpoint.csv', index=False)
    rawB.to_csv(RESULTS_DIR / 'display_cov_alpha_sweep.csv', index=False)
    rawC.to_csv(RESULTS_DIR / 'display_cov_sigma_sweep.csv', index=False)

    summary = summarize(rawA, endA, rawB, rawC)
    summary.to_csv(RESULTS_DIR / 'display_cov_summary.csv', index=False)

    if not SMOKE:
        make_figures(rawA, rawB, rawC)

    print(f'\nCSV -> {RESULTS_DIR}')
    print(f'Done in {time.time() - t0:.1f}s')
