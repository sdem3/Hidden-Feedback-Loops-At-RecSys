# theorem: Разрыв offline/true-риска ограничен KL (Пинскер) — проверка гипотезы
"""
exp_2026-06-15b_kl_risk_gap.py
==============================
Эмпирическая проверка гипотезы о связи offline/real-world-разрыва с дивергенцией
между наблюдаемым и истинным распределениями (Пинскер).

ТЕОРИЯ (проверяемая). Истинное и наблюдаемое распределения различаются только
экспозицией: P^true берёт целевую релевантность ρ_t(v|u), P^obs — политику показа
π_t(v|u;θ_t). При ограниченной потере ℓ∈[0,L]:
    |R^true_t(θ) − R^obs_t(θ)| ≤ L·TV(P^obs,P^true) ≤ L·sqrt(2·KL(P^obs‖P^true)).
Гипотеза: в замкнутом контуре π_t концентрируется ⇒ KL(P^obs‖P^true) РАСТЁТ ⇒
offline-метрика становится всё менее надёжным прокси истинного качества.

ВАЖНО (отличие от текста ВКР). Здесь дивергенция — между ЭКСПОЗИЦИЕЙ и ЦЕЛЕВОЙ
релевантностью на шаге t, а НЕ KL(P_t‖P_0) (дрейф вкусов от старта), который уже
есть в работе. Это разные объекты.
НАПРАВЛЕНИЕ KL: берём KL(P^obs‖P^true) (конечно, т.к. ρ имеет полный носитель),
а не KL(P^true‖P^obs) (расходится при концентрированной π). Пинскер симметричен
по TV, граница в силе.

ОПЕРАЦИОНАЛИЗАЦИЯ (faithful к §3 ВКР). «Наблюдаемое vs истинное» в работе —
это ЗАГРЯЗНЕНИЕ ОТКЛИКА η_t=(1−α)θ+α f_t (eq:contaminated), а не выбор экспозиции.
Поэтому P^obs и P^true различаются откликом: y~Bern(η) против y~Bern(θ) на
показанных парах. Экспозиция π_t(v|u)=softmax(logit f_θt) — веса показанных пар.

ЧТО МЕРИМ на каждом шаге t, по режимам {closed_loop, static, no_influence}:
  η(u,v)   = (1−α)θ(u,v)+α f_θt(u,v),  α=adherence (0 для no_influence);
  KL_t     = E_{(u,v)~π} KL( Bern(η(u,v)) ‖ Bern(θ(u,v)) )   — загрязнение отклика;
  R^obs_t  = E_{(u,v)~π} η   — наблюдаемый CTR показанного (на загрязнённых логах);
  R^true_t = E_{(u,v)~π} θ   — истинная релевантность показанного;
  gap_t    = R^obs_t − R^true_t = E_π[α(f−θ)]   — завышение метрики (Утверждение A);
  bound_t  = L·sqrt(2·KL_t),  L=1               — граница Пинскера (Следствие 1).
Проверки: (i) растёт ли KL_t в петле (closed) vs заморозка/контроль;
(ii) растёт ли gap_t; (iii) держится ли bound_t ≥ |gap_t| всюду;
(iv) коррелируют ли gap_t и sqrt(KL_t) (содержательность границы).

Запуск:
  python exp_2026-06-15b_kl_risk_gap.py --smoke
  python exp_2026-06-15b_kl_risk_gap.py
"""
import sys
import time
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'dispcov', HERE / 'exp_2026-06-12_display_covariance.py')
m = importlib.util.module_from_spec(spec)
_saved = sys.argv
sys.argv = [sys.argv[0]] + (['--smoke'] if '--smoke' in sys.argv else [])
spec.loader.exec_module(m)
sys.argv = _saved

import numpy as np
import pandas as pd
import torch

SMOKE = '--smoke' in sys.argv
N_SEEDS = 2 if SMOKE else 8
T_run = 30 if SMOKE else 100
TAU = 2.0           # температура истинного отклика θ=σ(<u,v>/τ)
EPS = 1e-12


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def model_prob_matrix(model, U, I, device):
    """f[u,v] = P(click) модели по всем парам, один батчевый прогон."""
    model.eval()
    n_u, n_i = U.shape[0], I.shape[0]
    Ut = torch.tensor(U, dtype=torch.float32, device=device)
    It = torch.tensor(I, dtype=torch.float32, device=device)
    with torch.no_grad():
        u_rep = Ut.repeat_interleave(n_i, dim=0)      # (n_u*n_i, d)
        i_rep = It.repeat(n_u, 1)
        p = model(u_rep, i_rep).cpu().numpy().reshape(n_u, n_i)
    return p


def _kl_bernoulli(eta, theta):
    """KL(Bern(η)‖Bern(θ)) поэлементно, с клипом."""
    e = np.clip(eta, EPS, 1 - EPS)
    th = np.clip(theta, EPS, 1 - EPS)
    return e * np.log(e / th) + (1 - e) * np.log((1 - e) / (1 - th))


def step_metrics(model, U, I, device, alpha, tau=TAU):
    """Загрязнение отклика η=(1−α)θ+αf; KL(Bern(η)‖Bern(θ)) и завышение CTR,
    взвешенные по экспозиции π=softmax(logit f). Возвращает (KL,TV,R_obs,R_true,gap,bound)."""
    f = model_prob_matrix(model, U, I, device)             # (n_u,n_i) вероятности f_θt
    true_score = (U @ I.T) / tau                           # logit(θ)=<u,v>/τ
    theta = 1.0 / (1.0 + np.exp(-np.clip(true_score, -60, 60)))
    eta = (1 - alpha) * theta + alpha * f                  # загрязнённый отклик (§3)
    pi = _softmax_rows(_logit(f))                          # экспозиция π(v|u)

    kl_pair = _kl_bernoulli(eta, theta)                    # KL(Bern(η)‖Bern(θ))
    kl_u = (pi * kl_pair).sum(axis=1)                      # взвеш. по экспозиции
    tv_u = (pi * np.abs(eta - theta)).sum(axis=1)          # TV(Bern(η),Bern(θ))=|η−θ|
    r_obs = (pi * eta).sum(axis=1)                         # наблюдаемый CTR показанного
    r_true = (pi * theta).sum(axis=1)                      # истинная релевантность

    KL = float(np.mean(kl_u))
    TV = float(np.mean(tv_u))
    R_obs = float(np.mean(r_obs))
    R_true = float(np.mean(r_true))
    gap = R_obs - R_true                                  # завышение метрики (Утв. A)
    bound = 1.0 * np.sqrt(2.0 * max(KL, 0.0))             # L=1, Следствие 1
    return KL, TV, R_obs, R_true, gap, bound


def run():
    device = torch.device('cpu')
    rows = []
    for mode in ('closed_loop', 'static', 'no_influence'):
        for seed in range(N_SEEDS):
            env = m.build_env(mode, seed, seen_filter=True)
            alpha = float(getattr(env.click_model, 'adherence', 0.7))
            for t in range(T_run):
                env.step(t)
                U = env.dataset.users_embeddings
                I = env.dataset.items_embeddings
                KL, TV, R_obs, R_true, gap, bound = step_metrics(env.rec_model, U, I, device, alpha)
                rows.append(dict(mode=mode, seed=seed, t=t,
                                 KL_obs_true=KL, TV=TV,
                                 R_obs=R_obs, R_true=R_true,
                                 gap=gap, abs_gap=abs(gap), pinsker_bound=bound,
                                 bound_holds=int(abs(gap) <= bound + 1e-9)))
            print(f'  {mode} seed {seed}: '
                  f'KL {rows[-1]["KL_obs_true"]:.3f}  |gap| {rows[-1]["abs_gap"]:.4f}  '
                  f'bound {rows[-1]["pinsker_bound"]:.3f}')
    return pd.DataFrame(rows)


def summarize(df):
    print('\n========== KL(P^obs‖P^true), РАЗРЫВ РИСКОВ И ГРАНИЦА ПИНСКЕРА ==========')
    print(f'  Граница |gap| ≤ L·sqrt(2·KL) держится: '
          f'{100.0 * df["bound_holds"].mean():.1f}% строк (должно быть 100%)')

    print('\n  --- рост во времени (среднее по seeds): t=0 → t=T ---')
    for mode in ('closed_loop', 'static', 'no_influence'):
        sub = df[df['mode'] == mode]
        g = sub.groupby('t')
        kl0 = g['KL_obs_true'].mean().iloc[0]
        klT = g['KL_obs_true'].mean().iloc[-1]
        ag0 = g['abs_gap'].mean().iloc[0]
        agT = g['abs_gap'].mean().iloc[-1]
        ro = g['R_obs'].mean(); rt = g['R_true'].mean()
        print(f'  {mode:13s}: KL {kl0:.3f}→{klT:.3f} (×{klT/max(kl0,1e-9):.2f})   '
              f'|gap| {ag0:.4f}→{agT:.4f}   '
              f'R_obs {ro.iloc[-1]:.3f} R_true {rt.iloc[-1]:.3f}')

    # эндпойнт closed − no_influence (петля vs контроль без влияния), парно по seed
    print('\n  --- эндпойнт t=T: разрыв closed_loop − no_influence (по KL и |gap|) ---')

    def gap_ci(a_mode, b_mode, col, n_boot=5000):
        rng = np.random.default_rng(0)
        A = df[(df['mode'] == a_mode) & (df['t'] == T_run - 1)].sort_values('seed')[col].values
        B = df[(df['mode'] == b_mode) & (df['t'] == T_run - 1)].sort_values('seed')[col].values
        n = min(len(A), len(B))
        d = A[:n] - B[:n]
        bs = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
        return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))

    for col, name in [('KL_obs_true', 'KL(obs‖true)'), ('abs_gap', '|risk gap|')]:
        for a, b in [('closed_loop', 'no_influence'), ('closed_loop', 'static')]:
            mn, lo, hi = gap_ci(a, b, col)
            sig = 'SIGNIF' if not (lo <= 0 <= hi) else 'n.s.'
            print(f'    {name:14s} {a}−{b}: {mn:+.4f} [{lo:+.4f};{hi:+.4f}] {sig}')

    # корреляция |gap| ~ sqrt(KL) (содержательный смысл границы)
    sk = np.sqrt(np.maximum(df['KL_obs_true'].values, 0))
    ag = df['abs_gap'].values
    if sk.std() > 0 and ag.std() > 0:
        r = float(np.corrcoef(sk, ag)[0, 1])
        print(f'\n  corr(|gap|, sqrt(KL)) = {r:+.3f}  (граница содержательна, если >0)')


def make_figure(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'closed_loop': 'C3', 'static': 'C0', 'no_influence': 'C2'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mode in ('closed_loop', 'static', 'no_influence'):
        sub = df[df['mode'] == mode]
        gk = sub.groupby('t')['KL_obs_true']; ga = sub.groupby('t')['abs_gap']
        gb = sub.groupby('t')['pinsker_bound']
        axes[0].plot(gk.mean().index, gk.mean().values, color=colors[mode], label=mode)
        axes[1].plot(ga.mean().index, ga.mean().values, color=colors[mode], label=mode)
        axes[2].plot(ga.mean().index, ga.mean().values, color=colors[mode], lw=2, label=f'{mode}: |gap|')
        axes[2].plot(gb.mean().index, gb.mean().values, color=colors[mode], ls='--', alpha=.6)
    axes[0].set_title('KL(P^obs‖P^true)'); axes[0].set_xlabel('шаг t'); axes[0].set_ylabel('KL')
    axes[1].set_title('|R^true − R^obs| (разрыв рисков)'); axes[1].set_xlabel('шаг t')
    axes[2].set_title('разрыв (—) и граница Пинскера (- -)'); axes[2].set_xlabel('шаг t')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle('Связь KL(обс‖ист) с разрывом offline/true-риска (граница Пинскера)')
    fig.tight_layout()
    out = m.FIG_DIR / 'DIAG_kl_risk_gap.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure -> {out}')


if __name__ == '__main__':
    t0 = time.time()
    print(f'[kl-risk-gap] SMOKE={SMOKE}  N_SEEDS={N_SEEDS}  T={T_run}')
    df = run()
    out = m.RESULTS_DIR / 'kl_risk_gap.csv'
    df.to_csv(out, index=False)
    summarize(df)
    if not SMOKE:
        make_figure(df)
    print(f'\nCSV -> {out}\nDone in {time.time()-t0:.1f}s')
