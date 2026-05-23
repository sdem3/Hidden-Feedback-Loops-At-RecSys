"""
Анализ результатов из out/*.csv и построение финальных диагностических графиков.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[3] / 'paper' / 'figures' / 'diagnostic'
OUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'diagnostic'

A = pd.read_csv(OUT_DIR / 'A_baseline.csv')
B = pd.read_csv(OUT_DIR / 'B_beta_sweep.csv')
C = pd.read_csv(OUT_DIR / 'C_alpha_sweep_static.csv')
D = pd.read_csv(OUT_DIR / 'D_factorial.csv')

COLORS = {'closed_loop': '#d62728', 'static': '#ff7f0e',
          'fresh_oracle': '#2ca02c', 'no_influence': '#1f77b4'}


# ---------------------------------------------------------------------------
# Plot 1: главный диагностический рисунок
# ---------------------------------------------------------------------------

def plot_main_diagnostic():
    metrics = [
        ('trace_sigma',        r'$\mathrm{tr}(\hat{\Sigma}_t^u)$',                'log'),
        ('kl_from_initial',    r'$KL(P_t\|P_0)$',                                  'linear'),
        ('gini_exposure',      'Gini показов рекомендаций',                       'linear'),
        ('exposure_entropy',   r'$H(\mathrm{exposure})$, нат',                    'linear'),
        ('catalog_coverage',   'Catalog coverage',                                'linear'),
        ('intra_list_diversity', r'intra-list diversity (rec.)',                  'linear'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (metric, ylab, scale) in zip(axes.flat, metrics):
        for mode in ['closed_loop', 'static', 'fresh_oracle', 'no_influence']:
            d = A[A['mode'] == mode]
            g = d.groupby('t')[metric].agg(['mean', 'std']).reset_index()
            ax.plot(g.t, g['mean'], label=mode, color=COLORS[mode], lw=2.0)
            ax.fill_between(g.t, g['mean']-g['std'], g['mean']+g['std'],
                            color=COLORS[mode], alpha=0.18)
        ax.set_xlabel('t', fontsize=11); ax.set_ylabel(ylab, fontsize=11)
        if scale == 'log' and (g['mean'].min() > 0):
            ax.set_yscale('log')
        ax.grid(True, ls='--', alpha=0.4)
    axes[0,0].legend(fontsize=9, loc='upper right')
    fig.suptitle(
        'H1+H3 диагностика: 6 метрик во всех четырёх режимах '
        '(N=300, T=100, 8 seeds, α=0.7, β=0.005)', y=1.01, fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'DIAG_main_diagnostic.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved DIAG_main_diagnostic.pdf')


# ---------------------------------------------------------------------------
# Plot 2: β-sweep (ключевой график)
# ---------------------------------------------------------------------------

def plot_beta_sweep():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for mode in ['closed_loop', 'static', 'fresh_oracle', 'no_influence']:
        d = B[B['mode'] == mode]
        g = d.groupby('beta').agg(
            ts_m=('trace_sigma_T', 'mean'), ts_s=('trace_sigma_T', 'std'),
            kl_m=('kl_T', 'mean'),         kl_s=('kl_T', 'std'),
            gi_m=('gini_T', 'mean'),       gi_s=('gini_T', 'std'),
        ).reset_index()
        # Сдвинем 0 в небольшое значение для логарифмической оси, либо нарисуем как отдельная точка
        betas = g['beta'].values
        axes[0].errorbar(np.where(betas == 0, 1e-4, betas), g['ts_m'], yerr=g['ts_s'],
                          label=mode, color=COLORS[mode], marker='o', capsize=3, lw=1.8)
        axes[1].errorbar(np.where(betas == 0, 1e-4, betas), g['kl_m'], yerr=g['kl_s'],
                          label=mode, color=COLORS[mode], marker='o', capsize=3, lw=1.8)
        axes[2].errorbar(np.where(betas == 0, 1e-4, betas), g['gi_m'], yerr=g['gi_s'],
                          label=mode, color=COLORS[mode], marker='o', capsize=3, lw=1.8)

    for ax, ylab in zip(axes,
                        [r'$\mathrm{tr}(\hat{\Sigma}_T)$',
                         r'$KL(P_T\|P_0)$',
                         'Gini показов (T)']):
        ax.set_xscale('log')
        ax.set_xlabel(r'$\beta$ (user drift; $10^{-4} \equiv 0$)')
        ax.set_ylabel(ylab)
        ax.grid(True, ls='--', alpha=0.4)
    axes[0].legend(fontsize=9, loc='best')
    fig.suptitle(r'Эксперимент B: эффект $\beta$ во всех режимах (5 seeds × 5 $\beta$)',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'DIAG_B_beta_sweep_v2.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved DIAG_B_beta_sweep_v2.pdf')


# ---------------------------------------------------------------------------
# Plot 3: α-sweep в static (без петли) — ключевой контрольный график
# ---------------------------------------------------------------------------

def plot_alpha_sweep_static():
    g = C.groupby('alpha').agg(
        ts_m=('trace_sigma_T', 'mean'), ts_s=('trace_sigma_T', 'std'),
        kl_m=('kl_T', 'mean'),         kl_s=('kl_T', 'std'),
        gi_m=('gini_T', 'mean'),       gi_s=('gini_T', 'std'),
    ).reset_index()
    # initial trace ≈ 28.6 для нормировки
    tr0 = A[A['t'] == 0].groupby('mode')['trace_sigma'].mean().mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    a = g['alpha']
    axes[0].errorbar(a, g['ts_m'], yerr=g['ts_s'],
                      marker='o', capsize=3, color='#ff7f0e', lw=2)
    axes[0].axhline(tr0, ls=':', color='gray', label=r'$\mathrm{tr}(\hat\Sigma_0) \approx 28.6$')
    axes[0].set_xlabel(r'$\alpha$ (adherence к рекомендации)', fontsize=11)
    axes[0].set_ylabel(r'$\mathrm{tr}(\hat\Sigma_T)$, mode=static', fontsize=11)
    axes[0].legend(fontsize=9)

    axes[1].errorbar(a, 100*(1-g['ts_m']/tr0),
                      yerr=100*g['ts_s']/tr0,
                      marker='o', capsize=3, color='#d62728', lw=2)
    axes[1].set_xlabel(r'$\alpha$')
    axes[1].set_ylabel('drop tr(Σ), %')
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.grid(True, ls='--', alpha=0.4)
    fig.suptitle(r'Эксперимент C: в режиме static (без переобучения) коллапс'
                 r' зависит ТОЛЬКО от $\alpha$', y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'DIAG_C_alpha_sweep_static_v2.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved DIAG_C_alpha_sweep_static_v2.pdf')


# ---------------------------------------------------------------------------
# Plot 4: factorial heatmap
# ---------------------------------------------------------------------------

def plot_factorial():
    modes = ['closed_loop', 'static', 'fresh_oracle']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mode in zip(axes, modes):
        d = D[D['mode'] == mode]
        pivot = d.groupby(['alpha', 'beta'])['drop_pct'].mean().unstack()
        im = ax.imshow(pivot.values, origin='lower', aspect='auto', cmap='RdYlBu_r',
                       vmin=-10, vmax=100)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{b:.3f}' for b in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{a:.1f}' for a in pivot.index])
        ax.set_xlabel(r'$\beta$'); ax.set_ylabel(r'$\alpha$')
        ax.set_title(mode)
        # annotate
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f'{pivot.values[i,j]:.0f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(pivot.values[i,j]) > 50 else 'black')
        plt.colorbar(im, ax=ax, label='drop tr(Σ), %')
    fig.suptitle(r'Эксперимент D: drop tr$(\hat\Sigma)$ в зависимости от $(\alpha, \beta)$ — '
                 r'влияние режима пренебрежимо мало', y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'DIAG_D_factorial_v2.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved DIAG_D_factorial_v2.pdf')


# ---------------------------------------------------------------------------
# Plot 5: Сводный рисунок-объяснение
# ---------------------------------------------------------------------------

def plot_explainer():
    """Один большой рисунок: показывает что коллапс обусловлен β а не петлёй."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.32)

    # (1) trace_sigma over time
    ax = fig.add_subplot(gs[0, 0])
    for mode in ['closed_loop', 'static', 'fresh_oracle', 'no_influence']:
        d = A[A['mode'] == mode]
        g = d.groupby('t')['trace_sigma'].agg(['mean', 'std']).reset_index()
        ax.plot(g.t, g['mean'], label=mode, color=COLORS[mode], lw=2)
        ax.fill_between(g.t, g['mean']-g['std'], g['mean']+g['std'],
                        color=COLORS[mode], alpha=0.18)
    ax.set_yscale('log')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\mathrm{tr}(\hat\Sigma_t^u)$')
    ax.set_title(r'(a) Baseline: все режимы с $\alpha>0$ дают одинаковый коллапс')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.4)

    # (2) gini exposure over time
    ax = fig.add_subplot(gs[0, 1])
    for mode in ['closed_loop', 'static', 'fresh_oracle', 'no_influence']:
        d = A[A['mode'] == mode]
        g = d.groupby('t')['gini_exposure'].agg(['mean', 'std']).reset_index()
        ax.plot(g.t, g['mean'], label=mode, color=COLORS[mode], lw=2)
    ax.set_xlabel('t'); ax.set_ylabel('Gini показов')
    ax.set_title('(b) Gini выдачи: все режимы концентрируют показы')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.4)

    # (3) catalog coverage
    ax = fig.add_subplot(gs[0, 2])
    for mode in ['closed_loop', 'static', 'fresh_oracle', 'no_influence']:
        d = A[A['mode'] == mode]
        g = d.groupby('t')['catalog_coverage'].agg(['mean', 'std']).reset_index()
        ax.plot(g.t, g['mean'], label=mode, color=COLORS[mode], lw=2)
    ax.set_xlabel('t'); ax.set_ylabel('catalog coverage')
    ax.set_title('(c) Покрытие каталога рекомендациями')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.4)

    # (4) B beta-sweep tr_T
    ax = fig.add_subplot(gs[1, 0])
    for mode in ['closed_loop', 'static', 'fresh_oracle', 'no_influence']:
        d = B[B['mode'] == mode]
        g = d.groupby('beta').agg(m=('trace_sigma_T', 'mean'),
                                  s=('trace_sigma_T', 'std')).reset_index()
        betas = g['beta'].values
        ax.errorbar(np.where(betas == 0, 1e-4, betas), g['m'], yerr=g['s'],
                    label=mode, color=COLORS[mode], marker='o', capsize=3, lw=1.5)
    ax.set_xscale('log'); ax.set_xlabel(r'$\beta$ ($10^{-4} \equiv 0$)')
    ax.set_ylabel(r'$\mathrm{tr}(\hat\Sigma_T)$')
    ax.set_title(r'(d) $\beta=0 \Rightarrow$ нет коллапса; $\beta>0 \Rightarrow$ коллапс'
                 '\nрежим не влияет на форму кривой')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.4)

    # (5) C alpha sweep
    ax = fig.add_subplot(gs[1, 1])
    g = C.groupby('alpha').agg(m=('trace_sigma_T', 'mean'),
                                s=('trace_sigma_T', 'std')).reset_index()
    ax.errorbar(g['alpha'], g['m'], yerr=g['s'],
                marker='o', capsize=3, color='#ff7f0e', lw=2,
                label='static, β=0.005')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$\mathrm{tr}(\hat\Sigma_T)$')
    ax.set_title(r'(e) В static (БЕЗ петли) коллапс растёт с $\alpha$')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.4)

    # (6) D heatmap closed_loop
    ax = fig.add_subplot(gs[1, 2])
    d = D[D['mode'] == 'static']
    pivot = d.groupby(['alpha', 'beta'])['drop_pct'].mean().unstack()
    im = ax.imshow(pivot.values, origin='lower', aspect='auto', cmap='RdYlBu_r',
                   vmin=-10, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{b:.3f}' for b in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{a:.1f}' for a in pivot.index])
    ax.set_xlabel(r'$\beta$'); ax.set_ylabel(r'$\alpha$')
    ax.set_title(r'(f) static: drop tr$(\hat\Sigma)$ как функция $(\alpha,\beta)$')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f'{pivot.values[i,j]:.0f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(pivot.values[i,j]) > 50 else 'black')
    plt.colorbar(im, ax=ax, label='drop %')

    fig.suptitle('Диагностика коллапса tr$(\\hat\\Sigma)$: причина — β-дрейф эмбеддингов '
                 'через концентрированную выдачу, а не петля переобучения',
                 fontsize=13, y=1.005)
    fig.savefig(FIG_DIR / 'DIAG_explainer_overview.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved DIAG_explainer_overview.pdf')


def main():
    plot_main_diagnostic()
    plot_beta_sweep()
    plot_alpha_sweep_static()
    plot_factorial()
    plot_explainer()
    print('All plots saved to:', FIG_DIR)


if __name__ == '__main__':
    main()
