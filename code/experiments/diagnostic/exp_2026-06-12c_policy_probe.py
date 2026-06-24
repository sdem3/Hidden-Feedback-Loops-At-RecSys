"""
exp_2026-06-12c_policy_probe.py
================================
РЕШАЮЩИЙ тест механизма гауссовой модели (§6, Теорема 2): сужает ли переобучение
саму ПОЛИТИКУ (резкость A растёт ⇒ ковариация выдачи C сужается)?

Проблема прямого замера показанного (exp_2026-06-12_display_covariance.py):
  - seen_filter=True (режим H7, где петлевой сигнал по вкусам есть): «показанное»
    портится исчерпанием каталога (юзеры всё уже видели → argmax почти случайный);
  - seen_filter=False: показанное чисто, но петлевой сигнал по вкусам исчезает.
Эти два условия взаимно исключают чистый тест.

РЕШЕНИЕ — ПРОБА ПОЛИТИКИ. Прогоняем петлю в режиме H7 (seen_filter=True), берём
ФИНАЛЬНУЮ модель (для closed — переобученную, для static — замороженную warm) и
оцениваем её top-K по ПОЛНОМУ каталогу на ФИКСИРОВАННЫХ пробных пользователях
(стартовые позиции, общие для обоих режимов), БЕЗ seen-фильтра. Ковариация этих
top-K айтемов по кластерам = собственная концентрация политики, без артефактов
исчерпания/оттока.

Предсказание §6: у closed проб-top-K у́же, чем у static (политика заострилась).

В тех же прогонах пишем разрыв по ВКУСАМ (trace, KL) — позитивный контроль
(должен воспроизвести H7/§21: KL closed-static > 0). Это даёт диссоциацию в
ОДНИХ прогонах: вкусы / политика.
"""
import sys
import time
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'dispcov', HERE / 'exp_2026-06-12_display_covariance.py')
m = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]] + (['--smoke'] if '--smoke' in sys.argv else [])
spec.loader.exec_module(m)

import numpy as np
import pandas as pd
import torch

SMOKE = '--smoke' in sys.argv
N_SEEDS = 2 if SMOKE else 10
T_run = 30 if SMOKE else 100
PROBE_PER_CLUSTER = 60   # пробных юзеров на кластер


def make_probe_users(seed, sigma_k=m.SIGMA_K):
    """Стартовые пользователи для seed — идентичны тем, что строит build_env."""
    np.random.seed(seed)
    means, covs, weights = m.make_gmm_params(m.K_GMM, m.EMB_DIM, m.INTER_DIST, sigma_k)
    gen = m.GMMUserGenerator(component_means=means, component_covs=covs,
                             component_weights=weights,
                             replacement_rate=m.REPLACE, memory_effect=6)
    np.random.seed(seed)
    ue, lbl = gen.initialize(m.N_USERS)
    # подвыборка фиксированного размера на кластер (детерминированно)
    rng = np.random.default_rng(seed + 7777)
    idx = []
    for k in range(m.K_GMM):
        ck = np.where(lbl == k)[0]
        take = min(PROBE_PER_CLUSTER, len(ck))
        idx.extend(rng.choice(ck, take, replace=False).tolist())
    idx = np.array(idx)
    return ue[idx], lbl[idx]


def policy_topk_cov(model, probe_u, probe_lbl, items, K, device):
    """Для каждого кластера: ковариация top-K айтемов (по полному каталогу),
    усреднённая по пробным юзерам кластера. Возвращает mean trace по кластерам."""
    model.eval()
    I = torch.tensor(items, dtype=torch.float32, device=device)
    n_items = items.shape[0]
    trs = []
    with torch.no_grad():
        for k in range(m.K_GMM):
            us = probe_u[probe_lbl == k]
            if len(us) < 2:
                continue
            shown = []
            for u in us:
                ut = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0).expand(n_items, -1)
                sc = model(ut, I).squeeze().cpu().numpy()
                topk = np.argsort(sc)[::-1][:K]
                shown.append(items[topk])
            V = np.vstack(shown)
            Sig = np.atleast_2d(np.cov(V, rowvar=False))
            trs.append(float(np.trace(Sig)))
    return float(np.mean(trs)) if trs else np.nan


def run():
    rows = []
    device = torch.device('cpu')
    for seed in range(N_SEEDS):
        probe_u, probe_lbl = make_probe_users(seed)
        rec = {}
        for mode in ('closed_loop', 'static'):
            env = m.build_env(mode, seed, seen_filter=True)   # режим H7
            for t in range(T_run):
                env.step(t)
            df = env.metrics.get_dataframe()
            taste_tr = float(df['trace_sigma'].iloc[-1])
            taste_kl = float(df['kl_from_initial'].iloc[-1])
            probe_tr = policy_topk_cov(env.rec_model, probe_u, probe_lbl,
                                       env.dataset.items_embeddings, m.K_REC, device)
            rec[mode] = (taste_tr, taste_kl, probe_tr)
        rows.append(dict(seed=seed,
                         taste_tr_closed=rec['closed_loop'][0], taste_tr_static=rec['static'][0],
                         taste_kl_closed=rec['closed_loop'][1], taste_kl_static=rec['static'][1],
                         probe_tr_closed=rec['closed_loop'][2], probe_tr_static=rec['static'][2]))
        print(f'  seed {seed}: taste_kl cl={rec["closed_loop"][1]:.3f} st={rec["static"][1]:.3f} | '
              f'probe_tr cl={rec["closed_loop"][2]:.3f} st={rec["static"][2]:.3f}')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    t0 = time.time()
    print(f'[policy-probe] SMOKE={SMOKE} N_SEEDS={N_SEEDS} T={T_run} '
          f'probe/cluster={PROBE_PER_CLUSTER}')
    df = run()
    out = m.RESULTS_DIR / 'policy_probe.csv'
    df.to_csv(out, index=False)

    def gap_ci(a, b, n_boot=5000, seed=0):
        rng = np.random.default_rng(seed)
        d = (df[a] - df[b]).values
        d = d[~np.isnan(d)]
        bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    print('\n----- POLICY PROBE (closed - static, paired, t=T, H7 regime) -----')
    for name, ca, cb in [('TASTE trace ', 'taste_tr_closed', 'taste_tr_static'),
                         ('TASTE KL    ', 'taste_kl_closed', 'taste_kl_static'),
                         ('POLICY probe', 'probe_tr_closed', 'probe_tr_static')]:
        mn, lo, hi = gap_ci(ca, cb)
        sig = 'SIGNIF' if not (lo <= 0 <= hi) else 'n.s.'
        print(f'  {name} gap = {mn:+.4f} [{lo:+.4f};{hi:+.4f}] {sig}')
    print(f'\n  mean probe tr: closed={df.probe_tr_closed.mean():.3f}  static={df.probe_tr_static.mean():.3f}')
    print(f'  mean taste KL: closed={df.taste_kl_closed.mean():.3f}  static={df.taste_kl_static.mean():.3f}')
    print(f'\nCSV -> {out}\nDone in {time.time() - t0:.1f}s')
