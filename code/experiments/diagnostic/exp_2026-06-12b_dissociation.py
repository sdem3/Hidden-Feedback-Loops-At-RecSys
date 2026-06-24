"""
exp_2026-06-12b_dissociation.py
================================
Позитивный контроль / тест диссоциации к exp_2026-06-12_display_covariance.py.

Цель: в ОДНИХ И ТЕХ ЖЕ прогонах показать, что
  (1) разрыв по ВКУСОВОЙ ковариации closed-static ОТРИЦАТЕЛЕН (воспроизводит H7/§21);
  (2) разрыв по ковариации ПОКАЗАННОГО Σξ — НУЛЕВОЙ.
Это делает вывод однозначным: петля углубляет коллапс вкусов НЕ через сужение
выдачи (механизм гауссовой модели §6 / бывшего постулата B3' отвергается).

Переиспользует build_env / Σξ-хелперы из exp_2026-06-12_display_covariance.py
(грузим как модуль через importlib из-за дефиса в имени).
"""
import sys
import time
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'dispcov', HERE / 'exp_2026-06-12_display_covariance.py')
m = importlib.util.module_from_spec(spec)
# не запускаем его __main__:
sys.argv = [sys.argv[0]] + (['--smoke'] if '--smoke' in sys.argv else [])
spec.loader.exec_module(m)

import numpy as np
import pandas as pd

SMOKE = '--smoke' in sys.argv
N_PC = 2 if SMOKE else 5
T_pc = 30 if SMOKE else 100
SEEN = False  # чистый замер концентрации политики


def run():
    rows = []
    for seed in range(N_PC):
        rec = {}
        for mode in ('closed_loop', 'static'):
            env = m.build_env(mode, seed, seen_filter=SEEN)
            for t in range(T_pc):
                env.step(t)
                if t == T_pc - 1:
                    covs = m.display_cov_per_cluster(
                        env._last_recs, env._last_labels, env._last_items, m.K_GMM)
                    disp_tr = float(np.mean([m.cov_scalars(S)[0] for S in covs.values()])) if covs else np.nan
            df = env.metrics.get_dataframe()
            taste_tr = float(df['trace_sigma'].iloc[-1])
            taste_kl = float(df['kl_from_initial'].iloc[-1])
            rec[mode] = (taste_tr, taste_kl, disp_tr)
        rows.append(dict(seed=seed,
                         taste_tr_closed=rec['closed_loop'][0], taste_tr_static=rec['static'][0],
                         taste_kl_closed=rec['closed_loop'][1], taste_kl_static=rec['static'][1],
                         disp_tr_closed=rec['closed_loop'][2], disp_tr_static=rec['static'][2]))
        print(f'  seed {seed}: taste_tr cl={rec["closed_loop"][0]:.3f} st={rec["static"][0]:.3f} | '
              f'disp_tr cl={rec["closed_loop"][2]:.3f} st={rec["static"][2]:.3f}')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    t0 = time.time()
    print(f'[dissociation] SMOKE={SMOKE} N_PC={N_PC} T={T_pc} seen_filter={SEEN}')
    df = run()
    out = m.RESULTS_DIR / 'display_cov_dissociation.csv'
    df.to_csv(out, index=False)

    def gap(a, b):
        d = (df[a] - df[b]).values
        return d.mean(), d.std() / max(1, np.sqrt(len(d)))

    print('\n----- DISSOCIATION (closed - static, paired, t=T) -----')
    for name, ca, cb in [('TASTE trace  ', 'taste_tr_closed', 'taste_tr_static'),
                         ('TASTE KL     ', 'taste_kl_closed', 'taste_kl_static'),
                         ('DISPLAY trace', 'disp_tr_closed', 'disp_tr_static')]:
        mn, se = gap(ca, cb)
        print(f'  {name} gap = {mn:+.4f} ± {se:.4f} (SE)')
    print(f'\nCSV -> {out}')
    print(f'Done in {time.time() - t0:.1f}s')
