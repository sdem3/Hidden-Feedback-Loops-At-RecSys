# theorem: Теорема 2 (петлевой канал) — проверка ПОД СОБСТВЕННЫМ оператором §6.3
"""
exp_2026-06-15_kl_regularized_retrain.py
========================================
Проверка Теоремы 2 (THEOREM_2.md §6) под её СОБСТВЕННЫМ допущением о форме
переобучения.

КОНТЕКСТ. §7.7 (проба политики) ОПРОВЕРГ механизм Теоремы 2: при обычном
переобучении (Adam + BCE, ERM) переобученная политика оказывается ШИРЕ, а не
уже; Loewner C^closed ⪯ C^stat держится лишь в 7 % случаев. Но §6.3 явно
оговаривает: оператор переобучения в теореме — это НЕ ERM, а
KL-РЕГУЛЯРИЗОВАННОЕ УЛУЧШЕНИЕ ПОЛИТИКИ (зеркальный спуск над политиками,
Лемма B / Гиббс):

    π_{k,t+1} = argmax_π { E_{V~π}[ c_{k,t}(V) ] − (1/κ) KL(π ‖ π_{k,t}) }
              ∝ exp( κ · c_{k,t}(v) ) · π_{k,t}(v)            (Гиббс),

где c_{k,t}(v) — кластерный reward (средний логируемый сигнал кластера k по
айтему v). Накопление: A_{k,t+1} = A_{k,t} + κ H_{k,t} ⪰ A_{k,t} ⇒ C ⪯.

ЭТОТ ЭКСПЕРИМЕНТ. Меняем ТОЛЬКО оператор переобучения: вместо ERM на логах —
точный Гиббс-апдейт политики из §6.3. Всё остальное (warm-start, β-дрейф,
кластеры, отток, клик-модель, режимы closed/static, парность по seed)
ИДЕНТИЧНО exp_2026-06-12c_policy_probe.py, чтобы числа были прямо сопоставимы
с опровержением.

РЕАЛИЗАЦИЯ. Политика — на уровне КЛАСТЕРА (гранулярность §6.2–6.4, где
Теорема 2 и формулируется): на каждый кластер k держим потенциал S[k, ·] над
каталогом. Старт S[k,v] = средний warm-MLP-логит cluster-k юзеров по айтему v
(= замороженная warm-политика, A^stat ≡ A_0). Выдача кластеру = top-K от S[k].
  - closed_loop: каждые T_ret шагов  S[k] ← S[k] + κ · ĉ_{k}(v),
    ĉ_{k}(v) = (клики по v в кластере k) / (показы v в кластере k) — измеренный
    кластерный reward за период. Это и есть Гиббс-апдейт §6.3.
  - static: S заморожен.
β-дрейф, отток, клики, метрики — через немодифицированный SimulationEnvironment.
Сигнатуры sim/ и models/ НЕ меняются.

ЧТО МЕРИМ (closed − static, парно, t=T, 95% бутстрап-ДИ):
  (1) ПРОБА ПОЛИТИКИ: tr(top-K) финальной политики по полному каталогу на
      фиксированных пробных юзерах. Теорема: closed УЖЕ (gap < 0).
      §7.7-ERM давал +0.21 (ШИРЕ) — ждём разворот знака.
  (2) Loewner: доля (seed,кластер), где Σξ^stat − Σξ^closed ⪰ 0.
      Теорема: →100 %. §7.7-ERM: 7 %.
  (3) Σξ показанного: tr, дисперсионное 𝒟(Σξ_T‖Σξ_0).
  (4) Вкусы (позит. контроль / сигнатура петли): tr Σ̂ (gap<0), KL (gap>0).
  (5) Вогнутость reward H⪰0: доля кластеров, где ĉ_{k}(v) убывает с удалением
      v от центра кластера (премиса §6.3/§6.7 — выполнена ли она в данных).

Запуск:
  python exp_2026-06-15_kl_regularized_retrain.py --smoke
  python exp_2026-06-15_kl_regularized_retrain.py
  python exp_2026-06-15_kl_regularized_retrain.py --kappa 5.0
"""
import sys
import time
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
# переиспользуем хелперы/конфиг/RecordingEnv из эксперимента прямого замера Σξ
spec = importlib.util.spec_from_file_location(
    'dispcov', HERE / 'exp_2026-06-12_display_covariance.py')
m = importlib.util.module_from_spec(spec)
_saved_argv = sys.argv
sys.argv = [sys.argv[0]] + (['--smoke'] if '--smoke' in sys.argv else [])
spec.loader.exec_module(m)
sys.argv = _saved_argv

import numpy as np
import pandas as pd
import torch

from sim.user_generator import GMMUserGenerator
from sim.environment import SimulationEnvironment, ExperimentDataset
from sim.click_model import ClickModel
from models.rec_models import RecModel
from models.serving import ServingPolicy


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# ----------------------------------------------------------------------------- #
# Окружение с KL-регуляризованным (Гиббс) переобучением политики §6.3
# ----------------------------------------------------------------------------- #
class KLPolicyEnv(SimulationEnvironment):
    """Переопределяет ТОЛЬКО оператор переобучения T: вместо ERM — Гиббс-апдейт
    кластерной политики S[k] ← S[k] + κ·ĉ_k. Выдача — top-K от S[k]."""

    def setup_policy(self, init_potentials, kappa, temp=1.0, tau=2.0):
        self.S = init_potentials.copy()            # (K_GMM, N_ITEMS)
        self.kappa = float(kappa)
        self.temp = float(temp)                    # температура serving-софтмакса
        self.tau = float(tau)                      # температура истинного отклика θ
        self._last_recs = {}
        self._last_labels = None
        self._last_items = None
        self.reward_concave_flags = []             # для проверки H⪰0
        self._rng = np.random.default_rng(20260615)

    def _policy_probs(self, k, mask=None):
        """π_k(v) ∝ exp(S[k,v]/temp) над каталогом (или маскированным)."""
        s = self.S[k] / self.temp
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        s = s - np.max(s)
        p = np.exp(s)
        z = p.sum()
        return p / z if z > 0 else None

    def _make_recommendations(self, U, I, n_users, n_items):
        """Выдача = СЭМПЛ K айтемов из политики π_k ∝ exp(S_k/temp) (§6.2,
        рандомизированная политика). Показанное тогда отражает ковариацию
        выдачи C — то, на чём построена Теорема 2 (тождество C = Σξ, §6.2)."""
        labels = np.asarray(self.user_gen.cluster_labels)
        recs = {}
        for uid in range(n_users):
            k = int(labels[uid]) if uid < len(labels) else 0
            mask = None
            if self.seen_filter:
                mask = np.isnan(self.dataset.matrix[uid])
                if mask.sum() <= self.K:
                    mask = None
            p = self._policy_probs(k, mask)
            if p is None:
                continue
            nz = int((p > 0).sum())
            kk = min(self.K, nz)
            sel = self._rng.choice(len(p), size=kk, replace=False, p=p)
            recs[uid] = sel
        self._last_recs = recs
        self._last_labels = labels.copy()
        self._last_items = I
        return recs

    def _cluster_reward(self, k, U, I):
        """Популяционный кластерный reward §6: c_k(v) = E_{u∈C_k}[η_t(u,v)],
        η_t = (1−α)·θ(u,v) + α·f_t(u,v) (загрязнение логов, §3.1).
        θ(u,v)=σ(⟨u,v⟩/τ) — истинный отклик; f_t — текущая политика (σ от S/temp,
        нормированного), что даёт самоподкрепление §6.5. Полный каталог, аналитически."""
        labels = np.asarray(self.user_gen.cluster_labels)
        us = U[labels == k]
        if len(us) == 0:
            return None
        theta_bar = _sigmoid((us @ I.T) / self.tau).mean(axis=0)   # (n_items,)
        s = self.S[k] / self.temp
        f_t = _sigmoid(s - s.mean())                              # текущая политика
        alpha = float(getattr(self.click_model, 'adherence', 0.7))
        return (1.0 - alpha) * theta_bar + alpha * f_t

    def _retrain(self, dataset, **kwargs):
        """Гиббс-апдейт §6.3: π_{t+1} ∝ exp(κ·c_k)·π_t ⇔ S[k] ← S[k] + κ·c_k(v)."""
        U = dataset.users_embeddings
        I = dataset.items_embeddings
        for k in range(self.S.shape[0]):
            c_k = self._cluster_reward(k, U, I)
            if c_k is None:
                continue
            self.S[k] += self.kappa * c_k                        # экспоненц. наклон
            self._record_concavity(k, c_k, I)

    def _record_concavity(self, k, c_k, I):
        """H⪰0 ⇔ reward пикирован у центра: корреляция(c_k, −dist_to_center) > 0."""
        center = np.asarray(self.user_gen.component_means[k]) \
            if hasattr(self.user_gen, 'component_means') else I.mean(0)
        d = np.linalg.norm(I - center, axis=1)
        if c_k.std() < 1e-12 or d.std() < 1e-12:
            return
        self.reward_concave_flags.append(float(np.corrcoef(c_k, -d)[0, 1]))


def build_kl_env(mode, seed, kappa, adherence=m.ADHERENCE, sigma_k=m.SIGMA_K,
                 seen_filter=True):
    """Аналог m.build_env, но окружение — KLPolicyEnv с Гиббс-переобучением.
    closed_loop и static при общем seed делят идентичный старт (парный разрыв)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    means, covs, weights = m.make_gmm_params(m.K_GMM, m.EMB_DIM, m.INTER_DIST, sigma_k)
    gen = GMMUserGenerator(component_means=means, component_covs=covs,
                           component_weights=weights,
                           replacement_rate=m.REPLACE, memory_effect=6)
    np.random.seed(seed)
    user_emb, init_lbl = gen.initialize(m.N_USERS)
    item_emb = m.make_items(m.N_ITEMS, m.K_GMM, means, sigma_k, m.EMB_DIM, rng)
    true_pref = m.make_true_pref(user_emb, item_emb)
    matrix = np.full((m.N_USERS, m.N_ITEMS), np.nan)
    dataset = ExperimentDataset(user_emb.copy(), item_emb.copy(), matrix)

    device = torch.device('cpu')
    model = RecModel(m.EMB_DIM, m.EMB_DIM, hidden_size=64).to(device)
    m.pretrain_model(model, user_emb, item_emb, true_pref, rng, device)  # warm-start

    # стартовый потенциал политики = средний warm-MLP-логит по кластеру (= A_0)
    S0 = _warm_cluster_potential(model, user_emb, init_lbl, item_emb, m.K_GMM, device)

    policy = ServingPolicy('top_k')
    alpha_c = 0.0 if mode == 'no_influence' else adherence
    click = ClickModel(adherence=alpha_c, usage_rate=0.8, noise_level=0.05)
    env = KLPolicyEnv(
        dataset=dataset, rec_model=model, user_generator=gen, click_model=click,
        serving_policy=policy, mode=mode, true_preference_matrix=true_pref,
        retrain_period=m.T_RET, K=m.K_REC, device=device, seen_filter=seen_filter,
        user_drift_beta=m.USER_DRIFT, drift_alpha=0.0)
    env.setup_policy(S0, kappa)
    return env


def _warm_cluster_potential(model, user_emb, labels, item_emb, K, device, eps=1e-6):
    """S0[k,v] = mean_{u in C_k} logit( warm-MLP(u,v) ) — замороженная warm-политика."""
    model.eval()
    I = torch.tensor(item_emb, dtype=torch.float32, device=device)
    n_items = item_emb.shape[0]
    S0 = np.zeros((K, n_items), dtype=float)
    with torch.no_grad():
        for k in range(K):
            us = user_emb[labels == k]
            if len(us) == 0:
                continue
            logits = np.zeros(n_items)
            for u in us:
                ut = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0).expand(n_items, -1)
                p = model(ut, I).squeeze().cpu().numpy()
                p = np.clip(p, eps, 1 - eps)
                logits += np.log(p / (1 - p))
            S0[k] = logits / len(us)
    return S0


def softmax_policy_cov(S, items, temp):
    """ГЛАВНЫЙ объект Теоремы 2: ковариация выдачи C_k под политикой
    π_k(v) ∝ exp(S[k,v]/temp) по ПОЛНОМУ каталогу (детерминированно, без
    сэмпл-шума и без исчерпания). Возвращает (dict k->C_k, mean_trace)."""
    Cs = {}
    trs = []
    for k in range(S.shape[0]):
        s = S[k] / temp
        s = s - np.max(s)
        p = np.exp(s)
        p = p / p.sum()
        mean = p @ items                      # (d,)
        d = items - mean
        C = (d * p[:, None]).T @ d            # Σ_v π(v)(v-μ)(v-μ)^T  = C_k
        C = np.atleast_2d(C)
        Cs[k] = C
        trs.append(float(np.trace(C)))
    return Cs, (float(np.mean(trs)) if trs else np.nan)


def policy_topk_cov_from_S(S, items, K):
    """Вторичная проба: ковариация дискретного top-K от S[k] по каталогу."""
    trs = []
    for k in range(S.shape[0]):
        top = np.argsort(S[k])[::-1][:K]
        V = items[top]
        Sig = np.atleast_2d(np.cov(V, rowvar=False))
        trs.append(float(np.trace(Sig)))
    return float(np.mean(trs)) if trs else np.nan


# ----------------------------------------------------------------------------- #
# Прогон
# ----------------------------------------------------------------------------- #
def run(n_seeds, T_run, kappa, seen_filter=True):
    rows = []
    concave = []
    for seed in range(n_seeds):
        rec = {}
        endpoint_mats = {}
        for mode in ('closed_loop', 'static'):
            env = build_kl_env(mode, seed, kappa, seen_filter=seen_filter)
            mats0 = matsT = None
            for t in range(T_run):
                env.step(t)
                covs = m.display_cov_per_cluster(env._last_recs, env._last_labels,
                                                 env._last_items, m.K_GMM)
                if t == 0:
                    mats0 = covs
                if t == T_run - 1:
                    matsT = covs
            df = env.metrics.get_dataframe()
            taste_tr = float(df['trace_sigma'].iloc[-1])
            taste_kl = float(df['kl_from_initial'].iloc[-1])
            items = env.dataset.items_embeddings
            C_k, C_tr = softmax_policy_cov(env.S, items, env.temp)   # ГЛАВНЫЙ объект C
            probe_tr = policy_topk_cov_from_S(env.S, items, m.K_REC)
            rec[mode] = (taste_tr, taste_kl, probe_tr, C_tr)
            endpoint_mats[mode] = (mats0, matsT, C_k)
            if env.reward_concave_flags:
                concave.append(dict(seed=seed, mode=mode,
                                    mean_corr=float(np.mean(env.reward_concave_flags)),
                                    frac_pos=float(np.mean(np.array(env.reward_concave_flags) > 0))))

        # Loewner + 𝒟 + tr Σξ + tr C, парно closed vs static, по кластерам на t=T-1
        m0_cl, mT_cl, C_cl = endpoint_mats['closed_loop']
        m0_st, mT_st, C_st = endpoint_mats['static']
        for k in range(m.K_GMM):
            if mT_cl and mT_st and k in mT_cl and k in mT_st:
                tr_cl = m.cov_scalars(mT_cl[k])[0]
                tr_st = m.cov_scalars(mT_st[k])[0]
                psd, mineig = m.loewner_psd_frac(mT_st[k], mT_cl[k])
                disp_cl = m.dispersion_div(mT_cl[k], m0_cl[k]) if (m0_cl and k in m0_cl) else np.nan
                disp_st = m.dispersion_div(mT_st[k], m0_st[k]) if (m0_st and k in m0_st) else np.nan
            else:
                tr_cl = tr_st = psd = mineig = disp_cl = disp_st = np.nan
            # ГЛАВНОЕ: ковариация выдачи C (софтмакс-политика) — объект Теоремы 2
            C_tr_cl = float(np.trace(C_cl[k])) if k in C_cl else np.nan
            C_tr_st = float(np.trace(C_st[k])) if k in C_st else np.nan
            if k in C_cl and k in C_st:
                C_psd, C_min = m.loewner_psd_frac(C_st[k], C_cl[k])   # C^stat − C^closed ⪰ 0 ?
            else:
                C_psd = C_min = np.nan

            rows.append(dict(
                seed=seed, cluster=k,
                taste_tr_closed=rec['closed_loop'][0], taste_tr_static=rec['static'][0],
                taste_kl_closed=rec['closed_loop'][1], taste_kl_static=rec['static'][1],
                probe_tr_closed=rec['closed_loop'][2], probe_tr_static=rec['static'][2],
                C_tr_closed=C_tr_cl, C_tr_static=C_tr_st,
                C_loewner_psd=C_psd, C_min_eig_diff=C_min,
                xi_tr_closed=tr_cl, xi_tr_static=tr_st, loewner_psd=psd, min_eig_diff=mineig,
                xi_disp_closed=disp_cl, xi_disp_static=disp_st))
        print(f'  seed {seed}: taste_kl cl={rec["closed_loop"][1]:.3f} st={rec["static"][1]:.3f} | '
              f'probe_tr cl={rec["closed_loop"][2]:.3f} st={rec["static"][2]:.3f}')
    return pd.DataFrame(rows), pd.DataFrame(concave)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--kappa', type=float, default=5.0)
    args, _ = ap.parse_known_args()

    smoke = args.smoke
    n_seeds = 2 if smoke else 10
    T_run = 30 if smoke else 100
    kappa = args.kappa

    t0 = time.time()
    print(f'[kl-retrain] SMOKE={smoke}  N_SEEDS={n_seeds}  T={T_run}  kappa={kappa}')
    df, dfc = run(n_seeds, T_run, kappa)

    out = m.RESULTS_DIR / 'kl_regularized_retrain.csv'
    df.to_csv(out, index=False)

    # один ряд на seed для парных вкусовых/проб-метрик (cluster-инвариантны)
    per_seed = df.drop_duplicates('seed')

    def gap_ci(frame, a, b, n_boot=5000, seed=0):
        rng = np.random.default_rng(seed)
        d = (frame[a] - frame[b]).values.astype(float)
        d = d[~np.isnan(d)]
        if len(d) == 0:
            return np.nan, np.nan, np.nan
        bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    print('\n========== KL-РЕГУЛЯРИЗОВАННОЕ ПЕРЕОБУЧЕНИЕ (§6.3) — closed − static ==========')
    print('  --- ГЛАВНОЕ: ковариация выдачи C (объект Теоремы 2, §6.2–6.4) ---')
    checks_main = [
        ('C = cov выдачи, tr  ', df, 'C_tr_closed', 'C_tr_static',
         'Теорема 2: <0 (C^closed у́же)'),
    ]
    checks_aux = [
        ('ПРОБА top-K tr      ', per_seed, 'probe_tr_closed', 'probe_tr_static',
         'вторичн.; §7.7-ERM: +0.212 (шире)'),
        ('ВКУСЫ trace Σ̂      ', per_seed, 'taste_tr_closed', 'taste_tr_static',
         'сигнатура петли: <0'),
        ('ВКУСЫ KL(P_T‖P_0)  ', per_seed, 'taste_kl_closed', 'taste_kl_static',
         'сигнатура петли: >0'),
        ('Σξ показанного tr   ', df, 'xi_tr_closed', 'xi_tr_static',
         'сужение показанного: <0'),
        ('Σξ дисперсионное 𝒟  ', df, 'xi_disp_closed', 'xi_disp_static',
         'closed дальше сжата: >0'),
    ]
    for name, frame, a, b, note in checks_main:
        mn, lo, hi = gap_ci(frame, a, b)
        sig = 'SIGNIF' if not (lo <= 0 <= hi) else 'n.s.'
        print(f'  {name} gap = {mn:+.4f} [{lo:+.4f};{hi:+.4f}] {sig:6s} | {note}')
    cloew = float(df['C_loewner_psd'].dropna().mean())
    print(f'  Loewner C^stat ⪰ C^closed:  {cloew*100:.0f}%  (Теорема 2 → 100%; §7.7-ERM: 7%)')
    print('  --- вспомогательные ---')
    for name, frame, a, b, note in checks_aux:
        mn, lo, hi = gap_ci(frame, a, b)
        sig = 'SIGNIF' if not (lo <= 0 <= hi) else 'n.s.'
        print(f'  {name} gap = {mn:+.4f} [{lo:+.4f};{hi:+.4f}] {sig:6s} | {note}')

    loew = float(df['loewner_psd'].dropna().mean())
    print(f'\n  Loewner Σξ^stat ⪰ Σξ^closed: {loew*100:.0f}%  (показанное)')
    print(f'  mean C tr:      closed={per_seed.C_tr_closed.mean():.3f}  '
          f'static={per_seed.C_tr_static.mean():.3f}')
    if len(dfc):
        print(f'  reward-вогнутость H⪰0 (corr ĉ,−dist): mean={dfc.mean_corr.mean():+.3f}  '
              f'доля кластеров с corr>0={dfc.frac_pos.mean()*100:.0f}%')

    print(f'\nCSV -> {out}\nDone in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
