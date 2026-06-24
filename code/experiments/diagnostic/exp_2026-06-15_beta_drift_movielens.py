#!/usr/bin/env python3
# theorem: beta_drift_observational_upper_bound
# exp: L0 observational estimation of the user-preference drift rate beta on
#      MovieLens-20M, in a content-attribute (genome-tag) space that is
#      independent of any behaviour-derived embedding.
#
# WHAT THIS MEASURES
# ------------------
# The simulation assumes per-interaction preference drift
#       u_i  <-  (1 - beta) u_i  +  beta * v_{consumed},   beta = 0.005.
# Here we ask: does a real user's *preference profile* actually wander toward
# consumed content, and how fast (upper bound on beta)?
#
# NON-TAUTOLOGY NOTE
# ------------------
# If we defined the profile as the running mean of consumed genome vectors and
# regressed its increment on consumption, the result would be an accounting
# identity (the profile moves toward consumption *by construction*). We avoid
# this entirely. Instead we model the sequence of *per-window consumption
# centroids* C_t with a local-level (random-walk-plus-noise) state space:
#
#       observation:  C_t      = theta_t + eps_t ,   eps ~ N(0, s_eps^2)   (within-window sampling noise)
#       state:        theta_t  = theta_{t-1} + nu_t,  nu  ~ N(0, s_nu^2)   (preference drift)
#
#   * stable preference  <=>  s_nu^2 = 0  (theta constant; C_t i.i.d. around a fixed mean)
#   * genuine drift      <=>  s_nu^2 > 0  (the consumption target itself wanders)
#
# Mapping to beta: under the simulation's own update theta_{t+1}=theta_t+beta*eps_t,
# so s_nu = beta * s_eps  =>  beta = s_nu / s_eps = sqrt(s_nu^2 / s_eps^2).
#
# Closed-form method of moments on the first difference d_t = C_t - C_{t-1},
# which is MA(1) under the local-level model:
#       gamma0 = Var(d)        = s_nu^2 + 2 s_eps^2
#       gamma1 = Cov(d_t,d_{t-1}) = -s_eps^2
# =>    s_eps^2 = -gamma1 ,   s_nu^2 = gamma0 + 2*gamma1 ,   beta^2 = (gamma0+2 gamma1)/(-gamma1)
#
# CONTROLS
#   * shuffle: permute window order within each user -> destroys the random-walk
#     temporal structure. If beta_observed >> beta_shuffled, drift is a genuine
#     temporal phenomenon, not a cross-sectional / sampling artefact.
#
# CAVEAT (selection, not influence): with no exogenous variation in exposure we
# cannot separate recommendation-INDUCED drift from autonomous taste evolution.
# All non-stationarity is attributed to drift, so beta_hat is an UPPER BOUND.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = Path(__file__).resolve().parents[3]
DATA      = ROOT / "data" / "movielens"
OUT_CSV   = ROOT / "code" / "experiments" / "results" / "diagnostic"
OUT_FIG   = ROOT / "paper" / "figures" / "diagnostic"
OUT_CSV.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

RNG_SEED   = 0
LIKE_THR   = 4.0      # rating >= LIKE_THR counts as positive consumption
N_PCA      = 20       # genome 1128-d -> N_PCA content dimensions
WIN_SIZES  = [10, 20] # window = this many liked movies (robustness sweep)
MIN_WIN    = 6        # require at least this many windows per user

np.random.seed(RNG_SEED)


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# 1. Genome content space (independent of behaviour) -> PCA
# ----------------------------------------------------------------------------
def build_genome_pca():
    log("[1] loading genome_scores.csv ...")
    g = pd.read_csv(DATA / "genome_scores.csv",
                    dtype={"movieId": np.int32, "tagId": np.int32,
                           "relevance": np.float32})
    n_tags = g["tagId"].max()
    movie_ids = np.sort(g["movieId"].unique())
    mid_to_row = {m: i for i, m in enumerate(movie_ids)}
    mat = np.zeros((len(movie_ids), n_tags), dtype=np.float32)
    rows = g["movieId"].map(mid_to_row).to_numpy()
    cols = (g["tagId"].to_numpy() - 1)
    mat[rows, cols] = g["relevance"].to_numpy()
    log(f"    genome matrix: {mat.shape[0]} movies x {mat.shape[1]} tags")

    pca = PCA(n_components=N_PCA, random_state=RNG_SEED)
    coords = pca.fit_transform(mat).astype(np.float32)
    evr = pca.explained_variance_ratio_.sum()
    log(f"    PCA -> {N_PCA} dims, explained variance = {evr:.3f}")
    return {int(m): coords[i] for i, m in enumerate(movie_ids)}, evr


# ----------------------------------------------------------------------------
# 2. Liked ratings with genome, sorted per user by time
# ----------------------------------------------------------------------------
def load_liked_ratings(genome_movies):
    log("[2] loading rating.csv ...")
    # timestamp kept as string: 'YYYY-MM-DD HH:MM:SS' sorts lexicographically
    r = pd.read_csv(DATA / "rating.csv",
                    usecols=["userId", "movieId", "rating", "timestamp"],
                    dtype={"userId": np.int32, "movieId": np.int32,
                           "rating": np.float32, "timestamp": "string"})
    log(f"    raw ratings: {len(r):,}")
    r = r[r["rating"] >= LIKE_THR]
    r = r[r["movieId"].isin(genome_movies)]
    log(f"    liked (>= {LIKE_THR}) with genome: {len(r):,}")
    r = r.sort_values(["userId", "timestamp"], kind="stable")
    return r


# ----------------------------------------------------------------------------
# 3. Per-user windowed consumption centroids in PCA space
# ----------------------------------------------------------------------------
def build_window_centroids(r, mvec, win_size):
    """Return list of per-user arrays [n_windows, N_PCA] of consumption centroids."""
    user_ids = r["userId"].to_numpy()
    movie_ids = r["movieId"].to_numpy()
    coords = np.stack([mvec[m] for m in movie_ids])  # [N_liked, N_PCA]

    # boundaries between users in the sorted array
    change = np.flatnonzero(np.diff(user_ids)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(user_ids)]])

    series = []
    for s, e in zip(starts, ends):
        n = e - s
        n_win = n // win_size
        if n_win < MIN_WIN:
            continue
        block = coords[s : s + n_win * win_size]
        block = block.reshape(n_win, win_size, N_PCA)
        series.append(block.mean(axis=1))  # [n_win, N_PCA]
    return series


# ----------------------------------------------------------------------------
# 4. Local-level method-of-moments beta from pooled within-user autocovariances
# ----------------------------------------------------------------------------
def estimate_beta(series, shuffle=False):
    """Pool gamma0, gamma1 per PCA dim across users; return beta + components."""
    g0 = np.zeros(N_PCA)   # sum d_t^2
    g0_n = 0
    g1 = np.zeros(N_PCA)   # sum d_t * d_{t-1}
    g1_n = 0
    rng = np.random.default_rng(RNG_SEED + (1 if shuffle else 0))
    for C in series:
        if shuffle:
            C = C[rng.permutation(C.shape[0])]
        d = np.diff(C, axis=0)               # [n_win-1, N_PCA]
        g0 += (d * d).sum(axis=0)
        g0_n += d.shape[0]
        if d.shape[0] >= 2:
            g1 += (d[1:] * d[:-1]).sum(axis=0)
            g1_n += d.shape[0] - 1
    gamma0 = g0 / g0_n
    gamma1 = g1 / g1_n
    s_eps2 = -gamma1                          # per dim
    s_nu2  = gamma0 + 2 * gamma1              # per dim
    # aggregate over dims by pooling variances (clamp tiny negatives from noise)
    S_eps2 = np.maximum(s_eps2, 1e-12).sum()
    S_nu2  = np.maximum(s_nu2, 0.0).sum()
    beta = float(np.sqrt(S_nu2 / S_eps2))
    return {
        "beta": beta,
        "S_nu2": float(S_nu2),
        "S_eps2": float(S_eps2),
        "n_users": len(series),
        "n_diffs": int(g0_n),
        "frac_dims_drift": float(np.mean(s_nu2 > 0)),
    }


def main():
    mvec, evr = build_genome_pca()
    genome_movies = set(mvec.keys())
    r = load_liked_ratings(genome_movies)

    rows = []
    traj_for_plot = None
    for win in WIN_SIZES:
        log(f"[3] windows of {win} liked movies (min {MIN_WIN} windows/user) ...")
        series = build_window_centroids(r, mvec, win)
        log(f"    qualifying users: {len(series):,}")
        obs = estimate_beta(series, shuffle=False)
        shu = estimate_beta(series, shuffle=True)
        win_lengths = np.array([s.shape[0] for s in series])
        per_int = obs["beta"] / win              # rough per-interaction conversion
        log(f"    beta_window(obs)     = {obs['beta']:.4f}")
        log(f"    beta_window(shuffle) = {shu['beta']:.4f}   (null)")
        log(f"    beta_per_interaction ~ {per_int:.5f}   (sim uses 0.005)")
        rows.append({
            "win_size": win,
            "n_users": obs["n_users"],
            "median_windows_per_user": int(np.median(win_lengths)),
            "n_window_diffs": obs["n_diffs"],
            "beta_window_obs": round(obs["beta"], 5),
            "beta_window_shuffle": round(shu["beta"], 5),
            "beta_per_interaction_obs": round(per_int, 6),
            "beta_per_interaction_shuffle": round(shu["beta"] / win, 6),
            "S_nu2_obs": round(obs["S_nu2"], 5),
            "S_eps2_obs": round(obs["S_eps2"], 5),
            "frac_dims_drift_obs": round(obs["frac_dims_drift"], 3),
            "frac_dims_drift_shuffle": round(shu["frac_dims_drift"], 3),
            "genome_pca_evr": round(evr, 3),
        })
        if win == 20:
            traj_for_plot = series

    df = pd.DataFrame(rows)
    out = OUT_CSV / "beta_drift_movielens.csv"
    df.to_csv(out, index=False)
    log(f"[4] saved {out}")
    log("\n" + df.to_string(index=False))

    # ---- figure: observed vs shuffled beta, plus example PC1 trajectories ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(WIN_SIZES))
    w = 0.35
    ax[0].bar(x - w / 2, df["beta_window_obs"], w, label="observed", color="#2c6fbb")
    ax[0].bar(x + w / 2, df["beta_window_shuffle"], w, label="shuffled (null)",
              color="#bbbbbb")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"win={w_}" for w_ in WIN_SIZES])
    ax[0].set_ylabel(r"$\hat\beta$ per window")
    ax[0].set_title("Drift rate: observed vs temporal-shuffle null")
    ax[0].legend()

    if traj_for_plot is not None:
        rng = np.random.default_rng(RNG_SEED)
        pick = rng.choice(len(traj_for_plot),
                          size=min(40, len(traj_for_plot)), replace=False)
        for idx in pick:
            c = traj_for_plot[idx][:, 0]
            ax[1].plot(np.arange(len(c)), c, color="#2c6fbb", alpha=0.25, lw=0.8)
        ax[1].set_xlabel("window (time)")
        ax[1].set_ylabel("consumption centroid, PC1")
        ax[1].set_title("Per-user consumption trajectories (PC1, win=20)")
    fig.tight_layout()
    figpath = OUT_FIG / "DIAG_beta_drift_movielens.pdf"
    fig.savefig(figpath)
    log(f"[4] saved {figpath}")


if __name__ == "__main__":
    main()
