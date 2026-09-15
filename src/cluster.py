#!/usr/bin/env python3
"""Cluster player styles using k-means, Gaussian mixtures, and spectral clustering.

Win rate is excluded from clustering and reported in cluster summaries.
Usage: python src/cluster.py players.csv
       python src/cluster.py --demo"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

STYLE = ["vpip", "pfr", "agg_freq", "wtsd"]


def make_demo(n=600, seed=0):
    """Generate synthetic player styles for smoke tests."""
    rng = np.random.default_rng(seed)
    centres = {                       # vpip, pfr, agg_freq, wtsd
        "TAG":  (.22, .18, .55, .28),
        "LAG":  (.40, .32, .62, .30),
        "nit":  (.12, .09, .40, .22),
        "fish": (.48, .08, .20, .40),
    }
    rows = []
    for name, mu in centres.items():
        x = rng.normal(mu, 0.04, size=(n // 4, 4)).clip(0, 1)
        for r in x:
            rows.append([name, *r, rng.normal(0, 1)])
    df = pd.DataFrame(rows, columns=["true", *STYLE, "net_per_hand"])
    df["n_hands"] = 200
    df["nick"] = [f"p{i}" for i in range(len(df))]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="players.csv from parse_features.py")
    ap.add_argument("--demo", action="store_true", help="run on synthetic fixture")
    ap.add_argument("--min-hands", type=int, default=100,
                    help="drop players with fewer hands (noisy stats)")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--force-k", type=int, default=None,
                    help="override silhouette and fit final k-means with this k")
    args = ap.parse_args()

    if args.demo:
        df = make_demo()
        print("[demo] synthetic data -- pipeline smoke test only")
    else:
        if not args.csv:
            ap.error("provide players.csv or use --demo")
        df = pd.read_csv(args.csv)
        df = df[df["n_hands"] >= args.min_hands].dropna(subset=STYLE)
    print(f"{len(df)} players after min-hands={args.min_hands} filter")

    X = df[STYLE].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    n = len(Xs)

    pca = PCA().fit(Xs)
    print("PCA explained variance ratio:",
          np.round(pca.explained_variance_ratio_, 3))
    Z = pca.transform(Xs)[:, :2]

    ks = list(range(2, args.kmax + 1))
    sample = min(5000, n)               # silhouette is O(n^2); subsample if big
    sils = []
    for k in ks:
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Xs)
        sils.append(silhouette_score(Xs, lab, sample_size=sample, random_state=0))
        print(f"k={k}  silhouette={sils[-1]:.3f}")
    best_k = ks[int(np.argmax(sils))]
    print(f"--> best k (silhouette) = {best_k}")
    k_used = args.force_k or best_k
    if args.force_k:
        print(f"fitting final k-means with k={k_used} (--force-k)")

    km = KMeans(k_used, n_init=10, random_state=0).fit(Xs)
    df["kmeans"] = km.labels_

    bics = [GaussianMixture(k, covariance_type="full", random_state=0)
            .fit(Xs).bic(Xs) for k in ks]
    print("GMM BIC per k:", [int(b) for b in bics])
    best_g = ks[int(np.argmin(bics))]
    gmm = GaussianMixture(best_g, covariance_type="full", random_state=0).fit(Xs)
    df["gmm"] = gmm.predict(Xs)
    df["gmm_conf"] = gmm.predict_proba(Xs).max(axis=1)   # soft assignment
    print(f"GMM best k (BIC) = {best_g}")

    if n <= 4000:
        df["spectral"] = SpectralClustering(
            k_used, affinity="nearest_neighbors", random_state=0).fit_predict(Xs)
    else:
        print("(skipping spectral: n>4000 makes it slow; subsample to enable)")

    centroids = scaler.inverse_transform(km.cluster_centers_)
    summary = pd.DataFrame(centroids, columns=STYLE)
    summary["n"] = df.groupby("kmeans").size().reindex(range(k_used)).values
    summary["net_median"] = (df.groupby("kmeans")["net_per_hand"].median()
                             .reindex(range(k_used)).values)
    print("\n=== k-means cluster profiles (original units) ===")
    print(summary.round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(ks, sils, "o-")
    ax[0].axvline(best_k, ls="--", c="grey")
    ax[0].set(xlabel="k", ylabel="silhouette", title="k-means model selection")
    ax[1].scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="tab10", s=8)
    ax[1].set(xlabel="PC1", ylabel="PC2", title="players in PCA space (k-means)")
    fig.tight_layout()
    fig.savefig("clusters.png", dpi=130)
    df.to_csv("players_labeled.csv", index=False)
    print("\nsaved clusters.png and players_labeled.csv")


if __name__ == "__main__":
    main()
