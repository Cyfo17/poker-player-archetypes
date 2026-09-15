#!/usr/bin/env python3
"""Compare k-means clusters with fixed VPIP/PFR rules and measure seed stability.

Usage: python src/baseline.py players_labeled.csv
       python src/baseline.py --demo"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

STYLE = ["vpip", "pfr", "agg_freq", "wtsd"]

VPIP_TIGHT = 0.28           # vpip <= 0.28 tight, else loose
PFR_RATIO  = 0.50           # pfr >= 0.5*vpip aggressive, else passive


def rule_buckets(df):
    tight = df["vpip"] <= VPIP_TIGHT
    aggr  = df["pfr"]  >= PFR_RATIO * df["vpip"].clip(lower=1e-9)
    return pd.Series(
        np.where(tight & aggr,  "TAG (tight-aggr)",
        np.where(tight & ~aggr, "nit (tight-pass)",
        np.where(~tight & aggr, "LAG (loose-aggr)",
                                "fish (loose-pass)"))),
        index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="players_labeled.csv from cluster.py")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    if args.demo:
        from cluster import make_demo
        df = make_demo()
        Xs0 = StandardScaler().fit_transform(df[STYLE])
        df["kmeans"] = KMeans(4, n_init=10, random_state=0).fit_predict(Xs0)
    else:
        if not args.csv:
            ap.error("provide players_labeled.csv or use --demo")
        df = pd.read_csv(args.csv)

    df = df.dropna(subset=STYLE)
    Xs = StandardScaler().fit_transform(df[STYLE].to_numpy(float))
    df["baseline"] = rule_buckets(df)
    samp = min(5000, len(df))

    bl = df["baseline"].astype("category").cat.codes.to_numpy()
    sil_km = silhouette_score(Xs, df["kmeans"], sample_size=samp, random_state=0)
    sil_bl = silhouette_score(Xs, bl,           sample_size=samp, random_state=0)
    print(f"silhouette   k-means = {sil_km:.3f}   |   2x2 rule = {sil_bl:.3f}")

    print(f"adjusted Rand index (k-means vs rule) = "
          f"{adjusted_rand_score(df['kmeans'], bl):.3f}  (1=identical, 0=chance)")
    print("\ncross-tab  (rows = k-means cluster, cols = 2x2 rule bucket):")
    print(pd.crosstab(df["kmeans"], df["baseline"]).to_string())

    if "net_per_hand" in df.columns:
        print("\nmedian net_per_hand by k-means cluster:")
        print(df.groupby("kmeans")["net_per_hand"]
              .agg(["median", "count"]).round(3).to_string())
        print("\nmedian net_per_hand by 2x2 rule bucket:")
        print(df.groupby("baseline")["net_per_hand"]
              .agg(["median", "count"]).round(3).to_string())

    k = int(df["kmeans"].nunique())
    labs, sils = [], []
    for s in range(args.seeds):
        l = KMeans(k, n_init=10, random_state=s).fit_predict(Xs)
        labs.append(l)
        sils.append(silhouette_score(Xs, l, sample_size=samp, random_state=0))
    aris = [adjusted_rand_score(labs[i], labs[j])
            for i in range(len(labs)) for j in range(i + 1, len(labs))]
    print(f"\nstability over {args.seeds} seeds (k={k}): "
          f"silhouette {np.mean(sils):.3f}+/-{np.std(sils):.3f}, "
          f"mean pairwise ARI {np.mean(aris):.3f}  (1=identical clusterings)")


if __name__ == "__main__":
    main()
