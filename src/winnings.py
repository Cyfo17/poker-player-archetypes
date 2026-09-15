#!/usr/bin/env python3
"""Analyze relationships between playing style and net winnings per hand.

Report linear regression coefficients, rank correlations, and cluster summaries.
The extended all_in_rate feature may reflect stack depletion after losses.

Usage: python src/winnings.py players.csv --labeled players_labeled.csv
       python src/winnings.py players.csv --extended --exclude all_in_rate"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

STYLE = ["vpip", "pfr", "agg_freq", "wtsd"]
EXTENDED = ["all_in_rate", "river_agg_freq", "limp_rate",
            "three_bet_rate", "cbet", "donk", "pfr_vpip_ratio"]


def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="players.csv from parse_features.py")
    ap.add_argument("--labeled", help="players_labeled.csv from cluster.py")
    ap.add_argument("--min-hands", type=int, default=100)
    ap.add_argument("--extended", action="store_true",
                    help="include the extended feature set")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="drop named features from the chosen set")
    args = ap.parse_args()

    full = pd.read_csv(args.csv).dropna(subset=STYLE)
    df = full[full["n_hands"] >= args.min_hands].copy()

    ols_feats = list(STYLE)
    if args.extended:
        missing = [c for c in EXTENDED if c not in df.columns]
        if missing:
            ap.error(f"missing columns {missing}; re-run parse_features.py")
        df[EXTENDED] = df[EXTENDED].fillna(0)
        full[EXTENDED] = full[EXTENDED].fillna(0)
        ols_feats = STYLE + EXTENDED + ["wsf"]
    if args.exclude:
        ols_feats = [f for f in ols_feats if f not in args.exclude]
    tag = "extended" if args.extended else "base"
    if args.exclude:
        tag += f" (excluded: {', '.join(args.exclude)})"
    print(f"{len(df)} players (>={args.min_hands} hands); "
          f"{len(full)} total for the hand-count distribution "
          f"-- {tag} feature set: {len(ols_feats)} features for OLS")

    y = winsorize(df["net_per_hand"])
    Xs = StandardScaler().fit_transform(df[ols_feats])
    ols = LinearRegression().fit(Xs, y)
    print(f"\n=== OLS ({tag}): win-rate (winsorized 1-99%) vs standardized style ===")
    for f_, c in zip(ols_feats, ols.coef_):
        print(f"  {f_:18s} coef = {c:+.3f}")
    print(f"  intercept = {ols.intercept_:+.3f}   R^2 = {ols.score(Xs, y):.3f}")

    Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(df[STYLE]))
    df["PC1"], df["PC2"] = Z[:, 0], Z[:, 1]
    df["gap"] = df["vpip"] - df["pfr"]

    cols = STYLE + (EXTENDED if args.extended else []) + ["gap", "PC1", "PC2"]
    if args.exclude:
        cols = [c for c in cols if c not in args.exclude]
    rho = (df[cols + ["net_per_hand"]].corr(method="spearman")["net_per_hand"]
           .drop("net_per_hand").sort_values())
    print(f"\n=== Spearman ({tag}) rank-correlation with win-rate ===")
    print(rho.round(3).to_string())

    if args.labeled:
        lab = pd.read_csv(args.labeled).dropna(subset=["net_per_hand"])
        print("\n=== median win-rate by archetype (k-means) ===")
        print(lab.groupby("kmeans")["net_per_hand"]
              .agg(["median", "count"]).round(3).to_string())

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    v = float(np.percentile(np.abs(winsorize(df["net_per_hand"], .05, .95)), 95))
    sc = ax[0, 0].scatter(df["PC1"], df["PC2"], c=df["net_per_hand"].clip(-v, v),
                          cmap="RdYlGn", s=8, vmin=-v, vmax=v)
    fig.colorbar(sc, ax=ax[0, 0], label="win-rate (clipped)")
    ax[0, 0].set(xlabel="PC1 (looseness)", ylabel="PC2 (aggression)",
                 title="profit gradient over style-space")

    ax[0, 1].barh(rho.index, rho.values,
                  color=["#c0392b" if x < 0 else "#27ae60" for x in rho.values])
    ax[0, 1].axvline(0, color="k", lw=.8)
    ax[0, 1].set(title=f"Spearman corr. of style vs win-rate ({tag})",
                 xlabel="rho")

    if args.labeled:
        ks = sorted(lab["kmeans"].unique())
        groups = [lab.loc[lab["kmeans"] == k, "net_per_hand"].values for k in ks]
        ax[1, 0].boxplot(groups, showfliers=False)
        ax[1, 0].set_xticks(range(1, len(ks) + 1))
        ax[1, 0].set_xticklabels([f"c{k}" for k in ks])
        lo, hi = lab["net_per_hand"].quantile([.05, .95])
        ax[1, 0].set_ylim(lo, hi)
        ax[1, 0].axhline(0, color="grey", ls="--", lw=.8)
        ax[1, 0].set(title="win-rate by archetype (k-means)", ylabel="net per hand")
    else:
        ax[1, 0].set_visible(False)

    net = full["net_per_hand"].values
    nh = full["n_hands"].values
    bins = np.logspace(np.log10(nh.min() + 1), np.log10(nh.max()), 16)
    cen = np.sqrt(bins[:-1] * bins[1:])
    idx = np.digitize(nh, bins)
    p10 = [np.percentile(net[idx == i], 10) if (idx == i).sum() > 5 else np.nan
           for i in range(1, len(bins))]
    p90 = [np.percentile(net[idx == i], 90) if (idx == i).sum() > 5 else np.nan
           for i in range(1, len(bins))]
    yv = float(np.percentile(np.abs(winsorize(full["net_per_hand"], .02, .98)), 98))
    ax[1, 1].scatter(nh, np.clip(net, -yv, yv), s=3, alpha=.15)
    ax[1, 1].fill_between(cen, p10, p90, alpha=.3, color="orange", label="10-90% band")
    ax[1, 1].axhline(0, color="k", lw=.8)
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_ylim(-yv, yv)
    ax[1, 1].legend()
    ax[1, 1].set(xlabel="# hands (log)", ylabel="net per hand",
                 title="win-rate distribution by hand count")

    fig.tight_layout()
    base_name = "winnings_extended" if args.extended else "winnings"
    if args.exclude:
        base_name += "_no_" + "_".join(args.exclude)
    out = f"{base_name}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
