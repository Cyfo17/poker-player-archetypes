#!/usr/bin/env python3
"""Classify players in the top and bottom win-rate tertiles from playing style.

Compare logistic regression and decision trees with a majority-class baseline.
The middle tertile is excluded. Winnings are used only to define the target.
The extended all_in_rate feature may reflect stack depletion after losses;
use --exclude all_in_rate to evaluate without it.

Usage: python src/supervised.py players.csv
       python src/supervised.py players.csv --extended --exclude all_in_rate"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, KFold

STYLE = ["vpip", "pfr", "agg_freq", "wtsd"]
EXTENDED = ["all_in_rate", "river_agg_freq", "limp_rate",
            "three_bet_rate", "cbet", "donk", "pfr_vpip_ratio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="players.csv from parse_features.py")
    ap.add_argument("--min-hands", type=int, default=500,
                    help="stability cutoff before tertile split")
    ap.add_argument("--depths", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 8, 10, 12])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--extended", action="store_true",
                    help="include seven additional playing-style features")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="drop named features from the chosen set "
                         "(e.g., --exclude all_in_rate)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=STYLE + ["net_per_hand"])
    df = df[df["n_hands"] >= args.min_hands].copy()
    df["gap"] = df["vpip"] - df["pfr"]
    feats = STYLE + ["wsf", "gap"]
    if args.extended:
        missing = [c for c in EXTENDED if c not in df.columns]
        if missing:
            ap.error(f"missing columns {missing}; re-run parse_features.py")
        df[EXTENDED] = df[EXTENDED].fillna(0)
        feats = feats + EXTENDED
    if args.exclude:
        feats = [f for f in feats if f not in args.exclude]
    tag = "extended" if args.extended else "base"
    if args.exclude:
        tag += f" (excluded: {', '.join(args.exclude)})"
    print(f"feature set: {tag} ({len(feats)} features) -> {feats}")

    q = df["net_per_hand"].quantile([1/3, 2/3])
    lo, hi = float(q.iloc[0]), float(q.iloc[1])
    df = df[(df["net_per_hand"] <= lo) | (df["net_per_hand"] >= hi)].copy()
    df["winner"] = (df["net_per_hand"] >= hi).astype(int)
    print(f"{len(df)} players after >={args.min_hands} hands + tertile split "
          f"(winners={int(df.winner.sum())}, losers={int((1-df.winner).sum())})")

    X = df[feats].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    y = df["winner"].to_numpy()
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=0)

    maj = float(max(y.mean(), 1 - y.mean()))
    print(f"\nmajority-class baseline accuracy = {maj:.3f}")

    s_lr = cross_val_score(LogisticRegression(max_iter=2000),
                           Xs, y, cv=cv, scoring="accuracy")
    print(f"logistic regression   CV accuracy = {s_lr.mean():.3f} +/- {s_lr.std():.3f}")

    print(f"\ndecision-tree depth sweep ({args.folds}-fold CV):")
    means, stds = [], []
    for d in args.depths:
        sc = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0),
                             X, y, cv=cv, scoring="accuracy")
        means.append(sc.mean()); stds.append(sc.std())
        print(f"  depth={d:2d}: {sc.mean():.3f} +/- {sc.std():.3f}")
    best = args.depths[int(np.argmax(means))]
    print(f"--> best depth = {best}")

    tree = DecisionTreeClassifier(max_depth=best, random_state=0).fit(X, y)
    imp = pd.Series(tree.feature_importances_, index=feats).sort_values(ascending=False)
    print(f"\nfinal tree (depth {best}) feature importances:")
    print(imp.round(3).to_string())

    print("\ntree structure (first 3 levels):")
    print(export_text(
        DecisionTreeClassifier(max_depth=min(3, best), random_state=0).fit(X, y),
        feature_names=feats))

    lr_fit = LogisticRegression(max_iter=2000).fit(Xs, y)
    print("logistic-regression coefficients (standardized features):")
    print(pd.Series(lr_fit.coef_[0], index=feats).round(3).sort_values().to_string())

    root_feat = feats[tree.tree_.feature[0]]
    root_thr = float(tree.tree_.threshold[0])
    print(f"\ntree's first split:  {root_feat} <= {root_thr:.3f}")
    print(f"hand-coded 2x2 rule first cut:  vpip <= 0.28 (tight) vs > 0.28 (loose)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].errorbar(args.depths, means, yerr=stds, fmt="o-", capsize=4,
                   label="decision tree")
    ax[0].axhline(s_lr.mean(), color="green", ls="--",
                  label=f"log. reg. ({s_lr.mean():.3f})")
    ax[0].axhline(maj, color="grey", ls=":",
                  label=f"majority baseline ({maj:.3f})")
    ax[0].axvline(best, color="black", ls=":", alpha=.4)
    ax[0].set(xlabel="tree max_depth", ylabel="CV accuracy",
              title=f"model selection ({args.folds}-fold CV) -- {tag}")
    ax[0].legend()
    ax[1].barh(imp.index[::-1], imp.values[::-1], color="#2980b9")
    ax[1].set(title=f"decision-tree (depth {best}) feature importances",
              xlabel="Gini importance")
    fig.tight_layout()
    base_name = "supervised_extended" if args.extended else "supervised"
    if args.exclude:
        base_name += "_no_" + "_".join(args.exclude)
    out = f"{base_name}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
