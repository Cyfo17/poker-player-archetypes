#!/usr/bin/env python3
"""Extract per-player statistics from IRC Poker Database pdb files.

Input rows: nick timestamp n_dealt pos preflop flop turn river bankroll bet won
            [card1 card2]
Action codes: B blind, f fold, k check, b bet, c call, r raise, A all-in,
              Q quit, K kicked, - no action.

Rates are per hand unless conditional on actions or reaching a street.
three_bet_rate counts hands with multiple preflop raises; cbet and donk use
flop actions conditioned on preflop aggression or calling.
net_per_hand is winnings minus bets, averaged over hands.

Usage: python src/parse_features.py data/holdem -o players.csv"""
from __future__ import annotations
import argparse, csv, glob, os
from collections import defaultdict

VOLUNTARY_PRE = set("cbrA")
RAISE_PRE     = set("brA")
AGGR          = set("brA")
CALL          = set("c")
FOLD          = set("f")


def street_counts(s: str):
    if not s or s == "-":
        return 0, 0, 0
    return (sum(c in AGGR for c in s),
            sum(c in CALL for c in s),
            sum(c in FOLD for c in s))


class P:
    __slots__ = (
        "hands", "vpip", "pfr", "saw_flop",
        "agg", "calls", "folds", "showdown", "net",
        "allin", "limp", "multi_raise",
        "pre_aggr_flop", "cbet",
        "caller_flop", "donk",
        "river_agg", "river_calls", "river_folds",
    )

    def __init__(self):
        self.hands = self.vpip = self.pfr = self.saw_flop = 0
        self.agg = self.calls = self.folds = self.showdown = 0
        self.net = 0.0
        self.allin = self.limp = self.multi_raise = 0
        self.pre_aggr_flop = self.cbet = 0
        self.caller_flop = self.donk = 0
        self.river_agg = self.river_calls = self.river_folds = 0


def parse_file(path, players):
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                bet, won = float(parts[9]), float(parts[10])
            except ValueError:
                continue
            pre, flop, turn, river = parts[4], parts[5], parts[6], parts[7]
            p = players[parts[0]]
            p.hands += 1
            p.net += won - bet

            if any(c in VOLUNTARY_PRE for c in pre):
                p.vpip += 1
            if any(c in RAISE_PRE for c in pre):
                p.pfr += 1
            saw_flop_this = bool(flop) and flop != "-"
            if saw_flop_this:
                p.saw_flop += 1

            for s in (flop, turn, river):
                a, c, f = street_counts(s)
                p.agg += a; p.calls += c; p.folds += f

            if len(parts) > 11:
                p.showdown += 1

            if any("A" in s for s in (pre, flop, turn, river)):
                p.allin += 1

            pre_aggr   = any(c in "brA" for c in pre)
            pre_caller = ("c" in pre) and not pre_aggr
            if pre_caller:
                p.limp += 1
            if pre.count("r") + pre.count("A") >= 2:
                p.multi_raise += 1

            bet_flop_first = saw_flop_this and len(flop) > 0 and flop[0] == "b"
            if pre_aggr and saw_flop_this:
                p.pre_aggr_flop += 1
                if bet_flop_first:
                    p.cbet += 1
            if pre_caller and saw_flop_this:
                p.caller_flop += 1
                if bet_flop_first:
                    p.donk += 1

            ra, rc, rf = street_counts(river)
            p.river_agg += ra; p.river_calls += rc; p.river_folds += rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with the extracted IRCdata")
    ap.add_argument("-o", "--out", default="players.csv")
    ap.add_argument("--min-hands", type=int, default=1)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.root, "**", "pdb", "*"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(args.root, "**", "pdb.*"), recursive=True)
    print(f"found {len(files)} player files")

    players = defaultdict(P)
    skipped = 0
    for i, f in enumerate(files):
        if not os.path.isfile(f):
            continue
        try:
            parse_file(f, players)
        except OSError:
            skipped += 1
            continue
        if i % 5000 == 0:
            print(f"  {i}/{len(files)} files...")
    if skipped:
        print(f"skipped {skipped} unreadable files")

    kept = 0
    with open(args.out, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow([
            "nick", "n_hands",
            "vpip", "pfr", "wsf", "agg_freq", "af", "wtsd", "net_per_hand",
            "all_in_rate", "river_agg_freq", "limp_rate",
            "three_bet_rate", "cbet", "donk", "pfr_vpip_ratio",
        ])
        for nick, p in players.items():
            if p.hands < args.min_hands:
                continue
            denom = p.agg + p.calls + p.folds
            agg_freq = p.agg / denom if denom else 0.0
            af = p.agg / p.calls if p.calls else ""

            rdenom = p.river_agg + p.river_calls + p.river_folds
            river_agg_freq = p.river_agg / rdenom if rdenom else 0.0
            cbet = p.cbet / p.pre_aggr_flop if p.pre_aggr_flop else ""
            donk = p.donk / p.caller_flop if p.caller_flop else ""
            pfr_vpip = p.pfr / p.vpip if p.vpip else ""

            w.writerow([
                nick, p.hands,
                p.vpip / p.hands, p.pfr / p.hands, p.saw_flop / p.hands,
                agg_freq, af, p.showdown / p.hands, p.net / p.hands,
                p.allin / p.hands, river_agg_freq, p.limp / p.hands,
                p.multi_raise / p.hands, cbet, donk, pfr_vpip,
            ])
            kept += 1
    print(f"wrote {kept} players to {args.out}")


if __name__ == "__main__":
    main()
