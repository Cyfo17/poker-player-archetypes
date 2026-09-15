# Poker Player Archetypes

A group project developed for the Machine Learning course (CSC_2S004_EP) at École Polytechnique.

This project studies whether poker players form distinct playing-style groups and how those styles relate to winnings. It extracts player statistics from historical hand logs, compares clustering methods, and evaluates models that distinguish players with high and low observed win rates.

## Data and methods

The analysis uses the limit hold'em subset of the **IRC Poker Database**, covering games from 1995–2001. The original run aggregated **23,269 nicknames**, including **10,599 with at least 100 hands** for clustering. These are play-money games, and a nickname does not necessarily identify a single person.

Players are represented by statistics such as:

- **VPIP:** fraction of hands with voluntary preflop participation.
- **PFR:** fraction of hands with a preflop raise.
- **Aggression frequency:** aggressive actions as a fraction of postflop bets, raises, calls, and folds, including all-ins as aggressive actions.
- **WTSD:** fraction of hands with recorded showdown cards.
- **Passivity gap:** VPIP minus PFR.

The clustering pipeline standardizes four style features (VPIP, PFR, aggression frequency, and WTSD), uses PCA for visualization, and compares k-means with Gaussian mixture models. Silhouette scores and BIC guide the cluster-count comparison. Spectral clustering is also available for samples of up to 4,000 players.

Winnings are excluded from the clustering inputs and examined afterward. A fixed VPIP/PFR rule provides a comparison, while repeated k-means runs measure sensitivity to random initialization.

## Results

The figures below are reported values from the original course analysis, not a new benchmark run.

### Playing styles

Silhouette scoring favored **two clusters**, while Gaussian mixture BIC favored approximately **ten components**. This disagreement suggests that there is no single clear number of player types; a continuous range of styles is a plausible interpretation. Four k-means clusters were also examined as a practical summary.

For that four-cluster solution:

- The silhouette score was **0.310**, compared with **0.081** for the fixed poker rules. Higher values indicate better separation in the selected feature space.
- Across ten random seeds, mean pairwise adjusted Rand index was **0.982**, indicating very similar assignments across runs.
- The two more selective groups had median net winnings of **+0.61** and **+0.81** per hand. The looser groups had medians of **−2.19** and **−5.05**.

These winnings are in the dataset's play-money units, not dollars or big blinds. The associations do not establish that a style causes a particular outcome.

### Classification and winnings

Classification used **3,326 players with at least 500 hands**, split evenly between the top and bottom win-rate tertiles. The middle tertile was excluded.

With five-fold cross-validation, the original run reported:

- **Majority-class baseline:** 50.0% accuracy.
- **Logistic regression:** 72.2% accuracy, with a fold standard deviation of 2.0 percentage points.
- **Decision tree, depth 4:** 71.7% accuracy, with a fold standard deviation of 1.1 percentage points.

The tree's first split used the passivity gap, suggesting that the balance between participation and raising was useful for distinguishing the two groups.

Linear regression on the four core style features gave an **in-sample R² of 0.191** for winnings clipped at the 1st and 99th percentiles. Style captured some variation in observed winnings, but much remained unexplained; this score alone cannot identify how much is due to luck.

### Additional features

Adding seven style features raised logistic regression accuracy to **82.0%**. Most of that improvement disappeared when `all_in_rate` was removed: accuracy fell to **73.3%**.

All-in frequency may reflect depleted stacks after losses rather than an independent playing preference. This makes the higher score difficult to interpret as evidence about playing style. Excluding it left a modest improvement over the base model. The corresponding regression R² changed from **0.191** to **0.396** with the extended features and **0.216** without all-in frequency.

## Run the project

Install Python 3 and the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the archive from the [IRC Poker Database](http://poker.cs.ualberta.ca/irc_poker_database.html). Extract the outer archive, then extract its base `holdem.*.tgz` monthly archives into `data/holdem/`. The parser expects the extracted player files, not the compressed monthly archives.

Run these commands from the repository root:

```bash
python src/parse_features.py data/holdem -o players.csv
python src/cluster.py players.csv --min-hands 100 --kmax 12 --force-k 4
python src/baseline.py players_labeled.csv
python src/winnings.py players.csv --labeled players_labeled.csv
python src/supervised.py players.csv
```

To repeat the feature comparison:

```bash
python src/supervised.py players.csv --extended
python src/supervised.py players.csv --extended --exclude all_in_rate
python src/winnings.py players.csv --labeled players_labeled.csv --extended
python src/winnings.py players.csv --labeled players_labeled.csv --extended --exclude all_in_rate
```

The scripts print summaries and save generated CSV files and PNG charts in the working directory. Data and generated outputs are excluded from version control.

For a quick check without downloading the dataset:

```bash
python src/cluster.py --demo
python src/baseline.py --demo
```

Demo mode uses synthetic data and does not reproduce the results above. Each script supports `--help`.

## Limitations

The dataset is historical, uses play money, and may contain incomplete logs or reused nicknames. Its results may not transfer to modern real-money poker. Features and winnings are aggregated over the same observation period, so classification measures an association rather than prediction of future performance.

The reported classification scores use the original evaluation procedure: logistic-regression scaling is fitted before cross-validation, and tree depth is selected using the same folds used to report its score. A stricter performance estimate would fit preprocessing within each fold and use a separate test set or nested cross-validation for model selection.

## References

- Michael Maurer and the University of Alberta Computer Poker Research Group: [IRC Poker Database](http://poker.cs.ualberta.ca/irc_poker_database.html).
- Allen Lin: [PokerHandsDataset](https://github.com/allenfrostline/PokerHandsDataset), related parsing work.
