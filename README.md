# Visible Polarization Paper Data

This repository provides the main numerical data and selected analysis code for the manuscript:

**From Private Opinions to Visible Polarization: A Layered Diagnostic Framework for Online Public Opinion**

## Contents

- `data/figure2_baseline`–`data/figure9_empirical`: source data for the main result figures.
- `data/validation`: summary results from the human validation of tweet stance labels.
- `code/src/m0_lps_epo`: core implementation of the layered visible-bias model (LVBM).
- `code/scripts`: selected simulation, plotting, and empirical-analysis scripts.
- `code/config/baseline.yaml`: baseline simulation configuration.

The empirical results use the 2020 ANES Time Series Study and the US Election 2020 Tweets dataset. To protect data subjects and respect source-data conditions, this repository does not redistribute respondent-level ANES records, tweet text, usernames, or tweet/user identifiers. The released CSV files contain aggregate estimates and figure-ready numerical data.

The ANES and Twitter analyses are based on independent samples and provide complementary evidence for different layers of the framework. Retweet-weighted visibility is used as a proxy for platform visibility rather than a direct measure of impressions.

## Data availability statement

The data underlying the main findings and the principal analysis code are publicly available at https://github.com/Zhaolin0912/visible-polarization-paper-data.

## Upstream data

- ANES 2020 Time Series Study: https://electionstudies.org/data-center/2020-time-series-study/
- US Election 2020 Tweets: https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets

The full Monte Carlo experiments can be computationally intensive. The archived CSV files are the data used for the reported figures and estimates.
