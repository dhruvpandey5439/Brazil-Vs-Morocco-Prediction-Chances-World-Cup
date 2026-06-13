# Brazil vs Morocco — World Cup Match Predictor

This is a machine learning pipeline that predicts the outcome of the **Brazil vs Morocco** 2026 FIFA World Cup match using historical international football data, logistic regression, and statistical feature engineering — inspired by quantitative sports modelling techniques used in financial prediction systems, and quantatative finance modeling.

---

## Overview

This project applies a full supervised learning pipeline to 44,000+ international football fixtures dating back to 1872. By treating each team's match history as a time series of outcomes, the model engineers predictive features, trains a calibrated classifier, and outputs win/draw/loss probabilities — similar in structure to a quant finance signal model.

---

## How It Works

### 1. Data Ingestion
Data is pulled directly from the open-source dataset maintained by [@martj42](https://github.com/martj42/international_results), which contains every recorded international football match. No API key required.

```
Columns: date, home_team, away_team, home_score, away_score, tournament, neutral
```

---

### 2. Data Parsing
Each raw fixture is rotated into a **team-perspective DataFrame** — so `goals_for`, `goals_against`, and `result` are always from that team's point of view, regardless of whether they were home or away.

---

### 3. Feature Engineering

| Feature | Description |
|---|---|
| `win_rate` | Overall historical win percentage |
| `draw_rate` | Overall historical draw percentage |
| `loss_rate` | Overall historical loss percentage |
| `avg_goals_for` | Mean goals scored per match |
| `avg_goals_against` | Mean goals conceded per match |
| `avg_goal_diff` | Mean goal differential per match |
| `recent_win_rate` | Win % across last 20 matches (form) |
| `recent_avg_gd` | Avg goal diff across last 20 matches |
| `wc_win_rate` | Win % in FIFA World Cup fixtures only |
| `is_home` | Binary home/away indicator |
| `neutral` | Binary neutral venue indicator |

---

### 4. Standard Scaling

Before training, all features are normalised using **StandardScaler** so that no single feature dominates due to differing scales.

$$z = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the feature mean and $\sigma$ is the standard deviation across the training set.

---

### 5. Logistic Regression Model

A **multinomial Logistic Regression** classifier is trained independently on each team's match history.

The model learns to predict the probability of each outcome class $k \in \{Win, Draw, Loss\}$ using the **softmax function**:

$$P(Y = k \mid X) = \frac{e^{X \cdot w_k}}{\sum_{j} e^{X \cdot w_j}}$$

Optimiser: **L-BFGS** | Regularisation: **L2 (C=1.0)** | Max iterations: **1000**

Each team's model is trained on that team's historical match data (80% train / 20% test split, stratified).

---

### 6. Cross-Validation

Model reliability is measured using **5-fold stratified cross-validation** to ensure the accuracy score is not a fluke of a single train/test split.

$$CV_{score} = \frac{1}{k} \sum_{i=1}^{k} \text{Accuracy}_i$$

---

### 7. Probability Blending

A neutral-ground feature vector is constructed for each team and passed through their respective trained model. The two teams' draw probabilities are averaged, and all three outcomes are re-normalised so they sum to 1.

$$P_{final}(Win_A) = \frac{P_A(Win)}{P_A(Win) + \bar{P}(Draw) + P_B(Win)}$$

---

## Dashboard Output

Running the model generates a **10-panel visual dashboard** saved as `charts/dashboard.png`:

| Panel | Description |
|---|---|
| 1 | Win / Draw / Loss rate comparison |
| 2 | Average goals scored vs conceded |
| 3 | Win rate breakdown (overall / recent / World Cup) |
| 4 | Brazil win rate over time (30-match rolling) |
| 5 | Morocco win rate over time (30-match rolling) |
| 6 | Goals scored distribution (histogram) |
| 7 | Feature correlation heatmap |
| 8 | Model confusion matrix |
| 9 | 5-fold cross-validation accuracy |
| 10 | Final prediction probability bar |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/brazil-morocco-predictor
cd brazil-morocco-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the model**
```bash
python model.py
```

The dashboard will open on screen and be saved to `charts/dashboard.png`.

---

## Data Source

- **Dataset:** [martj42/international_results](https://github.com/martj42/international_results)
- **Coverage:** 1872 – present, 44,000+ international fixtures
- **License:** Open Data

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data ingestion |
| `numpy` | Numerical calculations |
| `scikit-learn` | StandardScaler, LogisticRegression, cross-validation |
| `matplotlib` | Dashboard visualisation |
| `seaborn` | Heatmap and statistical plots |

---

## Disclaimer

This project is built for educational reasons and also because I have a passion for it. Predictions are probabilistic estimations based on historical data and do not account for real-time factors such as squad fitness, tactics, or referee decisions. Don't take this project to account for any bets or expectations of a team winning over the other, football will remain wonderfully unpredictable.

---

## Reflection

This was a very tedious project, as it is my first time interacting with Machine Learning, and understanding coding in a deepr level. However, with my passion and love for football, this tedious process became entertaining, and became something I found as valuable. Right as summer break started, which for me was June 3rd, I didn't waste anytime and tried to take action for something I was quite interested in for a very long time. The process was not perfect, and I became frustrated more than I can count, however I am satisfied with my project and now I am aiming to broaden this model into all teams and matches and not only Brazil and Morocco.
