=============================================================
  World Cup Match Predictor  |  Brazil vs Morocco
  Author  : [Your Name]
  Date    : 2026
  Dataset : github.com/martj42/international_results
=============================================================

Pipeline
--------
  1. Ingest open-source international match data (44 000+ fixtures)
  2. Filter and parse per-team match history into a DataFrame
  3. Engineer features: win rate, goal differential, recent form,
     World Cup form, home advantage
  4. Scale features with StandardScaler (zero mean, unit variance)
  5. Train a Logistic Regression classifier per team
  6. Validate with 5-fold stratified cross-validation
  7. Blend team-perspective probabilities and output prediction
  8. Produce a full 10-panel visual dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings, os
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
DATA_URL   = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
TEAM_A     = "Brazil"
TEAM_B     = "Morocco"
COLOR_A    = "#009C3B"   # Brazil green
COLOR_B    = "#C1272D"   # Morocco red
COLOR_DRAW = "#888888"
CHART_DIR  = "charts"
os.makedirs(CHART_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 1. DATA INGESTION
# ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Download the full international results dataset.
    Source: github.com/martj42/international_results
    Columns: date, home_team, away_team, home_score, away_score,
             tournament, city, country, neutral
    """
    print("[1/6] Downloading dataset ...")
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"])
    # ── FIX: drop rows where scores are missing so int() never sees NaN ──
    df = df.dropna(subset=["home_score", "away_score"])
    print(f"      Loaded {len(df):,} international matches\n")
    return df


# ──────────────────────────────────────────────────────────────
# 2. PARSING  —  build a per-team perspective DataFrame
# ──────────────────────────────────────────────────────────────
def get_team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Filter all matches for *team* and rotate every row so that
    goals_for / goals_against / result are always from that
    team's point of view.
    """
    mask    = (df["home_team"] == team) | (df["away_team"] == team)
    team_df = df[mask].copy()
    rows    = []

    for _, row in team_df.iterrows():
        is_home = (row["home_team"] == team)
        gf = row["home_score"] if is_home else row["away_score"]
        ga = row["away_score"] if is_home else row["home_score"]

        # ── FIX: skip any row that still has NaN scores ──
        if pd.isna(gf) or pd.isna(ga):
            continue

        result = "W" if gf > ga else ("L" if gf < ga else "D")

        rows.append({
            "date":          row["date"],
            "year":          row["date"].year,
            "tournament":    row["tournament"],
            "is_home":       int(is_home),
            "goals_for":     int(gf),
            "goals_against": int(ga),
            "goal_diff":     int(gf - ga),
            "result":        result,
            "neutral":       int(row["neutral"]),
        })

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"      {team}: {len(out)} matches in dataset")
    return out


# ──────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, team: str) -> dict:
    """
    Compute aggregate and rolling statistics used as model inputs.

    Features
    --------
    win_rate          overall historical win percentage
    draw_rate         overall draw percentage
    loss_rate         overall loss percentage
    avg_goals_for     mean goals scored per match
    avg_goals_against mean goals conceded per match
    avg_goal_diff     mean goal differential per match
    recent_win_rate   win % across last 20 matches (form)
    recent_avg_gd     avg goal diff across last 20 matches
    wc_win_rate       win % in FIFA World Cup fixtures only
    matches_played    total matches in dataset
    """
    n      = len(df)
    wins   = (df["result"] == "W").sum()
    draws  = (df["result"] == "D").sum()
    losses = (df["result"] == "L").sum()
    recent = df.tail(20)
    wc     = df[df["tournament"].str.contains("FIFA World Cup", na=False)]

    feats = {
        "win_rate":          wins / n,
        "draw_rate":         draws / n,
        "loss_rate":         losses / n,
        "avg_goals_for":     df["goals_for"].mean(),
        "avg_goals_against": df["goals_against"].mean(),
        "avg_goal_diff":     df["goal_diff"].mean(),
        "recent_win_rate":   (recent["result"] == "W").mean(),
        "recent_avg_gd":     recent["goal_diff"].mean(),
        "wc_win_rate":       (wc["result"] == "W").mean() if len(wc) else 0,
        "matches_played":    n,
        "total_wins":        int(wins),
        "total_draws":       int(draws),
        "total_losses":      int(losses),
    }

    print(f"      {team}: {wins}W / {draws}D / {losses}L  |  "
          f"{feats['avg_goals_for']:.2f} scored / {feats['avg_goals_against']:.2f} conceded per match  |  "
          f"WC win rate {feats['wc_win_rate']:.1%}")
    return feats


# ──────────────────────────────────────────────────────────────
# 4 & 5. STANDARD SCALING + LOGISTIC REGRESSION
# ──────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, team: str):
    """
    StandardScaler
    --------------
    Transforms each feature x to z = (x - mean) / std
    so that no single feature dominates due to scale.

    Logistic Regression
    -------------------
    Models P(Y = k | X) via the softmax function:
        P(Y=k|X) = exp(X * w_k) / sum_j[ exp(X * w_j) ]
    Optimised with L-BFGS.  Regularisation C=1.0.
    Labels: 1 = Win, 0 = Draw, -1 = Loss
    """
    df = df.copy()
    df["label"]  = df["result"].map({"W": 1, "D": 0, "L": -1})
    FEATURES     = ["is_home", "goals_for", "goals_against", "goal_diff", "neutral"]
    df           = df.dropna(subset=FEATURES + ["label"])
    X, y         = df[FEATURES], df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=1000, C=1.0, random_state=42
    )
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    acc    = accuracy_score(y_te, y_pred)
    cv     = cross_val_score(
        model, scaler.transform(X), y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    print(f"      {team} — test acc: {acc:.1%}  |  "
          f"5-fold CV: {cv.mean():.1%} +/- {cv.std():.2%}")

    return model, scaler, X_te_s, y_te, y_pred, acc, cv


# ──────────────────────────────────────────────────────────────
# 6. PROBABILITY BLENDING
# ──────────────────────────────────────────────────────────────
def get_probabilities(bf, mf, bmod, bscl, mmod, mscl):
    """
    Build a neutral-ground feature vector for each team,
    run it through that team's calibrated model, then blend
    and normalise so probabilities sum to 1.
    """
    def make_input(t_feats, o_feats):
        return pd.DataFrame([{
            "is_home":       0,
            "goals_for":     t_feats["avg_goals_for"],
            "goals_against": o_feats["avg_goals_for"],
            "goal_diff":     t_feats["avg_goal_diff"] - o_feats["avg_goal_diff"],
            "neutral":       1,
        }])

    def prob(model, scaler, feats, opp, label):
        X   = scaler.transform(make_input(feats, opp))
        prb = model.predict_proba(X)[0]
        cls = list(model.classes_)
        return prb[cls.index(label)] if label in cls else 0.0

    b_win  = prob(bmod, bscl, bf, mf,  1)
    b_draw = prob(bmod, bscl, bf, mf,  0)
    m_win  = prob(mmod, mscl, mf, bf,  1)
    m_draw = prob(mmod, mscl, mf, bf,  0)

    draw_blend = (b_draw + m_draw) / 2
    total      = b_win + draw_blend + m_win
    return b_win/total, draw_blend/total, m_win/total


# ──────────────────────────────────────────────────────────────
# 7. VISUALISATION DASHBOARD
# ──────────────────────────────────────────────────────────────
def plot_dashboard(bdf, mdf, bf, mf,
                   bmod, bscl, mmod, mscl,
                   bXte, byte_, bypred,
                   mXte, myte, mypred,
                   bacc, macc, bcv, mcv,
                   b_win, draw, m_win):

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        "World Cup 2026  |  Brazil vs Morocco  |  ML Prediction Dashboard",
        fontsize=20, fontweight="bold", color="white", y=0.995
    )
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)
    lbl = {1: "Win", 0: "Draw", -1: "Loss"}
    w   = 0.35

    def style_ax(ax):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Panel 1: Win / Draw / Loss ──
    ax1 = fig.add_subplot(gs[0, 0]); style_ax(ax1)
    cats   = ["Win %", "Draw %", "Loss %"]
    bv     = [bf["win_rate"]*100, bf["draw_rate"]*100, bf["loss_rate"]*100]
    mv     = [mf["win_rate"]*100, mf["draw_rate"]*100, mf["loss_rate"]*100]
    x      = np.arange(3)
    b1     = ax1.bar(x-w/2, bv, w, color=COLOR_A, alpha=0.9, label="Brazil")
    b2     = ax1.bar(x+w/2, mv, w, color=COLOR_B, alpha=0.9, label="Morocco")
    ax1.set_xticks(x); ax1.set_xticklabels(cats, color="white")
    ax1.set_ylabel("Percentage (%)", color="white")
    ax1.set_title("Overall Win / Draw / Loss Rate", color="white", fontweight="bold")
    ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)
    for bar in list(b1) + list(b2):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 color="white", fontsize=8)

    # ── Panel 2: Goals scored vs conceded ──
    ax2 = fig.add_subplot(gs[0, 1]); style_ax(ax2)
    x2 = np.arange(2)
    sc = [bf["avg_goals_for"], mf["avg_goals_for"]]
    co = [bf["avg_goals_against"], mf["avg_goals_against"]]
    g1 = ax2.bar(x2-w/2, sc, w, color="#FFD700", alpha=0.9, label="Avg Scored")
    g2 = ax2.bar(x2+w/2, co, w, color="#FF6B6B", alpha=0.9, label="Avg Conceded")
    ax2.set_xticks(x2); ax2.set_xticklabels(["Brazil","Morocco"], color="white")
    ax2.set_ylabel("Goals per Match", color="white")
    ax2.set_title("Avg Goals Scored vs Conceded", color="white", fontweight="bold")
    ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
    for bar in list(g1)+list(g2):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom",
                 color="white", fontsize=8)

    # ── Panel 3: Win rate breakdown ──
    ax3 = fig.add_subplot(gs[0, 2]); style_ax(ax3)
    mets = ["Overall\nWin %", "Recent\nWin %", "World Cup\nWin %"]
    bm   = [bf["win_rate"]*100, bf["recent_win_rate"]*100, bf["wc_win_rate"]*100]
    mm   = [mf["win_rate"]*100, mf["recent_win_rate"]*100, mf["wc_win_rate"]*100]
    x3   = np.arange(3)
    ax3.bar(x3-w/2, bm, w, color=COLOR_A, alpha=0.9, label="Brazil")
    ax3.bar(x3+w/2, mm, w, color=COLOR_B, alpha=0.9, label="Morocco")
    ax3.set_xticks(x3); ax3.set_xticklabels(mets, color="white", fontsize=8)
    ax3.set_ylabel("Win Rate (%)", color="white")
    ax3.set_title("Win Rate Breakdown", color="white", fontweight="bold")
    ax3.legend(facecolor="#222", labelcolor="white", fontsize=8)

    # ── Panel 4: Brazil rolling win rate ──
    ax4 = fig.add_subplot(gs[1, 0]); style_ax(ax4)
    bdf["roll"] = (bdf["result"]=="W").rolling(30).mean()*100
    ax4.plot(bdf["date"], bdf["roll"], color=COLOR_A, linewidth=2)
    ax4.fill_between(bdf["date"], bdf["roll"], alpha=0.2, color=COLOR_A)
    ax4.set_title("Brazil Win Rate Over Time\n(30-match rolling avg)", color="white", fontweight="bold")
    ax4.set_ylabel("Win Rate (%)", color="white"); ax4.set_xlabel("Year", color="white")

    # ── Panel 5: Morocco rolling win rate ──
    ax5 = fig.add_subplot(gs[1, 1]); style_ax(ax5)
    mdf["roll"] = (mdf["result"]=="W").rolling(30).mean()*100
    ax5.plot(mdf["date"], mdf["roll"], color=COLOR_B, linewidth=2)
    ax5.fill_between(mdf["date"], mdf["roll"], alpha=0.2, color=COLOR_B)
    ax5.set_title("Morocco Win Rate Over Time\n(30-match rolling avg)", color="white", fontweight="bold")
    ax5.set_ylabel("Win Rate (%)", color="white"); ax5.set_xlabel("Year", color="white")

    # ── Panel 6: Goals scored distribution ──
    ax6 = fig.add_subplot(gs[1, 2]); style_ax(ax6)
    ax6.hist(bdf["goals_for"], bins=range(0,12), alpha=0.7,
             color=COLOR_A, label="Brazil",  density=True)
    ax6.hist(mdf["goals_for"], bins=range(0,12), alpha=0.7,
             color=COLOR_B, label="Morocco", density=True)
    ax6.set_title("Goals Scored Distribution", color="white", fontweight="bold")
    ax6.set_xlabel("Goals in a Match", color="white")
    ax6.set_ylabel("Density", color="white")
    ax6.legend(facecolor="#222", labelcolor="white", fontsize=8)

    # ── Panel 7: Feature correlation heatmap ──
    ax7 = fig.add_subplot(gs[2, 0]); style_ax(ax7)
    fdf = bdf[["goals_for","goals_against","goal_diff","is_home","neutral"]].copy()
    fdf["result_num"] = bdf["result"].map({"W":1,"D":0,"L":-1})
    sns.heatmap(fdf.corr(), ax=ax7, cmap="RdYlGn", annot=True, fmt=".2f",
                linewidths=0.5, annot_kws={"size":8},
                cbar_kws={"shrink":0.8})
    ax7.set_title("Feature Correlation Heatmap\n(Brazil)", color="white", fontweight="bold")
    ax7.tick_params(colors="white", labelsize=7)

    # ── Panel 8: Confusion matrix ──
    ax8 = fig.add_subplot(gs[2, 1]); style_ax(ax8)
    labs = sorted(set(byte_)|set(bypred))
    cm   = confusion_matrix(byte_, bypred, labels=labs)
    sns.heatmap(cm, ax=ax8, annot=True, fmt="d", cmap="Blues",
                xticklabels=[lbl[l] for l in labs],
                yticklabels=[lbl[l] for l in labs],
                linewidths=0.5)
    ax8.set_title(f"Brazil Model — Confusion Matrix\nTest Accuracy: {bacc:.1%}",
                  color="white", fontweight="bold")
    ax8.set_xlabel("Predicted", color="white")
    ax8.set_ylabel("Actual",    color="white")
    ax8.tick_params(colors="white")

    # ── Panel 9: Cross-validation ──
    ax9 = fig.add_subplot(gs[2, 2]); style_ax(ax9)
    folds = [f"Fold {i+1}" for i in range(5)]
    x9    = np.arange(5)
    ax9.bar(x9-w/2, bcv*100, w, color=COLOR_A, alpha=0.9, label="Brazil")
    ax9.bar(x9+w/2, mcv*100, w, color=COLOR_B, alpha=0.9, label="Morocco")
    ax9.axhline(bcv.mean()*100, color=COLOR_A, linestyle="--", linewidth=1.5,
                label=f"Brazil avg {bcv.mean():.1%}")
    ax9.axhline(mcv.mean()*100, color=COLOR_B, linestyle="--", linewidth=1.5,
                label=f"Morocco avg {mcv.mean():.1%}")
    ax9.set_xticks(x9); ax9.set_xticklabels(folds, color="white", fontsize=8)
    ax9.set_ylabel("Accuracy (%)", color="white")
    ax9.set_title("5-Fold Cross-Validation Accuracy", color="white", fontweight="bold")
    ax9.legend(facecolor="#222", labelcolor="white", fontsize=7)

    # ── Panel 10: Final prediction ──
    ax10 = fig.add_subplot(gs[3, :]); ax10.axis("off")
    ax10.set_facecolor("#0d0d0d")
    ax10.set_xlim(0,1); ax10.set_ylim(0,1)

    by, bh = 0.3, 0.28
    ax10.add_patch(plt.Rectangle((0.05, by),           b_win*0.9,  bh, color=COLOR_A, zorder=3))
    ax10.add_patch(plt.Rectangle((0.05+b_win*0.9, by), draw*0.9,   bh, color=COLOR_DRAW, zorder=3))
    ax10.add_patch(plt.Rectangle((0.05+(b_win+draw)*0.9, by), m_win*0.9, bh, color=COLOR_B, zorder=3))

    def label_bar(cx, val, text):
        if val > 0.06:
            ax10.text(cx, by+bh/2, text, ha="center", va="center",
                      color="white", fontsize=13, fontweight="bold", zorder=4)

    label_bar(0.05+b_win*0.9/2,                   b_win, f"Brazil\n{b_win:.1%}")
    label_bar(0.05+(b_win+draw/2)*0.9,             draw,  f"Draw\n{draw:.1%}")
    label_bar(0.05+(b_win+draw+m_win/2)*0.9,       m_win, f"Morocco\n{m_win:.1%}")

    winner, wprob, wcol = max(
        [("Brazil", b_win, COLOR_A), ("Draw", draw, COLOR_DRAW), ("Morocco", m_win, COLOR_B)],
        key=lambda x: x[1]
    )
    ax10.text(0.5, 0.88, "FINAL PREDICTION", ha="center", va="center",
              color="white", fontsize=17, fontweight="bold")
    ax10.text(0.5, 0.15, f"Most likely outcome:  {winner}  ({wprob:.1%} probability)",
              ha="center", va="center", color=wcol, fontsize=14, fontweight="bold")
    ax10.text(0.5, 0.05,
              "Model: Logistic Regression  |  Scaling: StandardScaler  |  "
              "Validation: 5-fold Stratified CV  |  Data: martj42/international_results",
              ha="center", va="center", color="#666", fontsize=8)

    path = os.path.join(CHART_DIR, "dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"\n      Dashboard saved -> {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Brazil vs Morocco  |  World Cup ML Predictor")
    print("=" * 60)

    df = load_data()

    print("[2/6] Parsing team match histories ...")
    bdf = get_team_matches(df, TEAM_A)
    mdf = get_team_matches(df, TEAM_B)

    print("\n[3/6] Engineering features ...")
    bf  = build_features(bdf, TEAM_A)
    mf  = build_features(mdf, TEAM_B)

    print("\n[4/6] Scaling + training Logistic Regression models ...")
    bmod, bscl, bXte, byte_, bypred, bacc, bcv = train_model(bdf, TEAM_A)
    mmod, mscl, mXte, myte,  mypred, macc, mcv = train_model(mdf, TEAM_B)

    print("\n[5/6] Computing match probabilities ...")
    b_win, draw, m_win = get_probabilities(bf, mf, bmod, bscl, mmod, mscl)
    print(f"      Brazil wins : {b_win:.1%}")
    print(f"      Draw        : {draw:.1%}")
    print(f"      Morocco wins: {m_win:.1%}")

    print("\n[6/6] Rendering dashboard ...")
    plot_dashboard(bdf, mdf, bf, mf,
                   bmod, bscl, mmod, mscl,
                   bXte, byte_, bypred,
                   mXte, myte,  mypred,
                   bacc, macc, bcv, mcv,
                   b_win, draw, m_win)

    print("\nDone.")

if __name__ == "__main__":
    main()
