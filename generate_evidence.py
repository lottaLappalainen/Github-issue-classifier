"""
generate_evidence.py

Usage:
    python generate_evidence.py
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

# ── setup ──────────────────────────────────────────────────────────────────
ROOT     = Path(".")
OUT      = ROOT / "evidence"
OUT.mkdir(exist_ok=True)

COLORS   = {"high": "#d32f2f", "medium": "#f57c00", "low": "#388e3c"}
plt.rcParams["figure.dpi"]    = 150
plt.rcParams["font.family"]   = "sans-serif"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

def save(name):
    path = OUT / f"{name}.png"
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {path}")

print("Generating evidence graphs...\n")

# ══════════════════════════════════════════════════════════════════════════
# 1. DATA — Bronze layer
# ══════════════════════════════════════════════════════════════════════════
print("1. Bronze layer...")
bronze = pd.read_parquet(ROOT / "data/bronze/issues_raw.parquet")

fig, ax = plt.subplots(figsize=(13, 6))
repo_counts = bronze["repo"].value_counts()
bars = ax.barh(repo_counts.index[::-1], repo_counts.values[::-1], color="#2E5FA3", alpha=0.85)
ax.set_xlabel("Number of Issues", fontsize=11)
ax.set_title("Raw Issues per Repository — Bronze Layer", fontsize=14, fontweight="bold", pad=15)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(bars, repo_counts.values[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.set_xlabel("Issues", fontsize=11)
plt.tight_layout()
save("01_bronze_issues_per_repo")

# Label coverage
fig, ax = plt.subplots(figsize=(7, 5))
has_label = bronze["labels"].str.strip() != ""
ax.pie([has_label.sum(), (~has_label).sum()],
       labels=[f"Has label\n{has_label.mean():.1%}", f"No label\n{(~has_label).mean():.1%}"],
       colors=["#2E5FA3", "#BBBBBB"], autopct="%1.1f%%", startangle=90,
       textprops={"fontsize": 12})
ax.set_title("Label Coverage in Raw Data", fontsize=13, fontweight="bold")
save("02_bronze_label_coverage")

# ══════════════════════════════════════════════════════════════════════════
# 2. DATA — Silver layer
# ══════════════════════════════════════════════════════════════════════════
print("2. Silver layer...")
silver = pd.read_parquet(ROOT / "data/silver/issues_clean.parquet")
silver["title_len"] = silver["title"].str.split().str.len().fillna(0)
silver["body_len"]  = silver["body"].str.split().str.len().fillna(0)

# Priority distribution before balancing
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
priority_counts = silver["priority"].value_counts()

ax = axes[0]
bars = ax.bar(priority_counts.index, priority_counts.values,
              color=[COLORS[p] for p in priority_counts.index], alpha=0.85, width=0.5)
ax.set_title("Priority Class Distribution — Silver Layer", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Issues")
for bar, val in zip(bars, priority_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{val:,}", ha="center", fontsize=11, fontweight="bold")

ax = axes[1]
ax.pie(priority_counts.values, labels=priority_counts.index,
       colors=[COLORS[p] for p in priority_counts.index],
       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
ax.set_title("Priority Proportions", fontsize=12, fontweight="bold")
plt.suptitle("Class Imbalance Before Balancing", fontsize=13, y=1.02)
plt.tight_layout()
save("03_silver_class_distribution")

# Label hit rate per repo
bronze_per_repo = bronze.groupby("repo").size()
silver_per_repo = silver.groupby("repo").size()
hit_rate = (silver_per_repo / bronze_per_repo).fillna(0).sort_values()

fig, ax = plt.subplots(figsize=(13, 6))
colors_hr = ["#d32f2f" if r < 0.3 else "#f57c00" if r < 0.5 else "#388e3c"
             for r in hit_rate.values]
ax.barh(hit_rate.index, hit_rate.values, color=colors_hr, alpha=0.85)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.6, label="50% line")
ax.set_xlabel("Label Hit Rate")
ax.set_title("Label Hit Rate per Repository", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend()
plt.tight_layout()
save("04_silver_label_hit_rate")

# Text length distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, label in zip(axes, ["title_len", "body_len"], ["Title", "Body"]):
    for p in ["high", "medium", "low"]:
        data = silver[silver["priority"] == p][col].clip(upper=silver[col].quantile(0.95))
        ax.hist(data, bins=30, alpha=0.6, label=p, color=COLORS[p], density=True)
    ax.set_xlabel(f"{label} length (words)")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} Length by Priority", fontsize=11, fontweight="bold")
    ax.legend()
plt.suptitle("Text Length Distributions by Priority Class", fontsize=13, fontweight="bold")
plt.tight_layout()
save("05_silver_text_lengths")

# Distinctive vocabulary
import re
STOP = {"the","a","an","is","in","it","of","to","and","for","on","with","this",
        "that","be","are","was","has","have","not","but","or","at","by","as",
        "from","i","my","we","you","can","if","when","will","no","do","so"}

def top_tokens(series, n=15):
    tokens = []
    for t in series.fillna(""):
        tokens.extend([w for w in re.findall(r"\b[a-z]{3,}\b", t.lower()) if w not in STOP])
    return Counter(tokens).most_common(n)

all_tokens = top_tokens(silver["text"], n=99999)
total = dict(all_tokens)
total_n = sum(total.values())

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, priority in zip(axes, ["high", "medium", "low"]):
    subset = silver[silver["priority"] == priority]
    counts = dict(top_tokens(subset["text"], n=99999))
    n = sum(counts.values())
    scores = {}
    for word, count in counts.items():
        if count < 5:
            continue
        scores[word] = (count / n) / (total.get(word, 1) / total_n)
    top = sorted(scores.items(), key=lambda x: -x[1])[:15]
    words, vals = zip(*top)
    ax.barh(list(words)[::-1], list(vals)[::-1], color=COLORS[priority], alpha=0.85)
    ax.set_title(f"Top tokens\n({priority} priority)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative frequency")
plt.suptitle("Most Distinctive Vocabulary per Priority Class", fontsize=13, fontweight="bold")
plt.tight_layout()
save("06_silver_vocabulary")

# Comments and reactions boxplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, ["comments", "reactions_total"]):
    data = [silver[silver["priority"] == p][col].clip(upper=silver[col].quantile(0.95))
            for p in ["high", "medium", "low"]]
    bp = ax.boxplot(data, labels=["high", "medium", "low"], patch_artist=True)
    for patch, p in zip(bp["boxes"], ["high", "medium", "low"]):
        patch.set_facecolor(COLORS[p]); patch.set_alpha(0.7)
    ax.set_title(f"{col.replace('_',' ').title()} by Priority", fontsize=11, fontweight="bold")
plt.suptitle("Community Engagement by Priority Class", fontsize=12, fontweight="bold")
plt.tight_layout()
save("07_silver_engagement")

# ══════════════════════════════════════════════════════════════════════════
# 3. DATA — Gold layer
# ══════════════════════════════════════════════════════════════════════════
print("3. Gold layer...")
train = pd.read_parquet(ROOT / "data/gold/train.parquet")
test  = pd.read_parquet(ROOT / "data/gold/test.parquet")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, df, title in zip(axes, [train, test], ["Train", "Test"]):
    counts = df["priority"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[COLORS[p] for p in counts.index], alpha=0.85, width=0.5)
    ax.set_title(f"{title} Set — Class Distribution (Gold)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:,}", ha="center", fontsize=11)
plt.suptitle("Gold Layer — Balanced Classes After Upsampling", fontsize=13, fontweight="bold")
plt.tight_layout()
save("08_gold_class_balance")

# Pipeline funnel
fig, ax = plt.subplots(figsize=(9, 5))
stages  = ["Bronze\n(raw)", "Silver\n(labelled)", "Gold train\n(balanced)", "Gold test\n(balanced)"]
sizes   = [len(bronze), len(silver), len(train), len(test)]
colors  = ["#1565C0", "#2E7D32", "#6A1B9A", "#AD1457"]
bars    = ax.bar(stages, sizes, color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f"{val:,}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of rows")
ax.set_title("Data Pipeline Funnel — Rows at Each Layer", fontsize=13, fontweight="bold")
plt.tight_layout()
save("09_pipeline_funnel")

# ══════════════════════════════════════════════════════════════════════════
# 4. MODEL — train classifiers and produce all model graphs
# ══════════════════════════════════════════════════════════════════════════
print("4. Model graphs...")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score,
                              roc_curve, auc)
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize

X_train, y_train = train["text"], train["priority"]
X_test,  y_test  = test["text"],  test["priority"]
LABELS = ["high", "medium", "low"]

def make_lr(C=1.0, max_features=10000):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1,2),
                                  sublinear_tf=True, min_df=2)),
        ("clf",   LogisticRegression(C=C, max_iter=1000, class_weight="balanced", solver="lbfgs"))
    ])

def make_rf(n=100, max_features=10000):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1,2),
                                  sublinear_tf=True, min_df=2)),
        ("clf",   RandomForestClassifier(n_estimators=n, class_weight="balanced",
                                          random_state=42, n_jobs=-1))
    ])

print("  Training LR C=0.5...")
lr1 = make_lr(C=0.5, max_features=5000);  lr1.fit(X_train, y_train)
print("  Training LR C=1.0...")
lr2 = make_lr(C=1.0, max_features=10000); lr2.fit(X_train, y_train)
print("  Training RF...")
rf  = make_rf(n=100, max_features=10000); rf.fit(X_train, y_train)

models = {
    "LR C=0.5 (5k)":  lr1,
    "LR C=1.0 (10k)": lr2,
    "Random Forest":   rf,
}

# ── Confusion matrices ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, model) in zip(axes, models.items()):
    preds = model.predict(X_test)
    cm    = confusion_matrix(y_test, preds, labels=LABELS)
    disp  = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    ax.set_title(f"{name}\nF1 macro = {f1:.4f}", fontsize=11, fontweight="bold")
plt.suptitle("Confusion Matrices — All Three Classifiers", fontsize=13, fontweight="bold")
plt.tight_layout()
save("10_confusion_matrices")

# ── Per-class F1 comparison ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(LABELS))
width = 0.25
for i, (name, model) in enumerate(models.items()):
    preds = model.predict(X_test)
    f1s   = [f1_score(y_test, preds, labels=[l], average="macro", zero_division=0)
             for l in LABELS]
    bars  = ax.bar(x + i*width, f1s, width, label=name, alpha=0.85)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(LABELS)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.1)
ax.set_title("Per-Class F1 Score — Classifier Comparison", fontsize=13, fontweight="bold")
ax.axhline(0.70, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Production threshold (0.70)")
ax.legend()
plt.tight_layout()
save("11_per_class_f1_comparison")

# ── F1 macro comparison bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
names, f1s, cvs, cvstds = [], [], [], []
for name, model in models.items():
    preds = model.predict(X_test)
    f1    = f1_score(y_test, preds, average="macro", zero_division=0)
    cv    = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro")
    names.append(name); f1s.append(f1); cvs.append(cv.mean()); cvstds.append(cv.std())

x = np.arange(len(names))
w = 0.35
bars1 = ax.bar(x - w/2, f1s, w, label="Test F1 macro", color="#2E5FA3", alpha=0.85)
bars2 = ax.bar(x + w/2, cvs, w, label="CV F1 macro (mean)", color="#388e3c", alpha=0.85,
               yerr=cvstds, capsize=4)
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 Macro")
ax.set_title("Test F1 vs Cross-Validation F1 — All Classifiers", fontsize=13, fontweight="bold")
ax.axhline(0.70, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Production threshold")
ax.legend()
plt.tight_layout()
save("12_f1_comparison")

# ── Learning curve for best model ───────────────────────────────────────
print("  Learning curve (this takes a minute)...")
train_sizes, train_scores, val_scores = learning_curve(
    lr2, X_train, y_train, cv=3, scoring="f1_macro",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
)
fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#2E5FA3")
ax.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1), alpha=0.15, color="#388e3c")
ax.plot(train_sizes, train_scores.mean(1), "o-", color="#2E5FA3", label="Training score")
ax.plot(train_sizes, val_scores.mean(1),   "o-", color="#388e3c", label="CV score")
ax.set_xlabel("Training examples")
ax.set_ylabel("F1 Macro")
ax.set_title("Learning Curve — Logistic Regression (C=1.0, 10k features)", fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(0.5, 1.05)
plt.tight_layout()
save("13_learning_curve")

# ── Random Forest feature importance ────────────────────────────────────
print("  Feature importance...")
vocab     = rf.named_steps["tfidf"].get_feature_names_out()
importances = rf.named_steps["clf"].feature_importances_
top_idx   = np.argsort(importances)[-30:]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(vocab[top_idx], importances[top_idx], color="#7B1FA2", alpha=0.8)
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest — Top 30 Most Important TF-IDF Features", fontsize=12, fontweight="bold")
plt.tight_layout()
save("14_rf_feature_importance")

# ── LR coefficients per class ────────────────────────────────────────────
vocab_lr = lr2.named_steps["tfidf"].get_feature_names_out()
coefs    = lr2.named_steps["clf"].coef_
classes  = lr2.named_steps["clf"].classes_

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
for ax, cls, coef in zip(axes, classes, coefs):
    top    = np.argsort(coef)
    bottom10 = top[:10]
    top10    = top[-10:]
    idx    = np.concatenate([bottom10, top10])
    words  = vocab_lr[idx]
    vals   = coef[idx]
    colors_c = [COLORS[cls] if v > 0 else "#888888" for v in vals]
    ax.barh(words, vals, color=colors_c, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"LR Coefficients\n({cls} priority)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Coefficient")
plt.suptitle("Logistic Regression — Most Positive & Negative Coefficients per Class",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save("15_lr_coefficients")

# ── ROC curves ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
y_bin = label_binarize(y_test, classes=LABELS)

for ax, (name, model) in zip(axes, models.items()):
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)
        cls_order = list(model.classes_)
        for i, cls in enumerate(LABELS):
            ci = cls_order.index(cls)
            fpr, tpr, _ = roc_curve(y_bin[:, i], probas[:, ci])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=COLORS[cls], lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve\n{name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
plt.suptitle("ROC Curves — One vs Rest per Class", fontsize=13, fontweight="bold")
plt.tight_layout()
save("16_roc_curves")

# ── Prediction confidence distribution ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, model) in zip(axes, models.items()):
    if hasattr(model, "predict_proba"):
        probas     = model.predict_proba(X_test)
        confidence = probas.max(axis=1)
        ax.hist(confidence, bins=30, color="#2E5FA3", alpha=0.8, edgecolor="white")
        ax.axvline(confidence.mean(), color="red", linestyle="--",
                   label=f"mean={confidence.mean():.2f}")
        ax.set_xlabel("Max class probability (confidence)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
plt.suptitle("Prediction Confidence Distribution on Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
save("17_confidence_distribution")

# ── CV F1 stability boxplots ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
    cv_results.append(scores)
bp = ax.boxplot(cv_results, labels=list(models.keys()), patch_artist=True)
model_colors = ["#1565C0", "#2E5FA3", "#7B1FA2"]
for patch, c in zip(bp["boxes"], model_colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_ylabel("F1 Macro (5-fold CV)")
ax.set_title("Cross-Validation Stability — 5-Fold CV per Classifier", fontsize=12, fontweight="bold")
ax.axhline(0.70, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Threshold")
ax.legend()
plt.tight_layout()
save("18_cv_stability")

# ══════════════════════════════════════════════════════════════════════════
# 5. MONITORING
# ══════════════════════════════════════════════════════════════════════════
print("5. Monitoring graphs...")

# Drift report summary
drift_path = ROOT / "monitoring" / "drift_report.json"
if drift_path.exists():
    report = json.loads(drift_path.read_text())
    checks = report.get("checks", {})
    if checks:
        names  = list(checks.keys())
        drifts = [1 if v.get("drift_detected") or v.get("new_batch_detected") else 0
                  for v in checks.values()]
        colors_d = ["#d32f2f" if d else "#388e3c" for d in drifts]
        labels_d = ["DRIFT" if d else "OK" for d in drifts]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, [1]*len(names), color=colors_d, alpha=0.85)
        for bar, label in zip(bars, labels_d):
            ax.text(0.5, bar.get_y() + bar.get_height()/2,
                    label, va="center", ha="center", fontsize=12,
                    fontweight="bold", color="white")
        ax.set_xlim(0, 1.2)
        ax.set_xticks([])
        ax.set_title("Drift Monitor — Check Results", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save("19_drift_monitor_results")

# ── Predictions CSV summary ──────────────────────────────────────────────
pred_path = ROOT / "data" / "predictions.csv"
if pred_path.exists():
    preds_df = pd.read_csv(pred_path)
    if "predicted_priority" in preds_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        counts = preds_df["predicted_priority"].value_counts()
        axes[0].bar(counts.index, counts.values,
                    color=[COLORS.get(p, "#999") for p in counts.index], alpha=0.85, width=0.5)
        axes[0].set_title("Predicted Priority Distribution\n(Production inference)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Count")

        if "confidence" in preds_df.columns:
            for p in ["high", "medium", "low"]:
                sub = preds_df[preds_df["predicted_priority"] == p]["confidence"]
                if len(sub):
                    axes[1].hist(sub, bins=25, alpha=0.6, label=p, color=COLORS[p], density=True)
            axes[1].set_xlabel("Confidence score")
            axes[1].set_ylabel("Density")
            axes[1].set_title("Confidence by Predicted Priority\n(Production inference)", fontsize=11, fontweight="bold")
            axes[1].legend()
        plt.suptitle("Production Predictions Summary", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save("20_production_predictions")

# ══════════════════════════════════════════════════════════════════════════
print(f"\nDone! All graphs saved to: {OUT.resolve()}")
print(f"Total files: {len(list(OUT.glob('*.png')))}")