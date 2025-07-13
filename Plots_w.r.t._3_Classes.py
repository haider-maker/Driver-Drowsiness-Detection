#!/usr/bin/env python3
"""
Plot mean ±1 SD and also boxplots of each extracted feature by Sleepiness Class & Person.
Skips any Person/Class combos that have no data for a given feature,
but still draws each plot as long as someone has data.
"""

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# 1) Configuration
# ────────────────────────────────────────────────────────────────────────────────
BASE_DIR = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 4/Summer Project/LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems/Dataset/Features"
OUT_DIR  = os.path.join(BASE_DIR, "Summary_Plots")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    'lane_departure_freq',
    'lane_changes',
    'lane_keeping_ratio',
    'offset_std',
    'steering_entropy',
    'steering_reversal_rate',
    'steering_std'
]

# ────────────────────────────────────────────────────────────────────────────────
# 2) Gather all feature‐CSVs into one big DataFrame
# ────────────────────────────────────────────────────────────────────────────────
pattern = os.path.join(BASE_DIR, '*', '*', 'features_output_labeled_sync.csv')
paths = glob.glob(pattern)
if not paths:
    raise RuntimeError(f"No CSVs found under {BASE_DIR} matching {pattern}")

for p in sorted(paths):
    print("   ", p)

all_dfs = []
for path in paths:
    person  = os.path.basename(os.path.dirname(os.path.dirname(path)))
    session = os.path.basename(os.path.dirname(path))
    df = pd.read_csv(path)
    df['Person']  = person
    df['Session'] = session
    df['Class'] = pd.cut(df['kss_score'],
                         bins=[0,3,6,9],
                         labels=["Alert","Moderate Fatigue","Drowsy"])
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)

# ────────────────────────────────────────────────────────────────────────────────
# 3) Plot each feature in turn
# ────────────────────────────────────────────────────────────────────────────────
sns.set(style="whitegrid", font_scale=1.1)

for feat in FEATURES:
    if feat not in df.columns:
        print(f"[!] feature '{feat}' not found → skipping")
        continue

    df_feat = df[['Person','Class',feat]].dropna(subset=[feat])
    if df_feat.empty:
        print(f"[!] feature '{feat}' has no data → skipping")
        continue

    # ——— point‐plot of means ±1 SD —————————————————————————————————————
    summary = (
        df_feat
        .groupby(['Person','Class'], observed=True)[feat]
        .agg(mean='mean', std='std')
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7,4.5))
    sns.pointplot(
        data   = summary,
        x      = 'Class', y='mean', hue='Person',
        dodge  = 0.4, markers='o', linestyles='-',
        errorbar=None,
        ax     = ax
    )
    # add ±1 SD whiskers manually
    persons = summary['Person'].unique().tolist()
    for _, row in summary.iterrows():
        cls_idx = ["Alert","Moderate Fatigue","Drowsy"].index(row['Class'])
        hue_idx = persons.index(row['Person'])
        n = len(persons)
        if n > 1:
            total = 0.8
            sep   = total/(n-1)
            offset = (hue_idx - (n-1)/2) * sep
        else:
            offset = 0
        x = cls_idx + offset
        y = row['mean']
        yerr = row['std']
        ax.errorbar([x], [y], yerr=[[yerr], [yerr]],
                    fmt='none', ecolor='gray', capsize=3)

    ax.set_title(f"{feat.replace('_',' ').title()}  —  mean ±1 SD by Class & Person")
    ax.set_xlabel("Sleepiness Class")
    ax.set_ylabel(feat.replace('_',' ').title())
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.85,1])             # leave room on right for legend
    out1 = os.path.join(OUT_DIR, f"{feat}_meansd_by_person.png")
    plt.savefig(out1, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved", out1)

    # ——— boxplot of full distributions —————————————————————————————————
    fig, ax2 = plt.subplots(figsize=(8,5))
    sns.boxplot(
        data    = df_feat,
        x       = 'Class',
        y       = feat,
        hue     = 'Person',
        dodge   = True,
        palette = "Set2",
        ax      = ax2
    )
    ax2.set_title(f"{feat.replace('_',' ').title()}  —  distribution by Class & Person")
    ax2.set_xlabel("Sleepiness Class")
    ax2.set_ylabel(feat.replace('_',' ').title())
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.8,1])              # shrink to make space
    out2 = os.path.join(OUT_DIR, f"{feat}_box_by_person.png")
    plt.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved", out2)

print("\n✅ Done — all feature plots written to:", OUT_DIR)
