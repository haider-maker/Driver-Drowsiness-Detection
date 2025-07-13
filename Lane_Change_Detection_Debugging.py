#!/usr/bin/env python3
"""
Scan a root folder for CSVs, detect real “lane changes” whenever
the vehicle *crosses* |lane_offset| ≥ lane_width/2 (rising edge only),
AND also if it jumps directly from one side to the other in one sample,
and plot+save results into one flat folder with names like
<person>_<session>.png, preserving full session name.
"""

import os, re, warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────
ROOT_DIR   = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 4/Summer Project/LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems/Dataset/Sync_Extracted_Data"
OUTPUT_DIR = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 4/Summer Project/LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems/Dataset/Lane_Change"
LANE_WIDTH = 3.5   # full lane width (m)
# ────────────────────────────────────────────────────────────────────────────────

HALF_W  = LANE_WIDTH / 2.0
EARLY_W = HALF_W * 0.9

def detect_events(series, thresh):
    """Return (raw_mask, arrival_mask) for threshold crossings & flips."""
    above    = series.abs() >= thresh
    prev_abv = above.shift(1, fill_value=False)
    edge     = above & ~prev_abv

    p = series.shift(1)
    c = series
    flip = ((p <= -thresh) & (c >= +thresh)) | ((p >= +thresh) & (c <= -thresh))

    raw   = edge | flip
    clean = raw.copy()
    idxs  = np.nonzero(raw)[0]
    for a, b in zip(idxs, idxs[1:]):
        # suppress the first of any ± flip pair
        if b == a + 1 and series.iat[a] * series.iat[b] < 0:
            clean[a] = False
    return raw, clean

def plot_and_save(df, er_raw, er_arr, tr_raw, tr_arr, out_png):
    fig, ax = plt.subplots(figsize=(12,4))
    ts  = df["timestamp"]
    offs = df["lane_offset"]

    # offset + threshold lines
    ax.plot(ts, offs, lw=1)
    ax.axhline(+HALF_W, color="black", ls="--")
    ax.axhline(-HALF_W, color="black", ls="--")

    # fill left/right
    pos = offs >= 0
    ax.fill_between(ts, 0, offs, where=pos, facecolor="tab:orange", alpha=0.3)
    ax.fill_between(ts, 0, offs, where=~pos, facecolor="tab:cyan",   alpha=0.3)

    # early warnings (90%)
    ev = df.loc[er_raw]
    ax.scatter(ev.timestamp, ev.lane_offset,
               facecolors="none", edgecolors="gold", s=80,
               label="90% raw")
    ev = df.loc[er_arr]
    ax.scatter(ev.timestamp, ev.lane_offset,
               facecolors="none", edgecolors="green", s=120, lw=2,
               label="90% arrival")

    # true lane‐changes (100%)
    ev = df.loc[tr_raw]
    ax.scatter(ev.timestamp, ev.lane_offset,
               facecolors="none", edgecolors="darkgoldenrod", s=80,
               label="100% raw")
    ev = df.loc[tr_arr]
    ax.scatter(ev.timestamp, ev.lane_offset,
               facecolors="none", edgecolors="red", s=120, lw=2,
               label="100% arrival")

    ax.set_title("Lateral Offset – Early Warnings & Lane-Changes")
    ax.set_ylabel("Offset (m) ← Left    0 = Center    Right →")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.3)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02,1))
    plt.tight_layout(rect=[0,0,0.85,1])

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"    → saved plot as: {out_png}")

def make_png_name(csv_path):
    rel     = os.path.relpath(csv_path, ROOT_DIR).replace("\\","/")
    parts   = rel.split("/")
    person  = parts[0] if len(parts)>1 else ""
    base    = re.sub(r'(?i)_sync\.csv$', "", parts[-1])
    return f"{(person.lower() + '_') if person else ''}{base.lower()}.png"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    warnings.simplefilter("ignore", FutureWarning)

    for dirpath, _, files in os.walk(ROOT_DIR):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue

            path = os.path.join(dirpath, fn)
            try:
                df = pd.read_csv(path, parse_dates=["timestamp"])
            except pd.errors.EmptyDataError:
                # skip totally empty CSVs
                print(f"[SKIP] {os.path.relpath(path, ROOT_DIR)}: empty CSV")
                continue
            except Exception as e:
                print(f"[WARN] {os.path.relpath(path, ROOT_DIR)}: {e}")
                continue

            if "lane_offset" not in df.columns:
                continue

            df["lane_offset"] = pd.to_numeric(df["lane_offset"], errors="coerce")
            df = (
                df.dropna(subset=["timestamp","lane_offset"])
                  .sort_values("timestamp")
                  .reset_index(drop=True)
            )

            # detect early (90%) and true (100%) events
            er_raw, er_arr = detect_events(df.lane_offset, EARLY_W)
            tr_raw, tr_arr = detect_events(df.lane_offset, HALF_W)

            rel = os.path.relpath(path, ROOT_DIR)
            print(f"{rel}: early arrivals={er_arr.sum()}/{er_raw.sum()}  "
                  f"true arrivals={tr_arr.sum()}/{tr_raw.sum()}")
            for _, r in df.loc[tr_arr, ["timestamp","lane_offset"]].iterrows():
                side = "right" if r.lane_offset > 0 else "left"
                print(f"    • {r.timestamp} → {r.lane_offset:+.2f} m ({side})")

            out_png = os.path.join(OUTPUT_DIR, make_png_name(path))
            plot_and_save(df, er_raw, er_arr, tr_raw, tr_arr, out_png)

    print("\n✅ Done. All plots saved under:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
