#!/usr/bin/env python3
"""
Align Carla‐derived steering+offset to IR‐camera mid‐window timestamps,
computing sliding‐window features every STRIDE_SEC seconds.
Detects “real” lane‐changes by watching for |lane_offset| crossing **up**
past HALF_WIDTH, **or** jumping directly from one side to the other in one sample,
and only counts the *arrival* at the new side (not the departure). 

Prints:
  • GLOBAL lane‐change timestamps
  • per‐window arrival timestamps + feature summary
  • grand total of true lane‐changes

Finally writes out your aligned features CSV.

Just edit the USER CONFIGURATION below, then:

    python3 align_carla_to_ir.py
"""
import os, warnings
import pandas as pd
import numpy as np
from scipy.stats import entropy

# === USER CONFIGURATION ===
SYNC_CSV    = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 4/Summer Project/LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems/Dataset/Sync_Extracted_Data/Theresa/KSS_8_Vid_1/KSS_8_Vid_1_Sync.csv"
WINDOW_SEC  = 60.0      # window length in seconds
STRIDE_SEC  = 20.0      # stride between midpoints
LANE_WIDTH  = 3.5       # lane width (m)
KSS_SCORE   = 8        # KSS tag for every row
OUT_CSV     = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 4/Summer Project/LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems/Dataset/Features/Theresa/KSS_8_Vid_1/features_output_labeled_sync.csv"
# ===========================

HALF_W = LANE_WIDTH / 2.0 * 0.9  # 90% of half-lane width for sensitivity

def detect_lane_changes(offsets):
    """
    offsets: 1D numpy array of lateral_offset values.
    Returns two bool arrays:
      all_events: any rising‐edge ≥ HALF_W or direct sign‐flip across ±HALF_W
      true_events: arrival points only (drops the departure half of flips)
    """
    s = pd.Series(offsets)
    above    = s.abs() >= HALF_W
    prev_abv = above.shift(1, fill_value=False)
    thr_edge = above & ~prev_abv

    p = s.shift(1)
    c = s
    jump = ((p <= -HALF_W) & (c >= +HALF_W)) | ((p >= +HALF_W) & (c <= -HALF_W))

    all_ev = thr_edge | jump
    true_ev = all_ev.copy()
    idxs = np.nonzero(all_ev.values)[0]
    for a, b in zip(idxs, idxs[1:]):
        if b == a + 1 and (s.iat[a] * s.iat[b] < 0):
            true_ev.iat[a] = False

    return all_ev.values, true_ev.values

def compute_features(win):
    s = win['steering_angle'].to_numpy()
    o = win['lane_offset'].to_numpy()

    hist,_ = np.histogram(s, bins=10, density=True)
    s_ent  = float(entropy(hist[hist>0], base=2))
    rev    = float((np.diff(np.sign(s)) != 0).sum() / WINDOW_SEC)
    s_std  = float(np.std(s))
    o_std  = float(np.std(o))

    outside   = np.abs(o) > HALF_W
    dep_freq  = float(outside.sum() / WINDOW_SEC)
    keep_ratio= float(1 - outside.sum() / len(o))

    idxs = (pd.Series(o)
              .divide(LANE_WIDTH)
              .round()
              .dropna()
              .astype(int)
              .to_numpy())
    lc = int((np.diff(idxs) != 0).sum())

    return {
        'steering_entropy':       round(s_ent,4),
        'steering_reversal_rate': round(rev,4),
        'steering_std':           round(s_std,4),
        'offset_std':             round(o_std,4),
        'lane_departure_freq':    round(dep_freq,4),
        'lane_keeping_ratio':     round(keep_ratio,4),
        'lane_changes':           lc
    }

def main():
    warnings.simplefilter("ignore", category=FutureWarning)

    # 1) load & sort by float time
    df = (pd.read_csv(SYNC_CSV)
            .sort_values('timestamp_float')
            .reset_index(drop=True))

    times = df['timestamp_float'].to_numpy()

    # 2) GLOBAL true‐arrival lane‐changes
    _, true_glob = detect_lane_changes(df['lane_offset'].to_numpy())
    print("\n🔍 GLOBAL lane‐change timestamps (float):")
    if true_glob.sum():
        for t in df.loc[true_glob, 'timestamp_float']:
            print(f"  → {t:.3f} s")
    else:
        print("  (none)")

    # 3) build float‐based sliding windows
    filtered, last = [], None
    for t in times:
        if last is None or (t - last) >= STRIDE_SEC:
            filtered.append(t)
            last = t

    rows, grand = [], 0
    half_w = WINDOW_SEC / 2.0

    # 4) per‐window summaries
    print("\n📊 Per‐window summaries:")
    for mid in filtered:
        mask = (df['timestamp_float'] >= mid - half_w) & (df['timestamp_float'] < mid + half_w)
        win  = df.loc[mask]
        if win.empty:
            continue

        _, true_win = detect_lane_changes(win['lane_offset'].to_numpy())
        if true_win.any():
            print(f"\n[center={mid:.3f}s]  Detected at:")
            for t in win.loc[true_win, 'timestamp_float']:
                print(f"    • {t:.3f} s")
        else:
            print(f"\n[center={mid:.3f}s]  No lane‐changes")

        feats = compute_features(win)
        print(f"    departures/sec={feats['lane_departure_freq']:.3f}, "
              f"keep_ratio={feats['lane_keeping_ratio']:.3f}, "
              f"lane_changes={feats['lane_changes']}")

        grand += feats['lane_changes']
        feats['timestamp']      = pd.to_datetime(mid, unit='s')
        feats['timestamp_float']= mid
        feats['kss_score']      = KSS_SCORE
        rows.append(feats)

    # 5) write out CSV
    out = pd.DataFrame(rows).sort_values('timestamp_float')
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Grand total lane‐changes: {grand}")
    print(f"✅ Wrote {len(rows)} windows → {OUT_CSV}\n")

if __name__ == "__main__":
    main()