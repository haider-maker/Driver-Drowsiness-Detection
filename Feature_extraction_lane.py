#!/usr/bin/env python3
"""
compute_lane_features.py

Compute lane-position features from a CARLA-style log in one script,
with all parameters and file paths configured inside the code.

Features:
  - Standard Deviation of Lateral Position (SDLP, rounded to 3 decimals)
  - Lane Departure Frequency (Hz)
  - Lane Keeping Ratio (fraction)

Configuration is done via the constants below.
"""

import numpy as np
import pandas as pd

# === Configuration ===
INPUT_CSV        = 'D:\CARLA\CARLA_0.9.13\WindowsNoEditor\PythonAPI\examples\Final\carla_data_Awake.csv'  # path to your input file
OUTPUT_CSV       = 'lane_features_awake.csv'      # path to your output file
DT               = 0.05                     # sampling interval in seconds, because 20Hz is common in CARLA
WINDOW_SECONDS   = 5.0                      # sliding window length in seconds
DEPART_THRESHOLD = 1.0                      # lateral distance threshold for departure
LAT_COL          = 'lateral_offset'              # column name for lateral offset
LANE_COL         = 'lane_id'                # column name for lane ID

# === Feature computation functions ===

def compute_sdlp(lateral):
    """Standard Deviation of Lateral Position (SDLP), rounded to 3 decimals."""
    return round(float(np.std(lateral)), 3)

def compute_lane_departure_frequency(lane_ids, lateral, dt, threshold):
    """
    Lane Departure Frequency: events per second.
    Counts transitions from in-lane to off-lane.
    """
    lane_ids = np.asarray(lane_ids)
    lateral = np.asarray(lateral)
    off_lane = (lane_ids == -1) & (np.abs(lateral) > threshold)
    # count entry events
    events = np.sum((~off_lane[:-1]) & (off_lane[1:]))
    duration = len(lane_ids) * dt
    return float(events / duration)

def compute_lane_keeping_ratio(lane_ids, lateral, threshold):
    """
    Lane Keeping Ratio: fraction of samples within lane bounds.
    """
    lane_ids = np.asarray(lane_ids)
    lateral = np.asarray(lateral)
    in_lane = (lane_ids != -1) & (np.abs(lateral) <= threshold)
    return float(np.sum(in_lane) / len(in_lane))

def sliding_window_features(lateral, lane_ids, dt, window_seconds, threshold):
    """
    Generate a list of feature dicts for each sliding window.
    """
    window_size = int(window_seconds / dt)
    features = []
    for start in range(0, len(lateral) - window_size + 1):
        end = start + window_size
        lat_win  = lateral[start:end]
        lane_win = lane_ids[start:end]
        sdlp       = compute_sdlp(lat_win)
        depart_frq = compute_lane_departure_frequency(lane_win, lat_win, dt, threshold)
        keep_ratio = compute_lane_keeping_ratio(lane_win, lat_win, threshold)
        features.append({
            'sdlp':           sdlp,
            'departure_freq': depart_frq,
            'keeping_ratio':  keep_ratio
        })
    return features

def main():
    # Load the CARLA-style log
    df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
    lateral  = df[LAT_COL].values
    lane_ids = df[LANE_COL].values

    # Compute features
    feats = sliding_window_features(
        lateral, lane_ids,
        dt=DT,
        window_seconds=WINDOW_SECONDS,
        threshold=DEPART_THRESHOLD
    )

    # Save results
    out_df = pd.DataFrame(feats)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} windows of features to '{OUTPUT_CSV}'")

if __name__ == '__main__':
    main()
