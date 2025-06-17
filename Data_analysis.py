#!/usr/bin/env python3
"""
drowsiness_analysis.py

Perform exploratory data analysis on three driving datasets: normal, moderate fatigue, drowsy.
Extracts features and compares them to help detect drowsiness.

Usage:
    python drowsiness_analysis.py \
      --normal normal.csv \
      --moderate moderate.csv \
      --drowsy drowsy.csv \
      --output summary.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

def compute_descriptive(arr):
    return {
        'mean':     np.mean(arr),
        'std':      np.std(arr, ddof=1),
        'min':      np.min(arr),
        'max':      np.max(arr),
        'median':   np.median(arr),
        'skewness': float(skew(arr)),
        'kurtosis': float(kurtosis(arr))
    }

def compute_entropy(arr, bins=20):
    counts, _ = np.histogram(arr, bins=bins, density=True)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))

def compute_reversal_rate(arr, timestamps):
    signs = np.sign(arr)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i-1]
    reversals = np.sum((signs[:-1] * signs[1:]) < 0)
    dur_min = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 60.0
    return float(reversals / dur_min) if dur_min > 0 else np.nan

def compute_lane_metrics(lat, lane_id, dt):
    sdlp = float(np.std(lat, ddof=1))
    thresh = 1.0
    off = (lane_id == -1) & (np.abs(lat) > thresh)
    events = np.sum((~off[:-1]) & off[1:])
    freq = float(events / (len(lat) * dt)) if dt > 0 else np.nan
    in_lane = (lane_id != -1) & (np.abs(lat) <= thresh)
    keep_ratio = float(np.sum(in_lane) / len(in_lane))
    return sdlp, freq, keep_ratio

def analyze(df):
    # sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    # compute sampling interval (assuming uniform spacing)
    if len(df) > 1:
        dt = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    else:
        dt = np.nan

    steer = df['steering_angle_rad'].values
    lat   = df['lateral_offset'].values
    lane  = df['lane_id'].values
    times = df['timestamp']

    desc_steer = compute_descriptive(steer)
    desc_lat   = compute_descriptive(lat)
    ent_steer  = compute_entropy(steer)
    rev_rate   = compute_reversal_rate(pd.Series(steer), times)
    sdlp, dep_freq, keep_ratio = compute_lane_metrics(lat, lane, dt)

    features = {}
    features.update({f'steer_{k}': v for k, v in desc_steer.items()})
    features['steer_entropy'] = ent_steer
    features['steer_reversal_rate_per_min'] = rev_rate
    features.update({f'lat_{k}': v for k, v in desc_lat.items()})
    features['lateral_sdlp'] = sdlp
    features['lane_departure_freq_hz'] = dep_freq
    features['lane_keeping_ratio'] = keep_ratio

    return features, df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal',   '-n', required=True, help='normal driving CSV')
    parser.add_argument('--moderate', '-m', required=True, help='moderate fatigue driving CSV')
    parser.add_argument('--drowsy',   '-d', required=True, help='drowsy driving CSV')
    parser.add_argument('--output',   '-o', required=True, help='summary CSV to write')
    args = parser.parse_args()

    # Load datasets
    df_normal   = pd.read_csv(args.normal,   parse_dates=['timestamp'])
    df_moderate = pd.read_csv(args.moderate, parse_dates=['timestamp'])
    df_drowsy   = pd.read_csv(args.drowsy,   parse_dates=['timestamp'])

    # Compute features
    feat_normal,   raw_normal   = analyze(df_normal)
    feat_moderate, raw_moderate = analyze(df_moderate)
    feat_drowsy,   raw_drowsy   = analyze(df_drowsy)

    # Save summary
    summary = pd.DataFrame(
        [feat_normal, feat_moderate, feat_drowsy],
        index=['normal', 'moderate', 'drowsy']
    )
    summary.to_csv(args.output)
    print("Feature comparison:")
    print(summary)

    # Box plot: Steering angle
    plt.figure(figsize=(6,4))
    plt.boxplot(
        [ raw_normal['steering_angle_rad'],
          raw_moderate['steering_angle_rad'],
          raw_drowsy['steering_angle_rad'] ],
        labels=['normal','moderate','drowsy']
    )
    plt.ylabel('Steering Angle (rad)')
    plt.title('Steering Angle Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Box plot: Lateral offset
    plt.figure(figsize=(6,4))
    plt.boxplot(
        [ raw_normal['lateral_offset'],
          raw_moderate['lateral_offset'],
          raw_drowsy['lateral_offset'] ],
        labels=['normal','moderate','drowsy']
    )
    plt.ylabel('Lateral Offset (m)')
    plt.title('Lateral Offset Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.show()

if __name__ == "__main__":
    main()
