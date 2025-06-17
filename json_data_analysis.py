#!/usr/bin/env python3
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt

def load_and_label(path, label):
    """Load a JSON array from `path` and tag each entry with `label`."""
    with open(path, 'r') as f:
        records = json.load(f)
    for r in records:
        r['class'] = label
    return records

def main():
    p = argparse.ArgumentParser(
        description="Plot line graphs of steering & lateral offset for three driver states"
    )
    p.add_argument('--normal',   required=True, help='path to normal.json')
    p.add_argument('--moderate', required=True, help='path to moderate.json')
    p.add_argument('--drowsy',   required=True, help='path to drowsy.json')
    args = p.parse_args()

    # 1) Load & label
    data = []
    data += load_and_label(args.normal,   'normal')
    data += load_and_label(args.moderate, 'moderate')
    data += load_and_label(args.drowsy,   'drowsy')

    # 2) Build DataFrame
    df = pd.DataFrame(data)
    df['steering_angle']   = df['steering_angle'].astype(float)
    df['lateral_offset_m'] = df['lateral_offset_m'].astype(float)

    # 3) Plot steering_angle vs. index
    plt.figure(figsize=(10,4))
    for cls, group in df.groupby('class'):
        plt.plot(group.index, group['steering_angle'], label=cls, linewidth=1)
    plt.title('Steering Angle Over Time by Driver State')
    plt.xlabel('Sample Index')
    plt.ylabel('Steering Angle')
    plt.legend(title='Class')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 4) Plot lateral_offset_m vs. index
    plt.figure(figsize=(10,4))
    for cls, group in df.groupby('class'):
        plt.plot(group.index, group['lateral_offset_m'], label=cls, linewidth=1)
    plt.title('Lateral Offset Over Time by Driver State')
    plt.xlabel('Sample Index')
    plt.ylabel('Lateral Offset (m)')
    plt.legend(title='Class')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
