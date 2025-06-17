#!/usr/bin/env python3
import argparse

import pandas as pd
import matplotlib.pyplot as plt

def load_and_label_csv(path, label):
    """Load a CSV with `steering_angle_rad` & `lateral_offset`, tag each row with `label`."""
    df = pd.read_csv(path, usecols=['steering_angle_rad','lateral_offset'])
    df['class'] = label
    return df

def main():
    p = argparse.ArgumentParser(
        description="Plot steering-angle & lateral-offset from three driver-state CSVs"
    )
    p.add_argument('--normal',   required=True, help='path to normal.csv')
    p.add_argument('--moderate', required=True, help='path to moderate.csv')
    p.add_argument('--drowsy',   required=True, help='path to drowsy.csv')
    args = p.parse_args()

    # 1) Load & label
    df_normal   = load_and_label_csv(args.normal,   'normal')
    df_moderate = load_and_label_csv(args.moderate, 'moderate')
    df_drowsy   = load_and_label_csv(args.drowsy,   'drowsy')

    # 2) Concatenate
    df = pd.concat([df_normal, df_moderate, df_drowsy], ignore_index=True)

    # 3) Plot Steering Angle
    plt.figure(figsize=(10,4))
    for cls, group in df.groupby('class'):
        plt.plot(group.index, group['steering_angle_rad'], label=cls, linewidth=1)
    plt.title('Steering Angle Over Samples by Driver State')
    plt.xlabel('Sample Index')
    plt.ylabel('Steering Angle (rad)')
    plt.legend(title='Class')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 4) Plot Lateral Offset
    plt.figure(figsize=(10,4))
    for cls, group in df.groupby('class'):
        plt.plot(group.index, group['lateral_offset'], label=cls, linewidth=1)
    plt.title('Lateral Offset Over Samples by Driver State')
    plt.xlabel('Sample Index')
    plt.ylabel('Lateral Offset (m)')
    plt.legend(title='Class')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
