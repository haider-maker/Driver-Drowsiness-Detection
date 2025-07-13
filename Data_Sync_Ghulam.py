#!/usr/bin/env python3
import os
import glob
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────────
EXTRACTED_ROOT   = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 4/Summer Project/LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems/Dataset/extracted_data"         # where your per-person folders live
SYNC_ROOT        = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 4/Summer Project/LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems/Dataset/Sync_Extracted_Data"    # where to write the sync CSVs
# ────────────────────────────────────────────────────────────────────────────────

def load_csvs(person_dir, session):
    """
    Given a session folder, load its image_metadata and data_capture CSVs.
    Returns (cam_df, drive_df), or (None, None) if either is missing.
    """
    images_meta = os.path.join(person_dir, session, "images", "image_metadata.csv")
    drive_data  = os.path.join(person_dir, session, "data_capture.csv")
    if not (os.path.exists(images_meta) and os.path.exists(drive_data)):
        return None, None

    cam_df   = pd.read_csv(images_meta)
    drive_df = pd.read_csv(drive_data)
    # convert UNIX-float timestamps to pandas Timestamps
    cam_df  ['timestamp'] = pd.to_datetime(cam_df  ['timestamp'], unit='s')
    drive_df['timestamp'] = pd.to_datetime(drive_df['timestamp'], unit='s')
    return cam_df, drive_df

def sync_and_save(cam_df, drive_df, out_csv):
    """
    For each row in drive_df, find the nearest cam_df timestamp,
    then assemble the synced DataFrame and write to out_csv.
    """
    rows = []
    for _, drow in drive_df.iterrows():
        t = drow['timestamp']
        # nearest camera
        i = (cam_df['timestamp'] - t).abs().idxmin()
        c = cam_df.loc[i]
        rows.append({
            "timestamp":        t,
            "timestamp_float":  t.timestamp(),
            "ir_filename":      c['image_filename'],
            "steering_angle":   drow['steering'],
            "lane_offset":      drow['offset']
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv,
                  index=False,
                  date_format='%Y-%m-%d %H:%M:%S.%f',
                  float_format='%.3f')
    print(f"[Sync] -> {out_csv}  ({len(out_df)} rows)")

def main():
    # find each person/session under EXTRACTED_ROOT
    for person in sorted(os.listdir(EXTRACTED_ROOT)):
        person_dir = os.path.join(EXTRACTED_ROOT, person)
        if not os.path.isdir(person_dir):
            continue

        for session in sorted(os.listdir(person_dir)):
            session_dir = os.path.join(person_dir, session)
            if not os.path.isdir(session_dir):
                continue

            cam_df, drive_df = load_csvs(person_dir, session)
            if cam_df is None:
                print(f"[SKIP] missing CSVs for {person}/{session}")
                continue

            # output path mirrors input hierarchy
            out_dir = os.path.join(SYNC_ROOT, person, session)
            out_csv = os.path.join(out_dir, f"{session}_Sync.csv")
            sync_and_save(cam_df, drive_df, out_csv)

if __name__ == "__main__":
    main()
