#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import deque

class OnlineFeatureNode(Node):
    def __init__(self):
        super().__init__('online_feature_node')

        # ─── USER CONFIGURATION ──────────────────────────────────────
        self.WINDOW_SEC   = 60.0      # length of sliding window (s)
        self.STRIDE_SEC   = 20.0      # compute features every this many s
        self.LANE_WIDTH   = 3.5       # lane width in meters
        self.KSS_SCORE    = 0         # default if none provided
        half_lane         = self.LANE_WIDTH / 2.0
        self.EARLY_THRESH = 0.9 * half_lane   # 90% “early” warnings
        self.TRUE_THRESH  =       half_lane   # 100% true arrivals
        # ─────────────────────────────────────────────────────────────

        # circular buffer of recent samples
        # each entry: {'t': float, 'steering': float, 'offset': float}
        self.buf = deque()

        # subscribe to CARLA driving data (JSON-stringed std_msgs/String)
        self.create_subscription(
            String, '/synced_output',
            self.driving_callback, 10)

        # publisher for extracted features
        self.feat_pub = self.create_publisher(
            Float32MultiArray, '/features_carla', 10)

        # timer to fire every STRIDE_SEC
        self.create_timer(self.STRIDE_SEC, self.on_timer)

        self.get_logger().info('OnlineFeatureNode initialized')

    def driving_callback(self, msg: String):
        """Called for each incoming /data_capture/data JSON string."""
        try:
            js = json.loads(msg.data)
            t      = float(js.get('image_stamp', 0))
            steer  = float(js.get('steering_angle'))
            offset = float(js.get('lateral_offset_m'))
        except Exception as e:
            self.get_logger().warning(f'Bad data msg: {e}')
            return

        # append to buffer and purge older than WINDOW_SEC
        self.buf.append({'t': t, 'steering': steer, 'offset': offset})
        cutoff = t - self.WINDOW_SEC
        while self.buf and self.buf[0]['t'] < cutoff:
            self.buf.popleft()

    def detect_arrivals(self, offs: np.ndarray, thresh: float):
        """
        Returns a boolean mask of “arrival” points:
        whenever |offset| crosses ≥ thresh (rising edge)
        or flips sign across ±thresh in one sample,
        keeping only the arrival half of any flip-pair.
        """
        s = pd.Series(offs)
        above    = s.abs() >= thresh
        edge     = above & ~above.shift(1, fill_value=False)
        p, c     = s.shift(1), s
        flip     = ((p <= -thresh)&(c >= +thresh)) | ((p >= +thresh)&(c <= -thresh))
        raw      = edge | flip

        # drop the “departure” half of any direct flip
        true_mask = raw.copy()
        idxs     = np.nonzero(raw.values)[0]
        for a, b in zip(idxs, idxs[1:]):
            if b == a + 1 and s.iat[a] * s.iat[b] < 0:
                true_mask.iat[a] = False

        return true_mask.values

    def compute_features(self, df: pd.DataFrame):
        s = df['steering'].to_numpy()
        o = df['offset'].to_numpy()

        # a) steering entropy
        hist,_ = np.histogram(s, bins=10, density=True)
        s_ent  = float(entropy(hist[hist>0], base=2))

        # b) steering reversal rate
        rev    = float((np.diff(np.sign(s))!=0).sum() / self.WINDOW_SEC)

        # c) standard deviations
        s_std  = float(np.std(s))
        o_std  = float(np.std(o))

        # d) lane departure freq & keep ratio
        half = self.LANE_WIDTH / 2.0
        outside = np.abs(o) > half
        dep_freq   = float(outside.sum() / self.WINDOW_SEC)
        keep_ratio = float(1 - outside.sum() / len(o))

        # e) lane_changes via quantized‐offset stepping
        idxs = (pd.Series(o)
                  .divide(self.LANE_WIDTH)
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

    def on_timer(self):
        """Fires every STRIDE_SEC to compute & publish features."""
        if not self.buf:
            self.get_logger().info('Buffer empty → no features')
            return

        # snapshot buffer into DataFrame
        df = pd.DataFrame(self.buf)
        t_arr = df['offset'].to_numpy()

        # detect early and true arrivals
        early_mask = self.detect_arrivals(t_arr, self.EARLY_THRESH)
        true_mask  = self.detect_arrivals(t_arr, self.TRUE_THRESH)

        # log arrivals
        for t, offs in zip(df['t'][true_mask], df['offset'][true_mask]):
            self.get_logger().info(
                f'→ True arrival at {t:.3f}s, offset {offs:+.2f}m')
        for t, offs in zip(df['t'][early_mask & ~true_mask], df['offset'][early_mask & ~true_mask]):
            self.get_logger().info(
                f'→ Early warning at {t:.3f}s, offset {offs:+.2f}m')

        # compute features
        feats = self.compute_features(df)

        # publish as Float32MultiArray in fixed order
        arr = Float32MultiArray()
        arr.data = [
            feats['steering_entropy'],
            feats['steering_reversal_rate'],
            feats['steering_std'],
            feats['offset_std'],
            feats['lane_departure_freq'],
            feats['lane_keeping_ratio'],
            float(feats['lane_changes']),
        ]
        self.feat_pub.publish(arr)
        last_time = df['t'].iat[-1]
        self.get_logger().info(
            f'Published features at {last_time:.3f}s → {arr.data}')

def main(args=None):
    rclpy.init(args=args)
    node = OnlineFeatureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
