#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
from collections import deque

def ros_time_to_float(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_carla_ir')

        # ─── USER CONFIGURATION ────────────────────────────────────────────
        self.IMAGE_TOPIC    = '/camera/image_raw'
        self.DRIVE_TOPIC    = '/data_capture/data'
        self.SYNC_TOPIC     = '/sync_data'
        self.TOLERANCE_S    = 0.05     # max allowed delta for matching
        self.BUFFER_SIZE    = 200      # how many recent drive messages to keep
        # ────────────────────────────────────────────────────────────────────

        # small time‐buffer of driving samples
        # each entry = dict(t=float_seconds, steering=float, offset=float)
        self.drive_buf = deque(maxlen=self.BUFFER_SIZE)

        # subscribers
        self.create_subscription(Image, self.IMAGE_TOPIC,
                                 self.image_callback, 10)
        self.create_subscription(String, self.DRIVE_TOPIC,
                                 self.drive_callback, 10)

        # publisher for sync output
        self.sync_pub = self.create_publisher(String,
                                              self.SYNC_TOPIC, 10)

        self.get_logger().info('SyncNode initialized')

    def drive_callback(self, msg: String):
        """Receive JSON‐stringed driving data, append to buffer."""
        try:
            js = json.loads(msg.data)
            t     = float(js['timestamp'])
            steer = float(js['steering_angle'])
            off   = float(js['lateral_offset_m'])
        except Exception as e:
            self.get_logger().warning(f'bad drive msg: {e}')
            return
        self.drive_buf.append({'t': t, 'steering': steer, 'offset': off})

    def image_callback(self, msg: Image):
        """For each IR frame, find the nearest drive sample and publish merged JSON."""
        t_img = ros_time_to_float(msg.header.stamp)

        # find nearest driving sample
        best = None
        best_dt = None
        for d in self.drive_buf:
            dt = abs(d['t'] - t_img)
            if best is None or dt < best_dt:
                best, best_dt = d, dt

        if best is not None and best_dt <= self.TOLERANCE_S:
            steer = best['steering_angle']
            off   = best['lateral_offset_m']
        else:
            steer = None
            off   = None

        out = {
            'timestamp':       t_img,
            'image_stamp':     f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
            'steering_angle':  steer,
            'lateral_offset_m':  off
        }
        js = json.dumps(out)
        self.sync_pub.publish(String(data=js))

        if steer is None:
            self.get_logger().warn(
                f"No match for image @ {t_img:.3f}s (closest Δ={best_dt:.3f}s)" 
                if best_dt is not None else
                f"No drive data yet for image @ {t_img:.3f}s"
            )
        else:
            self.get_logger().info(
                f"Synced image @ {t_img:.3f}s  → steer={steer:+.3f}, offset={off:+.3f}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
