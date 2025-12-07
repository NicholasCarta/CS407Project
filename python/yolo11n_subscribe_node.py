#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime


class Yolo11nNode(Node):
    def __init__(self):
        super().__init__('yolo11n_subscribe_node')
        
        # Declare Parameters to be passed
        self.declare_parameter("model_path", "/home/pi/weights/best.pt")
        self.declare_parameter("conf_thres", 0.5)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("save_dir", "/home/pi/annotated_images")

        self.model_path = self.get_parameter("model_path").value
        self.conf_thres = self.get_parameter("conf_thres").value
        self.publish_annotated = self.get_parameter("publish_annotated").value
        self.save_dir = self.get_parameter("save_dir").value
        
        os.makedirs(self.save_dir, exist_ok=True)

        # Get yolo model
        self.get_logger().info(f'Loading YOLO model from: {self.model_path}')
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            err = f"Failed to load YOLO model: {e}"
            self.get_logger().error(err)
            raise RuntimeError(err)

        # Send OpenCV output to Ros2 for publishing
        self.bridge = CvBridge()

        # Yolo subscriber
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # Yolo image publisher
        if self.publish_annotated:
            self.annotated_image_pub = self.create_publisher(
                Image,
                'camera/image_yolo',
                10
            )
        else:
            self.annotated_image_pub = None

        self.get_logger().info(
            f"Yolo11nNode started. Listening on /camera/image_raw\n"
            f"Model: {self.model_path}\n"
            f"Confidence threshold: {self.conf_thres}\n"
            f"Publish annotated images: {self.publish_annotated}"
            f"Saving images to: {self.save_dir}"
        )

    def image_callback(self, msg: Image):
       # Send OpenCV output to Ros2 for publishing
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Run the yolo inference
        try:
            results = self.model(frame, verbose=False)[0]
        except Exception as e:
            self.get_logger().error(f'YOLO inference failed: {e}')
            return

        annotated = frame.copy()
        has_detections = False
        boxes = results.boxes
        names = self.model.names
        
        # Draw bounding box and label
        def draw_box(img, x1, y1, x2, y2, label, color=(0, 255, 0)):
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1, cv2.LINE_AA)

        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.conf_thres:
                continue
            
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, str(cls_id))
            # bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            draw_box(annotated, x1, y1, x2, y2, f"{class_name} {conf:.2f}")    
            has_detections = True
            
        if has_detections:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(self.save_dir, f"{timestamp}.jpg")
            cv2.imwrite(save_path, annotated)

        # Publish annotated image if enabled
        if self.annotated_image_pub is not None:
            try:
                out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                out_msg.header = msg.header
                self.annotated_image_pub.publish(out_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish annotated image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Yolo11nNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
