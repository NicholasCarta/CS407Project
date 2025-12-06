import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # Ros2 CMD parameters
        self.declare_parameter("device_id", 0)
        self.declare_parameter("frame_rate", 10.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)

        # Get parameters
        device_param = self.get_parameter("device_id")
        rate_param = self.get_parameter("frame_rate")
        width_param = self.get_parameter("width")
        height_param = self.get_parameter("height")

        # Assign parameters
        self.device_id = device_param.value
        self.frame_rate = rate_param.value
        self.width = width_param.value
        self.height = height_param.value

        # Access the camera at device_id
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            err = f"Camera {self.device_id} could not be opened"
            self.get_logger().error(err)
            raise RuntimeError(err)

        # Create Ros2 publisher
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        
        # Send OpenCV output to Ros2 for publishing
        self.bridge = CvBridge()

        # Timer for publishing images
        self.timer = self.create_timer(1.0 / self.frame_rate, self.timer_callback)

        self.get_logger().info(
            f'CameraPublisher started: device={self.device_id}, '
            f'{self.width}x{self.height} @ {self.frame_rate} Hz'
        )

    def timer_callback(self):
        # Grab a frame and publish it
        success, image = self.cap.read()
        if success is False:
            self.get_logger().warning("Camera did not give a frame")
            return
            
        # Get image with encoding for Ros2 and pass it to the bridge
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = "camera_frame"

        self.publisher_.publish(ros_image)

    def destroy_node(self):
        # Release the camera
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
