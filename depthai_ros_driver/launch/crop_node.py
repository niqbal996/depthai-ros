import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import depthai as dai
import cv2

class ImageManipNode(Node):
    def __init__(self):
        super().__init__('image_manip_node')
        self.publisher = self.create_publisher(Image, 'manipulated_image', 10)
        self.bridge = CvBridge()

        # Create DepthAI pipeline for image manipulation
        self.pipeline = dai.Pipeline()
        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        self.image_manip = self.pipeline.create(dai.node.ImageManip)

        # Configure camera
        self.cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080P)
        self.cam_rgb.setFps(30)

        # Set crop parameters
        self.image_manip.setCropRect(100, 100, 800, 600)

        # Link camera to ImageManip
        self.cam_rgb.video.link(self.image_manip.inputImage)

        # Link ImageManip to XLinkOut for sending output
        self.xlink_out = self.pipeline.create(dai.node.XLinkOut)
        self.xlink_out.setStreamName("manipulated_image")
        self.image_manip.out.link(self.xlink_out.input)

        # Run pipeline
        self.device = dai.Device(self.pipeline)
        self.q = self.device.getOutputQueue("manipulated_image", maxSize=8, blocking=False)

        # Timer to periodically process frames
        self.create_timer(1/30.0, self.timer_callback)

    def timer_callback(self):
        # Fetch the manipulated frame
        frame = self.q.get()
        frame_data = frame.getCvFrame()

        # Convert to ROS image message
        ros_image = self.bridge.cv2_to_imgmsg(frame_data, encoding='bgr8')

        # Publish the image
        self.publisher.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    node = ImageManipNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
