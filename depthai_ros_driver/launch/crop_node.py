import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import depthai as dai
import cv2

class CroppedVideoPublisher(Node):
    def __init__(self):
        super().__init__('cropped_video_publisher')
        self.raw_publisher = self.create_publisher(Image, '/oak/rgb/cropped_raw', 10)
        # self.compressed_publisher = self.create_publisher(Image, '/oak/rgb/compressed', 10)
        self.bridge = CvBridge()

        # DepthAI pipeline setup (from cam_test.py, only video stream)

        self.pipeline = dai.Pipeline()
        self.image_manip_cfg = dai.ImageManipConfig()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(30)
        crop_width = 640  # default crop width
        crop_height = 800  # default crop height
        crop_x = 100  # top-left x
        crop_y = 50   # top-left y
        self.camRgb.setVideoSize(crop_width, crop_height)
        # self.camRgb.setIspScale(1, 1)

        # Create XLinkOut for video stream
        videoOut = self.pipeline.create(dai.node.XLinkOut)
        videoOut.setStreamName('video')

        configIn = self.pipeline.create(dai.node.XLinkIn)
        configIn.setStreamName('config')

        # Linking
        self.camRgb.video.link(videoOut.input)
        configIn.out.link(self.camRgb.inputConfig)

        # Now create the device after pipeline is fully constructed
        self.device = dai.Device(self.pipeline)

        # Output queue
        self.videoQueue = self.device.getOutputQueue('video', maxSize=8, blocking=False)
        self.config_queue = self.device.getInputQueue('config')

        # Configure parameters
        # Get native ISP and video sizes
        isp_width = self.camRgb.getIspWidth()
        isp_height = self.camRgb.getIspHeight()
        # Specify top-left corner for crop (in pixels)
        # Calculate normalized crop rectangle for top-left crop
        xMin = crop_x / isp_width
        yMin = crop_y / isp_height
        xMax = (crop_x + crop_width) / isp_width
        yMax = (crop_y + crop_height) / isp_height
        self.image_manip_cfg.setCropRect(xMin, yMin, xMax, yMax)
        self.config_queue.send(self.image_manip_cfg)
        print(f"Native Resolution: {isp_width}x{isp_height}, Cropped Video: {crop_width}x{crop_height}, Crop top-left: ({crop_x},{crop_y})")
        # Timer to periodically process frames
        self.create_timer(1/30.0, self.timer_callback)

    def timer_callback(self):
        frame = self.videoQueue.tryGet()
        # compressed = self.compressedQueue.tryGet()
        if frame is not None:
            frame_data = frame.getCvFrame()
            ros_image = self.bridge.cv2_to_imgmsg(frame_data, encoding='bgr8')
            self.raw_publisher.publish(ros_image)
        # if compressed is not None:
        #     compressed_data = compressed.getData()
        #     ros_compressed = self.bridge.cv2_to_imgmsg(compressed_data, encoding='jpeg')
        #     self.compressed_publisher.publish(ros_compressed)

def main(args=None):
    rclpy.init(args=args)
    node = CroppedVideoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
