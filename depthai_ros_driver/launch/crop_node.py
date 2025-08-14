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
        video_width = 640  # default crop width
        video_height = 360  # default crop height
        self.image_manip_cfg.setCropRect(0, 0, video_width, video_height)
        self.config_queue.send(self.image_manip_cfg)
        maxCropX = (isp_width - video_width) / isp_width
        maxCropY = (isp_height - video_height) / isp_height
        print(f"maxCropX: {maxCropX}, maxCropY: {maxCropY}, ISP: {isp_width}x{isp_height}, Video: {video_width}x{video_height}")

        # Check if requested crop size is valid
        if video_width > isp_width or video_height > isp_height:
            print(f"Warning: Requested crop size {video_width}x{video_height} exceeds native ISP resolution {isp_width}x{isp_height}. Adjusting to max possible.")
            video_width = min(video_width, isp_width)
            video_height = min(video_height, isp_height)
            self.camRgb.setVideoSize(video_width, video_height)

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
