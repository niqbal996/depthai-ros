import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import depthai as dai
import cv2

class CroppedVideoPublisher(Node):
    def __init__(self):
        super().__init__('cropped_video_publisher')
        self.raw_publisher = self.create_publisher(Image, '/oak/rgb/cropped_raw', 10)
        self.compressed_publisher = self.create_publisher(CompressedImage, '/oak/rgb/compressed', 10)
        self.bridge = CvBridge()
        # self.compressed = self.get_parameter('compressed').get_parameter_value().bool_value
        # self.resolution = self.get_parameter('resolution').get_parameter_value().integer_value
        # DepthAI pipeline setup (from cam_test.py, only video stream)
        self.compressed = True  # Set to True for compressed output, False for raw
        self.resolution = '4K'
        self.pipeline = dai.Pipeline()
        self.image_manip_cfg = dai.ImageManipConfig()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.encoder = self.pipeline.create(dai.node.VideoEncoder)
        if self.resolution == '1080p':
            self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        elif self.resolution == '4K':
            self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        else:
            self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(15)
        crop_width = 640  # default crop width multiple of 32
        crop_height = 800  # default crop height multiple of 32
        crop_tl_x = 100  # top-left x
        crop_tl_y = 50   # top-left y
        self.camRgb.setVideoSize(crop_width, crop_height)
        # self.camRgb.setIspScale(1, 1)

        # Create XLinkOut for video stream
        videoOut = self.pipeline.create(dai.node.XLinkOut)
        videoOut.setStreamName('video')

        configIn = self.pipeline.create(dai.node.XLinkIn)
        configIn.setStreamName('config')

        encoderOut = self.pipeline.create(dai.node.XLinkOut)
        encoderOut.setStreamName('still')
        self.encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG) 

        # Linking
        # self.camRgb.video.link(videoOut.input)    # no compression
        self.camRgb.video.link(self.encoder.input)  # with compression
        configIn.out.link(self.camRgb.inputConfig)
        self.encoder.bitstream.link(encoderOut.input)

        # Now create the device after pipeline is fully constructed
        self.device = dai.Device(self.pipeline)

        # Output queue
        self.videoQueue = self.device.getOutputQueue('video', maxSize=8, blocking=False)
        self.encoderQueue = self.device.getOutputQueue('still', maxSize=30, blocking=False)
        self.config_queue = self.device.getInputQueue('config')

        # Configure parameters
        # Get native ISP and video sizes
        isp_width = self.camRgb.getIspWidth()
        isp_height = self.camRgb.getIspHeight()
        # Specify top-left corner for crop (in pixels)
        # Calculate normalized crop rectangle for top-left crop
        xMin = crop_tl_x / isp_width
        yMin = crop_tl_y / isp_height
        xMax = (crop_tl_x + crop_width) / isp_width
        yMax = (crop_tl_y + crop_height) / isp_height
        self.image_manip_cfg.setCropRect(xMin, yMin, xMax, yMax)
        self.config_queue.send(self.image_manip_cfg)
        print(f"Native Resolution: {isp_width}x{isp_height}, Cropped Video: {crop_width}x{crop_height}, Crop top-left: ({crop_tl_x},{crop_tl_y})")
        # Timer to periodically process frames
        self.create_timer(1/30.0, self.timer_callback)

    def timer_callback(self):
        if self.compressed:
            compressed_packet = self.encoderQueue.tryGet()
            frame = cv2.imdecode(compressed_packet.getData(), cv2.IMREAD_UNCHANGED) if compressed_packet else None
            if frame is not None:
                ros_compressed = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='png')
                self.compressed_publisher.publish(ros_compressed)
        else:
            frame = self.videoQueue.tryGet()
            if frame is not None:
                frame_data = frame.getCvFrame()
                ros_image = self.bridge.cv2_to_imgmsg(frame_data, encoding='bgr8')
                self.raw_publisher.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    node = CroppedVideoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
