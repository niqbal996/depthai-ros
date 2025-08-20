import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
import depthai as dai
import cv2
import numpy as np

class CroppedVideoPublisher(Node):
    def __init__(self):
        super().__init__('cropped_video_publisher')
        self.raw_publisher = self.create_publisher(Image, '/oak/rgb/cropped_raw', 10)
        self.compressed_publisher = self.create_publisher(CompressedImage, '/oak/rgb/compressed', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/oak/rgb/camera_info', 10)
        self.bridge = CvBridge()
        # self.compressed = self.get_parameter('compressed').get_parameter_value().bool_value
        # self.resolution = self.get_parameter('resolution').get_parameter_value().integer_value
        # DepthAI pipeline setup (from cam_test.py, only video stream)
        self.compressed = False  # Set to True for compressed output, False for raw
        self.resolution = '4K'
        self.framerate = 15
        self.auto_config = False
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
        self.camRgb.setFps(self.framerate)
        crop_width = 800  # default crop width multiple of 32
        crop_height = 1000  # default crop height multiple of 32
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

        self.controlIn = self.pipeline.create(dai.node.XLinkIn)
        self.controlIn.setStreamName('control')

        if self.compressed:
            self.camRgb.video.link(self.encoder.input)  # with compression
        else:
            self.camRgb.video.link(videoOut.input)  # without compression
        configIn.out.link(self.camRgb.inputConfig)
        self.encoder.bitstream.link(encoderOut.input)
        self.controlIn.out.link(self.camRgb.inputControl)
        # Now create the device after pipeline is fully constructed
        self.device = dai.Device(self.pipeline)

        # Get calibration data for current resolution
        calibData = self.device.readCalibration()
        
        # Setup CameraInfo message
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.width = crop_width
        self.camera_info_msg.height = crop_height
        self.camera_info_msg.distortion_model = 'plumb_bob'
        
        # Get intrinsics for 4K resolution and scale for crop size
        if self.resolution == '4K':
            M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 3840, 2160))
            # Scale intrinsics to crop size
            scale_x = crop_width / 3840.0
            scale_y = crop_height / 2160.0
            M_rgb[0, 0] *= scale_x  # fx
            M_rgb[1, 1] *= scale_y  # fy
            M_rgb[0, 2] *= scale_x  # cx
            M_rgb[1, 2] *= scale_y  # cy
        else:  # 1080p
            M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1920, 1080))
            # Scale intrinsics to crop size
            scale_x = crop_width / 1920.0
            scale_y = crop_height / 1080.0
            M_rgb[0, 0] *= scale_x  # fx
            M_rgb[1, 1] *= scale_y  # fy
            M_rgb[0, 2] *= scale_x  # cx
            M_rgb[1, 2] *= scale_y  # cy
        
        # Fill K matrix (3x3 intrinsic matrix)
        self.camera_info_msg.k = [
            M_rgb[0, 0], M_rgb[0, 1], M_rgb[0, 2],
            M_rgb[1, 0], M_rgb[1, 1], M_rgb[1, 2],
            M_rgb[2, 0], M_rgb[2, 1], M_rgb[2, 2]
        ]
        
        # Get distortion coefficients (if available for RGB camera)
        try:
            D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
            self.camera_info_msg.d = D_rgb[:5].tolist()  # Use first 5 coefficients (k1, k2, p1, p2, k3)
        except:
            self.camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
        
        # R matrix (rectification matrix) - identity for monocular
        self.camera_info_msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        
        # P matrix (projection matrix) - same as K for monocular
        self.camera_info_msg.p = [
            M_rgb[0, 0], M_rgb[0, 1], M_rgb[0, 2], 0.0,
            M_rgb[1, 0], M_rgb[1, 1], M_rgb[1, 2], 0.0,
            M_rgb[2, 0], M_rgb[2, 1], M_rgb[2, 2], 0.0
        ]

        # Output queue
        self.videoQueue = self.device.getOutputQueue('video', maxSize=self.framerate, blocking=False)
        self.encoderQueue = self.device.getOutputQueue('still', maxSize=self.framerate, blocking=False)
        self.config_queue = self.device.getInputQueue('config')
        self.control_queue = self.device.getInputQueue('control')

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
        ctrl = dai.CameraControl()
        if not self.auto_config:
            ctrl_manual_exposure_time = 10000   # [1, 33000]
            ctrl_sens_iso = 800                 # [100, 1600]
            ctrl_manual_white_balance = 8000    # [1000, 12000]
            ctrl_manual_focus = 120             # [0, 255]
            ctrl.setManualFocus(ctrl_manual_focus)
            ctrl.setManualExposure(ctrl_manual_exposure_time, ctrl_sens_iso)
            ctrl.setManualWhiteBalance(ctrl_manual_white_balance)
            self.control_queue.send(ctrl)
        else:
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            ctrl.setAutoExposureEnable()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            ctrl.setAutoFocusRegion(xMin, yMin, xMax, yMax)  # Set focus region to the crop area
            self.control_queue.send(ctrl)

        print(f"Native Resolution: {isp_width}x{isp_height}, Cropped Video: {crop_width}x{crop_height}, Crop top-left: ({crop_tl_x},{crop_tl_y})")
        # Timer to periodically process frames
        self.create_timer(1/30.0, self.timer_callback)

    def timer_callback(self):
        if self.compressed:
            compressed_packets = self.encoderQueue.tryGetAll()
            for compressed_packet in compressed_packets:
                frame = cv2.imdecode(compressed_packet.getData(), cv2.IMREAD_UNCHANGED) if compressed_packet else None
                if frame is not None:
                    ros_compressed = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='png')
                    self.compressed_publisher.publish(ros_compressed)
                    # Publish CameraInfo synchronized with compressed image
                    self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
                    self.camera_info_msg.header.frame_id = 'oak_rgb_camera_frame'
                    self.camera_info_pub.publish(self.camera_info_msg)
        else:
            frame = self.videoQueue.tryGet()
            if frame is not None:
                frame_data = frame.getCvFrame()
                ros_image = self.bridge.cv2_to_imgmsg(frame_data, encoding='bgr8')
                self.raw_publisher.publish(ros_image)
                # Publish synchronized CameraInfo
                self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
                self.camera_info_msg.header.frame_id = 'oak_rgb_camera_frame'
                self.camera_info_pub.publish(self.camera_info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CroppedVideoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
