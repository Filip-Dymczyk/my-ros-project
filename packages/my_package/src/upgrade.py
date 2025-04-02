import os
import rospy
import cv2
import numpy as np
import time
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage

class LineDetector(DTROS):
    def __init__(self, node_name):
        super(LineDetector, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        vehicle_name = os.environ['VEHICLE_NAME']

        # Wheels control:
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        self._wheels_publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        self.throttle_left = 0.5
        self.throttle_right = 0.5
        self.vel_left = self.throttle_left
        self.vel_right = self.throttle_right

        # State variables
        self.stop_time = None  # Timer for red line waiting
        self.spinning = False  # State for white line reaction

        # Image receiving:
        compressed_image_sub = 'camera_node/image/compressed'
        self.image_sub = rospy.Subscriber(compressed_image_sub, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)

        # Proper bridging of compressed image:
        self.bridge = CvBridge()

    def image_callback(self, compressed_image):
        img_bgr = self.bridge.compressed_imgmsg_to_cv2(compressed_image, desired_encoding="bgr8")

        # Crop the image to get only the bottom part (where lines should be) and convert to hsv:
        height, _, _ = img_bgr.shape
        cropped_img = img_bgr[int(height * 0.5):height, :]
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        red_detected = self.detect_red_lines(hsv)
        yellow_detected = self.detect_yellow_lines(hsv)
        white_detected = self.detect_white_lines(hsv)
        rospy.loginfo(f"Detected: red-{red_detected}, yellow-{yellow_detected}, white-{white_detected}")

        if red_detected and self.stop_time is None:
            self.stop_time = time.time()  # Start a 3-second timer
            self.vel_left = 0.0
            self.vel_right = 0.0
            return

        if white_detected:
            self.spinning = True  # Start spinning
        elif self.spinning:
            self.spinning = False  # Stop spinning when white line disappears
            self.vel_left = 0.5  # Resume forward movement
            self.vel_right = 0.5

    def detect_red_lines(self, image_hsv) -> bool:
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        return self.detect_lines(red_mask)

    def detect_yellow_lines(self, image_hsv) -> bool:
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
        return self.detect_lines(yellow_mask)
    
    def detect_white_lines(self, image_hsv) -> bool:
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(image_hsv, lower_white, upper_white)
        return self.detect_lines(white_mask)

    def detect_lines(self, mask) -> bool:
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
        return lines is not None
    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.stop_time:
                elapsed = time.time() - self.stop_time
                if elapsed >= 3:
                    self.stop_time = None  # Reset timer after 3 seconds
                    self.vel_left = 0.5
                    self.vel_right = 0.5

            if self.spinning:
                self.vel_left = 0.1  # Spin right
                self.vel_right = -0.1
            
            message = WheelsCmdStamped(vel_left=self.vel_left, vel_right=self.vel_right)
            self._wheels_publisher.publish(message)
            rate.sleep()

if __name__ == '__main__':
    line_detector = LineDetector(node_name='line_detector')
    line_detector.run()