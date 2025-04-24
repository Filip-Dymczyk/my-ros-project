#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from enum import Enum

class LineDetector(DTROS):
    class Direction(Enum):
        FORWARD = 1
        BACKWARD = -1

    def __init__(self, node_name):
        super(LineDetector, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        vehicle_name = os.environ['VEHICLE_NAME']

        # Wheels control:
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        self._wheels_publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        self.throttle_left = 0.0
        self.throttle_right = 0.0
        self.direction = self.Direction.FORWARD
        self.vel_left = 0.0
        self.vel_right = 0.0

        # Image receiving:
        compressed_image_sub = f"{vehicle_name}/camera_node/image/compressed"
        self.image_sub = rospy.Subscriber(compressed_image_sub, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)

        self.red_detected = False
        self.yellow_detected = False
        self.white_detected = False
        self.falses_detected = 0
        self.max_falses_count = 20
        self.turning = False
        self.turn_counter = 0
        self.max_turn_count = 5

        # Proper bridging of compressed image:
        self.bridge = CvBridge()
        self.img_bgr = None

    def image_callback(self, compressed_image) -> None:
        if self.img_bgr is None:
            self.img_bgr = self.bridge.compressed_imgmsg_to_cv2(compressed_image, desired_encoding="bgr8")
    
    def detect_lines(self, mask) -> bool:
        # Prepare for Hough transform:
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines using probability Hough transform:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

        return lines is not None

    def detect_red_lines(self, image_hsv) -> bool:
        # HSV ranges for red color:
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks and join them (red has two separate ranges):
        mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        return self.detect_lines(red_mask)

    def detect_white_lines(self, image_hsv) -> bool:
        # HSV ranges for white color:
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])

        white_mask = cv2.inRange(image_hsv, lower_white, upper_white)

        return self.detect_lines(white_mask)

    def detect_yellow_lines(self, image_hsv) -> bool:
        # HSV ranges for yellow color:
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

        return self.detect_lines(yellow_mask)
    
    def start_lines_detection(self) -> None:
        # Crop the image to get only the bottom part (where lines should be) and convert to hsv:
        height, width, _ = self.img_bgr.shape
        
        center_x = width // 2
        crop_width = width
        cropped_img = self.img_bgr[int(height * 0.6):int(height * 1.0), center_x - crop_width // 4 : center_x + crop_width // 4] # May require more cropping.

        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        self.red_detected = self.detect_red_lines(hsv)
        self.yellow_detected = self.detect_yellow_lines(hsv)
        self.white_detected = self.detect_white_lines(hsv)
        rospy.loginfo(f"Detected: red-yellow-white -> {self.red_detected}-{self.yellow_detected}-{self.white_detected}")
    
    def set_speed(self, turn_right = False) -> None:
        if turn_right:
            self.vel_right = self.throttle_right * float(self.Direction.BACKWARD.value)
            self.vel_left = self.throttle_left * float(self.Direction.FORWARD.value)
        else:
            self.vel_right = self.throttle_right * float(self.direction.value)
            self.vel_left = self.throttle_left * float(self.direction.value)

    def handle_red_lines(self) -> None:
        # Already turning.
        if self.turning:
            return
        
        if self.red_detected:
            self.throttle_left = 0.0
            self.throttle_right = 0.0
            self.falses_detected = 0
        else:
            self.falses_detected += 1
            if self.falses_detected == self.max_falses_count:
                self.throttle_left = 0.1
                self.throttle_right = 0.1
                self.direction = self.Direction.FORWARD
        self.set_speed()
        
    
    def handle_white_lines(self) -> None:
        if self.white_detected:
            self.turning = True
            self.turn_cycles = 0 

        if self.turning:
            if self.turn_cycles < self.max_turn_count:
                self.throttle_left = 0.2
                self.throttle_right = 0.2
                self.set_speed(turn_right=True)
                self.turn_cycles += 1
            else:
                self.turning = False
                self.turn_cycles = 0

        else:
            self.turning = False
            self.turn_cycles = 0

    def update_control(self) -> None:
        self.handle_red_lines()

        # Red detection terminates other behaviors:
        if not self.red_detected:
            self.handle_white_lines()
    
    def run(self) -> None:
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.img_bgr is not None:
                self.start_lines_detection()
                self.img_bgr = None
                self.update_control()
            message = WheelsCmdStamped(vel_left=self.vel_left, vel_right=self.vel_right)
            self._wheels_publisher.publish(message)
            rate.sleep()

if __name__ == '__main__':
    line_detector = LineDetector(node_name='line_detector')
    line_detector.run()