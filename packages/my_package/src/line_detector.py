import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage

# https://sites.uml.edu/paul-robinette/teaching/eece-5560-spring-2021/assignments/lab-4-lane-detection/
# https://studentuml-my.sharepoint.com/:p:/g/personal/paul_robinette_uml_edu/EYNnMyiti2JElAKPnppe4j0BGMCEgfuVNojenOs5K7ZsrA?rtime=1adcUyFy3Ug
# https://studentuml-my.sharepoint.com/:p:/g/personal/paul_robinette_uml_edu/EU9aPbwFeD9Mp670XuVsHM8BYOJ2ibnXKxafNbM3_h5KqA?e=4UA6C2

class LineDetector(DTROS):
    def __init__(self, node_name):
        super(LineDetector, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        vehicle_name = os.environ['VEHICLE_NAME']

        # Wheels control:
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        self._wheels_publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        self.throttle_left = 0.0
        self.throttle_right = 0.0
        self.forward = 1.0
        self.backward = -1.0
        self.vel_left = 0.0
        self.vel_right = 0.0

        # Image receiving:
        compressed_image_sub = 'camera_node/image/compressed'
        self.image_sub = rospy.Subscriber(compressed_image_sub, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)

        # Proper bridging of compressed image:
        self.bridge = CvBridge()

    def image_callback(self, compressed_image):
        img_bgr = self.bridge.compressed_imgmsg_to_cv2(compressed_image, desired_encoding="bgr8")

        # Crop the image to get only the bottom part (where lines should be) and convert to hsv:
        height, _, _ = img_bgr.shape
        cropped_img = img_bgr[int(height * 0.5):height, :] # May require more cropping.

        # Can be tested - from tips:
        # image_size = (160, 120)
        # offset = 40
        # resized_image = cv2.resize(img_bgr, image_size, interpolation=cv2.INTER_NEAREST)
        # cropped_img = resized_image[offset:, :]

        # Show cropped image (debug):
        cv2.imshow("Cropped Red Line Detection", cropped_img)
        cv2.waitKey(1)

        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        red_detected = self.detect_red_lines(hsv)
        yellow_detected = self.detect_yellow_lines(hsv)
        white_detected = self.detect_white_lines(hsv)
        rospy.loginfo(f"Detected: red-yellow-white -> {red_detected}-{yellow_detected}-{white_detected}")

        # These might need to be done inside run:
        if red_detected:
            self.throttle_left = 0.0
            self.throttle_right = 0.0

            self.vel_left = self.throttle_left
            self.vel_right = self.throttle_right
            return
        else:
            self.throttle_left = 0.5
            self.throttle_right = 0.5

            self.vel_left = self.throttle_left * self.forward
            self.vel_right = self.throttle_right * self.forward
        
        # if white_detected:
        # set throttles to 0.1 - spin right in place (left to forward, right to backward) 
        # if white line is not detected no more  - 0.5 forward to both

    def detect_red_lines(self, image_hsv) -> bool:
        # HSV ranges for red color:
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create masks and join them (red has two separate ranges):
        mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        return self.detect_lines(red_mask)

    def detect_yellow_lines(self, image_hsv) -> bool:
        # HSV ranges for yellow color:
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

        return self.detect_lines(yellow_mask)
    
    def detect_white_lines(self, image_hsv) -> bool:
        # HSV ranges for white color:
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(image_hsv, lower_white, upper_white)

        return self.detect_lines(white_mask)

    def detect_lines(self, mask) -> bool:
        # Prepare for Hough transform:
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines using probability Hough transform:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

        return lines is not None
    
    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            message = WheelsCmdStamped(vel_left=self.vel_left, vel_right=self.vel_right)
            self._wheels_publisher.publish(message)
            rate.sleep()

if __name__ == '__main__':
    line_detector = LineDetector(node_name='line_detector')
    line_detector.run()