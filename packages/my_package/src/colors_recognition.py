#!/usr/bin/env python3
import os
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, AntiInstagramThresholds, SegmentList, Segment
from sensor_msgs.msg import CompressedImage
from .line_detector import LineDetector
from .detections import Detections
from .color_range import ColorRange
from .plot_detections import plotSegments, plotMaps
import numpy as np

# throttle and direction for each wheel
THROTTLE_LEFT = 0.5  # 50% throttle
FORWARD = 1   # forward
THROTTLE_RIGHT = 0.5 # 30% throttle
BACKWARD = -1 # backward
#https://github.com/duckietown/sim-duckiebot-lanefollowing-demo/blob/master/custom_line_detector/include/line_detector/line_detector2.py
class LineDetectorNode(DTROS):
    def __init__(self, node_name):
        super(LineDetectorNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"

        self._line_detector = LineDetector()
        self._detections = Detections()
        
        self._yellow_range = ColorRange(low=np.array([20, 100, 100]), high=np.array([40, 255, 255]))
        self._white_range = ColorRange(low=np.array([0, 0, 200]), high=np.array([180, 20, 255]))
        self._red_range = ColorRange(low=np.array([0, 120, 70]), high=np.array([10, 255, 255]))

        self._wheels_publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

        compressed_image_sub = '~camera_node/image/compressed'
        self.image_sub = rospy.Subscriber(compressed_image_sub, CompressedImage, self.image_callback)
        self.threshold_sub = rospy.Subscriber('~anti_instagram_node/thresholds', AntiInstagramThresholds, self.threshold_callback)
        
        # Debug publishers
        self.debug_segments_pub = rospy.Publisher('~debug/segments/compressed', CompressedImage, queue_size=1)
        self.debug_edges_pub = rospy.Publisher('~debug/edges/compressed', CompressedImage, queue_size=1)

        # Segment result publisher
        self.segment_list_pub = rospy.Publisher('~segment_list', SegmentList, queue_size=1)
    
    def image_callback(self, image):
        # Convert the image to a NumPy array
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # # Resize the image and apply top cutoff
        # image_resized = cv2.resize(image, tuple(self.img_size))
        # image_cropped = image_resized[self.top_cutoff:, :]

        # Process the image using the LineDetector
        self.line_detector.setImage(image)

        # Detect lines for each color range (yellow, white, and red)
        # yellow_detections = self.line_detector.detectLines(self.yellow_range)
        # white_detections = self.line_detector.detectLines(self.white_range)
        red_detections = self._line_detector.detectLines(self._red_range)

        self._line_detector.plotMaps(image, red_detections)
        self._line_detector.plotSegments(image, red_detections)
        # # Prepare SegmentList message
        # segment_list = SegmentList()
        # segment_list.header = Header(stamp=rospy.Time.now(), frame_id="camera_frame")

        # # Add the detected line segments to the SegmentList message
        # segment_list.segments = []
        # self.add_segments_to_list(segment_list, yellow_detections)
        # self.add_segments_to_list(segment_list, white_detections)
        # self.add_segments_to_list(segment_list, red_detections)

        # # Publish the segment list
        # self.segment_list_pub.publish(segment_list)

        # # Debugging (optional)
        # self.publish_debug_images(image_cropped, yellow_detections, white_detections, red_detections)

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():            
            rate.sleep()


if __name__ == "__main__":
    line_detector = LineDetectorNode(node_name='line')
    line_detector.run()