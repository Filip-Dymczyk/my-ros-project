#!/bin/bash
source /environment.sh
# initialize launch file
dt-launchfile-init
# launch publisher
rosrun my_package line_detector.py
# wait for app to end
dt-launchfile-join