from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([Node(package="feature_detection_py",executable="LineDetector.py")])