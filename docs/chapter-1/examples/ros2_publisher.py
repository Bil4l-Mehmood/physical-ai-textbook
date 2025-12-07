#!/usr/bin/env python3
# Copyright (c) 2025 Physical AI Course
# License: MIT
# Target: NVIDIA Jetson Orin Nano - ROS 2 Humble

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    """
    A simple ROS 2 publisher that sends String messages.

    This node demonstrates the basic publish/subscribe pattern:
    - Creates a publisher on topic 'topic'
    - Publishes a message every 0.5 seconds
    - Runs indefinitely until interrupted
    """

    def __init__(self):
        """Initialize the publisher node."""
        # Call parent class constructor with node name
        super().__init__('minimal_publisher')

        # Create a publisher that publishes String messages
        # Arguments: message_type, topic_name, queue_size
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # Create a timer that calls timer_callback every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter for messages
        self.i = 0

    def timer_callback(self):
        """
        Called periodically by the timer.
        Publishes a message and logs to console.
        """
        # Create a String message
        msg = String()
        msg.data = f'Hello ROS 2: {self.i}'

        # Publish the message
        self.publisher_.publish(msg)

        # Log to console (appears in stdout)
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Increment counter
        self.i += 1


def main(args=None):
    """
    Main function: Initialize ROS 2, create node, spin, cleanup.
    """
    # Initialize ROS 2 communication
    rclpy.init(args=args)

    # Create the publisher node
    minimal_publisher = MinimalPublisher()

    # Spin the node (keep it running and processing callbacks)
    # This blocks until the node is shut down (Ctrl+C)
    rclpy.spin(minimal_publisher)

    # Cleanup when done (Ctrl+C pressed)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
