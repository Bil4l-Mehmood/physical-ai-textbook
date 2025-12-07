#!/usr/bin/env python3
# Copyright (c) 2025 Physical AI Course
# License: MIT
# Target: NVIDIA Jetson Orin Nano - ROS 2 Humble

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):
    """
    A simple ROS 2 subscriber that listens to String messages.

    This node demonstrates the subscribe pattern:
    - Subscribes to topic 'topic'
    - Receives messages published by MinimalPublisher
    - Logs received messages to console
    """

    def __init__(self):
        """Initialize the subscriber node."""
        # Call parent class constructor with node name
        super().__init__('minimal_subscriber')

        # Create a subscription to topic 'topic'
        # Arguments: message_type, topic_name, callback_function, queue_size
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)

        # Suppress unused variable warning
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        """
        Callback function called whenever a message is received.

        Args:
            msg: The received message (String in this case)
        """
        # Log the received message
        self.get_logger().info(f'Heard: "{msg.data}"')


def main(args=None):
    """
    Main function: Initialize ROS 2, create node, spin, cleanup.
    """
    # Initialize ROS 2 communication
    rclpy.init(args=args)

    # Create the subscriber node
    minimal_subscriber = MinimalSubscriber()

    # Spin the node (keep it running and calling callbacks)
    # This blocks until the node is shut down (Ctrl+C)
    rclpy.spin(minimal_subscriber)

    # Cleanup when done (Ctrl+C pressed)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
