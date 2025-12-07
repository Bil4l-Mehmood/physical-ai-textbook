---
sidebar_position: 2
sidebar_label: "Lesson 1.2: ROS 2 Core Concepts"
title: "ROS 2 Core Concepts: Nodes, Topics, Services"
description: "Learn ROS 2 publish/subscribe communication and build practical Python examples"
duration: 60
difficulty: Beginner
hardware: ["Jetson Orin Nano", "ROS 2 Humble"]
prerequisites: ["Lesson 1.1: Foundations of Physical AI"]
---

# Lesson 1.2: ROS 2 Core Concepts: Nodes, Topics, Services

:::info Lesson Overview
**Duration**: 60 minutes | **Difficulty**: Beginner | **Hardware**: Jetson Orin Nano + ROS 2 Humble

**Prerequisites**: Complete Lesson 1.1 (Foundations of Physical AI)

**Learning Outcome**: Understand ROS 2's publish/subscribe architecture and write working Python examples
:::

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand ROS 2 nodes, topics, and publish/subscribe pattern
- Explain how robots communicate using topics and services
- Write a Python ROS 2 publisher that sends messages
- Write a Python ROS 2 subscriber that receives messages
- Use ROS 2 command-line tools to monitor topics
- Understand synchronous services and when to use them

## Hardware & Prerequisites

**Required Hardware**:
- NVIDIA Jetson Orin Nano with ROS 2 Humble installed
- SSH access or local terminal

**Required Software**:
- ROS 2 Humble (Humble Hawksbill LTS)
- Python 3.10+
- rclpy (ROS 2 Python library)
- std_msgs package

**Verification**: Confirm ROS 2 is installed and sourced

```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

**Expected Output**:
```
ROS 2 Humble Hawksbill
```

## ROS 2 Fundamentals

### What is ROS 2?

**ROS 2** (Robot Operating System 2) is middleware that enables robots to have multiple processes (nodes) communicate reliably and efficiently. It handles:

- **Communication**: Nodes send messages to each other
- **Scheduling**: Nodes can run in parallel
- **Hardware abstraction**: Same code works across different robots
- **Tools**: Visualization, debugging, simulation

### Nodes

A **node** is a single process that does one thing. Examples:

- `motor_controller` - Sends commands to motors
- `camera_reader` - Captures images from RealSense
- `object_detector` - Identifies objects in images
- `path_planner` - Calculates routes to goals

Multiple nodes run simultaneously and communicate through ROS 2 middleware.

### Topics: Publish/Subscribe

A **topic** is a named channel for asynchronous one-way communication.

**Example**: The `robot_velocity` topic

```
[motor_controller node]
         ↓ publishes
    /robot_velocity topic
         ↑
[path_planner node] subscribes
```

- **Publisher**: Sends messages to a topic (doesn't care who listens)
- **Subscriber**: Listens to messages on a topic
- **Topic**: Named channel (e.g., `/robot_velocity`, `/camera/image`)

**Advantages**:
- Loose coupling: Publisher and subscriber don't need to know each other
- One-to-many: One publisher, multiple subscribers
- Non-blocking: Publisher doesn't wait for subscriber to receive

**Disadvantages**:
- No guarantee of delivery
- No acknowledgment of success

### Services: Request/Response

A **service** is synchronous, request-response communication.

**Example**: The `plan_path` service

```
[client node]
    │ calls service
    │ (waits for response)
    ↓
[service server node]
    │ processes request
    │ sends response
    ↓
[client node]
    receives response
    (continues)
```

**Use cases**:
- Getting current robot state: `get_position` service
- Planning operations: `plan_path` service
- Configuration: `set_parameter` service

**Advantages**:
- Synchronous: Caller waits for response
- Acknowledgment: Server must respond
- Bidirectional: Both request and response

**Disadvantages**:
- Blocking: Caller must wait
- One-to-one: Not suitable for continuous streams

---

## Publish/Subscribe Architecture

### Publisher Example: Simple Talker

**File**: `ros2_publisher.py`

**Hardware**: Jetson Orin Nano with ROS 2 Humble

**Purpose**: Send "Hello ROS 2" messages every 0.5 seconds

```python
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
```

**Expected Output** (when run):

```
[INFO] [minimal_publisher]: Publishing: "Hello ROS 2: 0"
[INFO] [minimal_publisher]: Publishing: "Hello ROS 2: 1"
[INFO] [minimal_publisher]: Publishing: "Hello ROS 2: 2"
[INFO] [minimal_publisher]: Publishing: "Hello ROS 2: 3"
...
```

**How to run**:

```bash
# Source ROS 2 setup
source /opt/ros/humble/setup.bash

# Run the publisher
python3 ros2_publisher.py
```

**What happens**:
1. Node initializes ROS 2 communication
2. Creates publisher on topic named `topic`
3. Every 0.5 seconds, publishes message "Hello ROS 2: N"
4. Runs indefinitely (press Ctrl+C to stop)

---

### Subscriber Example: Simple Listener

**File**: `ros2_subscriber.py`

**Hardware**: Jetson Orin Nano with ROS 2 Humble

**Purpose**: Receive messages from the publisher

```python
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
```

**Expected Output** (when publisher is also running):

```
[INFO] [minimal_subscriber]: Heard: "Hello ROS 2: 0"
[INFO] [minimal_subscriber]: Heard: "Hello ROS 2: 1"
[INFO] [minimal_subscriber]: Heard: "Hello ROS 2: 2"
[INFO] [minimal_subscriber]: Heard: "Hello ROS 2: 3"
...
```

**How to run** (in separate terminals):

```bash
# Terminal 1: Run the publisher
source /opt/ros/humble/setup.bash
python3 ros2_publisher.py

# Terminal 2: Run the subscriber
source /opt/ros/humble/setup.bash
python3 ros2_subscriber.py
```

**What happens**:
1. Subscriber node initializes and subscribes to `topic`
2. When publisher publishes a message, subscriber's callback is triggered
3. Callback receives the message and logs it
4. Receiver runs indefinitely (press Ctrl+C to stop)

---

## ROS 2 Command-Line Tools

### Monitoring Topics

**List all active topics**:

```bash
ros2 topic list
```

**Output**:
```
/topic
/parameter_events
/rosout
```

**View messages on a topic** (subscribe and display):

```bash
ros2 topic echo /topic
```

**Output**:
```
data: 'Hello ROS 2: 0'
---
data: 'Hello ROS 2: 1'
---
data: 'Hello ROS 2: 2'
---
```

**Check topic message rate and type**:

```bash
ros2 topic info /topic
```

**Output**:
```
Type: std_msgs/msg/String
Publisher count: 1
Subscription count: 1
```

### Monitoring Nodes

**List all active nodes**:

```bash
ros2 node list
```

**Output**:
```
/minimal_publisher
/minimal_subscriber
```

### Services (Preview)

ROS 2 also supports services for request/response communication. Services will be covered in detail in Lesson 1.3.

---

## Hands-On Exercise

**Task**: Run the publisher and subscriber, monitor with ROS 2 tools, and modify the message.

### Step 1: Create the Python files

Save the publisher code as `ros2_publisher.py` and subscriber code as `ros2_subscriber.py`.

### Step 2: Run publisher (Terminal 1)

```bash
source /opt/ros/humble/setup.bash
python3 ros2_publisher.py
```

Expected: Messages printed every 0.5 seconds.

### Step 3: Run subscriber (Terminal 2)

```bash
source /opt/ros/humble/setup.bash
python3 ros2_subscriber.py
```

Expected: Subscriber receives and logs messages.

### Step 4: Monitor with ROS 2 tools (Terminal 3)

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /topic
```

Expected: See raw message data flowing.

### Step 5: Modify the message

Edit `ros2_publisher.py` and change:

```python
msg.data = f'Hello ROS 2: {self.i}'
```

to:

```python
msg.data = f'[Jetson Orin Nano] Message #{self.i}'
```

Stop the publisher (Ctrl+C), run it again, and observe the new message format in the subscriber output.

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'rclpy'` | ROS 2 not sourced | Run `source /opt/ros/humble/setup.bash` |
| `This node has already been initialized` | Created multiple nodes | Ensure only one `rclpy.init()` call |
| Subscriber doesn't receive messages | Topic name mismatch | Use `ros2 topic list` to verify topic names |
| `Permission denied` | File not executable | Run `chmod +x ros2_publisher.py` or use `python3 ros2_publisher.py` |
| Ctrl+C doesn't stop node | Interrupted during cleanup | Press Ctrl+C again or use `pkill -f ros2_` |

---

## Key Takeaways

✅ **Nodes** are independent processes that communicate via ROS 2

✅ **Topics** enable asynchronous publish/subscribe communication (one-to-many)

✅ **Publishers** send messages to a topic without waiting for receivers

✅ **Subscribers** listen to topics and process messages via callbacks

✅ **Services** provide synchronous request/response communication (one-to-one)

✅ **ROS 2 tools** (`ros2 topic`, `ros2 node`) help monitor and debug running systems

✅ **Decoupling** is key: Nodes don't need to know each other; communication goes through topics

## Code Example Files

Complete code examples are provided:
- `ros2_publisher.py` - Full working publisher
- `ros2_subscriber.py` - Full working subscriber

Both are executable and tested on Jetson Orin Nano with ROS 2 Humble.

---

## Further Reading

- 📖 [ROS 2 Publish and Subscribe](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html)
- 📖 [ROS 2 Services](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Services/Understanding-ROS2-Services.html)
- 📖 [rclpy Node Documentation](https://docs.ros.org/en/humble/Concepts/Intermediate/About-ROS-2-Node.html)
- 📖 [std_msgs Message Types](https://docs.ros.org/en/humble/Concepts/Intermediate/About-ROS-2-Message-Types.html)

---

## Next Steps

Ready to organize your code? Head to **[Lesson 1.3: Building rclpy Packages and Launch Files](/docs/chapter-1/1-3-rclpy-packages)** to learn how to structure ROS 2 projects and manage multiple nodes.

**Questions?** Ask on [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions) or check [Common Errors & Fixes](#common-errors--fixes)

---

*Last Updated: 2025-12-05*
*Code Examples: Tested on Jetson Orin Nano with ROS 2 Humble*
*Execution Time: ~15 min (reading) + ~20 min (hands-on) = 35 minutes*
