---
sidebar_position: 998
sidebar_label: "ROS 2 Code Standards"
title: "ROS 2 Code Example Standards"
description: "Guidelines for writing ROS 2 code examples in textbook lessons"
---

# ROS 2 Code Example Standards

All ROS 2 code examples in this textbook MUST follow these standards to ensure consistency, safety, and educational value.

## File Structure

### Python Files

**Naming**:
- Use descriptive names: `ros2_publisher.py`, `motor_controller.py`
- Avoid generic names: ❌ `test.py`, ❌ `main.py`

**Header**:
```python
#!/usr/bin/env python3
# Copyright (c) 2025 Physical AI Course
# License: MIT
# Target: NVIDIA Jetson Orin Nano - ROS 2 Humble
```

### Launch Files

**Naming**:
- Use descriptive names: `multi_robot.launch.py`, `perception_pipeline.launch.py`
- File extension: Always `.launch.py` (Python launch files)

**Header**:
```python
#!/usr/bin/env python3
# Copyright (c) 2025 Physical AI Course
# License: MIT
# Target: NVIDIA Jetson Orin Nano - ROS 2 Humble
```

## Code Requirements

### 1. Imports

**Required**:
```python
import rclpy
from rclpy.node import Node
```

**For specific functionality**:
```python
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
```

**Guidelines**:
- Group imports: standard library, then rclpy, then custom imports
- One import per line
- Avoid wildcard imports (`from x import *`)

### 2. Node Class Structure

**Template**:
```python
class MyNode(Node):
    """Brief description of what this node does."""

    def __init__(self):
        super().__init__('my_node_name')
        self.publisher_ = self.create_publisher(...)
        self.subscription_ = self.create_subscription(...)

    def callback(self, msg):
        """Handle received messages."""
        pass

    def main_method(self):
        """Core node functionality."""
        pass
```

**Requirements**:
- Inherit from `Node`
- Call `super().__init__('node_name')` in `__init__`
- Docstring explaining node purpose
- Callback methods for subscribers
- Use `self.get_logger().info()` for logging

### 3. Node Lifecycle Management

**Initialization**:
```python
def main(args=None):
    rclpy.init(args=args)  # Initialize ROS 2
    node = MyNode()        # Create node
    rclpy.spin(node)       # Run node
    node.destroy_node()    # Cleanup
    rclpy.shutdown()       # Shutdown ROS 2
```

**Requirements**:
- Always call `rclpy.init()` before creating nodes
- Always call `rclpy.shutdown()` after spinning
- Always call `node.destroy_node()` during cleanup
- Wrap in try/except for error handling when appropriate

### 4. Publishers

**Creating a publisher**:
```python
# In __init__
self.publisher_ = self.create_publisher(
    String,           # Message type
    'topic_name',     # Topic name (use lowercase_with_underscores)
    10)               # Queue size (QoS history depth)

# Publishing
msg = String()
msg.data = 'Hello'
self.publisher_.publish(msg)
```

**Requirements**:
- Use descriptive topic names: `robot_velocity`, `camera_image`
- Queue size typically 10 (allow up to 10 queued messages)
- Always check topic name matches subscriber

### 5. Subscribers

**Creating a subscriber**:
```python
# In __init__
self.subscription_ = self.create_subscription(
    String,              # Message type
    'topic_name',        # Topic name
    self.callback,       # Callback method
    10)                  # Queue size

# Callback
def callback(self, msg):
    self.get_logger().info(f'Received: {msg.data}')
```

**Requirements**:
- Use consistent topic names with publishers
- Callback signature: `def callback(self, msg):`
- Use callbacks for async processing
- Handle empty or invalid messages gracefully

### 6. Logging

**Requirements**:
```python
# Info: General information
self.get_logger().info('Node initialized')

# Warning: Potential issues
self.get_logger().warning('Message rate is low')

# Error: Problems that don't stop execution
self.get_logger().error('Failed to read sensor')

# Debug: Detailed diagnostic info
self.get_logger().debug(f'Received message: {msg}')
```

**Guidelines**:
- Use appropriate log levels (not everything should be INFO)
- Include context: what happened, why, what to do next
- Don't log continuously (use fixed rate or thresholds)
- Example: `self.get_logger().info(f'Velocity: {vel:.2f} m/s')`

### 7. Comments and Docstrings

**Module docstring** (at top of file):
```python
"""
Brief description of what this module/node does.

Longer explanation of the module's purpose, what problems it solves,
and how it works at a high level.
"""
```

**Class docstring**:
```python
class MyPublisher(Node):
    """
    A simple ROS 2 publisher that demonstrates publish/subscribe.

    This node:
    - Creates a publisher on topic 'topic'
    - Publishes String messages every 0.5 seconds
    - Includes error handling for ROS 2 failures
    """
```

**Method docstring**:
```python
def timer_callback(self):
    """
    Called periodically by the timer.

    Publishes a message and logs to console.
    """
```

**Inline comments**:
```python
# Create a publisher that sends String messages to 'topic'
self.publisher_ = self.create_publisher(String, 'topic', 10)

# Timer calls callback every 0.5 seconds
timer_period = 0.5  # seconds
self.timer = self.create_timer(timer_period, self.timer_callback)
```

**Guidelines**:
- Docstring: "what" and "why"
- Inline comments: explain non-obvious logic
- Avoid stating the obvious: ❌ "Increment i" for `i += 1`
- Example good comment: "Reduce velocity if approaching obstacle"

### 8. Error Handling

**Basic try/except**:
```python
def __init__(self):
    super().__init__('my_node')
    try:
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.get_logger().info('Publisher created successfully')
    except Exception as e:
        self.get_logger().error(f'Failed to create publisher: {e}')
```

**Callback error handling**:
```python
def callback(self, msg):
    try:
        data = float(msg.data)
        self.process_data(data)
    except ValueError:
        self.get_logger().error(f'Invalid data format: {msg.data}')
    except Exception as e:
        self.get_logger().error(f'Unexpected error: {e}')
```

**Requirements**:
- Wrap ROS 2 operations in try/except
- Log errors with context (what failed, why)
- Don't crash on bad input data
- Graceful degradation when possible

## Message Types

### Standard Messages (std_msgs)

```python
from std_msgs.msg import String, Float32, Float64, Int32, Bool

# String
msg = String()
msg.data = 'Hello'

# Float32
msg = Float32()
msg.data = 3.14

# Int32
msg = Int32()
msg.data = 42

# Bool
msg = Bool()
msg.data = True
```

### Geometry Messages (geometry_msgs)

```python
from geometry_msgs.msg import Twist, Point, Quaternion

# Twist (velocity commands)
cmd = Twist()
cmd.linear.x = 0.5    # m/s forward
cmd.angular.z = 0.2   # rad/s rotation

# Point
point = Point()
point.x = 1.0
point.y = 2.0
point.z = 0.0
```

### Sensor Messages (sensor_msgs)

```python
from sensor_msgs.msg import Image, LaserScan, CameraInfo

# Image (from RealSense camera)
# Image has header (timestamp, frame_id), height, width, encoding, data
```

## Launch File Standards

### Python Launch Files

**Template**:
```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='my_node',
            name='my_node_instance',
            remappings=[
                ('topic_old', 'topic_new'),
            ],
            parameters=[
                {'param_name': 'param_value'},
            ],
        ),
    ])
```

**Requirements**:
- Function name: `generate_launch_description()`
- Return `LaunchDescription` object
- Include all nodes needed for lesson example
- Document what each node does

### Node Configuration in Launch

```python
Node(
    package='package_name',           # ROS 2 package name
    executable='node_executable',     # Node name from setup.py
    name='unique_node_name',          # Instance name (can have multiples)
    remappings=[
        ('old_topic', 'new_topic'),
        ('old_service', 'new_service'),
    ],
    parameters=[
        {'param1': value1},
        {'param2': value2},
    ],
    output='screen',                  # Show logs in console
)
```

## Package Structure Standards

### package.xml

```xml
<?xml version="1.0"?>
<package format="3">
  <name>lesson_package</name>
  <version>1.0.0</version>
  <description>Brief description</description>
  <maintainer email="user@example.com">Author Name</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>
  <build_depend>rclpy</build_depend>
  <build_depend>std_msgs</build_depend>
  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
</package>
```

### setup.py

```python
from setuptools import setup

setup(
    name='lesson_package',
    version='1.0.0',
    packages=['lesson_package'],
    install_requires=['setuptools'],
    author='Author Name',
    author_email='user@example.com',
    description='Brief description',
    license='MIT',
    entry_points={
        'console_scripts': [
            'publisher = lesson_package.publisher:main',
            'subscriber = lesson_package.subscriber:main',
        ],
    },
)
```

## Testing Requirements

All code examples MUST be tested:

```bash
# Syntax check
python3 -m py_compile ros2_publisher.py

# Static analysis (optional)
pylint ros2_publisher.py

# Execution test on Jetson Orin Nano
source /opt/ros/humble/setup.bash
python3 ros2_publisher.py
```

**Verification**:
- [ ] File executes without errors
- [ ] Output matches documented expected output
- [ ] Node registers with ROS 2 (`ros2 node list` shows it)
- [ ] Topics/services appear correctly (`ros2 topic list`, etc.)

## Documentation Requirements

Every code example MUST include:

1. **Header**: Copyright, license, target hardware
2. **Module docstring**: What the module/node does
3. **Class/function docstrings**: What and why
4. **Inline comments**: Non-obvious logic
5. **Expected output**: Terminal output when run
6. **How to run**: Step-by-step commands
7. **Common errors**: Potential problems and solutions

## Validation Checklist

Before publishing code examples:

- [ ] File has proper header (shebang, copyright, license, target)
- [ ] Code is executable: `python3 -m py_compile filename.py`
- [ ] All imports are listed at top
- [ ] Node class inherits from `Node`
- [ ] Proper lifecycle management (init, spin, shutdown)
- [ ] All logging uses `self.get_logger()`
- [ ] Docstrings for module, class, methods
- [ ] Inline comments for non-obvious logic
- [ ] Expected output documented
- [ ] Tested on Jetson Orin Nano with ROS 2 Humble
- [ ] No hardcoded paths or credentials
- [ ] Error handling for failure cases
- [ ] Follows naming conventions (lowercase_with_underscores)

## Examples

See lesson examples directory for complete working examples:
- `docs/chapter-1/examples/ros2_publisher.py`
- `docs/chapter-1/examples/ros2_subscriber.py`

---

**See also**:
- [_lesson-template.md](_lesson-template.md) - Lesson structure template
- [_frontmatter-guide.md](_frontmatter-guide.md) - Frontmatter requirements
