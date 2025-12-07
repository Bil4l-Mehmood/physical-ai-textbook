---
sidebar_position: 3
sidebar_label: "Lesson 1.3: rclpy Packages & Launch"
title: "Building rclpy Packages and Launch Files"
description: "Learn how to structure ROS 2 Python packages and create launch files for multi-node systems"
duration: 90
difficulty: Intermediate
hardware: ["Jetson Orin Nano", "ROS 2 Humble"]
prerequisites: ["Lesson 1.2: ROS 2 Core Concepts"]
---

# Lesson 1.3: Building rclpy Packages and Launch Files

:::info Lesson Overview
**Duration**: 90 minutes | **Difficulty**: Intermediate | **Hardware**: Jetson Orin Nano + ROS 2 Humble

**Prerequisites**: Complete Lesson 1.2 (ROS 2 Core Concepts)

**Learning Outcome**: Understand ROS 2 package structure and create launch files to manage multi-node systems
:::

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand the ROS 2 package directory structure and conventions
- Create a Python ROS 2 package with proper `package.xml` and `setup.py`
- Write ROS 2 launch files in Python (`.launch.py`)
- Manage multiple nodes and configure node parameters
- Use environment variables and remapping in launch files

## Hardware & Prerequisites

**Required Hardware**:
- NVIDIA Jetson Orin Nano with ROS 2 Humble installed
- Development environment (SSH or local terminal)

**Required Software**:
- ROS 2 Humble (or newer)
- Python 3.10+
- colcon build system
- Text editor (nano, vim, or VS Code)

**Verification**: Confirm ROS 2 is installed
```bash
ros2 --version
```

Expected output: `ROS 2 Humble`

## ROS 2 Package Structure

:::note Package Anatomy
A ROS 2 Python package contains:
- `package.xml` - Package metadata and dependencies
- `setup.py` - Python installation configuration
- `setup.cfg` - Setup configuration file
- `resource/` - Package resources
- `{package_name}/` - Python source code directory
- `launch/` - Launch file directory
- `test/` - Unit tests directory
:::

### The `package.xml` File

Every ROS 2 package starts with `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_ros2_package</name>
  <version>0.0.1</version>
  <description>My first ROS 2 package</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache License 2.0</license>

  <!-- Build dependencies -->
  <build_depend>rclpy</build_depend>
  <build_depend>std_msgs</build_depend>

  <!-- Runtime dependencies -->
  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>

  <buildtool_depend>ament_python</buildtool_depend>
</package>
```

### The `setup.py` File

```python
from setuptools import setup

setup(
    name='my_ros2_package',
    version='0.0.1',
    packages=['my_ros2_package'],
    py_modules=[],
    install_requires=['setuptools'],
    author='Your Name',
    author_email='user@example.com',
    description='My first ROS 2 package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_ros2_package.publisher:main',
            'listener = my_ros2_package.subscriber:main',
        ],
    },
)
```

## Creating a ROS 2 Package

### Step 1: Create Package Structure

```bash
# Navigate to ROS 2 workspace
cd ~/ros2_ws/src

# Create a new package
ros2 pkg create --build-type ament_python my_first_package
```

### Step 2: Create Node Files

Create `my_first_package/my_first_package/talker.py`:

```python
#!/usr/bin/env python3
# Jetson Orin Nano target - ROS 2 Humble

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello ROS 2: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3: Update `setup.py`

```python
entry_points={
    'console_scripts': [
        'talker = my_first_package.talker:main',
    ],
},
```

### Step 4: Build the Package

```bash
# From workspace root
cd ~/ros2_ws
colcon build --packages-select my_first_package
source install/setup.bash
```

## Launch Files

:::tip Launch File Purpose
Launch files manage complex multi-node systems by:
- Starting multiple nodes
- Setting parameters
- Remapping topics
- Managing node lifecycle
:::

### Creating a Python Launch File

Create `my_first_package/launch/talker_listener.launch.py`:

```python
#!/usr/bin/env python3
# Jetson Orin Nano target - ROS 2 Humble

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_first_package',
            executable='talker',
            name='talker_node',
            remappings=[
                ('topic', 'chatter'),
            ],
        ),
        Node(
            package='my_first_package',
            executable='listener',
            name='listener_node',
            remappings=[
                ('topic', 'chatter'),
            ],
        ),
    ])
```

### Using the Launch File

```bash
ros2 launch my_first_package talker_listener.launch.py
```

## Complete Example: Multi-Node Package

### Full Package Listing

```
lesson_1_3_package/
├── package.xml                        # Package metadata
├── setup.py                           # Python setup
├── setup.cfg                          # Setup configuration
├── lesson_1_3_package/
│   ├── __init__.py                   # Python package marker
│   ├── talker.py                     # Publisher node
│   └── listener.py                   # Subscriber node
└── launch/
    └── talker_listener.launch.py      # Launch file
```

### Complete Code Files

**`lesson_1_3_package/talker.py`**:

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
    A simple ROS 2 publisher in a reusable package.

    This is the talker node - it publishes messages to the 'chatter' topic.
    """

    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.get_logger().info('Talker node started')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello ROS 2: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**`lesson_1_3_package/listener.py`**:

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
    A simple ROS 2 subscriber in a reusable package.

    This is the listener node - it receives messages from the 'chatter' topic.
    """

    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Listener node started')

    def listener_callback(self, msg):
        self.get_logger().info(f'Heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**`launch/talker_listener.launch.py`**:

```python
#!/usr/bin/env python3
# Copyright (c) 2025 Physical AI Course
# License: MIT
# Target: NVIDIA Jetson Orin Nano - ROS 2 Humble

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Create launch description for talker and listener nodes.

    This launch file starts both the publisher (talker) and subscriber (listener)
    nodes, which communicate through the 'chatter' topic.
    """
    return LaunchDescription([
        # Start the talker node
        Node(
            package='lesson_1_3_package',
            executable='talker',
            name='talker_node',
            output='screen',  # Show logs in terminal
        ),
        # Start the listener node
        Node(
            package='lesson_1_3_package',
            executable='listener',
            name='listener_node',
            output='screen',  # Show logs in terminal
        ),
    ])
```

## Hands-On Exercise

**Task**: Create a complete ROS 2 package with publisher and subscriber nodes, and a launch file to run both together.

### Step-by-Step Instructions

#### 1. Create Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python lesson_1_3_package
cd lesson_1_3_package
```

#### 2. Create Nodes

Copy the talker and listener code above into:
- `lesson_1_3_package/talker.py`
- `lesson_1_3_package/listener.py`

#### 3. Update setup.py

Edit `setup.py` to add entry points:

```python
entry_points={
    'console_scripts': [
        'talker = lesson_1_3_package.talker:main',
        'listener = lesson_1_3_package.listener:main',
    ],
},
```

#### 4. Create Launch Directory and File

```bash
mkdir -p launch
# Copy launch file code above to launch/talker_listener.launch.py
chmod +x launch/talker_listener.launch.py
```

#### 5. Build Package

```bash
cd ~/ros2_ws
colcon build --packages-select lesson_1_3_package
source install/setup.bash
```

#### 6. Run Launch File

```bash
ros2 launch lesson_1_3_package talker_listener.launch.py
```

**Expected Output**:
```
[INFO] [talker_node]: Talker node started
[INFO] [listener_node]: Listener node started
[INFO] [talker_node]: Publishing: 'Hello ROS 2: 0'
[INFO] [listener_node]: Heard: 'Hello ROS 2: 0'
[INFO] [talker_node]: Publishing: 'Hello ROS 2: 1'
[INFO] [listener_node]: Heard: 'Hello ROS 2: 1'
...
```

### Testing

Verify the package works:

```bash
# List active nodes
ros2 node list
# Output: /talker_node, /listener_node

# List active topics
ros2 topic list
# Output: /chatter, /parameter_events, /rosout

# Monitor the topic
ros2 topic echo /chatter
# Output: data: 'Hello ROS 2: 0', data: 'Hello ROS 2: 1', ...
```

### Exercises

1. **Modify topic name**: Change 'chatter' to '/my_robot_messages' in both nodes and launch file
2. **Change publish rate**: Edit timer_period in talker.py (try 0.2 for faster)
3. **Add a third node**: Create a "repeater" node that subscribes to 'chatter', processes messages, and publishes to a new topic
4. **Add parameters**: Use `declare_parameter` and `get_parameter` to configure timer period from launch file
5. **Error handling**: Add try/except blocks and test graceful shutdown (Ctrl+C)

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Package not found` | Package not built or sourced | Run `colcon build` and `source install/setup.bash` |
| `Could not find executable` | Entry point not in `setup.py` | Update `setup.py` with correct node name |
| `Topic mismatch` | Publisher and subscriber on different topics | Use `remappings` in launch file or verify topic names match |
| `ModuleNotFoundError: No module named 'rclpy'` | ROS 2 not sourced | Run `source /opt/ros/humble/setup.bash` |

## Key Takeaways

✅ ROS 2 packages require `package.xml`, `setup.py`, and source code organized in a package directory

✅ `entry_points` in `setup.py` define executables that can be run with `ros2 run`

✅ Launch files (`.launch.py`) manage complex multi-node systems and parameter configuration

✅ Use `remappings` to dynamically change topic and service names without code changes

✅ `colcon build` compiles packages; remember to source the install setup.bash

## Further Reading

- 📖 [ROS 2 Package Concepts](https://docs.ros.org/en/humble/Concepts/Intermediate/About-ROS-2-Package-Creation.html)
- 📖 [ROS 2 Launch Documentation](https://docs.ros.org/en/humble/Concepts/Intermediate/Launch/Basic-Launch.html)
- 📖 [ROS 2 Python Package Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html)

---

**Ready for the next lesson?** Continue to [Lesson 1.4: URDF/XACRO Basics](1-4-urdf-xacro)

**Have questions?** Check [Common Errors & Fixes](#common-errors--fixes) or ask in [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
