---
sidebar_position: 1
sidebar_label: "Lesson 2.1: Gazebo Fundamentals"
title: "Gazebo Simulation: Building Digital Twins"
description: "Learn to simulate robots in Gazebo with physics, collision detection, and realistic environment simulation"
duration: 90
difficulty: Intermediate
hardware: ["Ubuntu 22.04 LTS", "ROS 2 Humble", "Gazebo Harmonic (recommended)"]
prerequisites: ["Lesson 1.4: URDF/XACRO Basics"]
---

# Lesson 2.1: Gazebo Simulation Fundamentals

:::info Lesson Overview
**Duration**: 90 minutes | **Difficulty**: Intermediate | **Hardware**: Ubuntu 22.04 + ROS 2 Humble + Gazebo

**Prerequisites**: Complete Lesson 1.4 (URDF/XACRO Basics)

**Learning Outcome**: Build and simulate robots in Gazebo with physics, sensors, and realistic environments
:::

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand Gazebo architecture and simulation concepts
- Install Gazebo Harmonic with ROS 2 Humble integration
- Create and configure 3D simulation environments
- Load URDF robots into Gazebo with physics simulation
- Configure gravity, friction, and collision dynamics
- Simulate basic robot movement and interaction
- Monitor simulation state with ROS 2 topics
- Debug simulation issues and optimize performance

## Hardware & Prerequisites

**Required**:
- Ubuntu 22.04 LTS (native or VM with 8GB+ RAM)
- ROS 2 Humble installed and sourced
- URDF robots from Lesson 1.4 (simple_robot.urdf)
- Text editor (VS Code recommended)

**Optional (for visualization)**:
- Monitor with GPU support (RTX-capable machine for RTX rendering)
- 2 displays (one for Gazebo, one for monitoring)

**Verification**: Confirm prerequisites
```bash
ros2 --version
cat /etc/os-release | grep PRETTY_NAME  # Ubuntu 22.04
```

---

## Part 1: Gazebo Architecture & Concepts

:::tip What is Gazebo?
Gazebo is an open-source **physics simulator** for robotics:
- Simulates **rigid body dynamics** (gravity, friction, collisions)
- Models **sensors** (cameras, lidars, IMUs, contact sensors)
- Provides **3D visualization** and interactive environment control
- Integrates seamlessly with **ROS 2** for robot control
- Offers both **classical physics** (ODE) and **modern physics** (Bullet, DART)
:::

### Gazebo vs Other Simulators

| Feature | Gazebo | V-REP | Webots | CoppeliaSim |
|---------|--------|-------|--------|------------|
| **ROS 2 Integration** | ✅ Native | ⚠️ Limited | ⚠️ Limited | ✅ Good |
| **Physics Engine** | ✅ Multiple | ✅ Multiple | ✅ Built-in | ✅ Built-in |
| **Open Source** | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| **Humanoid Support** | ✅ Excellent | ✅ Good | ✅ Good | ✅ Good |
| **Learning Curve** | ⚠️ Steep | ✅ Easy | ✅ Easy | ✅ Easy |

**Why Gazebo for this course**:
- 100% ROS 2 compatible
- Industry standard for humanoid research
- Extensive URDF support
- Free and open-source
- Active community with excellent documentation

### Gazebo Terminology

**World**: The complete simulation environment (buildings, ground, robots)

**Model**: A robot or object in the simulation (URDF converted to SDF)

**Link**: A rigid body within a model (from URDF `<link>`)

**Joint**: A connection between links (from URDF `<joint>`)

**Sensor**: A device that generates data (camera, lidar, IMU)

**Plugin**: A dynamic library that extends Gazebo functionality

**Physics Engine**: The solver for rigid body dynamics (ODE, Bullet, DART, TPE)

---

## Part 2: Installation & Setup

### Step 1: Install Gazebo

```bash
# Update package list
sudo apt update

# Install Gazebo Harmonic (latest version)
sudo apt install gazebo-harmonic

# Install ROS 2 Gazebo bridge
sudo apt install ros-humble-gazebo-ros-pkgs

# Verify installation
gazebo --version

# Expected output:
# Gazebo version 8.0.0
```

### Step 2: Verify ROS 2 Integration

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Check if gazebo_ros package is available
ros2 pkg list | grep gazebo

# Expected output should show:
# gazebo_ros
# gazebo_plugins
# gazebo_ros2_control
```

### Step 3: Create Gazebo Workspace

```bash
# Create workspace
mkdir -p ~/gazebo_ws/src
cd ~/gazebo_ws

# Source ROS 2 and add to bashrc
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc

# Verify workspace
echo $ROS_DISTRO  # Should output: humble
```

### Step 4: Test Gazebo Launch

```bash
# Terminal 1: Launch an empty Gazebo world
ros2 launch gazebo_ros empty_world.launch.py

# Terminal 2: Verify Gazebo is running
ros2 topic list | grep gazebo

# Expected topics:
# /clock
# /gazebo/model_states
# /gazebo/set_entity_state
# /gazebo/get_entity_state
```

---

## Part 3: Simulation Description Format (SDF)

:::note SDF vs URDF
- **URDF**: Description for visualization and ROS 2 (from Lesson 1.4)
- **SDF**: Simulation-specific format with physics, sensors, and plugins
- **Relationship**: URDF is converted to SDF when loaded into Gazebo
:::

### SDF World File Structure

Create `gazebo_ws/src/worlds/simple_world.sdf`:

```xml
<?xml version="1.0" ?>
<!-- Copyright (c) 2025 Physical AI Course
     License: MIT
     Target: Ubuntu 22.04 + ROS 2 Humble + Gazebo
-->
<sdf version="1.10">
  <world name="simple_world">

    <!-- Physics configuration -->
    <physics name="physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.81</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>50</iters>
          <precon_iters>0</precon_iters>
          <sor>1.400000</sor>
        </solver>
        <constraints>
          <cfm>0.000000</cfm>
          <erp>0.200000</erp>
          <contact_max_correcting_vel>100.000000</contact_max_correcting_vel>
          <contact_surface_layer>0.001000</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Light source -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.8 0.8 0.8 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.6</mu>
                <mu2>0.6</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

### Key SDF Elements Explained

**`<physics>`**: Simulation parameters
- `max_step_size`: Time step for physics calculations (0.001 = 1ms)
- `real_time_factor`: 1.0 = real-time, 2.0 = 2x faster
- `gravity`: Earth gravity = (0, 0, -9.81) m/s²

**`<light>`**: Illumination for visualization
- `directional`: Sun-like light
- `pose`: Position and orientation (x, y, z, roll, pitch, yaw)

**`<collision>`**: Physics interaction geometry
- Used for collision detection and contact forces
- Often matches `<visual>` but can be simplified for performance

**`<friction>`**: Material properties
- `mu`: Coefficient of friction
- Critical for walking, grasping, and sliding

---

## Part 4: Loading Robots into Gazebo

### Method 1: Using ROS 2 Launch File

Create `gazebo_ws/src/launch/gazebo_robot.launch.py`:

```python
#!/usr/bin/env python3
# Copyright (c) 2025 Physical AI Course
# License: MIT
# Target: Ubuntu 22.04 + ROS 2 Humble + Gazebo

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch Gazebo with a robot loaded from URDF.

    This launch file:
    1. Starts Gazebo simulator
    2. Loads a robot URDF into the simulation
    3. Spawns necessary ROS 2 nodes for robot control
    """

    # Get path to URDF file
    urdf_file = os.path.join(
        os.path.expanduser('~'),
        'gazebo_ws', 'src', 'urdf', 'simple_robot.urdf'
    )

    # Read URDF content as string
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Start Gazebo with empty world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        )
    )

    # Spawn robot into Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'  # Start 0.5m above ground
        ],
        output='screen'
    )

    # Publish robot description
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
    ])
```

### Method 2: Direct SDF Loading

Add robot model to SDF world file:

```xml
<!-- Add this inside <world> in simple_world.sdf -->
<model name="simple_robot">
  <pose>0 0 0.5 0 0 0</pose>
  <!-- Convert URDF to SDF and include here -->
  <link name="base_link">
    <!-- ... link definition ... -->
  </link>
  <joint name="joint_1">
    <!-- ... joint definition ... -->
  </joint>
</model>
```

### Launching Gazebo with Robot

```bash
# Create URDF directory
mkdir -p ~/gazebo_ws/src/urdf
mkdir -p ~/gazebo_ws/src/launch

# Copy robot URDF from Chapter 1
cp ~/ai_robotics/chapter_1/simple_robot.urdf ~/gazebo_ws/src/urdf/

# Make launch file executable
chmod +x ~/gazebo_ws/src/launch/gazebo_robot.launch.py

# Source ROS 2
source /opt/ros/humble/setup.bash

# Launch Gazebo with robot
ros2 launch ~/gazebo_ws/src/launch/gazebo_robot.launch.py
```

**Expected Output**:
1. Gazebo window opens with empty world
2. Robot appears at coordinates (0, 0, 0.5)
3. Robot drops to ground due to gravity
4. ROS 2 topics appear for robot control

---

## Part 5: Physics Simulation & Control

### Controlling Robot Joints

```python
#!/usr/bin/env python3
"""
Robot Control via Gazebo Joint Commands
Control joints by publishing to gazebo_ros2_control topics
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time


class RobotController(Node):
    """Send joint commands to Gazebo simulated robot"""

    def __init__(self):
        super().__init__('robot_controller')

        # Publisher for joint commands
        self.joint_command_pub = self.create_publisher(
            Float64MultiArray,
            '/simple_robot_controller/commands',
            10
        )

        self.get_logger().info('Robot controller initialized')

    def send_joint_command(self, joint_positions):
        """
        Send position commands to all joints

        Args:
            joint_positions: List of target angles [joint_1, joint_2, ...]
        """
        msg = Float64MultiArray()
        msg.data = joint_positions
        self.joint_command_pub.publish(msg)
        self.get_logger().info(f'Sent joint command: {joint_positions}')

    def wave_motion(self):
        """Make robot wave (oscillate joint 1)"""
        self.get_logger().info('Starting wave motion...')

        for cycle in range(3):  # 3 complete waves
            # Move joint 1 to +90 degrees
            self.send_joint_command([1.57, 0.0])  # +90°, 0°
            time.sleep(1.0)

            # Move joint 1 to -90 degrees
            self.send_joint_command([-1.57, 0.0])  # -90°, 0°
            time.sleep(1.0)

        # Return to neutral
        self.send_joint_command([0.0, 0.0])
        self.get_logger().info('Wave motion complete')


def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    # Wait for Gazebo to fully load
    time.sleep(3)

    # Execute wave motion
    controller.wave_motion()

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Monitoring Simulation State

```bash
# Terminal 1: Launch Gazebo with robot
ros2 launch ~/gazebo_ws/src/launch/gazebo_robot.launch.py

# Terminal 2: Monitor robot state topics
ros2 topic list | grep gazebo

# View model positions
ros2 topic echo /gazebo/model_states

# View link states
ros2 topic echo /gazebo/link_states

# Monitor joint states
ros2 topic echo /joint_states
```

### Physics Tuning Parameters

| Parameter | Effect | Typical Value |
|-----------|--------|---------------|
| **gravity** | Acceleration due to gravity | -9.81 m/s² (down) |
| **max_step_size** | Physics simulation timestep | 0.001 s (1 ms) |
| **mu** (friction) | Static friction coefficient | 0.6 (rubber on concrete) |
| **erp** | Error Reduction Parameter | 0.2 (stability vs. accuracy) |
| **cfm** | Constraint Force Mixing | 0.0 (stiffer constraints) |

---

## Hands-On Exercise

**Task**: Create a Gazebo simulation with a robot and observe physics behavior.

### Step 1: Setup Environment

```bash
# Create workspace structure
mkdir -p ~/gazebo_ws/src/{urdf,launch,src,worlds}
cd ~/gazebo_ws/src

# Copy robot URDF from Chapter 1
cp ~/ai_robotics/chapter_1/simple_robot.urdf urdf/

# Copy world file
cat > worlds/physics_test.sdf << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="physics_test">
    <physics name="physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.81</gravity>
    </physics>
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.8 0.8 0.8 1</specular>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
EOF
```

### Step 2: Launch Gazebo

```bash
source /opt/ros/humble/setup.bash

# Launch with custom world
ros2 launch gazebo_ros gazebo.launch.py world:=$HOME/gazebo_ws/src/worlds/physics_test.sdf
```

### Step 3: Verify Robot Loads

```bash
# In another terminal
ros2 service list | grep gazebo

# Get model state
ros2 service call /gazebo/get_entity_state gazebo_msgs/GetEntityState "name: 'simple_robot'"
```

### Step 4: Observe Physics Behavior

- ✅ Robot falls from spawn point due to gravity
- ✅ Robot collides with ground and stops
- ✅ No penetration through ground plane
- ✅ Contact forces visible in simulation

### Exercises

1. **Gravity Experiment**: Modify world SDF to have 0 gravity and observe robot floating
2. **Friction Test**: Change ground friction from 0.6 to 0.1 and observe sliding behavior
3. **Timestep Impact**: Change max_step_size from 0.001 to 0.01 and observe stability
4. **Real-time Factor**: Set real_time_factor to 2.0 and observe simulation speed increase
5. **Multiple Objects**: Add a ball (sphere) model to the world and watch it interact with robot

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Gazebo: command not found` | Gazebo not installed | `sudo apt install gazebo-harmonic` |
| `robot not spawning` | URDF path wrong or malformed | Check file exists: `ls ~/gazebo_ws/src/urdf/simple_robot.urdf` |
| `robot falls through ground` | Collision geometry missing | Add `<collision>` element to ground plane link |
| `simulation very slow` | Timestep too small (0.0001) | Increase to 0.001 or 0.01 |
| `robot jerky movement` | Physics instability (high ERP) | Reduce ERP from 0.5 to 0.2 |
| `No ROS 2 topics from Gazebo` | gazebo_ros not installed | `sudo apt install ros-humble-gazebo-ros-pkgs` |
| `URDF to SDF conversion failed` | URDF has SDF-incompatible elements | Check for unsupported URDF tags |

---

## Key Takeaways

✅ **Gazebo Architecture**: World → Physics Engine → Models → Links & Joints

✅ **SDF Format**: Gazebo's native format with physics, sensors, and plugins

✅ **Physics Simulation**: ODE solver with configurable gravity, friction, constraints

✅ **ROS 2 Integration**: Gazebo publishes state topics, accepts command topics

✅ **Performance Tuning**: Timestep, ERP, CFM, and friction affect realism vs. speed

✅ **Robot Loading**: Convert URDF to SDF automatically when spawning into Gazebo

✅ **Debugging**: Use ROS 2 topics to monitor simulation state in real-time

---

## Further Reading

- 📖 [Gazebo Documentation](https://gazebosim.org/docs)
- 📖 [ROS 2 Gazebo Integration](https://docs.ros.org/en/humble/Tutorials/Intermediate/Gazebo.html)
- 📖 [SDF Format Specification](http://sdformat.org/)
- 📖 [ODE Physics Engine](http://www.ode.org/wiki/index.php/Manual)

---

**Next Lesson**: [Lesson 2.2: URDF/SDF & Physics Configuration](2-2-urdf-sdf-physics.md)

**Questions?** See [FAQ](../faq.md) or post in [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
