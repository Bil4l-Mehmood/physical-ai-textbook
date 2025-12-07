# Chapter 1: Code Examples

This directory contains executable code examples for all Chapter 1 lessons.

## Files by Lesson

### Lesson 1.2: ROS 2 Core Concepts

- **`ros2_publisher.py`** - Simple ROS 2 publisher (String messages to `topic`)
- **`ros2_subscriber.py`** - Simple ROS 2 subscriber (listens to `topic`)

### Lesson 1.4: URDF/XACRO Basics

- **`simple_robot.urdf`** - Basic 2-link robot in pure URDF format
- **`simple_robot.urdf.xacro`** - Same robot with XACRO properties and macros
- **`humanoid_leg.urdf.xacro`** - Parameterized 3-DOF humanoid leg (hip, knee, ankle)

## Quick Start Guide

### ROS 2 Publisher/Subscriber Examples

**Terminal 1 - Source ROS 2 and Run Publisher**:

```bash
source /opt/ros/humble/setup.bash
python3 ros2_publisher.py
```

**Terminal 2 - Source ROS 2 and Run Subscriber**:

```bash
source /opt/ros/humble/setup.bash
python3 ros2_subscriber.py
```

**Terminal 3 - Monitor Topic**:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /topic
```

Expected output (Terminal 1):
```
[INFO] [minimal_publisher]: Publishing: "Hello ROS 2: 0"
[INFO] [minimal_publisher]: Publishing: "Hello ROS 2: 1"
```

Expected output (Terminal 2):
```
[INFO] [minimal_subscriber]: Heard: "Hello ROS 2: 0"
[INFO] [minimal_subscriber]: Heard: "Hello ROS 2: 1"
```

### URDF/XACRO Examples

**Test URDF Syntax**:

```bash
# Check for parsing errors
urdf_parser simple_robot.urdf
```

**Convert XACRO to URDF**:

```bash
# Process XACRO file
xacro simple_robot.urdf.xacro > simple_robot_generated.urdf

# Create humanoid leg variants
xacro humanoid_leg.urdf.xacro leg_side:=right > humanoid_leg_right.urdf
xacro humanoid_leg.urdf.xacro leg_side:=left > humanoid_leg_left.urdf
```

**Visualize Robot Structure**:

```bash
# Generate dot file showing kinematic chain
urdf_to_graphviz simple_robot.urdf > robot_structure.dot

# Convert to SVG (requires graphviz)
dot -Tsvg robot_structure.dot > robot_structure.svg
```

**View in RViz**:

```bash
# Terminal 1: Launch RViz
ros2 run rviz2 rviz2

# Terminal 2: Publish robot description and state
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$(xacro simple_robot.urdf.xacro)"
```

## File Descriptions

### ros2_publisher.py
- Node name: `minimal_publisher`
- Publishes to: `topic` (String messages)
- Rate: Every 0.5 seconds
- Message format: `Hello ROS 2: {counter}`

### ros2_subscriber.py
- Node name: `minimal_subscriber`
- Subscribes to: `topic`
- Callback: Logs each received message
- Message format: `Heard: {message data}`

### simple_robot.urdf
- Pure URDF format (no preprocessing)
- 3 links: base_link (box), link_1 (cylinder), link_2 (cylinder)
- 2 joints: joint_1 (revolute), joint_2 (revolute)
- Joint limits: ±90° (±1.5708 radians)
- Demonstrates: Basic URDF structure

### simple_robot.urdf.xacro
- Same robot as simple_robot.urdf
- Uses XACRO properties for parameters
- Includes `arm_segment` macro for code reuse
- Demonstrates: XACRO features (properties, macros, math)

### humanoid_leg.urdf.xacro
- 4 links: hip, thigh, calf, foot
- 3 joints: hip_knee, knee_ankle, ankle_foot
- Parameterized: `leg_side` argument (right/left)
- Dimensions: Thigh/calf ~0.3m, foot ~0.15m
- Demonstrates: Parameterized XACRO models

## Requirements

- **Python 3.10+**
- **ROS 2 Humble** (with rclpy and robot_state_publisher)
- **URDF tools**: `sudo apt install ros-humble-urdf-parser`
- **Optional**: RViz 2 for visualization

## Exercises

### Lesson 1.2 Exercises

1. **Modify message format**: Change `msg.data` in publisher (try timestamps)
2. **Change topic name**: Update both publisher and subscriber to `/my_messages`
3. **Change publish rate**: Modify `timer_period` (try 0.1 or 2.0 seconds)
4. **Multiple subscribers**: Run several subscriber instances simultaneously
5. **Monitor with CLI**: Use `ros2 topic list`, `ros2 node list`, `ros2 topic info`

### Lesson 1.4 Exercises

1. **Expand simple_robot**: Add a third link and joint using the macro pattern
2. **Create robot variants**: Use XACRO `${...}` expressions to create 2, 3, and 4 DOF versions
3. **Modify humanoid_leg**: Change thigh/calf lengths and regenerate both legs
4. **Add geometry**: Create a humanoid arm using the same pattern
5. **Collision testing**: Load into Gazebo and test contact dynamics

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: rclpy` | ROS 2 not sourced | Run `source /opt/ros/humble/setup.bash` |
| `URDF parse error` | Malformed XML | Check closing tags and element names |
| `xacro: command not found` | XACRO not installed | Install with `apt install ros-humble-xacro` |
| `urdf_parser: command not found` | Parser not installed | Install with `apt install ros-humble-urdf-parser` |
| Subscriber doesn't receive | Topic names mismatch | Verify exact topic name (case-sensitive) |
| "undefined reference to link" | Joint references missing link | Check all link names in joint definitions |

## Hardware

**Tested on**:
- NVIDIA Jetson Orin Nano with ROS 2 Humble
- Ubuntu 22.04 x86_64 with ROS 2 Humble (for development)

## Next Steps

- **Lesson 1.3**: Package these examples into reusable ROS 2 packages
- **Lesson 1.4**: Create complex robot models using URDF and XACRO
- **Chapter 2**: Load robot models into Isaac Sim for physics simulation

## See Also

- [Lesson 1.2: ROS 2 Core Concepts](../1-2-ros2-core.md)
- [Lesson 1.3: rclpy Packages & Launch Files](../1-3-rclpy-packages.md)
- [Lesson 1.4: URDF/XACRO Basics](../1-4-urdf-xacro.md)
- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
- [URDF Tutorial](http://wiki.ros.org/urdf/Tutorials)
