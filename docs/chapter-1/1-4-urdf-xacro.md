---
sidebar_position: 4
sidebar_label: "Lesson 1.4: URDF/XACRO Basics"
title: "Defining the Physical Body: URDF/XACRO"
description: "Learn how to describe robot structure using URDF and parameterize with XACRO"
duration: 90
difficulty: Intermediate
hardware: ["Jetson Orin Nano", "Isaac Sim (optional)"]
prerequisites: ["Lesson 1.3: rclpy Packages & Launch Files"]
---

# Lesson 1.4: Defining the Physical Body: URDF/XACRO

:::info Lesson Overview
**Duration**: 90 minutes | **Difficulty**: Intermediate | **Hardware**: Jetson Orin Nano + Isaac Sim (optional)

**Prerequisites**: Complete Lesson 1.3 (rclpy Packages & Launch Files)

**Learning Outcome**: Create robot descriptions using URDF and parameterize them with XACRO for reusability
:::

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand URDF (Unified Robot Description Format) XML structure
- Define robot links and joints in URDF with geometry and inertia
- Create parameterized robot descriptions using XACRO macros
- Use XACRO properties and macros for code reusability
- Load and validate URDF files with ROS 2 tools
- Visualize URDF in RViz and Isaac Sim

## Hardware & Prerequisites

**Required**:
- Jetson Orin Nano with ROS 2 Humble installed
- URDF parser tools (included with ROS 2)
- Text editor (nano, vim, or VS Code)

**Optional (for visualization)**:
- RViz 2 (comes with ROS 2 Desktop)
- NVIDIA Isaac Sim 2024.1+ for 3D simulation

**Verification**: Confirm URDF tools are available
```bash
ros2 pkg list | grep urdf
```

## URDF: Unified Robot Description Format

:::tip URDF Purpose
URDF is an XML format that describes a robot's physical structure:
- **Links**: Rigid bodies with geometry, inertia, and visual properties
- **Joints**: Connections between links (revolute, prismatic, fixed)
- **Frames**: Coordinate systems for transformations
- **Collision and Visual Models**: For simulation and visualization
:::

### URDF Anatomy

Every URDF robot has this basic structure:

```xml
<?xml version="1.0" ?>
<robot name="my_robot">
  <!-- Define robot links (bodies) -->
  <link name="base_link">
    <!-- Visual geometry for rendering -->
    <visual>
      <geometry>
        <box size="0.1 0.1 0.2"/>
      </geometry>
    </visual>
    <!-- Collision geometry for simulation -->
    <collision>
      <geometry>
        <box size="0.1 0.1 0.2"/>
      </geometry>
    </collision>
    <!-- Mass and inertia for physics -->
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Define joints (connections between links) -->
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="3.14159" effort="10" velocity="1"/>
  </joint>

  <!-- More links and joints... -->
</robot>
```

### Key URDF Elements

**Link Element**:
- `<visual>`: How the link appears in RViz/simulation
- `<collision>`: Collision geometry for physics
- `<inertial>`: Mass and inertia properties

**Joint Element**:
- `type`: revolute (rotates), prismatic (slides), fixed, continuous
- `<parent>`: Link this joint is attached to
- `<child>`: Link being connected
- `<axis>`: Direction of rotation/translation (x, y, or z)
- `<limit>`: Range of motion (lower, upper, effort, velocity)

**Origin (Transform)**:
- `xyz`: Position relative to parent (x y z in meters)
- `rpy`: Rotation in roll-pitch-yaw (radians)

## Creating Your First URDF

### Simple Two-Link Robot

Create `simple_robot.urdf`:

```xml
<?xml version="1.0" ?>
<!-- Copyright (c) 2025 Physical AI Course
     License: MIT
     Target: NVIDIA Jetson Orin Nano - ROS 2 Humble
-->
<robot name="simple_robot">
  <!-- Base link (fixed to world) -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="base_material">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting base to first link -->
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="10" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- First arm link -->
  <link name="link_1">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <cylinder radius="0.02" length="0.1"/>
      </geometry>
      <material name="link_material">
        <color rgba="0.2 0.2 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <cylinder radius="0.02" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting first to second link -->
  <joint name="joint_2" type="revolute">
    <parent link="link_1"/>
    <child link="link_2"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="10" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Second arm link (end effector) -->
  <link name="link_2">
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.0005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <cylinder radius="0.015" length="0.1"/>
      </geometry>
      <material name="endeffector_material">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <cylinder radius="0.015" length="0.1"/>
      </geometry>
    </collision>
  </link>
</robot>
```

### Understanding the Example

**Links**:
- `base_link`: Main body (box geometry)
- `link_1`: First segment (cylinder)
- `link_2`: Second segment (end effector)

**Joints**:
- `joint_1`: Connects base to first link (revolute, ±90°)
- `joint_2`: Connects first to second link (revolute, ±90°)

**Properties**:
- `mass`: Weight in kg (affects physics simulation)
- `inertia`: Resistance to rotation (simplified as diagonal matrix)
- `color`: RGBA values (0-1 scale) for visualization

## XACRO: eXtensible Robot Configuration

:::note XACRO Purpose
XACRO is a preprocessing language that extends URDF with:
- **Variables**: Define parameters once, use everywhere
- **Macros**: Reusable blocks of XML code
- **Conditionals**: If-then blocks for flexibility
- **Math expressions**: Calculate values dynamically
:::

### Basic XACRO Syntax

XACRO files use `*.urdf.xacro` extension and are preprocessed into `.urdf`:

```bash
xacro simple_robot.urdf.xacro > simple_robot.urdf
```

### XACRO Properties

Create `simple_robot.urdf.xacro`:

```xml
<?xml version="1.0" ?>
<!-- Copyright (c) 2025 Physical AI Course
     License: MIT
     Target: NVIDIA Jetson Orin Nano - ROS 2 Humble
-->
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Define reusable properties -->
  <xacro:property name="base_mass" value="1.0"/>
  <xacro:property name="link_mass" value="0.5"/>
  <xacro:property name="ee_mass" value="0.3"/>

  <xacro:property name="base_length" value="0.1"/>
  <xacro:property name="base_width" value="0.1"/>
  <xacro:property name="base_height" value="0.1"/>

  <xacro:property name="link_length" value="0.1"/>
  <xacro:property name="link_radius" value="0.02"/>

  <xacro:property name="joint_limit_min" value="-1.5708"/>
  <xacro:property name="joint_limit_max" value="1.5708"/>
  <xacro:property name="joint_effort" value="10"/>
  <xacro:property name="joint_velocity" value="1"/>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="${base_mass}"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 ${base_height/2}"/>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="base_material">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 ${base_height/2}"/>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
  </link>

  <!-- Macro for creating arm segments -->
  <xacro:macro name="arm_segment" params="name parent joint_name x_offset">
    <joint name="${joint_name}" type="revolute">
      <parent link="${parent}"/>
      <child link="${name}"/>
      <origin xyz="${x_offset} 0 ${link_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${joint_limit_min}" upper="${joint_limit_max}"
             effort="${joint_effort}" velocity="${joint_velocity}"/>
      <dynamics damping="0.1" friction="0.0"/>
    </joint>

    <link name="${name}">
      <inertial>
        <mass value="${link_mass}"/>
        <origin xyz="0 0 ${link_length/2}"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
      </inertial>
      <visual>
        <origin xyz="0 0 ${link_length/2}"/>
        <geometry>
          <cylinder radius="${link_radius}" length="${link_length}"/>
        </geometry>
        <material name="link_material">
          <color rgba="0.2 0.2 0.8 1.0"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 ${link_length/2}"/>
        <geometry>
          <cylinder radius="${link_radius}" length="${link_length}"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Create arm segments using macro -->
  <xacro:arm_segment name="link_1" parent="base_link"
                     joint_name="joint_1" x_offset="0"/>
  <xacro:arm_segment name="link_2" parent="link_1"
                     joint_name="joint_2" x_offset="0"/>

</robot>
```

### Key XACRO Features

| Feature | Syntax | Purpose |
|---------|--------|---------|
| Property | `<xacro:property name="var" value="10"/>` | Define constant |
| Reference | `${var_name}` | Use property value |
| Math | `${3.14159 / 2}` | Calculate values |
| Macro | `<xacro:macro name="name" params="...">` | Define reusable block |
| Macro call | `<xacro:macro_name param="value"/>` | Use macro |

## Complete Example: Parameterized Humanoid Leg

Create a reusable humanoid leg that can be instantiated with different parameters:

**`humanoid_leg.urdf.xacro`**:

```xml
<?xml version="1.0" ?>
<!-- Copyright (c) 2025 Physical AI Course
     License: MIT
     Target: NVIDIA Jetson Orin Nano - ROS 2 Humble
-->
<robot name="humanoid_leg" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Configurable parameters for leg -->
  <xacro:arg name="leg_side" default="right"/>

  <xacro:property name="leg_side" value="$(arg leg_side)"/>

  <!-- Leg dimensions (in meters) -->
  <xacro:property name="hip_length" value="0.08"/>
  <xacro:property name="thigh_length" value="0.3"/>
  <xacro:property name="calf_length" value="0.3"/>
  <xacro:property name="foot_length" value="0.15"/>
  <xacro:property name="foot_width" value="0.08"/>

  <xacro:property name="segment_radius" value="0.02"/>

  <!-- Mass distribution -->
  <xacro:property name="hip_mass" value="0.5"/>
  <xacro:property name="thigh_mass" value="1.5"/>
  <xacro:property name="calf_mass" value="1.0"/>
  <xacro:property name="foot_mass" value="0.5"/>

  <!-- Hip (pelvis connection point) -->
  <link name="hip_${leg_side}">
    <inertial>
      <mass value="${hip_mass}"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="${segment_radius * 1.5}"/>
      </geometry>
      <material name="hip_material">
        <color rgba="0.3 0.3 0.3 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="${segment_radius * 1.5}"/>
      </geometry>
    </collision>
  </link>

  <!-- Hip-knee joint -->
  <joint name="hip_knee_${leg_side}" type="revolute">
    <parent link="hip_${leg_side}"/>
    <child link="thigh_${leg_side}"/>
    <origin xyz="0 0 -${hip_length}" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="50" velocity="2"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <!-- Thigh -->
  <link name="thigh_${leg_side}">
    <inertial>
      <mass value="${thigh_mass}"/>
      <origin xyz="0 0 -${thigh_length/2}"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -${thigh_length/2}"/>
      <geometry>
        <cylinder radius="${segment_radius}" length="${thigh_length}"/>
      </geometry>
      <material name="thigh_material">
        <color rgba="0.2 0.2 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -${thigh_length/2}"/>
      <geometry>
        <cylinder radius="${segment_radius}" length="${thigh_length}"/>
      </geometry>
    </collision>
  </link>

  <!-- Knee joint -->
  <joint name="knee_ankle_${leg_side}" type="revolute">
    <parent link="thigh_${leg_side}"/>
    <child link="calf_${leg_side}"/>
    <origin xyz="0 0 -${thigh_length}" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.5" effort="50" velocity="2"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <!-- Calf -->
  <link name="calf_${leg_side}">
    <inertial>
      <mass value="${calf_mass}"/>
      <origin xyz="0 0 -${calf_length/2}"/>
      <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -${calf_length/2}"/>
      <geometry>
        <cylinder radius="${segment_radius}" length="${calf_length}"/>
      </geometry>
      <material name="calf_material">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -${calf_length/2}"/>
      <geometry>
        <cylinder radius="${segment_radius}" length="${calf_length}"/>
      </geometry>
    </collision>
  </link>

  <!-- Ankle joint -->
  <joint name="ankle_foot_${leg_side}" type="revolute">
    <parent link="calf_${leg_side}"/>
    <child link="foot_${leg_side}"/>
    <origin xyz="0 0 -${calf_length}" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.7854" upper="0.7854" effort="20" velocity="1"/>
    <dynamics damping="0.3" friction="0.0"/>
  </joint>

  <!-- Foot (contact with ground) -->
  <link name="foot_${leg_side}">
    <inertial>
      <mass value="${foot_mass}"/>
      <origin xyz="${foot_length/2} 0 -${foot_width/2}"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="${foot_length/2} 0 -${foot_width/2}"/>
      <geometry>
        <box size="${foot_length} 0.04 ${foot_width}"/>
      </geometry>
      <material name="foot_material">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="${foot_length/2} 0 -${foot_width/2}"/>
      <geometry>
        <box size="${foot_length} 0.04 ${foot_width}"/>
      </geometry>
    </collision>
  </link>

</robot>
```

## Loading and Validating URDF

### Converting XACRO to URDF

```bash
# Convert XACRO to URDF
xacro humanoid_leg.urdf.xacro > humanoid_leg.urdf

# Create right and left legs
xacro humanoid_leg.urdf.xacro leg_side:=right > humanoid_leg_right.urdf
xacro humanoid_leg.urdf.xacro leg_side:=left > humanoid_leg_left.urdf
```

### Validating URDF Syntax

```bash
# Check for parsing errors
urdf_parser humanoid_leg.urdf

# Print robot structure
urdf_to_graphviz humanoid_leg.urdf
```

### Viewing in RViz

```bash
# Launch RViz with URDF
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix rviz2)/share/rviz2/rviz/default.rviz

# In another terminal, publish the robot description
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(xacro ~/humanoid_leg.urdf.xacro)"
```

## Hands-On Exercise

**Task**: Create a parameterized 2-DOF robotic arm that can be instantiated at different sizes.

### Step-by-Step Instructions

#### 1. Create XACRO Template

Create `robotic_arm.urdf.xacro`:

```xml
<?xml version="1.0" ?>
<robot name="robotic_arm" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Exercise: Fill in the parameters and macro definitions -->

  <!-- TODO: Define properties for arm length, mass, joint limits -->

  <!-- TODO: Create arm_segment macro with parameters -->

  <!-- TODO: Instantiate multiple segments using macro -->
</robot>
```

#### 2. Parameterize the Design

```bash
# Create URDF with different configurations
xacro robotic_arm.urdf.xacro > arm_standard.urdf
xacro robotic_arm.urdf.xacro segment_count:=3 > arm_3dof.urdf
xacro robotic_arm.urdf.xacro segment_count:=4 > arm_4dof.urdf
```

#### 3. Validate

```bash
# Check for syntax errors
urdf_parser arm_standard.urdf

# View structure
urdf_to_graphviz arm_standard.urdf | dot -Tsvg > arm_structure.svg
```

#### 4. Visualize

```bash
# Launch RViz and visualize the arm
ros2 launch robot_state_publisher robot_state_publisher.launch.py
  model:=arm_standard.urdf
```

### Exercises

1. **Parameter Study**: Create arm configurations with 2, 3, and 4 DOF by changing `segment_count`
2. **Mass Distribution**: Modify link masses and observe how inertia properties change
3. **Payload Capacity**: Add an end effector and calculate maximum payload
4. **Joint Limits**: Change joint limits and verify in visualization
5. **Collision Detection**: Add collision geometries and test with Gazebo

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `URDF parse error` | Malformed XML | Check closing tags and quotes |
| `undefined reference to 'link_name'` | Joint references non-existent link | Verify all link names match exactly |
| `inertia value is negative` | Inertia matrix not positive definite | Use positive diagonal values |
| `XACRO: undefined property` | Typo in property name | Check `${variable_name}` spelling |
| `Unable to load robot model` | File path incorrect | Use absolute path or check working directory |
| `TF tree has a cycle` | Circular parent-child relationship | Verify joint hierarchy is a tree (no loops) |

## Key Takeaways

✅ **URDF Structure**: Links (bodies) connected by joints (constraints) form robot kinematic chain

✅ **Geometry Types**: Boxes, cylinders, spheres, and meshes provide flexible collision/visual representation

✅ **Inertia Properties**: Mass, inertia tensor, and center of mass are crucial for physics simulation

✅ **XACRO Power**: Properties and macros eliminate repetition and enable parameterized robot designs

✅ **Validation**: Always validate URDF syntax with `urdf_parser` before visualization

✅ **Visualization**: RViz, Gazebo, and Isaac Sim display URDF models and enable collision testing

---

## Further Reading

- 📖 [URDF Documentation](http://wiki.ros.org/urdf)
- 📖 [XACRO Documentation](http://wiki.ros.org/xacro)
- 📖 [Building a Visual Robot Model with URDF](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)

---

**Coming Next**: Chapter 2 - The Digital Twin

**Questions?** Ask in [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
