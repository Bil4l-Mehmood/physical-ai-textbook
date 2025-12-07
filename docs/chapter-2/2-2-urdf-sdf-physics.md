---
sidebar_position: 2
sidebar_label: "Lesson 2.2: URDF/SDF & Physics"
title: "Converting URDF to SDF and Configuring Physics"
description: "Master URDF-to-SDF conversion and tune physics parameters for realistic robot simulation"
duration: 90
difficulty: Intermediate
hardware: ["Ubuntu 22.04 LTS", "ROS 2 Humble", "Gazebo Harmonic"]
prerequisites: ["Lesson 1.4: URDF/XACRO Basics", "Lesson 2.1: Gazebo Fundamentals"]
---

# Lesson 2.2: URDF/SDF & Physics Configuration

:::info Lesson Overview
**Duration**: 90 minutes | **Difficulty**: Intermediate | **Hardware**: Ubuntu 22.04 + ROS 2 Humble + Gazebo

**Prerequisites**: Lessons 1.4 and 2.1

**Learning Outcome**: Convert URDF to SDF and configure realistic physics for simulation
:::

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand URDF limitations for simulation
- Convert URDF files to SDF format using tools
- Add simulation-specific parameters to models
- Configure inertia properties for accurate physics
- Tune friction, damping, and contact parameters
- Test and validate physics configurations
- Optimize simulation stability and realism
- Debug physics errors and instabilities

## Part 1: URDF vs SDF for Simulation

### Key Differences

| Feature | URDF | SDF |
|---------|------|-----|
| **Purpose** | Visualization + ROS 2 communication | Physics simulation |
| **Inertia Requirement** | Optional | **Required** for accurate physics |
| **Friction** | Basic (single value) | Advanced (separate directions) |
| **Damping** | Not specified | Full support (linear + angular) |
| **Sensors** | Basic declaration | Full plugin integration |
| **Collision Mesh** | Simple geometries | Complex meshes supported |
| **Physics Engine** | Not specified | Engine-specific options |
| **Extensibility** | Limited | Plugins for custom behavior |

### URDF to SDF Conversion Process

**URDF (From Lesson 1.4)**:
- Defines **structure** (links & joints)
- Specifies **inertia** (mass, moment of inertia)
- Describes **appearance** (visual geometry)

**SDF (For Gazebo)**:
- Adds **physics parameters** (friction, damping, contact)
- Specifies **simulation properties** (solver settings)
- Includes **sensor plugins** (camera, lidar, IMU)
- Defines **actuator constraints** (effort, velocity limits)

---

## Part 2: Automatic URDF to SDF Conversion

### Method 1: Using `urdf_to_sdf` Tool

```bash
# Install conversion tool
sudo apt install ros-humble-urdf-to-sdf

# Convert your URDF
urdf_to_sdf input.urdf -o output.sdf

# Verify conversion
gazebo output.sdf
```

### Method 2: Using Python Script

```python
#!/usr/bin/env python3
"""
Convert URDF to SDF with custom physics parameters
"""

from urdf_parser_py.urdf import URDF
import xml.etree.ElementTree as ET


def urdf_to_sdf(urdf_file, output_file):
    """
    Convert URDF file to SDF format

    Args:
        urdf_file: Path to input URDF
        output_file: Path to output SDF
    """

    # Parse URDF
    robot = URDF.from_xml_file(urdf_file)

    # Create SDF root
    sdf_root = ET.Element('sdf', version='1.10')
    world = ET.SubElement(sdf_root, 'world', name='converted_world')

    # Add physics
    physics = ET.SubElement(world, 'physics', type='ode')
    ET.SubElement(physics, 'max_step_size').text = '0.001'
    ET.SubElement(physics, 'real_time_factor').text = '1.0'
    ET.SubElement(physics, 'real_time_update_rate').text = '1000'

    # Add light
    light = ET.SubElement(world, 'light', name='sun', type='directional')
    ET.SubElement(light, 'pose').text = '0 0 10 0 0 0'
    ET.SubElement(light, 'diffuse').text = '0.8 0.8 0.8 1'

    # Convert robot to SDF model
    model = ET.SubElement(world, 'model', name=robot.name)
    ET.SubElement(model, 'pose').text = '0 0 0 0 0 0'

    # Convert URDF links to SDF links
    for link in robot.links:
        sdf_link = ET.SubElement(model, 'link', name=link.name)

        # Add inertial (required for physics)
        if link.inertial:
            inertial = ET.SubElement(sdf_link, 'inertial')
            inertial_pose = ET.SubElement(inertial, 'pose')
            inertial_pose.text = f"{link.inertial.origin.xyz[0]} {link.inertial.origin.xyz[1]} {link.inertial.origin.xyz[2]} 0 0 0"

            mass = ET.SubElement(inertial, 'mass')
            mass.text = str(link.inertial.mass)

            inertia = ET.SubElement(inertial, 'inertia')
            ET.SubElement(inertia, 'ixx').text = str(link.inertial.inertia.ixx)
            ET.SubElement(inertia, 'iyy').text = str(link.inertial.inertia.iyy)
            ET.SubElement(inertia, 'izz').text = str(link.inertial.inertia.izz)

        # Add visual (from URDF)
        if link.visual:
            visual = ET.SubElement(sdf_link, 'visual', name='visual')
            visual_pose = ET.SubElement(visual, 'pose')
            visual_pose.text = f"{link.visual[0].origin.xyz[0]} {link.visual[0].origin.xyz[1]} {link.visual[0].origin.xyz[2]} 0 0 0"
            # ... add geometry ...

        # Add collision with friction
        if link.collision:
            collision = ET.SubElement(sdf_link, 'collision', name='collision')
            collision_pose = ET.SubElement(collision, 'pose')
            collision_pose.text = f"{link.collision[0].origin.xyz[0]} {link.collision[0].origin.xyz[1]} {link.collision[0].origin.xyz[2]} 0 0 0"
            # ... add geometry and friction ...

    # Convert joints
    for joint in robot.joints:
        sdf_joint = ET.SubElement(model, 'joint', name=joint.name, type=joint.type)
        ET.SubElement(sdf_joint, 'parent').text = joint.parent.link
        ET.SubElement(sdf_joint, 'child').text = joint.child.link
        ET.SubElement(sdf_joint, 'pose').text = f"{joint.origin.xyz[0]} {joint.origin.xyz[1]} {joint.origin.xyz[2]} 0 0 0"

        # Add axis
        axis = ET.SubElement(sdf_joint, 'axis')
        ET.SubElement(axis, 'xyz').text = f"{joint.axis[0]} {joint.axis[1]} {joint.axis[2]}"

        # Add limits
        limits = ET.SubElement(sdf_joint, 'axis')
        if hasattr(joint, 'limit'):
            ET.SubElement(limits, 'lower').text = str(joint.limit.lower)
            ET.SubElement(limits, 'upper').text = str(joint.limit.upper)
            ET.SubElement(limits, 'effort').text = str(joint.limit.effort)
            ET.SubElement(limits, 'velocity').text = str(joint.limit.velocity)

    # Write SDF
    tree = ET.ElementTree(sdf_root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

    print(f"Converted {urdf_file} → {output_file}")


if __name__ == '__main__':
    urdf_to_sdf('simple_robot.urdf', 'simple_robot.sdf')
```

---

## Part 3: Enhanced SDF with Physics Parameters

### Complete Robot SDF with Physics Tuning

Create `simple_robot_physics.sdf`:

```xml
<?xml version="1.0" ?>
<!-- Copyright (c) 2025 Physical AI Course
     License: MIT
     Target: Ubuntu 22.04 + ROS 2 Humble + Gazebo
-->
<sdf version="1.10">
  <world name="robot_physics_world">

    <!-- Physics Engine Configuration -->
    <physics name="physics" type="ode">
      <!-- Simulation timestep: smaller = more accurate but slower -->
      <max_step_size>0.001</max_step_size>

      <!-- Realtime factor: 1.0 = normal speed, 2.0 = 2x faster -->
      <real_time_factor>1.0</real_time_factor>

      <!-- Update rate: how often to publish states -->
      <real_time_update_rate>1000</real_time_update_rate>

      <!-- Gravity (m/s^2): Earth = -9.81 in z direction -->
      <gravity>0 0 -9.81</gravity>

      <!-- ODE Solver Settings -->
      <ode>
        <solver>
          <!-- 'quick' is faster but less accurate; 'world' is more accurate -->
          <type>quick</type>

          <!-- Number of iterations for solver -->
          <iters>50</iters>

          <!-- Preconditioning iterations (0 = faster, 1 = more stable) -->
          <precon_iters>0</precon_iters>

          <!-- Successive Over-Relaxation: 1.3 = default, 1.0 = slower convergence -->
          <sor>1.400000</sor>
        </solver>

        <!-- Contact and constraint settings -->
        <constraints>
          <!-- Constraint Force Mixing: higher = softer contacts -->
          <cfm>0.000000</cfm>

          <!-- Error Reduction Parameter: higher = faster stabilization -->
          <erp>0.200000</erp>

          <!-- Max velocity for contact correction (m/s) -->
          <contact_max_correcting_vel>100.000000</contact_max_correcting_vel>

          <!-- Contact surface layer depth (m) -->
          <contact_surface_layer>0.001000</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground Plane with Friction -->
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
          <!-- Ground friction -->
          <surface>
            <friction>
              <ode>
                <!-- Coefficient of friction (0.6 = rubber on concrete) -->
                <mu>0.6</mu>
                <!-- Second friction direction -->
                <mu2>0.6</mu2>
                <!-- Slip velocity threshold -->
                <slip1>0.001</slip1>
                <slip2>0.001</slip2>
              </ode>
            </friction>
            <contact>
              <!-- Soft contact parameters for stability -->
              <ode>
                <soft_cfm>0.0</soft_cfm>
                <soft_erp>0.2</soft_erp>
              </ode>
            </contact>
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
          </material>
        </visual>
      </link>
    </model>

    <!-- Robot Model -->
    <model name="simple_robot_physics">
      <pose>0 0 0.5 0 0 0</pose>

      <!-- Base Link -->
      <link name="base_link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <iyy>0.01</iyy>
            <izz>0.01</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.4</mu>
                <mu2>0.4</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
      </link>

      <!-- Joint 1 with Dynamics -->
      <joint name="joint_1" type="revolute">
        <parent>base_link</parent>
        <child>link_1</child>
        <pose>0 0 0.1 0 0 0</pose>
        <axis>
          <xyz>0 1 0</xyz>
          <!-- Joint damping: resists motion (b in F = -b*v) -->
          <dynamics>
            <damping>0.1</damping>
            <!-- Friction: constant resistance to motion -->
            <friction>0.0</friction>
          </dynamics>
        </axis>
        <!-- Joint limits -->
        <limit>
          <lower>-1.5708</lower>
          <upper>1.5708</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </joint>

      <!-- Link 1 -->
      <link name="link_1">
        <inertial>
          <mass>0.5</mass>
          <origin xyz="0 0 0.05"/>
          <inertia>
            <ixx>0.005</ixx>
            <iyy>0.005</iyy>
            <izz>0.001</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <origin xyz="0 0 0.05"/>
          <geometry>
            <cylinder>
              <radius>0.02</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.8 1</ambient>
            <diffuse>0.2 0.2 0.8 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <origin xyz="0 0 0.05"/>
          <geometry>
            <cylinder>
              <radius>0.02</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.4</mu>
                <mu2>0.4</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
      </link>

      <!-- Joint 2 -->
      <joint name="joint_2" type="revolute">
        <parent>link_1</parent>
        <child>link_2</child>
        <pose>0 0 0.1 0 0 0</pose>
        <axis>
          <xyz>0 1 0</xyz>
          <dynamics>
            <damping>0.1</damping>
            <friction>0.0</friction>
          </dynamics>
        </axis>
        <limit>
          <lower>-1.5708</lower>
          <upper>1.5708</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </joint>

      <!-- Link 2 (End Effector) -->
      <link name="link_2">
        <inertial>
          <mass>0.3</mass>
          <origin xyz="0 0 0.05"/>
          <inertia>
            <ixx>0.003</ixx>
            <iyy>0.003</iyy>
            <izz>0.0005</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <origin xyz="0 0 0.05"/>
          <geometry>
            <cylinder>
              <radius>0.015</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <origin xyz="0 0 0.05"/>
          <geometry>
            <cylinder>
              <radius>0.015</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.4</mu>
                <mu2>0.4</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
      </link>

    </model>

  </world>
</sdf>
```

---

## Part 4: Physics Parameter Tuning Guide

### Understanding Key Parameters

**Damping** (Linear & Angular):
```
Force = -damping * velocity
Effect: Dissipates energy over time (like air resistance)
Range: 0.0 (no damping) to 1.0+ (heavy damping)
Robot arm: 0.1 (light damping for smooth motion)
Humanoid: 0.5+ (heavier damping for stability)
```

**Friction Coefficient (μ)**:
```
Friction = μ * Normal_Force
Effect: Prevents sliding on surfaces
Common values:
  - 0.1: Ice/slippery surface
  - 0.4: Plastic on concrete
  - 0.6: Rubber on concrete
  - 1.0+: High-grip materials
```

**Error Reduction Parameter (ERP)**:
```
ERP = Fraction of error corrected per timestep
Effect: How aggressively constraints are enforced
Range: 0.0 (no correction) to 1.0 (full correction)
Value: 0.2 (default, good balance)
Lower: More accurate but slower convergence
Higher: Faster but may oscillate
```

**Constraint Force Mixing (CFM)**:
```
CFM = Softness of constraints
Effect: Allows slight constraint violations for stability
Range: 0.0 (very stiff) to 0.1 (soft)
Value: 0.0 (default, rigid constraints)
Higher CFM: More flexible, slower convergence
```

### Tuning Strategy for Different Robots

**Stable Robot (Low velocity, robust)**:
```xml
<damping>0.5</damping>        <!-- Higher damping -->
<friction>0.8</friction>       <!-- High friction -->
<erp>0.2</erp>                 <!-- Standard ERP -->
<cfm>0.0</cfm>                 <!-- Rigid constraints -->
<max_step_size>0.001</max_step_size>
```

**Fast Robot (High velocity, agile)**:
```xml
<damping>0.1</damping>         <!-- Low damping -->
<friction>0.4</friction>       <!-- Lower friction -->
<erp>0.5</erp>                 <!-- Higher ERP for stability -->
<cfm>0.0</cfm>                 <!-- Rigid -->
<max_step_size>0.0005</max_step_size>  <!-- Smaller timestep -->
```

**Humanoid Robot (Balanced, realistic)**:
```xml
<damping>0.3</damping>         <!-- Moderate damping -->
<friction>0.6</friction>       <!-- Standard friction -->
<erp>0.2</erp>                 <!-- Standard ERP -->
<cfm>0.0</cfm>                 <!-- Rigid -->
<max_step_size>0.001</max_step_size>
```

---

## Part 5: Testing and Validation

### Simulation Validation Script

```python
#!/usr/bin/env python3
"""
Validate physics simulation of robot
Check for common physics issues
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelState, LinkState
from geometry_msgs.msg import Twist
import math


class PhysicsValidator(Node):
    """Validate robot physics in Gazebo"""

    def __init__(self):
        super().__init__('physics_validator')

        # Subscribers
        self.model_states_sub = self.create_subscription(
            ModelState,
            '/gazebo/model_states',
            self.model_states_callback,
            10
        )

        self.link_states_sub = self.create_subscription(
            LinkState,
            '/gazebo/link_states',
            self.link_states_callback,
            10
        )

        self.initial_height = None
        self.falling = False
        self.get_logger().info('Physics validator initialized')

    def model_states_callback(self, msg):
        """Monitor model states for physics validation"""
        if 'simple_robot' not in msg.name:
            return

        idx = msg.name.index('simple_robot')
        pose = msg.pose[idx]

        # Check 1: Robot falls due to gravity
        if self.initial_height is None:
            self.initial_height = pose.position.z
            self.get_logger().info(f'Initial height: {self.initial_height:.3f}m')

        # Robot should fall in first seconds
        if pose.position.z < self.initial_height - 0.3:
            if not self.falling:
                self.falling = True
                self.get_logger().info('✅ Gravity working: Robot is falling')

        # Check 2: Robot doesn't fall through ground (z > 0)
        if pose.position.z < 0:
            self.get_logger().error('❌ PHYSICS ERROR: Robot penetrated ground!')

        # Check 3: Robot velocity reasonable
        vel_magnitude = math.sqrt(
            msg.twist[idx].linear.x**2 +
            msg.twist[idx].linear.y**2 +
            msg.twist[idx].linear.z**2
        )
        if vel_magnitude > 10.0:  # > 10 m/s is too fast
            self.get_logger().warn(f'⚠️  High velocity: {vel_magnitude:.2f} m/s')

        # Check 4: Robot rotation reasonable
        angular_vel = math.sqrt(
            msg.twist[idx].angular.x**2 +
            msg.twist[idx].angular.y**2 +
            msg.twist[idx].angular.z**2
        )
        if angular_vel > 20.0:  # > 20 rad/s is spinning too fast
            self.get_logger().warn(f'⚠️  High rotation: {angular_vel:.2f} rad/s')

    def link_states_callback(self, msg):
        """Monitor link states for contact forces"""
        # Could check contact forces here
        pass

    def report_validation(self):
        """Print validation report"""
        self.get_logger().info('=== PHYSICS VALIDATION REPORT ===')
        if self.falling:
            self.get_logger().info('✅ Gravity: OK')
        else:
            self.get_logger().info('❌ Gravity: NOT DETECTED')


def main(args=None):
    rclpy.init(args=args)
    validator = PhysicsValidator()

    # Run for 10 seconds
    rclpy.spin_once(validator)
    for _ in range(10):
        rclpy.spin_once(validator)
        validator.get_logger().info('...')

    validator.report_validation()
    validator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Hands-On Exercise

**Task**: Create an SDF world with a physics-configured robot and validate behavior.

### Step 1: Create Physics Test World

```bash
# Copy SDF file
mkdir -p ~/gazebo_ws/src/worlds
cat > ~/gazebo_ws/src/worlds/simple_robot_physics.sdf << 'EOF'
# (Paste the complete SDF from Part 3 above)
EOF
```

### Step 2: Launch with Physics Validation

```bash
# Terminal 1: Launch Gazebo
ros2 launch gazebo_ros gazebo.launch.py world:=$HOME/gazebo_ws/src/worlds/simple_robot_physics.sdf

# Terminal 2: Run physics validator
python3 ~/gazebo_ws/src/physics_validator.py
```

### Step 3: Experiments

1. **Gravity Test**: Verify robot falls from spawn height
2. **Friction Test**: Observe ground contact friction
3. **Damping Test**: Change damping from 0 to 1.0 and observe motion
4. **Timestep Test**: Reduce max_step_size to 0.0001 and check stability

### Exercises

1. **High Damping**: Set damping to 1.0 and observe sluggish motion
2. **No Friction**: Set ground friction to 0.0 and observe sliding
3. **High ERP**: Set ERP to 0.5 and observe constraint stiffness
4. **Small Timestep**: Use max_step_size 0.0001 for maximum accuracy
5. **Real-time Factor**: Set to 2.0 and watch simulation run 2x faster

---

## Common Physics Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Robot falls through ground | No collision or bad inertia | Add collision geometry, check inertia values |
| Robot oscillates/shakes | ERP too high, small timestep | Reduce ERP to 0.2, increase timestep to 0.001 |
| Joints move too fast | Low damping, low effort limit | Increase damping to 0.3, reduce velocity limit |
| Robot doesn't move | High friction/damping | Reduce both to 0.1, check joint effort |
| Simulation very slow | Timestep too small | Increase from 0.0001 to 0.001 |
| Contacts not working | CFM/ERP bad values | Use defaults: cfm=0.0, erp=0.2 |
| Joint limits not enforced | Missing limit element in SDF | Add `<limit>` with lower/upper/effort |

---

## Key Takeaways

✅ **URDF Limitations**: Designed for visualization, missing simulation physics details

✅ **SDF Advantages**: Purpose-built for physics simulation with full parameter control

✅ **Conversion Process**: Automatic tools exist, but manual tuning needed for accuracy

✅ **Physics Parameters**: Damping, friction, ERP, CFM control simulation behavior

✅ **Tuning Strategy**: Start with defaults, adjust iteratively based on validation

✅ **Trade-offs**: Accuracy vs. speed—smaller timestep = more accurate but slower

✅ **Validation**: Always test gravity, collisions, and motion before deployment

---

## Further Reading

- 📖 [Gazebo SDF Specification](http://sdformat.org/)
- 📖 [ODE Physics Tuning Guide](http://www.ode.org/wiki/index.php/Manual)
- 📖 [URDF to SDF Conversion](https://classic.gazebosim.org/tutorials?tut=ros2_urdf&cat=ign&ver=igni)
- 📖 [Humanoid Physics in Gazebo](https://gazebosim.org/docs)

---

**Next Lesson**: [Lesson 2.3: Sensor Simulation & Unity Integration](2-3-sensors-unity.md)

**Questions?** See [FAQ](../faq.md) or [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
