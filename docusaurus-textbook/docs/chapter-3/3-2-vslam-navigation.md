---
sidebar_position: 2
sidebar_label: "Lesson 3.2: VSLAM & Navigation"
title: "Isaac ROS: Hardware-Accelerated VSLAM and Autonomous Navigation"
description: "Implement visual SLAM and autonomous navigation using Isaac ROS and Nav2"
duration: 120
difficulty: Advanced
hardware: ["Ubuntu 22.04 LTS", "Jetson Orin Nano/NX", "Intel RealSense D435i", "ROS 2 Humble"]
prerequisites: ["Lesson 3.1: NVIDIA Isaac Sim Basics"]
---

# Lesson 3.2: VSLAM & Navigation with Isaac ROS

:::info Lesson Overview
**Duration**: 120 minutes | **Difficulty**: Advanced | **Hardware**: Jetson Orin + RealSense + ROS 2 Humble

**Prerequisites**: Lesson 3.1 complete

**Learning Outcome**: Deploy hardware-accelerated VSLAM and autonomous navigation on edge devices
:::

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand Visual SLAM (Simultaneous Localization and Mapping)
- Deploy Isaac ROS VSLAM on Jetson hardware
- Build real-time occupancy maps from camera and depth data
- Use Nav2 for autonomous path planning and navigation
- Configure costmaps and planners for humanoid locomotion
- Integrate SLAM output with ROS 2 navigation stack
- Validate mapping and localization in real environments
- Optimize performance on edge devices (Jetson Orin Nano)

## Part 1: VSLAM Fundamentals

:::tip What is VSLAM?
**Visual SLAM** = Computer Vision + Simultaneous Localization and Mapping
- **Localization**: Where am I? (uses visual features)
- **Mapping**: What's around me? (builds environment map)
- **Simultaneous**: Solves both problems together
- **Visual**: Uses only camera (RGB-D) as sensor
:::

### SLAM vs Pure Localization

| Aspect | SLAM | Localization Only |
|--------|------|-------------------|
| **Map** | Builds map while moving | Uses pre-existing map |
| **Computational Cost** | High (continuous mapping) | Lower (matching to map) |
| **Accuracy** | Accumulates errors over time | Stable with good map |
| **Use Case** | Unknown environments | Familiar environments |
| **Failure Mode** | Loop closure errors | Global position loss |

### Isaac ROS vs Standard ROS 2 SLAM

| Feature | Isaac ROS | Standard ROS 2 (gmapping) |
|---------|-----------|--------------------------|
| **Acceleration** | GPU-accelerated (RTX cores) | CPU only |
| **Latency** | <50ms per frame | 200-500ms per frame |
| **Accuracy** | Sub-centimeter | Decimeter |
| **Loop Closure** | Learned (neural network) | Geometric matching |
| **Edge Device** | Jetson Orin Nano (40 TOPS) | Requires i7+ workstation |
| **Real-time Speed** | 30 FPS at full resolution | 5-10 FPS downsample |

### Visual Features for SLAM

```
RGB Image               ORB Features             Feature Matching
┌─────────────┐        ┌─────────────┐          ┌─────────────┐
│             │        │ • Oriented  │          │ Keypoint 1  │
│    Robot    │  ───→  │   FAST      │  ──→     │ matches     │
│   Camera    │        │ • Rotation  │          │ Keypoint 2  │
│             │        │   invariant │          │    in 3D    │
└─────────────┘        │ • Depth     │          └─────────────┘
                       │   clues     │                  ↓
                       └─────────────┘          ┌─────────────┐
                                                │ Estimate    │
                                                │ camera pose │
                                                │ (SfM)       │
                                                └─────────────┘
```

**ORB Features** (used by Isaac ROS):
- **O**: Oriented
- **R**: FAST corner detector
- **B**: BRIEF binary descriptor
- Advantages: Fast (100 FPS), rotation-invariant, GPU-optimizable

---

## Part 2: Installing Isaac ROS VSLAM

### Prerequisites Check

```bash
# Verify Jetson Orin Nano
cat /etc/nv_tegra_release | head -1
# Expected: # R35 (release), REVISION: ...

# Verify ROS 2 Humble
source /opt/ros/humble/setup.bash
ros2 --version

# Verify RealSense camera
realsense-viewer
# Should show D435i RGB + Depth streams
```

### Installation on Jetson

```bash
# Add Isaac ROS sources
source /opt/ros/humble/setup.bash

# Install Isaac ROS VSLAM GEMs
sudo apt update
sudo apt install -y \
  ros-humble-isaac-ros-visual-slam \
  ros-humble-isaac-ros-depth-image-proc \
  ros-humble-isaac-ros-ess \
  ros-humble-realsense2-camera

# Install Nav2 (navigation stack)
sudo apt install -y \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-cartographer
```

### Verification

```bash
# Check Isaac ROS installation
ros2 pkg list | grep isaac_ros

# Expected output:
# isaac_ros_common
# isaac_ros_visual_slam
# isaac_ros_depth_image_proc
```

---

## Part 3: SLAM Configuration and Launch

### SLAM Configuration File

Create `~/ros2_ws/config/visual_slam_config.yaml`:

```yaml
# Isaac ROS Visual SLAM Configuration
visual_slam_node:
  ros__parameters:
    # Camera calibration (from RealSense D435i)
    camera_info_topic: '/camera/color/camera_info'
    rgb_image_topic: '/camera/color/image_raw'
    depth_image_topic: '/camera/depth/image_rect_raw'

    # Output topics
    pose_topic: '/visual_slam/pose'
    odometry_topic: '/visual_slam/odometry'
    slam_status_topic: '/visual_slam/status'

    # VSLAM parameters
    enable_depth_regularization: true
    enable_loop_closure: true
    enable_loop_closure_visualization: false  # Disable for performance

    # Feature detection
    min_features_for_tracking: 100
    max_features_for_tracking: 500

    # Keyframe settings
    keyframe_translation_threshold: 0.1  # meters
    keyframe_rotation_threshold: 0.1     # radians

    # Graph optimization
    loop_closure_confidence_threshold: 0.75
    optimization_frequency: 10  # Hz

    # Performance tuning for Jetson
    enable_imu_fusion: false  # Jetson may not have IMU
    map_update_frequency: 5   # Hz (lower = faster)

    # Output map format
    map_type: 'occupancy_grid'

    # Debug
    enable_debug: false
    verbosity: 'info'
```

### SLAM Launch File

Create `~/ros2_ws/launch/visual_slam.launch.py`:

```python
#!/usr/bin/env python3
"""
Launch Visual SLAM with RealSense camera on Jetson Orin Nano
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """
    Launch Visual SLAM with:
    1. RealSense D435i camera
    2. Isaac ROS VSLAM node
    3. RViz visualization
    """

    # Get paths
    realsense_pkg = FindPackageShare('realsense2_camera')
    isaac_slam_pkg = FindPackageShare('isaac_ros_visual_slam')

    # Arguments
    camera_namespace = LaunchConfiguration('camera_namespace', default='camera')
    use_rviz = LaunchConfiguration('use_rviz', default='true')

    return LaunchDescription([
        # Launch RealSense camera
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(realsense_pkg, 'launch', 'rs_launch.py')
            ),
            launch_arguments={
                'camera_name': camera_namespace,
                'depth_module.profile': '640x480x30',  # Jetson-friendly resolution
                'rgb_camera.color_profile': '640x480x30',
                'enable_gyro': 'false',
                'enable_accel': 'false',
                'align_depth': 'true',
                'decimation_filter.enable': 'true',  # Reduce computation
                'spatial_filter.enable': 'true',
            }.items()
        ),

        # Isaac ROS Visual SLAM
        Node(
            package='isaac_ros_visual_slam',
            executable='visual_slam_node',
            namespace='visual_slam',
            parameters=[{
                'camera_info_topic': f'/{camera_namespace}/color/camera_info',
                'rgb_image_topic': f'/{camera_namespace}/color/image_raw',
                'depth_image_topic': f'/{camera_namespace}/depth/image_rect_raw',
                'enable_loop_closure': True,
                'map_type': 'occupancy_grid',
                'map_publish_rate': 5.0,
            }],
            output='screen'
        ),

        # RViz visualization
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(isaac_slam_pkg, 'config', 'visual_slam.rviz')],
            condition=LaunchConfigurationEquals('use_rviz', 'true'),
            output='screen'
        ),

        # TF publisher for base_link → camera transformation
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', f'{camera_namespace}_link'],
            output='screen'
        ),
    ])
```

### Launch VSLAM

```bash
# Terminal 1: Launch VSLAM
source /opt/ros/humble/setup.bash
ros2 launch ~/ros2_ws/launch/visual_slam.launch.py

# Terminal 2: Monitor SLAM status
ros2 topic echo /visual_slam/status

# Terminal 3: View odometry
ros2 topic echo /visual_slam/odometry
```

---

## Part 4: Nav2 Navigation Stack

### Nav2 Configuration

Create `~/ros2_ws/config/nav2_params.yaml`:

```yaml
amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2      # Rotation noise from rotation
    alpha2: 0.2      # Rotation noise from translation
    alpha3: 0.2      # Translation noise from translation
    alpha4: 0.2      # Translation noise from rotation
    z_max: 4.0
    z_min: 0.05
    z_rand: 0.05
    scan_matching_uncertainty: 0.1

bt_navigator:
  ros__parameters:
    use_sim_time: False
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /visual_slam/odometry
    bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "navigate_to_pose.xml"
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_assisted_teleop_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_drive_on_heading_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stopped_condition_bt_node
      - nav2_goal_updated_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_truncate_path_local_action_bt_node
      - nav2_goal_checker_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_would_a_loop_be_formed_condition_bt_node
      - nav2_if_led_bt_node
      - nav2_sleep_bt_node
      - nav2_wait_uptime_bt_node

controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 10.0  # Lower for Jetson
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugins: ["general_goal_checker"]
    controller_plugins: ["FollowPath"]

    # Progress checker
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker
    general_goal_checker:
      stateful: True
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: true

    # DWA (Dynamic Window Approach) local planner
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.5    # m/s
      lookahead_dist: 0.6        # meters
      min_approach_linear_vel: -0.05
      approach_velocity_scaling_dist: 1.0
      max_allowed_time_error: 1.0
      use_velocity_scaled_lookahead_dist: false
      min_lookahead_dist: 0.3
      max_lookahead_dist: 0.9
      lookahead_time: 1.5
      use_approach_linear_velocity_scaling: true
      max_allowed_time_error: 1.0
      use_regulated_linear_velocity_scaling: true
      use_cost_regulated_linear_velocity_scaling: false
      regulated_linear_scaling_min_radius: 0.9
      regulated_linear_scaling_min_speed: 0.25
      use_rotate_to_heading: true
      rotate_to_heading_min_angle: 0.785
      max_angular_accel: 3.2
      max_robot_pose_search_dist: 2.0
      cost_scaling_dist: 0.6
      cost_scaling_gain: 1.0
      inflation_cost_scaling_gain: 3.0

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: False
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: true
      allow_unknown: true

costmap_server:
  ros__parameters:
    update_frequency: 5.0  # Lower frequency for Jetson
    publish_frequency: 2.0
    global_frame: map
    robot_base_frame: base_link
    use_sim_time: False
    rolling_window: true
    width: 100
    height: 100
    resolution: 0.05
    robot_radius: 0.22
    plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.55
    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0
    static_layer:
      plugin: "nav2_costmap_2d::StaticLayer"
      map_subscribe_transient_local: True
```

### Nav2 Launch File

```python
#!/usr/bin/env python3
"""
Launch Nav2 navigation stack
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nav2_dir = get_package_share_directory('nav2_bringup')
    config_dir = os.path.expanduser('~/ros2_ws/config')

    return LaunchDescription([
        # Map server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': os.path.join(config_dir, 'occupancy_map.yaml'),
            }]
        ),

        # Localization (AMCL)
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[os.path.join(config_dir, 'nav2_params.yaml')],
        ),

        # Path planning
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[os.path.join(config_dir, 'nav2_params.yaml')],
        ),

        # Local controller
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[os.path.join(config_dir, 'nav2_params.yaml')],
        ),

        # Navigation behavior tree
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[{
                'bt_xml_filename': os.path.join(nav2_dir, 'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
            }]
        ),

        # Navigation lifecycle manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': ['planner_server', 'controller_server', 'bt_navigator']
            }]
        ),
    ])
```

---

## Part 5: End-to-End VSLAM + Navigation

### Complete System Script

```python
#!/usr/bin/env python3
"""
End-to-end VSLAM + Navigation system
Integrates visual SLAM mapping with autonomous navigation
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import math


class SLAMNavigationSystem(Node):
    """Combines SLAM mapping with autonomous navigation"""

    def __init__(self):
        super().__init__('slam_navigation_system')

        # Subscribe to SLAM outputs
        self.odometry_sub = self.create_subscription(
            Odometry,
            '/visual_slam/odometry',
            self.odometry_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/occupancy_grid',
            self.map_callback,
            10
        )

        # Publish navigation goals
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.current_position = None
        self.current_map = None

        self.get_logger().info('SLAM Navigation System initialized')

    def odometry_callback(self, msg):
        """Update current robot position"""
        self.current_position = msg.pose.pose.position
        self.get_logger().debug(
            f'Robot position: x={self.current_position.x:.2f}, '
            f'y={self.current_position.y:.2f}'
        )

    def map_callback(self, msg):
        """Update occupancy map"""
        self.current_map = msg
        occupied_cells = sum(1 for cell in msg.data if cell > 50)
        self.get_logger().info(
            f'Map updated: {msg.info.width}x{msg.info.height}, '
            f'{occupied_cells} occupied cells'
        )

    def send_navigation_goal(self, goal_x, goal_y, goal_yaw=0.0):
        """Send goal to Nav2 navigation system"""
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()

        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = 0.0

        # Convert yaw to quaternion
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(goal_yaw / 2)
        goal.pose.orientation.w = math.cos(goal_yaw / 2)

        self.goal_pub.publish(goal)
        self.get_logger().info(f'✅ Goal sent: ({goal_x:.2f}, {goal_y:.2f})')

    def explore_environment(self):
        """Automatically explore environment by sending sequential goals"""
        exploration_sequence = [
            (1.0, 0.0, 0.0),      # Move forward 1m
            (1.0, 1.0, math.pi/4),  # Move diagonal
            (0.0, 1.0, math.pi/2),  # Move left
            (-1.0, 0.0, math.pi),   # Move back
        ]

        for goal_x, goal_y, goal_yaw in exploration_sequence:
            self.send_navigation_goal(goal_x, goal_y, goal_yaw)
            self.get_logger().info(f'Exploring: ({goal_x}, {goal_y})')

            # Wait for goal completion
            import time
            time.sleep(10)


def main(args=None):
    rclpy.init(args=args)
    system = SLAMNavigationSystem()

    # Run automated exploration
    system.explore_environment()

    system.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Hands-On Exercise

**Task**: Deploy VSLAM on Jetson, map an indoor environment, and navigate autonomously.

### Step 1: Launch VSLAM

```bash
ros2 launch ~/ros2_ws/launch/visual_slam.launch.py
```

### Step 2: Map Environment

Walk robot around room for 2-3 minutes while SLAM maps:
- Observe RViz showing feature tracking
- Watch occupancy grid build
- Check loop closure detected

### Step 3: Save Map

```bash
ros2 service call /map_server/save_map std_srvs/Empty
# Saved to: ~/map.pgm and ~/map.yaml
```

### Step 4: Launch Navigation

```bash
ros2 launch ~/ros2_ws/launch/nav2_bringup.launch.py map:=~/map.yaml
```

### Step 5: Send Navigation Goals

```bash
python3 ~/slam_navigation_system.py
```

### Exercises

1. **Mapping Accuracy**: Compare SLAM map to actual floor plan (measure error)
2. **Loop Closure**: Revisit starting position and check loop closure detection
3. **Autonomous Exploration**: Implement frontier-based exploration
4. **Collision Avoidance**: Add obstacles and verify costmap adjustment
5. **Performance**: Profile CPU/memory on Jetson Orin Nano

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Low feature count | Poor lighting | Improve illumination or use depth + RGB |
| Loop closure failure | Texture-less environment | Use visual features from different angles |
| Navigation oscillation | Controller gains wrong | Tune nav2_params.yaml controller settings |
| Memory full on Jetson | Map too large | Reduce map resolution or use rolling window |
| SLAM drift | Accumulated error | Loop closure helps; revisit areas |

---

## Key Takeaways

✅ **VSLAM = Localization + Mapping**: Simultaneous solving of both problems

✅ **Hardware Acceleration**: Isaac ROS uses Jetson's Tensor cores for 10x speedup

✅ **Visual Features**: ORB features enable real-time, robust tracking

✅ **Loop Closure**: Neural networks detect revisited areas to eliminate drift

✅ **Nav2 Integration**: Standardized ROS 2 navigation stack with SLAM output

✅ **Edge Deployment**: Real-time VSLAM on Jetson Orin Nano (40 TOPS)

✅ **Autonomous Navigation**: Combines mapping + planning + control

---

**Next Lesson**: [Lesson 3.3: Computer Vision with Isaac ROS](3-3-computer-vision.md)

**Questions?** See [FAQ](../faq.md) or [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
