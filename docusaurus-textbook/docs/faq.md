---
sidebar_position: 1002
sidebar_label: "FAQ"
title: "Frequently Asked Questions"
description: "Common questions and answers about the Physical AI course"
---

# Frequently Asked Questions

## Course Structure & Content

### Q: How long does this course take?
**A**: The full course is designed for **13 weeks**, with 4-6 hours per week of study and hands-on work:
- Chapter 1 (Weeks 1-3): Nervous System basics
- Chapter 2 (Weeks 4-6): Digital Twin and simulation
- Chapter 3 (Weeks 7-10): AI Brain and learning
- Chapter 4 (Weeks 11-13): Embodied intelligence and multimodal AI

You can progress at your own pace. Some modules can be completed in a weekend sprint.

### Q: Do I need prior robotics experience?
**A**: No, this course is designed for beginners. You should have:
- Basic Python programming skills (variables, functions, classes)
- Comfort with Linux terminal commands
- Curiosity about AI and robotics

If you're new to Python, complete a basic Python tutorial first (2-3 hours).

### Q: What if I only have limited hardware?
**A**: You can do the course on any Linux machine:
- **Jetson Orin Nano** (recommended): Full hands-on with real hardware
- **x86_64 Linux** (alternative): All code examples work on desktop/laptop
- **Isaac Sim**: Virtual environment for simulation-only learning
- **Cloud VM**: Ubuntu 22.04 on AWS/GCP for remote access

### Q: Can I skip chapters?
**A**: Yes, chapters have some independence:
- **Chapter 1**: Must complete (foundation for everything)
- **Chapter 2**: Recommended before Chapter 3
- **Chapter 3**: Can start after Chapter 1 (covers learning)
- **Chapter 4**: Requires Chapters 1-3 for full understanding

However, skipping creates knowledge gaps. We recommend full sequential progression.

### Q: How much does this cost?
**A**: The textbook and code are **100% free** and open-source. Costs are only for:
- **Hardware** (Jetson Orin Nano): ~$250
- **Optional sensors** (RealSense, ReSpeaker): ~$150-200 each
- **Electricity**: Minimal (Jetson uses ~15W)

## Technical Setup

### Q: I'm on Windows/macOS. Can I follow this course?
**A**: Partially:
- **Windows**: Use Windows Subsystem for Linux 2 (WSL2) or VirtualBox
- **macOS**: Use UTM or Parallels Desktop with Ubuntu VM
- **Cloud**: Use cloud Linux VMs (AWS EC2, Google Cloud, DigitalOcean)

For best experience, use native Linux or Jetson hardware.

### Q: What if ROS 2 installation fails?
**A**: This is the most common issue. Try these steps:

1. **Check Ubuntu version**:
   ```bash
   lsb_release -a
   # Should be Ubuntu 20.04 or 22.04
   ```

2. **Verify network connectivity**:
   ```bash
   ping 8.8.8.8
   sudo apt update
   ```

3. **Follow official ROS 2 instructions**:
   - Go to https://docs.ros.org/en/humble/Installation.html
   - Choose your OS carefully
   - Copy-paste exact commands (don't modify)

4. **Ask for help**: Post your error in [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions) with:
   - OS and version
   - Full error message
   - Commands you ran

### Q: How do I update ROS 2?
**A**: ROS 2 Humble is LTS (Long Term Support) and receives security updates:

```bash
sudo apt update
sudo apt upgrade  # Existing packages
sudo apt dist-upgrade  # Upgrade to new ROS 2 patch versions

# Verify version
ros2 --version
```

Don't skip major version updates. Pin to Humble (don't upgrade to Iron, etc.).

### Q: Can I use ROS 2 Rolling (development version)?
**A**: Not recommended. This course uses **Humble LTS** for stability. Rolling changes frequently and may break examples.

If you want to contribute improvements to the course, use a separate Humble installation.

### Q: How much disk space do I need?
**A**: Minimum recommendations:
- **ROS 2**: 500 MB
- **Development tools**: 1-2 GB
- **Examples code**: 100 MB
- **Working space**: 5 GB

**Recommended**: 20+ GB for comfortable development and testing.

## ROS 2 & Programming

### Q: What's the difference between `rclpy` and `rclcpp`?
**A**: Both are ROS 2 client libraries:
- **rclpy**: Python client library (this course uses this)
  - Easier to learn
  - Slower execution
  - Better for prototyping

- **rclcpp**: C++ client library
  - Harder to learn
  - Faster execution
  - Better for performance-critical code

This course uses Python (rclpy) for accessibility. C++ is optional for optimization later.

### Q: How do I debug a ROS 2 node?
**A**: Several approaches:

```bash
# 1. Enable logging
ROS_LOG_DIR=./logs ros2 run my_package my_node

# 2. Use print debugging (logging)
self.get_logger().info(f"Variable value: {my_var}")

# 3. Monitor with ROS 2 CLI
ros2 topic echo /my_topic
ros2 node info /my_node
ros2 interface show std_msgs/String

# 4. Use Python debugger
import pdb; pdb.set_trace()  # Breakpoint in code

# 5. Monitor CPU/memory
ros2 run rqt_py_common rqt_py_common  # ROS 2 GUI tools
```

### Q: How do I install custom Python packages?
**A**: Use `pip` within ROS 2 environment:

```bash
# Ensure ROS 2 is sourced
source /opt/ros/humble/setup.bash

# Install packages
pip install numpy scipy opencv-python

# For system-wide installation
sudo apt install python3-numpy  # If available

# Add to project requirements
echo "opencv-python>=4.5" > requirements.txt
pip install -r requirements.txt
```

### Q: What if two packages conflict?
**A**: Use Python virtual environments:

```bash
# Create venv
python3 -m venv ~/ros2_venv

# Activate
source ~/ros2_venv/bin/activate

# Source ROS 2 after activating venv
source /opt/ros/humble/setup.bash

# Install packages
pip install package1 package2
```

## Code & Examples

### Q: Where can I find more code examples?
**A**:
- **This textbook**: `/docs/chapter-X/examples/`
- **ROS 2 tutorials**: https://docs.ros.org/en/humble/Tutorials.html
- **Official demos**: `ros2 run demo_nodes_py` (comes with ROS 2)
- **Community repos**: GitHub search "ROS 2 examples"

### Q: Can I modify example code for my own project?
**A**: Yes! All examples are MIT licensed. You can:
- ✅ Copy and modify for your own use
- ✅ Create derivative works
- ✅ Use commercially
- ⚠️ Must keep license and attribution

Just credit the original source.

### Q: How do I convert XACRO to URDF?
**A**:
```bash
# Basic conversion
xacro robot.urdf.xacro > robot.urdf

# With parameters
xacro robot.urdf.xacro param1:=value1 param2:=value2 > robot.urdf

# Check syntax without converting
xacro robot.urdf.xacro --check-syntax

# Debug XACRO processing
xacro robot.urdf.xacro --verbosity 2
```

### Q: My URDF has "undefined reference" errors. How do I fix it?
**A**: URDF references must be exact (case-sensitive):

```xml
<!-- ❌ Wrong: names don't match -->
<link name="base_link">...</link>
<joint name="my_joint">
  <parent link="Base_Link"/>  <!-- Case mismatch! -->
</joint>

<!-- ✅ Correct: exact match -->
<link name="base_link">...</link>
<joint name="my_joint">
  <parent link="base_link"/>  <!-- Exact match -->
</joint>
```

Check all `<parent>` and `<child>` link references exactly match `<link name="">`.

## Hardware & Sensors

### Q: Do I need the RealSense camera to complete the course?
**A**: No, it's optional (needed for Chapter 3, Lesson 3.2):
- **Chapters 1-2**: No camera needed
- **Chapter 3**: Can use simulated cameras or other vision approaches
- **Optional**: RealSense examples provided if you have one

### Q: Can I use a different camera instead of RealSense?
**A**: Yes! Many cameras work with ROS 2:
- **USB webcams**: Simplest, widely supported
- **RealSense**: Depth + RGB (what we show)
- **Kinect**: Older but capable
- **OAK-D**: Similar to RealSense, cheaper
- **Simulated**: Isaac Sim provides virtual cameras

Examples may need modification, but concepts are transferable.

### Q: What if my Jetson runs out of storage?
**A**: Jetson Orin Nano typically has 256GB eMMC. To free space:

```bash
# Check disk usage
df -h
du -sh ~/.*  # Hidden directories
du -sh ~/*  # Top-level directories

# Clean package manager cache
sudo apt clean
sudo apt autoclean

# Remove old ROS 2 packages
sudo apt remove ros-humble-unused-package

# Check for large files
find ~/ -type f -size +100M

# Uninstall development tools if not needed
sudo apt remove python3-dev build-essential
```

### Q: Can I use the course with a different single-board computer?
**A**: Probably, but with caveats:
- **Raspberry Pi**: Limited (~1GB RAM for Humble - too small)
- **NVIDIA Jetson Xavier**: Yes, more powerful
- **x86_64 Linux**: Yes, for development/simulation

Jetson Orin Nano is the sweet spot (8GB, $250, good performance).

## Learning & Progress

### Q: How do I know if I'm on track?
**A**: Each lesson has:
- ✅ Learning objectives (what you'll learn)
- ✅ Hands-on exercises (practice skills)
- ✅ Expected output (how to verify)
- ✅ Common errors section (troubleshoot)

If you can complete exercises and explain concepts, you're on track.

### Q: What if I get stuck on a lesson?
**A**: Try this process:

1. **Re-read the lesson** - Often the answer is in the text
2. **Check Common Errors section** - Might describe your exact issue
3. **Try different approaches** - Modify the example slightly
4. **Search online** - "ROS 2 [your error message]"
5. **Ask for help**:
   - GitHub Discussions (course-specific)
   - ROS 2 Discourse (general ROS 2)
   - Stack Overflow (if programming)

When asking for help, include:
- Full error message (copy-paste)
- Exact steps you followed
- What you tried already
- Your OS/hardware setup

### Q: Can I skip exercises and just read?
**A**: Technically yes, but learning suffers. Hands-on practice is essential:
- Reading explains concepts (20% learning)
- Coding reinforces concepts (80% learning)

The "Embodied Intelligence" theme of this course means you learn by **doing**, not just reading.

### Q: How do I set up to work on multiple lessons in parallel?
**A**: Create separate workspaces:

```bash
# Workspace 1: Chapter 1 exercises
mkdir -p ~/ros2_ws_ch1/src
cd ~/ros2_ws_ch1

# Workspace 2: Chapter 2 exercises
mkdir -p ~/ros2_ws_ch2/src
cd ~/ros2_ws_ch2

# Switch between them
source ~/ros2_ws_ch1/install/setup.bash  # Use ch1
source ~/ros2_ws_ch2/install/setup.bash  # Use ch2

# Source both (overlay)
source ~/ros2_ws_ch1/install/setup.bash
source ~/ros2_ws_ch2/install/setup.bash  # ch2 overlays on top
```

## Contributing & Community

### Q: Can I contribute improvements to the textbook?
**A**: Absolutely! This is an open-source project. To contribute:

1. **Fork the repository**: GitHub "Fork" button
2. **Create a branch**: `git checkout -b fix/typo-in-lesson-1`
3. **Make changes**: Fix typos, improve code, add content
4. **Create Pull Request**: Describe what you fixed
5. **Review process**: Maintainers review and merge

See `CONTRIBUTING.md` in the repository for details.

### Q: How do I report a bug or suggest a feature?
**A**: Use GitHub Issues:

1. **Check existing issues**: Someone may have already reported it
2. **Create new issue**: Click "New Issue"
3. **Describe clearly**: Include error message, steps to reproduce
4. **Suggest fix** (optional): If you know how to fix it

Examples of good issues:
- "Lesson 1.2 code example has syntax error (line 42)"
- "ROS 2 commands in Getting Started don't work on Ubuntu 20.04"
- "Add exercise for topic remapping"

### Q: Is there a community forum?
**A**: Yes, multiple:
- **GitHub Discussions**: Course-specific
- **ROS 2 Discourse**: General ROS 2 help
- **Discord servers**: Some community ROS 2 communities
- **Reddit**: r/robotics, r/ROS

## Troubleshooting Commands

### Quick Diagnostic

```bash
# System check
echo "=== System Info ===" && uname -a
echo "=== Ubuntu Version ===" && lsb_release -a
echo "=== ROS 2 Version ===" && ros2 --version
echo "=== Python Version ===" && python3 --version
echo "=== ROS 2 Installation ===" && ros2 pkg list | wc -l

# ROS 2 communication test
echo "=== ROS 2 Communication ===" && ros2 topic list
```

### Environment Variables

```bash
# Check ROS 2 environment
printenv | grep ROS

# Enable ROS 2 logging
export ROS_LOG_DIR=./ros_logs

# Enable colcon caching for faster builds
export COLCON_DEFAULTS_CONFIG_PATH=~/.colcon

# Disable IPC shared memory (if getting permission errors)
export ROS_LOCALHOST_ONLY=1
```

---

**Still have questions?** Join our community:
- [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
- [ROS 2 Discourse](https://discourse.ros.org/)
- [Contact the course](mailto:course@physical-ai.org)
