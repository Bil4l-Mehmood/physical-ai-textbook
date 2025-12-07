---
sidebar_position: 0
sidebar_label: Home
title: Physical AI & Humanoid Robotics Textbook
description: A comprehensive AI-Native course for building intelligent robots with NVIDIA Jetson Orin Nano
---

# Welcome to the Physical AI & Humanoid Robotics Textbook

This is a comprehensive, production-ready textbook for learning **Physical AI and Humanoid Robotics** on the **NVIDIA Jetson Orin Nano**. The course is designed for hands-on learning with ROS 2 middleware, NVIDIA Isaac Sim for simulation, and cutting-edge AI techniques including reinforcement learning and large language models.

## Course Overview

**Duration**: 13 weeks | **4 Modules** | **13 Lessons**

This textbook covers the complete journey from robotics fundamentals to advanced AI-robot integration:

### Chapter 1: The Robotic Nervous System (Weeks 1-5)
Learn the foundations of Physical AI and how robots communicate using ROS 2 middleware.

- **Lesson 1.1**: Foundations of Physical AI & Embodied Intelligence
- **Lesson 1.2**: ROS 2 Core Concepts: Nodes, Topics, Services
- **Lesson 1.3**: Building rclpy Packages and Launch Files
- **Lesson 1.4**: Defining the Physical Body: URDF/XACRO

### Chapter 2: The Digital Twin (Weeks 6-7)
Understand simulation-to-reality workflows using NVIDIA Isaac Sim.

- **Lesson 2.1**: Introduction to Simulation: Isaac Sim & Gazebo
- **Lesson 2.2**: Sim-to-Real: Bridging the Reality Gap
- **Lesson 2.3**: Robot Physics and Kinematics

### Chapter 3: The AI-Robot Brain (Weeks 8-10)
Integrate perception and deep learning for autonomous robot behavior.

- **Lesson 3.1**: Deep Reinforcement Learning for Locomotion
- **Lesson 3.2**: Perception and Sensor Fusion (RealSense D435i)
- **Lesson 3.3**: Computer Vision with Isaac ROS

### Chapter 4: Vision-Language-Action (Weeks 11-13)
Build intelligent robot systems that understand and respond to natural language.

- **Lesson 4.1**: LLMs as the Robot's Cognitive Core
- **Lesson 4.2**: Conversational AI & VLA Agents (ReSpeaker Mic)
- **Lesson 4.3**: HRI, Safety, and Ethical Frameworks

## Target Hardware

This textbook is specifically designed for:

- **Compute**: NVIDIA Jetson Orin Nano (8GB or 4GB)
- **Simulation**: NVIDIA Isaac Sim 2024.1+
- **Middleware**: ROS 2 Humble
- **Perception**: Intel RealSense D435i (depth + color camera)
- **Audio**: ReSpeaker USB Mic Array v2.0
- **Development**: Ubuntu 22.04 LTS, Python 3.10+

## Learning Approach

Each lesson includes:

✅ **Clear Learning Objectives** - Specific, measurable outcomes you'll achieve
✅ **Hardware & Prerequisites** - What you need before starting
✅ **Theoretical Foundations** - Concepts explained with diagrams
✅ **Executable Code Examples** - Python code you can run on Jetson Orin Nano
✅ **Hands-On Exercises** - Practical activities to reinforce learning
✅ **Common Errors & Fixes** - Troubleshooting guidance
✅ **Key Takeaways** - Summary of essential concepts

## Getting Started

1. **Start with Chapter 1, Lesson 1.1** if you're new to robotics and Physical AI
2. **Review the Hardware Setup Guide** to prepare your Jetson Orin Nano
3. **Work through lessons sequentially** - each lesson builds on prior knowledge
4. **Run all code examples** on your hardware to maximize learning

## Prerequisites

This course assumes:

- **Intermediate Python knowledge** (comfortable with classes, modules, imports)
- **Basic understanding of robotics** (coordinate frames, actuators, sensors)
- **Linux familiarity** (command line, file system, environment variables)
- **Access to course hardware** (Jetson Orin Nano + peripherals)

If you're new to these topics, supplementary resources are linked throughout the textbook.

## How to Use This Textbook

### For Students

- Use the **Chapters** navigation on the left sidebar to browse lessons
- Read each lesson fully before attempting code examples
- Run code examples on your Jetson Orin Nano to verify understanding
- Complete hands-on exercises before moving to the next lesson
- Refer to "Common Errors & Fixes" if you encounter issues

### For Instructors

- Each lesson is independently testable (can be taught in any order with prerequisites noted)
- Code examples are peer-reviewed and verified on Jetson Orin Nano hardware
- Estimated time per lesson is documented (typically 45-120 minutes)
- Difficulty levels (Beginner/Intermediate/Advanced) help with pacing

## Support & Contributing

- 🐛 **Report issues** on [GitHub Issues](https://github.com/physical-ai-course/physical-ai-textbook/issues)
- 💬 **Ask questions** on [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
- 📝 **Contribute improvements** via [GitHub Pull Requests](https://github.com/physical-ai-course/physical-ai-textbook)

## Licensing & Attribution

This textbook is distributed under the **MIT License**. You are free to:
- Use it for learning and teaching
- Modify and adapt content for your needs
- Share and distribute with attribution

See the LICENSE file for full details.

## Quick Links

- 🤖 [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- 🏗️ [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- 💻 [Jetson Orin Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
- 📷 [Intel RealSense SDK](https://www.intelrealsense.com/sdk-2/)
- 🎙️ [ReSpeaker Documentation](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/)

## Acknowledgments

This textbook was developed with careful attention to:

- ✅ **Academic rigor** - Content verified against official documentation
- ✅ **Reproducibility** - All examples tested on Jetson Orin Nano hardware
- ✅ **Hardware specificity** - No generic abstractions; concrete examples for your hardware
- ✅ **Professional standards** - ROS 2 conventions and best practices throughout
- ✅ **Peer review** - All content reviewed before publication

---

**Ready to begin?** Start with [Chapter 1, Lesson 1.1: Foundations of Physical AI](/docs/chapter-1/1-1-foundations-pai)

**Questions?** Check the [Hardware Setup Guide](./hardware-setup-guide) or [FAQ](./faq) (coming soon)

---

*Last Updated: 2025-12-05*
*Docusaurus Version: 2.4.3*
*ROS 2 Version: Humble (LTS)*
