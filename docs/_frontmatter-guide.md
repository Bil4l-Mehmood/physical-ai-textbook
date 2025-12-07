---
sidebar_position: 999
sidebar_label: "Frontmatter Guide"
title: "Lesson Frontmatter Guide"
description: "How to properly configure lesson metadata for Docusaurus"
---

# Lesson Frontmatter Guide

Every lesson markdown file must begin with YAML frontmatter that configures metadata for Docusaurus and defines lesson properties.

## Complete Example

```yaml
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
```

## Field Descriptions

### `sidebar_position` (Required)

**Type**: Integer

**Purpose**: Controls lesson order within the chapter

**Example**:
- First lesson in chapter: `1`
- Second lesson in chapter: `2`
- Third lesson: `3`

**Note**: Docusaurus automatically sorts children by this value

### `sidebar_label` (Required)

**Type**: String

**Purpose**: Text displayed in the sidebar navigation

**Format**: `"Lesson X.X: Short Title"`

**Example**: `"Lesson 1.2: ROS 2 Core Concepts"`

**Note**: Keep to ~50 characters for sidebar readability

### `title` (Required)

**Type**: String

**Purpose**: Full lesson title displayed as page heading

**Format**: `"Full Title: Subtitle"` (up to ~60 characters)

**Example**: `"ROS 2 Core Concepts: Nodes, Topics, Services"`

### `description` (Required)

**Type**: String

**Purpose**: One-line summary shown in metadata and search

**Format**: Plain text, 1-2 sentences, ~100 characters max

**Example**: `"Learn ROS 2 publish/subscribe communication and build practical Python examples"`

### `duration` (Required)

**Type**: Integer

**Purpose**: Estimated time to complete lesson (in minutes)

**Values**:
- Beginner: 30-60 minutes
- Intermediate: 60-120 minutes
- Advanced: 90-180 minutes

**Example**: `60` (for 60 minutes)

**Guidelines**:
- Reading time: ~1 minute per 250 words
- Code examples: ~5 minutes each
- Hands-on exercise: ~20-30 minutes
- Total: Add reading + examples + exercise

### `difficulty` (Required)

**Type**: String (enumerated)

**Values**:
- `Beginner` - Assumes ROS 2 knowledge but not robotics
- `Intermediate` - Assumes Chapter 1 completion
- `Advanced` - Assumes Chapter 2+ completion

**Example**: `Beginner`

**Guidelines**:
- Chapter 1: Mostly Beginner/Intermediate
- Chapter 2-3: Intermediate/Advanced
- Chapter 4: Advanced

### `hardware` (Required)

**Type**: List of strings

**Purpose**: Hardware components needed for this lesson

**Values**: Specific model numbers and versions

**Examples**:
```yaml
# Theory only (no hardware needed)
hardware: ["None (theory)"]

# Jetson + ROS 2
hardware: ["Jetson Orin Nano", "ROS 2 Humble"]

# With specific camera
hardware: ["Jetson Orin Nano", "ROS 2 Humble", "Intel RealSense D435i"]

# Optional alternatives
hardware: ["Jetson Orin Nano", "ROS 2 Humble", "Isaac Sim 2024.1+ (optional)"]
```

**Guidelines**:
- Always include specific model/version
- Use format: `"Component Name X.Y.Z"`
- Mark alternatives with `(optional)` or `(alternative)`
- Use exact naming from official specs

### `prerequisites` (Required)

**Type**: List of strings

**Purpose**: Required lessons before taking this lesson

**Format**: `"Lesson X.X: Full Title"`

**Examples**:
```yaml
# First lesson in course
prerequisites: ["None"]

# Depends on prior lesson
prerequisites: ["Lesson 1.1: Foundations of Physical AI"]

# Depends on multiple lessons
prerequisites: ["Lesson 2.1: Isaac Sim Setup", "Lesson 2.2: Sim-to-Real"]
```

**Guidelines**:
- First lesson: `["None"]`
- Use exact lesson titles from corresponding `.md` files
- List in order of importance
- Validate links work in final review

## Valid Combinations

### Theory Lesson (No Hardware)

```yaml
---
sidebar_position: 1
sidebar_label: "Lesson 1.1: Foundations"
title: "Foundations of Physical AI & Embodied Intelligence"
description: "Learn the fundamentals of Physical AI and why robots need bodies"
duration: 45
difficulty: Beginner
hardware: ["None (theory)"]
prerequisites: ["None"]
---
```

### Hands-On ROS 2 Lesson

```yaml
---
sidebar_position: 2
sidebar_label: "Lesson 1.2: ROS 2 Core"
title: "ROS 2 Core Concepts: Nodes, Topics, Services"
description: "Write ROS 2 publishers and subscribers in Python"
duration: 60
difficulty: Beginner
hardware: ["Jetson Orin Nano", "ROS 2 Humble"]
prerequisites: ["Lesson 1.1: Foundations of Physical AI"]
---
```

### Advanced Multi-Hardware Lesson

```yaml
---
sidebar_position: 3
sidebar_label: "Lesson 3.2: Perception Fusion"
title: "Perception and Sensor Fusion (RealSense D435i)"
description: "Integrate RealSense depth camera with ROS 2 perception pipelines"
duration: 90
difficulty: Advanced
hardware: ["Jetson Orin Nano", "ROS 2 Humble", "Intel RealSense D435i"]
prerequisites: ["Lesson 3.1: Reinforcement Learning", "Lesson 2.2: Sim-to-Real"]
---
```

### Simulation-Based Lesson with Optional Hardware

```yaml
---
sidebar_position: 1
sidebar_label: "Lesson 2.1: Isaac Sim"
title: "Introduction to Simulation: Isaac Sim & Gazebo"
description: "Learn simulation-to-reality workflows"
duration: 120
difficulty: Intermediate
hardware: ["Jetson Orin Nano (optional)", "Isaac Sim 2024.1+ or Gazebo 2"]
prerequisites: ["Lesson 1.4: URDF/XACRO Basics"]
---
```

## Validation Checklist

Before publishing a lesson, verify:

- [ ] `sidebar_position` is unique within chapter (1, 2, 3...)
- [ ] `sidebar_label` matches pattern `"Lesson X.X: Title"`
- [ ] `title` is descriptive and specific
- [ ] `description` is 1-2 sentences, ~100 chars
- [ ] `duration` is realistic estimate (verify with actual reading/hands-on time)
- [ ] `difficulty` is Beginner|Intermediate|Advanced (no typos)
- [ ] `hardware` lists specific models (not generic "GPU" or "camera")
- [ ] `prerequisites` use exact lesson titles from actual files
- [ ] All referenced lessons exist in the textbook

## Common Mistakes

❌ **Wrong**: `hardware: ["Any GPU"]` - Too vague
✅ **Right**: `hardware: ["NVIDIA Jetson Orin Nano"]` - Specific model

❌ **Wrong**: `difficulty: "Hard"` - Invalid value
✅ **Right**: `difficulty: Advanced` - Correct enumeration

❌ **Wrong**: `sidebar_label: "ROS 2 Core Concepts"` - Missing lesson number
✅ **Right**: `sidebar_label: "Lesson 1.2: ROS 2 Core Concepts"` - Includes lesson number

❌ **Wrong**: `prerequisites: ["Chapter 1"]` - Too vague
✅ **Right**: `prerequisites: ["Lesson 1.1: Foundations of Physical AI"]` - Specific lesson

❌ **Wrong**: `duration: "1 hour"` - String instead of integer
✅ **Right**: `duration: 60` - Integer (minutes)

## Updating Frontmatter

When updating a lesson:

1. Keep `sidebar_position` the same (don't change order)
2. Update `description` if major content changes
3. Update `duration` if content significantly changes
4. Keep `difficulty` consistent with chapter context
5. Add to `prerequisites` if content now depends on new lessons
6. Update `hardware` if new equipment added

## Tool Support

The frontmatter is used by:

- **Docusaurus**: Renders sidebar, metadata, page title
- **Search**: Uses `description` for search results
- **Validation scripts**: Checks hardware specs, duration realism
- **Link validators**: Verifies prerequisites exist

---

**See also**: [_lesson-template.md](_lesson-template.md) for complete lesson structure
