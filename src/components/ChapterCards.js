import React from 'react';
import Link from '@docusaurus/Link';
import styles from './ChapterCards.module.css';

const chapters = [
  {
    number: '01',
    title: 'The Robotic Nervous System',
    description: 'Learn the foundations of Physical AI and how robots communicate using ROS 2 middleware.',
    duration: 'Weeks 1-5',
    lessons: 4,
    icon: '🧬',
    color: '#667eea',
    link: '/docs/chapter-1/1-1-foundations-pai',
    topics: ['Physical AI Foundations', 'ROS 2 Core Concepts', 'Building rclpy Packages', 'URDF/XACRO'],
  },
  {
    number: '02',
    title: 'The Digital Twin',
    description: 'Understand simulation-to-reality workflows using NVIDIA Isaac Sim.',
    duration: 'Weeks 6-7',
    lessons: 3,
    icon: '🌐',
    color: '#f093fb',
    link: '/docs/chapter-2/2-1-gazebo-fundamentals',
    topics: ['Isaac Sim & Gazebo', 'Sim-to-Real Transfer', 'Robot Physics & Kinematics'],
  },
  {
    number: '03',
    title: 'The AI-Robot Brain',
    description: 'Integrate perception and deep learning for autonomous robot behavior.',
    duration: 'Weeks 8-10',
    lessons: 3,
    icon: '🧠',
    color: '#4facfe',
    link: '/docs/chapter-3/3-1-isaac-sim-basics',
    topics: ['Deep Reinforcement Learning', 'Sensor Fusion', 'Computer Vision with Isaac ROS'],
  },
  {
    number: '04',
    title: 'Vision-Language-Action',
    description: 'Build intelligent robot systems that understand and respond to natural language.',
    duration: 'Weeks 11-13',
    lessons: 3,
    icon: '💬',
    color: '#43e97b',
    link: '/docs/chapter-4/4-1-llm-brain',
    topics: ['LLMs for Robotics', 'Conversational AI & VLA', 'HRI & Safety Frameworks'],
  },
];

export default function ChapterCards() {
  return (
    <div className={styles.chaptersSection}>
      <div className={styles.chapterBackground}>
        <div className={styles.gradientOrb1Chapter}></div>
        <div className={styles.gradientOrb2Chapter}></div>
        <div className={styles.gradientOrb3Chapter}></div>
      </div>

      <div className={styles.sectionHeader}>
        <h2 className={styles.sectionTitle}>Learning Roadmap</h2>
        <p className={styles.sectionSubtitle}>
          Four comprehensive chapters taking you from basics to advanced Physical AI
        </p>
      </div>

      <div className={styles.chaptersGrid}>
        {chapters.map((chapter, index) => (
          <Link
            key={index}
            to={chapter.link}
            className={styles.chapterCard}
            style={{ animationDelay: `${index * 0.15}s` }}>
            <div className={styles.cardHeader}>
              <div
                className={styles.chapterIcon}
                style={{ background: chapter.color }}>
                {chapter.icon}
              </div>
              <div className={styles.chapterMeta}>
                <div className={styles.chapterNumber}>Chapter {chapter.number}</div>
                <div className={styles.chapterDuration}>{chapter.duration}</div>
              </div>
            </div>

            <h3 className={styles.chapterTitle}>{chapter.title}</h3>
            <p className={styles.chapterDescription}>{chapter.description}</p>

            <div className={styles.topicsList}>
              {chapter.topics.map((topic, idx) => (
                <div key={idx} className={styles.topicItem}>
                  <span className={styles.topicBullet}>•</span>
                  {topic}
                </div>
              ))}
            </div>

            <div className={styles.cardFooter}>
              <div className={styles.lessonCount}>
                <span className={styles.lessonIcon}>📖</span>
                {chapter.lessons} Lessons
              </div>
              <div className={styles.arrowIcon}>→</div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
