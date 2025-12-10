import React from 'react';
import Link from '@docusaurus/Link';
import ProtectedLink from './Auth/ProtectedLink';
import styles from './Hero.module.css';

export default function Hero() {
  return (
    <div className={styles.heroContainer}>
      <div className={styles.heroBackground}>
        <div className={styles.gradientOrb1}></div>
        <div className={styles.gradientOrb2}></div>
        <div className={styles.gradientOrb3}></div>
      </div>

      <div className={styles.heroContent}>
        <div className={styles.badge}>
          <span className={styles.badgeIcon}>🤖</span>
          PANAVERSITY AI-NATIVE BOOK SERIES
        </div>

        <h1 className={styles.heroTitle}>
          Physical AI & Humanoid
          <br />
          <span className={styles.gradient}>Robotics Textbook</span>
        </h1>

        <p className={styles.heroSubtitle}>
          Master the future of robotics with hands-on Physical AI training.
          Build intelligent robots using NVIDIA Jetson Orin Nano, ROS 2, and cutting-edge AI.
        </p>

        <div className={styles.heroFeatures}>
          <div className={styles.featureItem}>
            <span className={styles.featureIcon}>✨</span>
            <span>13-Week Curriculum</span>
          </div>
          <div className={styles.featureItem}>
            <span className={styles.featureIcon}>🎯</span>
            <span>Hands-On Projects</span>
          </div>
          <div className={styles.featureItem}>
            <span className={styles.featureIcon}>🚀</span>
            <span>Industry-Ready Skills</span>
          </div>
        </div>

        <div className={styles.heroButtons}>
          <ProtectedLink
            className={styles.primaryButton}
            to="/docs/chapter-1/1-1-foundations-pai">
            Start Learning
            <span className={styles.buttonIcon}>→</span>
          </ProtectedLink>
          <ProtectedLink
            className={styles.secondaryButton}
            to="/docs/getting-started">
            View Curriculum
            <span className={styles.buttonIcon}>📚</span>
          </ProtectedLink>
        </div>

        <div className={styles.heroStats}>
          <div className={styles.statItem}>
            <div className={styles.statNumber}>4</div>
            <div className={styles.statLabel}>Chapters</div>
          </div>
          <div className={styles.statDivider}></div>
          <div className={styles.statItem}>
            <div className={styles.statNumber}>13</div>
            <div className={styles.statLabel}>Lessons</div>
          </div>
          <div className={styles.statDivider}></div>
          <div className={styles.statItem}>
            <div className={styles.statNumber}>13</div>
            <div className={styles.statLabel}>Weeks</div>
          </div>
          <div className={styles.statDivider}></div>
          <div className={styles.statItem}>
            <div className={styles.statNumber}>100+</div>
            <div className={styles.statLabel}>Code Examples</div>
          </div>
        </div>
      </div>
    </div>
  );
}
