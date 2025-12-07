// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'AI-Native Course for NVIDIA Jetson Orin Nano',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-physical-ai-textbook.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'physical-ai-course', // Usually your GitHub org/user name.
  projectName: 'physical-ai-textbook', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/physical-ai-course/physical-ai-textbook/edit/main/',
          lastVersion: 'current',
          versions: {
            current: {
              label: 'Current',
            },
          },
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      metadata: [
        {
          name: 'description',
          content: 'Comprehensive AI-Native Textbook on Physical AI & Humanoid Robotics for NVIDIA Jetson Orin Nano',
        },
        {
          name: 'keywords',
          content: 'robotics, ROS2, physical AI, Jetson Orin Nano, humanoid, machine learning',
        },
      ],
      navbar: {
        title: 'Physical AI Textbook',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'index',
            position: 'left',
            label: 'Chapters',
          },
          {
            href: 'https://github.com/physical-ai-course/physical-ai-textbook',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Chapters',
            items: [
              {
                label: 'Chapter 1: Robotic Nervous System',
                to: '/chapter-1/1-1-foundations-pai',
              },
              {
                label: 'Chapter 2: Digital Twin',
                to: '/chapter-2/2-1-isaac-sim-intro',
              },
              {
                label: 'Chapter 3: AI-Robot Brain',
                to: '/chapter-3/3-1-reinforcement-learning',
              },
              {
                label: 'Chapter 4: Vision-Language-Action',
                to: '/chapter-4/4-1-llm-brain',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'ROS 2 Documentation',
                href: 'https://docs.ros.org/en/humble/',
              },
              {
                label: 'NVIDIA Isaac Sim',
                href: 'https://developer.nvidia.com/isaac-sim',
              },
              {
                label: 'Jetson Orin Nano',
                href: 'https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub Issues',
                href: 'https://github.com/physical-ai-course/physical-ai-textbook/issues',
              },
              {
                label: 'Discussions',
                href: 'https://github.com/physical-ai-course/physical-ai-textbook/discussions',
              },
            ],
          },
        ],
        copyright: `Copyright © 2025 Physical AI Course. Built with Docusaurus 2.4.3.`,
      },
      prism: {
        theme: lightCodeTheme, // <-- Use the variable defined at the top
        darkTheme: darkCodeTheme, // <-- Use the variable defined at the top
        additionalLanguages: ['python', 'bash', 'yaml', 'json'],
      },
    }),
};

export default config;
