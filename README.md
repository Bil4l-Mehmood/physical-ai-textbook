# Physical AI & Humanoid Robotics Textbook

A comprehensive AI-Native course for learning Physical AI and Humanoid Robotics on NVIDIA Jetson Orin Nano.

## Quick Start

### Prerequisites

- Node.js 14.0 or later
- npm 6.0 or later

### Build and Serve Locally

```bash
# Install dependencies
npm install

# Start development server (http://localhost:3000)
npm start

# Build for production
npm run build

# Serve built site
npm run serve
```

### Build Output

The production build is generated in the `build/` directory.

## Project Structure

```
docusaurus-textbook/
├── docs/                    # Markdown lesson content
│   ├── index.md            # Homepage
│   ├── chapter-1/          # Chapter 1: Robotic Nervous System
│   ├── chapter-2/          # Chapter 2: Digital Twin
│   ├── chapter-3/          # Chapter 3: AI-Robot Brain
│   └── chapter-4/          # Chapter 4: Vision-Language-Action
├── static/                 # Static assets (images, downloads)
├── src/                    # Custom components and styling
├── docusaurus.config.js    # Docusaurus configuration
├── sidebars.js             # Sidebar navigation structure
└── package.json            # Node.js dependencies
```

## Deployment

### GitHub Pages

Update `docusaurus.config.js` with your GitHub organization and project name:

```javascript
organizationName: 'your-org', // Usually your GitHub org/user name.
projectName: 'physical-ai-textbook', // Usually your repo name.
```

Then deploy:

```bash
npm run deploy
```

### Vercel

Deploy the `build/` directory to Vercel:

1. Connect your GitHub repository to Vercel
2. Set build command: `npm run build`
3. Set output directory: `build`

### Self-Hosted

Build the site and serve the `build/` directory with any static web server:

```bash
npm run build
# Serve build/ with nginx, Apache, or Python:
cd build
python -m http.server 8080
```

## Contributing

1. Create a feature branch: `git checkout -b feature/lesson-name`
2. Write lesson content in Markdown
3. Test locally: `npm start`
4. Submit a pull request

### Lesson Writing Guidelines

See `/docs/_lesson-template.md` for the standard lesson structure and conventions.

### Code Example Standards

All code examples MUST:
- Be executable on Jetson Orin Nano without modification
- Include inline comments
- Have documented expected output
- Include error handling guidance
- Target ROS 2 Humble (or specified version)

## Versioning

This textbook uses semantic versioning:

- **MAJOR**: Significant curriculum changes or new modules
- **MINOR**: New lessons or major lesson updates
- **PATCH**: Typo fixes, small clarifications, code updates

## License

MIT License - See LICENSE file for details

## Authors

Physical AI Course Team

## Support

- 📖 [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- 🏗️ [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- 💻 [Jetson Orin Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

---

For more information, visit the [textbook homepage](https://your-physical-ai-textbook.com)
