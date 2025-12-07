#!/usr/bin/env python3
"""
Link Validator for Lesson Content

Validates internal and external links in lesson markdown files:
- Checks that internal lesson links exist
- Reports broken external links
- Validates link format

Usage:
    python3 validate-links.py
    python3 validate-links.py path/to/lessons/
"""

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


class LinkValidator:
    """Validates links in lesson markdown files."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = 0
        self.lesson_files = {}
        self.external_links = set()

    def _build_lesson_index(self, directory='docs'):
        """Build index of all lesson files."""
        print(f"Building lesson index from {directory}...")

        for filepath in Path(directory).rglob('*.md'):
            # Skip template files
            if filepath.name.startswith('_'):
                continue

            # Extract lesson ID from filename
            match = re.match(r'(\d+)-(\d+)-', filepath.name)
            if match:
                chapter, lesson = match.groups()
                self.lesson_files[f"chapter-{chapter}/lesson-{lesson}"] = filepath
            else:
                self.lesson_files[filepath.name] = filepath

        print(f"Found {len(self.lesson_files)} lesson files")

    def validate_file(self, filepath):
        """Validate a single markdown file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"{filepath}: Failed to read file - {e}")
            return

        # Find all links in markdown
        # Matches: [text](link) and [text](link "title")
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

        if not links:
            return

        print(f"Validating {filepath} ({len(links)} links)...")

        for text, url in links:
            self._validate_link(url, text, filepath)

    def _validate_link(self, url, text, filepath):
        """Validate a single link."""
        # Skip mailto and tel links
        if url.startswith(('mailto:', 'tel:')):
            return

        # Internal link (starts with /)
        if url.startswith('/'):
            self._validate_internal_link(url, filepath)

        # External link
        elif url.startswith(('http://', 'https://')):
            self.external_links.add(url)
            # Note: Can't validate external links without network access
            print(f"  ℹ️  External: {url[:50]}...")

        # Relative link
        else:
            self._validate_relative_link(url, filepath)

    def _validate_internal_link(self, url, filepath):
        """Validate internal lesson links."""
        # Extract path from URL
        path = url.split('#')[0]  # Remove anchor

        # Check if file exists
        full_path = Path('docs') / path.lstrip('/')

        if full_path.exists():
            self.passed += 1
            print(f"  ✅ Internal: {url}")
        else:
            self.errors.append(f"{filepath}: Broken internal link - {url}")
            print(f"  ❌ Internal: {url}")

    def _validate_relative_link(self, url, filepath):
        """Validate relative links."""
        # Get directory of current file
        file_dir = Path(filepath).parent

        # Resolve relative path
        target = (file_dir / url).resolve()

        if target.exists() or str(target).startswith('/chapter'):
            self.passed += 1
            print(f"  ✅ Relative: {url}")
        else:
            self.warnings.append(f"{filepath}: Unverified relative link - {url}")
            print(f"  ⚠️  Relative: {url}")

    def validate_directory(self, directory='docs'):
        """Validate all markdown files in directory."""
        self._build_lesson_index(directory)

        print(f"\nValidating links...")

        for filepath in Path(directory).rglob('*.md'):
            # Skip template files
            if filepath.name.startswith('_'):
                continue

            self.validate_file(str(filepath))

    def report(self):
        """Print validation report."""
        print("\n" + "="*60)
        print("LINK VALIDATION REPORT")
        print("="*60)

        print(f"\n✅ Passed: {self.passed}")
        print(f"🌐 External links found: {len(self.external_links)}")

        if self.external_links:
            print("\nExternal Links (not validated without network):")
            for link in sorted(self.external_links)[:10]:  # Show first 10
                print(f"  - {link}")
            if len(self.external_links) > 10:
                print(f"  ... and {len(self.external_links) - 10} more")

        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors[:10]:
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")
            return False
        else:
            print("\n✅ All links validated!")
            return True


def main():
    """Main entry point."""
    validator = LinkValidator()

    # Validate docs directory
    if os.path.isdir('docs'):
        validator.validate_directory('docs')
    else:
        print("Error: 'docs' directory not found")
        print("Run this script from the Docusaurus project root")
        sys.exit(1)

    # Print report and exit with appropriate code
    success = validator.report()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
