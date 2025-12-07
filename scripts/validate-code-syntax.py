#!/usr/bin/env python3
"""
Code Syntax Validator for Lesson Examples

Validates Python code examples in lesson markdown files:
- Checks Python syntax correctness
- Ensures required headers and structure
- Validates code example formatting

Usage:
    python3 validate-code-syntax.py
    python3 validate-code-syntax.py path/to/lessons/
"""

import os
import sys
import re
import py_compile
from pathlib import Path


class CodeValidator:
    """Validates code examples in lesson markdown files."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = 0

    def validate_file(self, filepath):
        """Validate a single markdown file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"{filepath}: Failed to read file - {e}")
            return

        # Extract Python code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)

        if not code_blocks:
            return

        print(f"Validating {filepath}...")

        for i, code in enumerate(code_blocks, 1):
            self._validate_code_block(code, filepath, i)

    def _validate_code_block(self, code, filepath, block_num):
        """Validate a single code block."""
        try:
            # Write to temporary file
            temp_file = f"/tmp/code_block_{block_num}.py"
            with open(temp_file, 'w') as f:
                f.write(code)

            # Check syntax
            py_compile.compile(temp_file, doraise=True)

            # Validate structure
            self._validate_structure(code, filepath, block_num)

            # Cleanup
            os.remove(temp_file)

            self.passed += 1
            print(f"  ✅ Code block {block_num} - Syntax OK")

        except py_compile.PyCompileError as e:
            self.errors.append(f"{filepath} block {block_num}: Syntax error - {e}")
            print(f"  ❌ Code block {block_num} - Syntax error")

    def _validate_structure(self, code, filepath, block_num):
        """Validate code structure and best practices."""
        checks = [
            (r'#!/usr/bin/env python3', f"{filepath} block {block_num}: Missing shebang"),
            (r'# Copyright.*2025', f"{filepath} block {block_num}: Missing/invalid copyright"),
            (r'# Target:.*Orin.*Nano', f"{filepath} block {block_num}: Missing hardware target"),
        ]

        # Only check for shebang if block starts like a script
        if code.strip().startswith('#!/'):
            for pattern, warning in checks:
                if not re.search(pattern, code):
                    self.warnings.append(warning)

    def validate_directory(self, directory='docs'):
        """Validate all markdown files in directory."""
        print(f"Scanning {directory} for lessons...")

        for filepath in Path(directory).rglob('*.md'):
            # Skip template files
            if filepath.name.startswith('_'):
                continue

            self.validate_file(str(filepath))

    def report(self):
        """Print validation report."""
        print("\n" + "="*60)
        print("CODE SYNTAX VALIDATION REPORT")
        print("="*60)

        print(f"\n✅ Passed: {self.passed}")

        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
            return False
        else:
            print("\n✅ All code examples passed validation!")
            return True


def main():
    """Main entry point."""
    validator = CodeValidator()

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
