# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Setup script for Anlord Abliterator package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="AnlordAbliterator",
    version="1.2.0",
    description="Anlord Abliterator — automated LLM abliteration and evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anlord Abliterator Development Team",
    author_email="anlord@example.com",
    url="https://github.com/justbedwarsplay/AnlordAbliterator",
    license="AGPL-3.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "ruff>=0.14.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "anlord=anlord.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: GPU",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="llm abliteration benchmark evaluation",
    project_urls={
        "Documentation": "https://github.com/anlord-project/anlord#readme",
        "Source": "https://github.com/anlord-project/anlord",
        "Tracker": "https://github.com/anlord-project/anlord/issues",
    },
)
