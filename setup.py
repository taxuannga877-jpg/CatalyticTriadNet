#!/usr/bin/env python3
"""
CatalyticTriadNet: Geometric Deep Learning for Enzyme Catalytic Site Identification
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="catalytic-triad-net",
    version="2.0.0",
    author="Your Name",
    author_email="your-email@example.com",
    description="Geometric Deep Learning Framework for Enzyme Catalytic Site Identification and Nanozyme Design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CatalyticTriadNet",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/CatalyticTriadNet/issues",
        "Documentation": "https://github.com/yourusername/CatalyticTriadNet#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "torch>=1.12.0",
        "biopython>=1.79",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "requests>=2.26.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "full": [
            "torch-geometric>=2.0.0",
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "catalytic-triad-net=catalytic_triad_net.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
