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
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "torch>=2.5.0",  # RTX 50 Series requires PyTorch 2.5+
        "biopython>=1.81",
        "matplotlib>=3.7.0",
        "networkx>=3.1",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "full": [
            "torch-geometric>=2.5.0",
            "torch-scatter>=2.1.2",
            "torch-sparse>=0.6.18",
            "torch-cluster>=1.6.3",
            "torch-spline-conv>=1.2.2",
            "plotly>=5.14.0",
            "seaborn>=0.12.0",
            "rdkit>=2023.3.1",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
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
