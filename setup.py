"""
Setup file for SHAPE package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="shape-xai",
    version="1.0.0",
    author="Prithwijit Chowdhury, Mohit Prabhushankar, Ghassan AlRegib, Mohamed Deriche",
    author_email="pchowdhury6@gatech.edu",
    description="SHAPE: SHifted Adversaries using Pixel Elimination - Adversarial Explanations for XAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SHAPE-adversarial-explanations",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.19.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=6.0.0",
            "black>=21.0",
        ],
        "gradcam": [
            "pytorch-grad-cam>=1.4.0",
        ],
    },
)
