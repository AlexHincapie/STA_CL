from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="sta_cl",
    version="1.2.0",
    author="Dane Alexander Hincapié Arango",
    author_email="dane.hincapie@nabucod.com",
    description="Spacetime Algebra Cl(1,3) library in PyTorch with Grassmannian projections and Lie-group rotor integrators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexHincapie/STA_CL",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
    },
    python_requires=">=3.9",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
