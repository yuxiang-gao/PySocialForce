import pathlib
from setuptools import setup, find_packages

# read the contents of your README file
ROOT = pathlib.Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding = 'utf-8')

# extract version from __init__.py
with open(ROOT / "pysocialforce/__init__.py", "r", encoding = 'utf-8') as f:
    VERSION_LINE = [l for l in f if l.startswith("__version__")][0]
    VERSION = VERSION_LINE.split("=")[1].strip()[1:-1]


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    requirements = []
    with open(filename) as f:
        for line in f:
            if line and not line.startswith("#"):
                requirements.append(line.strip())
    return requirements


setup(
    name="PySocialForce",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    license="MIT",
    description="Numpy implementation of the Extended Social Force model.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Yuxiang Gao",
    author_email="yuxiang.gao@jhu.edu",
    url="https://github.com/yuxiang-gao/PySocialForce",
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": ["black", "jupyter"],
        "test": ["pylint", "pytest",],
        "plot": ["matplotlib",],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
