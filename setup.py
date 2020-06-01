from setuptools import setup, find_packages

# extract version from __init__.py
with open("socialforce/__init__.py", "r") as f:
    VERSION_LINE = [l for l in f if l.startswith("__version__")][0]
    VERSION = VERSION_LINE.split("=")[1].strip()[1:-1]


setup(
    name="extended-social-force",
    version=VERSION,
    packages=find_packages(),
    license="MIT",
    description="Numpy implementation of the Extended Social Force model.",
    long_description=open("README.rst").read(),
    author="Sven Kreiss, Yuxiang Gao",
    author_email="me@svenkreiss.com, yuxiang.gao@jhu.edu",
    url="https://github.com/yuxiang-gao/socialforce",
    install_requires=["numpy", "scipy", "numba"],
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
